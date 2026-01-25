#include "common.h"

template <
  int BLOCK_Q,
  int BLOCK_KV,
  int NUM_WARPS,
  int QK_DIM,
  int V_DIM,
  typename Type
>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attn_sm80_kernel(
  const Type *Q_gmem,  // [bs, q_len, q_heads, QK_DIM]
  const Type *K_gmem,  // [bs, kv_len, kv_heads, QK_DIM]
  const Type *V_gmem,  // [bs, kv_len, kv_heads, V_DIM]
        Type *O_gmem,  // [bs, q_len, q_heads, V_DIM]
  // strides
  int Q_s0, int Q_s1, int Q_s2,
  int K_s0, int K_s1, int K_s2,
  int V_s0, int V_s1, int V_s2,
  int O_s0, int O_s1, int O_s2,
  // problem shape
  int q_len,
  int kv_len,
  int q_heads,
  int kv_heads
) {
  static_assert(cuda::std::is_same_v<Type, nv_bfloat16>
              || cuda::std::is_same_v<Type, half>);

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int q_head_id = blockIdx.x;
  const int q_block_id = blockIdx.y;
  const int bs_id = blockIdx.z;

  const int kv_head_id = q_head_id / (q_heads / kv_heads);

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // m16n8k16
  constexpr int MMA_M = SM80::MMA_M;
  constexpr int MMA_N = SM80::MMA_N;
  constexpr int MMA_K = 16;

  const int q_offset = q_block_id * BLOCK_Q;
  Q_gmem += (bs_id * Q_s0) + (q_offset * Q_s1) + (q_head_id * Q_s2);
  O_gmem += (bs_id * O_s0) + (q_offset * O_s1) + (q_head_id * O_s2);
  K_gmem += (bs_id * K_s0) + (kv_head_id * K_s2);
  V_gmem += (bs_id * V_s0) + (kv_head_id * V_s2);

  // we overlap Q_smem with (K_smem + V_smem)
  extern __shared__ char smem[];
  const int Q_smem = __cvta_generic_to_shared(smem);

  // double buffer for K
  const int K_smem = Q_smem;
  const int V_smem = K_smem + 2 * BLOCK_KV * QK_DIM * sizeof(Type);

  // pre-compute address and swizzling for ldmatrix
  //   swizzle<STRIDE in bytes>(row, col in 16-byte unit)
  const int Q_smem_thread = Q_smem + swizzle<QK_DIM * sizeof(Type)>(warp_id * WARP_Q + (lane_id % 16), lane_id / 16);  // A tile
  const int K_smem_thread = K_smem + swizzle<QK_DIM * sizeof(Type)>(lane_id % 8, lane_id / 8);  // B tile
  const int V_smem_thread = V_smem + swizzle<QK_DIM * sizeof(Type)>(lane_id % 16, lane_id / 16);  // B tile trans

  // set up registers
  int Q_rmem[WARP_Q / MMA_M][QK_DIM / MMA_K][4];
  int K_rmem[BLOCK_KV / MMA_N][QK_DIM / MMA_K][2];
  float scale_Q_rmem[WARP_Q / MMA_M][2];
  float scale_K_rmem[BLOCK_KV / MMA_N][2];

  // let compiler decide register reuse?
  int P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  int V_rmem[BLOCK_KV / MMA_K][V_DIM / MMA_N][2];

  // we use the same registers for O_regs and PV_regs
  // rescale O_regs once we obtain new rowmax, then accumulate to O_regs
  float O_rmem[WARP_Q / MMA_M][V_DIM / MMA_N][4] = {};

  // exp(x) = exp(log(2) * x / log(2)) = exp2(x / log2)
  const float softmax_scale = rsqrtf(static_cast<float>(QK_DIM)) * 1.4426950408889634f;

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int q = 0; q < WARP_Q / MMA_M; q++) {
    rowmax[q][0] = -3.4e38;  // ~FLT_MAX
    rowmax[q][1] = -3.4e38;
  }

  // Q gmem->smem
  cp_async_g2s_swizzle<BLOCK_Q, QK_DIM, TB_SIZE>(Q_smem, Q_gmem, Q_s1, tid);
  asm volatile("cp.async.commit_group;\n");

  // Q smem->rmem
  asm volatile("cp.async.wait_all;\n");
  __syncthreads();
  for (int q = 0; q < WARP_Q / MMA_M; q++)
    for (int d = 0; d < QK_DIM / MMA_K; d++) {
      int addr = Q_smem_thread;
      addr += q * MMA_M * QK_DIM * sizeof(Type);  // row
      addr ^= d * MMA_K * sizeof(Type);  // col
      ldmatrix<4>(Q_rmem[q][d], addr);
    }

  const int num_kv_iter = cdiv(kv_len, BLOCK_KV);

  // TODO: more flexible prefetch
  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      const int dst = K_smem + (kv_id % 2) * (BLOCK_KV * QK_DIM) * sizeof(Type);
      cp_async_g2s_swizzle<BLOCK_KV, QK_DIM, TB_SIZE>(dst, K_gmem, K_s1, tid);
      K_gmem += BLOCK_KV * K_s1;
    }
    asm volatile("cp.async.commit_group;\n");
  };
  auto load_V = [&](int kv_id) {
    cp_async_g2s_swizzle<BLOCK_KV, V_DIM, TB_SIZE>(V_smem, V_gmem, V_s1, tid);
    V_gmem += BLOCK_KV * V_s1;
    asm volatile("cp.async.commit_group;\n");
  };

  // prefetch K
  __syncthreads();  // make sure finish Q smem->rmem
  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // prefetch V
    __syncthreads();  // make sure finish V from prev iteration
    load_V(kv_id);

    // K smem->rmem
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int kv = 0; kv < BLOCK_KV / MMA_N; kv++)
      for (int d = 0; d < QK_DIM / MMA_K; d += 2) {
        int addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * QK_DIM) * sizeof(Type);
        addr += kv * MMA_N * QK_DIM * sizeof(Type);  // row
        addr ^= d * MMA_K * sizeof(Type);  // col
        ldmatrix<4>(K_rmem[kv][d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int q = 0; q < WARP_Q / MMA_M; q++)
      for (int kv = 0; kv < BLOCK_KV / MMA_N; kv++)
        for (int d = 0; d < QK_DIM / MMA_K; d++)
          mma<Type, Type, float>(Q_rmem[q][d], K_rmem[kv][d], S_rmem[q][kv]);

    // prefetch K
    // do we need sync here?
    load_K(kv_id + 1);

    // softmax
    for (int q = 0; q < WARP_Q / MMA_M; q++) {
      // apply softmax scale
      for (int kv = 0; kv < BLOCK_KV / MMA_N; kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[q][kv][reg_id] *= softmax_scale;

      // rowmax
      float this_rowmax[2];

      // unroll 1st iteration
      this_rowmax[0] = max(S_rmem[q][0][0], S_rmem[q][0][1]);
      this_rowmax[1] = max(S_rmem[q][0][2], S_rmem[q][0][3]);

      for (int kv = 1; kv < BLOCK_KV / MMA_N; kv++) {
        float *regs = S_rmem[q][kv];
        this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
        this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = exp2f(rowmax[q][0] - this_rowmax[0]);
      rescale[1] = exp2f(rowmax[q][1] - this_rowmax[1]);
      for (int d = 0; d < V_DIM / MMA_N; d++) {
        O_rmem[q][d][0] *= rescale[0];
        O_rmem[q][d][1] *= rescale[0];
        O_rmem[q][d][2] *= rescale[1];
        O_rmem[q][d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[q][0] = this_rowmax[0];
      rowmax[q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      for (int kv = 0; kv < BLOCK_KV / MMA_N; kv++) {
        float *regs = S_rmem[q][kv];
        float c0 = exp2f(regs[0] - rowmax[q][0]);
        float c1 = exp2f(regs[1] - rowmax[q][0]);
        float c2 = exp2f(regs[2] - rowmax[q][1]);
        float c3 = exp2f(regs[3] - rowmax[q][1]);

        if (kv == 0) {
          this_rowsumexp[0] = c0 + c1;
          this_rowsumexp[1] = c2 + c3;
        } else {
          this_rowsumexp[0] += c0 + c1;
          this_rowsumexp[1] += c2 + c3;
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        // equivalent to __float22bfloat162_rn() / __float22half2_rn()
        asm volatile("cvt.rn.satfinite.%6x2.f32 %0, %2, %3;\n"
                     "cvt.rn.satfinite.%6x2.f32 %1, %4, %5;\n"
                    : "=r"(P_rmem[q][kv / 2][(kv % 2) * 2]),
                      "=r"(P_rmem[q][kv / 2][(kv % 2) * 2 + 1])
                    : "f"(c1), "f"(c0), "f"(c3), "f"(c2),
                      "C"(Type_str<Type>::value));
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[q][0] = rowsumexp[q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[q][1] = rowsumexp[q][1] * rescale[1] + this_rowsumexp[1];
    }

    // V smem->rmem
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int kv = 0; kv < BLOCK_KV / MMA_K; kv++)
      for (int d = 0; d < V_DIM / MMA_N; d += 2) {
        int addr = V_smem_thread;
        addr += kv * MMA_K * V_DIM * sizeof(Type);  // row
        addr ^= d * MMA_N * sizeof(Type);  // col
        ldmatrix<4, TRANS>(V_rmem[kv][d], addr);
      }

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int q = 0; q < WARP_Q / MMA_M; q++)
      for (int d = 0; d < V_DIM / MMA_N; d++)
        for (int kv = 0; kv < BLOCK_KV / MMA_K; kv++)
          mma<Type, Type, float>(P_rmem[q][kv], V_rmem[kv][d], O_rmem[q][d]);
  }

  // write to O
  for (int q = 0; q < WARP_Q / MMA_M; q++)
    for (int d = 0; d < V_DIM / MMA_N; d++) {
      const int row = warp_id * WARP_Q + (q * MMA_M) + (lane_id / 4);
      const int col = (d * MMA_N) + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[q][d];
      regs[0] /= rowsumexp[q][0];
      regs[1] /= rowsumexp[q][0];
      regs[2] /= rowsumexp[q][1];
      regs[3] /= rowsumexp[q][1];

      if constexpr (cuda::std::is_same_v<Type, nv_bfloat16>) {
        reinterpret_cast<nv_bfloat162 *>(O_gmem + (row + 0) * O_s1 + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
        reinterpret_cast<nv_bfloat162 *>(O_gmem + (row + 8) * O_s1 + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
      }
      else if constexpr (cuda::std::is_same_v<Type, half>) {
        reinterpret_cast<half2 *>(O_gmem + (row + 0) * O_s1 + col)[0] = __float22half2_rn({regs[0], regs[1]});
        reinterpret_cast<half2 *>(O_gmem + (row + 8) * O_s1 + col)[0] = __float22half2_rn({regs[2], regs[3]});
      }
    }
}
