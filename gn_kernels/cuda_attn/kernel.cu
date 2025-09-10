#include "common.h"

constexpr int QK_DIM = 128;
constexpr int V_DIM = 128;
using Type = nv_bfloat16;

constexpr int BLOCK_Q = 64;
constexpr int BLOCK_KV = 64;
constexpr int NUM_WARPS = 4;

// start of kernel
extern "C"
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attn_kernel(
  const Type *Q_gmem,  // [bs, len_q, num_heads, QK_DIM]
  const Type *K_gmem,  // [bs, len_kv, num_heads, QK_DIM]
  const Type *V_gmem,  // [bs, len_kv, num_heads, V_DIM]
        Type *O_gmem,  // [bs, len_q, num_heads, V_DIM]
  // strides
  int Q_s0, int Q_s1, int Q_s2,
  int K_s0, int K_s1, int K_s2,
  int V_s0, int V_s1, int V_s2,
  int O_s0, int O_s1, int O_s2,
  // problem shape
  int bs,
  int len_q,
  int len_kv,
  int num_heads
) {
  static_assert(cuda::std::is_same_v<Type, nv_bfloat16>
              || cuda::std::is_same_v<Type, half>);

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int q_block_id = blockIdx.x;
  const int head_id = blockIdx.y;
  const int bs_id = blockIdx.z;

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // m16n8k16
  constexpr int MMA_K = 16;

  const int q_offset = q_block_id * BLOCK_Q;
  Q_gmem += (bs_id * Q_s0) + (q_offset * Q_s1) + (head_id * Q_s2);
  O_gmem += (bs_id * O_s0) + (q_offset * O_s1) + (head_id * O_s2);
  K_gmem += (bs_id * K_s0) + (head_id * K_s2);
  V_gmem += (bs_id * V_s0) + (head_id * V_s2);

  // we overlap Q_smem with (K_smem + V_smem)
  extern __shared__ char smem[];
  const int Q_smem = __cvta_generic_to_shared(smem);

  // double buffer for K
  const int K_smem = Q_smem;
  const int V_smem = K_smem + 2 * BLOCK_KV * QK_DIM * sizeof(Type);

  // pre-compute address and swizzling for ldmatrix
  int Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row = warp_id * WARP_Q + (lane_id % 16);
    const int col = (lane_id / 16) * 16;
    Q_smem_thread = swizzle<QK_DIM * sizeof(Type)>(Q_smem + (row * QK_DIM * sizeof(Type) + col));
  }
  {
    // B tile
    const int row = lane_id % 8;
    const int col = (lane_id / 8) * 16;
    K_smem_thread = swizzle<QK_DIM * sizeof(Type)>(K_smem + (row * QK_DIM * sizeof(Type) + col));
  }
  {
    // B tile trans
    const int row = lane_id % 16;
    const int col = (lane_id / 16) * 16;
    V_smem_thread = swizzle<QK_DIM * sizeof(Type)>(V_smem + (row * V_DIM * sizeof(Type) + col));
  }

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

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -3.4e38;  // ~FLT_MAX
    rowmax[mma_id_q][1] = -3.4e38;
  }

  // Q gmem->smem
  global_to_shared_swizzle<BLOCK_Q, QK_DIM, TB_SIZE>(Q_smem, Q_gmem, Q_s1, tid);
  asm volatile("cp.async.commit_group;\n");

  // Q smem->rmem
  asm volatile("cp.async.wait_all;\n");
  __syncthreads();
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < QK_DIM / MMA_K; mma_id_d++) {
      int addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * QK_DIM * sizeof(Type);  // row
      addr ^= mma_id_d * MMA_K * sizeof(Type);  // col
      ldmatrix<4>(Q_rmem[mma_id_q][mma_id_d], addr);
    }

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  // TODO: more flexible prefetch
  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      const int dst = K_smem + (kv_id % 2) * (BLOCK_KV * QK_DIM) * sizeof(Type);
      global_to_shared_swizzle<BLOCK_KV, QK_DIM, TB_SIZE>(dst, K_gmem, K_s1, tid);
      K_gmem += BLOCK_KV * K_s1;
    }
    asm volatile("cp.async.commit_group;\n");
  };
  auto load_V = [&](int kv_id) {
    global_to_shared_swizzle<BLOCK_KV, V_DIM, TB_SIZE>(V_smem, V_gmem, V_s1, tid);
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
    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < QK_DIM / MMA_K; mma_id_d += 2) {
        int addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * QK_DIM) * sizeof(Type);
        addr += mma_id_kv * MMA_N * QK_DIM * sizeof(Type);  // row
        addr ^= mma_id_d * MMA_K * sizeof(Type);  // col
        ldmatrix<4>(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < QK_DIM / MMA_K; mma_id_d++)
          mma<Type, Type, float>(Q_rmem[mma_id_q][mma_id_d],
                                 K_rmem[mma_id_kv][mma_id_d],
                                 S_rmem[mma_id_q][mma_id_kv]);

    // prefetch K
    // do we need sync here?
    load_K(kv_id + 1);

    // softmax
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]);
          this_rowmax[1] = max(regs[2], regs[3]);
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
        }
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = exp2f(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = exp2f(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < V_DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        float c0 = exp2f(regs[0] - rowmax[mma_id_q][0]);
        float c1 = exp2f(regs[1] - rowmax[mma_id_q][0]);
        float c2 = exp2f(regs[2] - rowmax[mma_id_q][1]);
        float c3 = exp2f(regs[3] - rowmax[mma_id_q][1]);

        if (mma_id_kv == 0) {
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
                    : "=r"(P_rmem[mma_id_q][mma_id_kv / 2][(mma_id_kv % 2) * 2]),
                      "=r"(P_rmem[mma_id_q][mma_id_kv / 2][(mma_id_kv % 2) * 2 + 1])
                    : "f"(c1), "f"(c0), "f"(c3), "f"(c2),
                      "C"(Type_str<Type>::value));
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // V smem->rmem
    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < V_DIM / MMA_N; mma_id_d += 2) {
        int addr = V_smem_thread;
        addr += mma_id_kv * MMA_K * V_DIM * sizeof(Type);  // row
        addr ^= mma_id_d * MMA_N * sizeof(Type);  // col
        ldmatrix_trans<4>(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < V_DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma<Type, Type, float>(P_rmem[mma_id_q][mma_id_kv],
                                 V_rmem[mma_id_kv][mma_id_d],
                                 O_rmem[mma_id_q][mma_id_d]);
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < V_DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + (mma_id_q * MMA_M) + (lane_id / 4);
      const int col = (mma_id_d * MMA_N) + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

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
