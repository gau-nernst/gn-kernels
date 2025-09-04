#include "../common.h"
#include "../host_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <float.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

using Type      = int8_t;
using TypeScale = float;

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void sm80_attn_int8_kernel(
  const Type *Q,             // [bs, len_q, DIM]
  const Type *K,             // [bs, len_kv, DIM]
  const Type *V,             // [bs, DIM, len_kv]
  const TypeScale *scale_Q,  // [bs, len_q]
  const TypeScale *scale_K,  // [bs, len_kv]
  const TypeScale *scale_V,  // [bs, len_kv/BLOCK_KV, DIM]
  nv_bfloat16 *O,            // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // m16n8k32
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 32;

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * DIM * len_kv;
  scale_Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q;
  scale_K += bs_id * len_kv;
  scale_V += bs_id * (len_kv / BLOCK_KV) * DIM;
  O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

  // we overlap (Q_smem + scale_Q_smem) with (K_smem + scale_K_smem + V_smem)
  // since we only need to load (Q_smem + scale_Q_smem) once
  extern __shared__ uint8_t smem[];
  const int Q_smem = __cvta_generic_to_shared(smem);
  const int scale_Q_smem = Q_smem + BLOCK_Q * DIM * sizeof(Type);

  // double buffer for K and scale_K
  const int K_smem = Q_smem;
  const int scale_K_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(Type);

  const int V_smem = scale_K_smem + 2 * BLOCK_KV * sizeof(TypeScale);
  const int scale_V_smem = V_smem + DIM * BLOCK_KV * sizeof(Type);

  // pre-compute address and swizzling for ldmatrix
  int Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row = warp_id * WARP_Q + (lane_id % 16);
    const int col = (lane_id / 16) * (16 / sizeof(Type));
    Q_smem_thread = swizzle<DIM * sizeof(Type)>(Q_smem + (row * DIM + col) * sizeof(Type));
  }
  {
    // B tile
    const int row = lane_id % 8;
    const int col = (lane_id / 8) * (16 / sizeof(Type));
    K_smem_thread = swizzle<DIM * sizeof(Type)>(K_smem + (row * DIM + col) * sizeof(Type));
  }
  {
    // B tile
    const int row = lane_id % 8;
    const int col = (lane_id / 8) * (16 / sizeof(Type));
    V_smem_thread = swizzle<BLOCK_KV * sizeof(Type)>(V_smem + (row * BLOCK_KV + col) * sizeof(Type));
  }

  // set up registers
  // [WARP_Q, DIM] x [BLOCK_KV, DIM].T = [WARP_Q, BLOCK_KV]
  int Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  int K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
  float scale_Q_rmem[WARP_Q / MMA_M][2];
  float scale_K_rmem[BLOCK_KV / MMA_N][2];

  // let compiler decide register reuse?
  // [WARP_Q, BLOCK_KV] x [DIM, BLOCK_KV].T = [WARP_Q, DIM]
  int P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  int V_rmem[DIM / MMA_N][BLOCK_KV / MMA_K][2];
  float scale_V_rmem[DIM / MMA_N][2];

  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  // exp(x) = exp(log(2) * x / log(2)) = exp2(x / log2)
  const float softmax_scale = rsqrtf(static_cast<float>(DIM)) * 1.4426950408889634f;

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM]
  {
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    constexpr int width = 16 / sizeof(TypeScale);  // no swizzling
    global_to_shared_swizzle<BLOCK_Q / width, width, TB_SIZE>(scale_Q_smem, scale_Q, width, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
  }
  __syncthreads();

  // shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      int addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(Type);  // row
      addr ^= mma_id_d * MMA_K * sizeof(Type);  // col
      ldmatrix<4>(Q_rmem[mma_id_q][mma_id_d], addr);
    }

    const int addr = scale_Q_smem
                   + warp_id * WARP_Q * sizeof(TypeScale)
                   + mma_id_q * MMA_M * sizeof(TypeScale)
                   + (lane_id / 4) * sizeof(TypeScale);
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_Q_rmem[mma_id_q][0]) : "r"(addr));
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_Q_rmem[mma_id_q][1]) : "r"(addr + (int)(8 * sizeof(TypeScale))));

    // fuse softmax scale into scale_Q
    scale_Q_rmem[mma_id_q][0] *= softmax_scale;
    scale_Q_rmem[mma_id_q][1] *= softmax_scale;
  }

  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      // double buffer for K
      const int K_dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM) * sizeof(Type);
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_dst, K, DIM, tid);
      K += BLOCK_KV * DIM;

      const int scale_K_dst = scale_K_smem + (kv_id % 2) * BLOCK_KV * sizeof(TypeScale);
      constexpr int width = 16 / sizeof(TypeScale);  // no swizzling
      global_to_shared_swizzle<BLOCK_KV / width, width, TB_SIZE>(scale_K_dst, scale_K, width, tid);
      scale_K += BLOCK_KV;
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_id) {
    // single buffer for V
    global_to_shared_swizzle<DIM, BLOCK_KV, TB_SIZE>(V_smem, V, len_kv, tid);
    V += BLOCK_KV;

    constexpr int width = 16 / sizeof(TypeScale);
    global_to_shared_swizzle<DIM / width, width, TB_SIZE>(scale_V_smem, scale_V, width, tid);
    scale_V += DIM;

    asm volatile("cp.async.commit_group;");
  };

  // prefetch K
  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    // use this as accumulator for both QK and PV
    constexpr int num_mma_n = (BLOCK_KV > DIM ? BLOCK_KV : DIM) / MMA_N;
    int QK_rmem[WARP_Q / MMA_M][num_mma_n][4] = {};

    // prefetch V
    // __syncthreads() here is required to make sure we finish using V_shm
    // from the previous iteration, since there is only 1 shared buffer for V.
    __syncthreads();
    load_V(kv_id);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        int addr = K_smem_thread
                 + (kv_id % 2) * (BLOCK_KV * DIM) * sizeof(Type)
                 + (mma_id_kv * MMA_N) * DIM * sizeof(Type);  // row
        addr ^= (mma_id_d * MMA_K) * sizeof(Type);  // col
        ldmatrix<4>(K_rmem[mma_id_kv][mma_id_d], addr);
      }

      const int addr = scale_K_smem
                     + (kv_id % 2) * BLOCK_KV * sizeof(TypeScale)
                     + (mma_id_kv * MMA_N) * sizeof(TypeScale)
                     + (lane_id % 4) * 2 * sizeof(TypeScale);
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_K_rmem[mma_id_kv][0]) : "r"(addr));
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_K_rmem[mma_id_kv][1]) : "r"(addr + (int)sizeof(TypeScale)));
    }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k32_s8s8(Q_rmem[mma_id_q][mma_id_d],
                            K_rmem[mma_id_kv][mma_id_d],
                            QK_rmem[mma_id_q][mma_id_kv]);

    // prefetch K
    load_K(kv_id + 1);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        int *regs = QK_rmem[mma_id_q][mma_id_kv];

        float c0 = (float)regs[0] * scale_K_rmem[mma_id_kv][0];
        float c1 = (float)regs[1] * scale_K_rmem[mma_id_kv][1];
        float c2 = (float)regs[2] * scale_K_rmem[mma_id_kv][0];
        float c3 = (float)regs[3] * scale_K_rmem[mma_id_kv][1];

        float rowmax0 = max(c0, c1) * scale_Q_rmem[mma_id_q][0];
        float rowmax1 = max(c2, c3) * scale_Q_rmem[mma_id_q][1];

        if (mma_id_kv == 0) {
          this_rowmax[0] = rowmax0;
          this_rowmax[1] = rowmax1;
        } else {
          this_rowmax[0] = max(this_rowmax[0], rowmax0);
          this_rowmax[1] = max(this_rowmax[1], rowmax1);
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
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
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
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv += 2) {
        int *regs = QK_rmem[mma_id_q][mma_id_kv];

        // recompute
        // TODO: check if we can avoid recompute (but don't use too much registers)
        float tmp[8];
        for (int reg_id = 0; reg_id < 8; reg_id++) {
          tmp[reg_id] = (float)regs[reg_id]
                      * scale_Q_rmem[mma_id_q][(reg_id / 2) % 2]
                      * scale_K_rmem[mma_id_kv + (reg_id / 4)][reg_id % 2];

          // scale [0,1] to [0,255]
          tmp[reg_id] = exp2f(tmp[reg_id] - rowmax[mma_id_q][(reg_id / 2) % 2]) * 255.0f;
        }

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = tmp[0] + tmp[1] + tmp[4] + tmp[5];
          this_rowsumexp[1] = tmp[2] + tmp[3] + tmp[6] + tmp[7];
        } else {
          this_rowsumexp[0] += tmp[0] + tmp[1] + tmp[4] + tmp[5];
          this_rowsumexp[1] += tmp[2] + tmp[3] + tmp[6] + tmp[7];
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 FP32 to m16k32 INT8
        // thread 0: 0 4
        // thread 1: 1 5
        // thread 2: 2 6
        // thread 3: 3 7
        int row1[2], row2[2];
        row1[0] = (__float2uint_rn(tmp[1]) << 8u) | __float2uint_rn(tmp[0]);
        row2[0] = (__float2uint_rn(tmp[3]) << 8u) | __float2uint_rn(tmp[2]);
        row1[1] = (__float2uint_rn(tmp[5]) << 8u) | __float2uint_rn(tmp[4]);
        row2[1] = (__float2uint_rn(tmp[7]) << 8u) | __float2uint_rn(tmp[6]);

        // thread 0: 0 1 -> done
        // thread 1: 4 5
        // thread 2: 2 3
        // thread 3: 6 7 -> done
        row1[1 ^ (lane_id % 2)] = __shfl_xor_sync(0xFFFF'FFFF, row1[1 ^ (lane_id % 2)], 1);
        row2[1 ^ (lane_id % 2)] = __shfl_xor_sync(0xFFFF'FFFF, row2[1 ^ (lane_id % 2)], 1);

        // (mma_id_kv % 4) is either 0 or 2
        int *this_P_regs = P_rmem[mma_id_q][mma_id_kv / 4];
        this_P_regs[(mma_id_kv % 4) + 0] = (row1[1] << 16u) | row1[0];
        this_P_regs[(mma_id_kv % 4) + 1] = (row2[1] << 16u) | row2[0];

        // swap content of thread 1 and thread 2
        if ((lane_id % 4) == 1 || (lane_id % 4) == 2) {
          this_P_regs[(mma_id_kv % 4) + 0] = __shfl_xor_sync(0x6666'6666, this_P_regs[(mma_id_kv % 4) + 0], 3);
          this_P_regs[(mma_id_kv % 4) + 1] = __shfl_xor_sync(0x6666'6666, this_P_regs[(mma_id_kv % 4) + 1], 3);
        }
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

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv += 2) {
        int addr = V_smem_thread
                      + (mma_id_d * MMA_N) * BLOCK_KV * sizeof(Type);  // row
        addr ^= (mma_id_kv * MMA_K) * sizeof(Type);  // col
        ldmatrix<4>(V_rmem[mma_id_d][mma_id_kv], addr);
      }

      const int addr = scale_V_smem
                     + (mma_id_d * MMA_N) * sizeof(TypeScale)
                     + (lane_id % 4) * 2 * sizeof(TypeScale);
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_V_rmem[mma_id_d][0]) : "r"(addr));
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_V_rmem[mma_id_d][1]) : "r"(addr + (int)sizeof(TypeScale)));
    }

    // reset QK_rmem
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          QK_rmem[mma_id_q][mma_id_d][reg_id] = 0;

    // MMA O = P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k32_u8s8(P_rmem[mma_id_q][mma_id_kv],
                            V_rmem[mma_id_d][mma_id_kv],
                            QK_rmem[mma_id_q][mma_id_d]);

    // accumulate to master O_rmem
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          O_rmem[mma_id_q][mma_id_d][reg_id] += (float)QK_rmem[mma_id_q][mma_id_d][reg_id]
                                                * scale_V_rmem[mma_id_d][reg_id % 2];
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + (mma_id_q * MMA_M) + (lane_id / 4);
      const int col = (mma_id_d * MMA_N) + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

      reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

at::Tensor sm80_attn_int8(
  const at::Tensor& Q,        // [bs, num_heads, len_q, dim]
  const at::Tensor& K,        // [bs, num_heads, len_kv, dim]
  const at::Tensor& V,        // [bs, num_heads, dim, len_kv]
  const at::Tensor& scale_Q,  // [bs, num_heads, len_q]
  const at::Tensor& scale_K,  // [bs, num_heads, len_kv]
  const at::Tensor& scale_V) {// [bs, num_heads, len_kv/64, dim]

  int bs = Q.size(0) * Q.size(1);
  int len_q = Q.size(2);
  int len_kv = K.size(2);
  int dim = Q.size(3);

  TORCH_CHECK(dim == 128, "Only supports dim=128");
  TORCH_CHECK(scale_Q.dtype() == at::kFloat);
  TORCH_CHECK(scale_K.dtype() == at::kFloat);
  TORCH_CHECK(scale_V.dtype() == at::kFloat);
  TORCH_CHECK(Q.is_contiguous());
  TORCH_CHECK(K.is_contiguous());
  TORCH_CHECK(V.is_contiguous());
  TORCH_CHECK(scale_Q.is_contiguous());
  TORCH_CHECK(scale_K.is_contiguous());
  TORCH_CHECK(scale_V.is_contiguous());

  at::Tensor O = at::empty_like(Q, Q.options().dtype(at::kBFloat16));

  auto Q_ptr = reinterpret_cast<const Type *>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<const Type *>(K.data_ptr());
  auto V_ptr = reinterpret_cast<const Type *>(V.data_ptr());
  auto scale_Q_ptr = reinterpret_cast<const TypeScale *>(scale_Q.data_ptr());
  auto scale_K_ptr = reinterpret_cast<const TypeScale *>(scale_K.data_ptr());
  auto scale_V_ptr = reinterpret_cast<const TypeScale *>(scale_V.data_ptr());
  auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 64;
  const int DIM = 128;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int Q_smem_size = BLOCK_Q * (DIM * sizeof(Type) + sizeof(TypeScale));
  const int K_smem_size = 2 * BLOCK_KV * (DIM * sizeof(Type) + sizeof(TypeScale));
  const int V_smem_size = DIM * (BLOCK_KV * sizeof(Type) + sizeof(TypeScale));
  const int smem_size = max(Q_smem_size, K_smem_size + V_smem_size);

  auto kernel = sm80_attn_int8_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, stream,
                Q_ptr, K_ptr, V_ptr, scale_Q_ptr, scale_K_ptr, scale_V_ptr, O_ptr, bs, len_q, len_kv);

  return O;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m)
{
  m.impl("gn_kernels::sm80_attn_int8", &sm80_attn_int8);
}
