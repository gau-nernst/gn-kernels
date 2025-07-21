#include "../common.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <float.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

using TypeQK    = int8_t;
using TypeV     = nv_bfloat16;
using TypeScale = float;

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void sm80_attn_int8_qk_kernel(
  const TypeQK *Q,           // [bs, len_q, DIM]
  const TypeQK *K,           // [bs, len_kv, DIM]
  const TypeV *V,            // [bs, len_kv, DIM]
  const TypeScale *scale_Q,  // [bs, len_q]
  const TypeScale *scale_K,  // [bs, len_kv]
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

  // m16n8k16 BF16 and m16n8k32 FP8
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K_BF16 = 16;
  constexpr int MMA_K_INT8 = 32;

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;
  scale_Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q;
  scale_K += bs_id * len_kv;
  O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

  // we overlap (Q_smem + scale_Q_smem) with (K_smem + scale_K_smem + V_smem)
  // since we only need to load (Q_smem + scale_Q_smem) once
  extern __shared__ uint8_t smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t scale_Q_smem = Q_smem + BLOCK_Q * DIM * sizeof(TypeQK);

  // double buffer for K and scale_K
  const uint32_t K_smem = Q_smem;
  const uint32_t scale_K_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(TypeQK);

  const uint32_t V_smem = scale_K_smem + 2 * BLOCK_KV * sizeof(TypeScale);

  // pre-compute address and swizzling for ldmatrix
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row = warp_id * WARP_Q + (lane_id % 16);
    const int col = (lane_id / 16) * (16 / sizeof(TypeQK));
    Q_smem_thread = swizzle<DIM * sizeof(TypeQK)>(Q_smem + (row * DIM + col) * sizeof(TypeQK));
  }
  {
    // B tile
    const int row = lane_id % 8;
    const int col = (lane_id / 8) * (16 / sizeof(TypeQK));
    K_smem_thread = swizzle<DIM * sizeof(TypeQK)>(K_smem + (row * DIM + col) * sizeof(TypeQK));
  }
  {
    // B tile trans
    const int row = lane_id % 16;
    const int col = (lane_id / 16) * 8;
    V_smem_thread = swizzle<DIM * sizeof(TypeV)>(V_smem + (row * DIM + col) * sizeof(TypeV));
  }

  // set up registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K_INT8][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K_INT8][2];
  float scale_Q_rmem[WARP_Q / MMA_M][2];
  float scale_K_rmem[BLOCK_KV / MMA_N][2];

  // let compiler decide register reuse?
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K_BF16][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K_BF16][DIM / MMA_N][2];

  // we use the same registers for O_regs and PV_regs
  // rescale O_regs once we obtain new rowmax, then accumulate to O_regs
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
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K_INT8; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(TypeQK);  // row
      addr ^= mma_id_d * MMA_K_INT8 * sizeof(TypeQK);  // col
      ldmatrix<4>(Q_rmem[mma_id_q][mma_id_d], addr);
    }

    const uint32_t addr = scale_Q_smem
                        + warp_id * WARP_Q * sizeof(TypeScale)
                        + mma_id_q * MMA_M * sizeof(TypeScale)
                        + (lane_id / 4) * sizeof(TypeScale);
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_Q_rmem[mma_id_q][0]) : "r"(addr));
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_Q_rmem[mma_id_q][1]) : "r"(addr + (uint32_t)(8 * sizeof(TypeScale))));

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
      const uint32_t K_dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM) * sizeof(TypeQK);
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_dst, K, DIM, tid);
      K += BLOCK_KV * DIM;

      const uint32_t scale_K_dst = scale_K_smem + (kv_id % 2) * BLOCK_KV * sizeof(TypeScale);
      constexpr int width = 16 / sizeof(TypeScale);  // no swizzling
      global_to_shared_swizzle<BLOCK_KV / width, width, TB_SIZE>(scale_K_dst, scale_K, width, tid);
      scale_K += BLOCK_KV;
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_id) {
    // single buffer for V
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
    V += BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");
  };

  // prefetch K
  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    int32_t QK_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // prefetch V
    // __syncthreads() here is required to make sure we finish using V_shm
    // from the previous iteration, since there is only 1 shared buffer for V.
    __syncthreads();
    load_V(kv_id);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K_INT8; mma_id_d += 2) {
        uint32_t addr = K_smem_thread
                      + (kv_id % 2) * (BLOCK_KV * DIM) * sizeof(TypeQK)
                      + mma_id_kv * MMA_N * DIM * sizeof(TypeQK);  // row
        addr ^= mma_id_d * MMA_K_INT8 * sizeof(TypeQK);  // col
        ldmatrix<4>(K_rmem[mma_id_kv][mma_id_d], addr);
      }

      const uint32_t addr = scale_K_smem
                          + (kv_id % 2) * BLOCK_KV * sizeof(TypeScale)
                          + (mma_id_kv * MMA_N) * sizeof(TypeScale)
                          + (lane_id % 4) * 2 * sizeof(TypeScale);
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_K_rmem[mma_id_kv][0]) : "r"(addr));
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_K_rmem[mma_id_kv][1]) : "r"(addr + (uint32_t)sizeof(TypeScale)));
    }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K_INT8; mma_id_d++)
          mma_m16n8k32_s8s8(Q_rmem[mma_id_q][mma_id_d],
                            K_rmem[mma_id_kv][mma_id_d],
                            QK_rmem[mma_id_q][mma_id_kv]);

    // prefetch K
    load_K(kv_id + 1);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        int32_t *regs = QK_rmem[mma_id_q][mma_id_kv];

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
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        int32_t *regs = QK_rmem[mma_id_q][mma_id_kv];

        // recompute
        // TODO: check if we can avoid recompute (but don't use too much registers)
        float c0 = (float)regs[0] * scale_Q_rmem[mma_id_q][0] * scale_K_rmem[mma_id_kv][0];
        float c1 = (float)regs[1] * scale_Q_rmem[mma_id_q][0] * scale_K_rmem[mma_id_kv][1];
        float c2 = (float)regs[2] * scale_Q_rmem[mma_id_q][1] * scale_K_rmem[mma_id_kv][0];
        float c3 = (float)regs[3] * scale_Q_rmem[mma_id_q][1] * scale_K_rmem[mma_id_kv][1];

        c0 = exp2f(c0 - rowmax[mma_id_q][0]);
        c1 = exp2f(c1 - rowmax[mma_id_q][0]);
        c2 = exp2f(c2 - rowmax[mma_id_q][1]);
        c3 = exp2f(c3 - rowmax[mma_id_q][1]);

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = c0 + c1;
          this_rowsumexp[1] = c2 + c3;
        } else {
          this_rowsumexp[0] += c0 + c1;
          this_rowsumexp[1] += c2 + c3;
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        nv_bfloat162 *this_P_regs = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_regs[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({c0, c1});
        this_P_regs[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({c2, c3});
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
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K_BF16; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread
                      + mma_id_kv * MMA_K_BF16 * DIM * sizeof(TypeV);  // row
        addr ^= mma_id_d * MMA_N * sizeof(TypeV);  // col
        ldmatrix_trans<4>(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA P = S @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K_BF16; mma_id_kv++)
          mma_m16n8k16_bf16(P_rmem[mma_id_q][mma_id_kv],
                            V_rmem[mma_id_kv][mma_id_d],
                            O_rmem[mma_id_q][mma_id_d]);
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

at::Tensor sm80_attn_int8_qk(
  const at::Tensor& Q,        // [bs, num_heads, len_q, dim]
  const at::Tensor& K,        // [bs, num_heads, len_kv, dim]
  const at::Tensor& V,        // [bs, num_heads, len_kv, dim]
  const at::Tensor& scale_Q,  // [bs, num_heads, len_q, dim/32] - kinda
  const at::Tensor& scale_K) {

  int bs = Q.size(0) * Q.size(1);
  int len_q = Q.size(2);
  int len_kv = K.size(2);
  int dim = Q.size(3);

  TORCH_CHECK(dim == 128, "Only supports dim=128");
  TORCH_CHECK(scale_Q.dtype() == at::kFloat);
  TORCH_CHECK(scale_K.dtype() == at::kFloat);

  // possibly make a copy inside the kernel. this is undesirable.
  // workaround until figure out how to force torch.compile() to use
  // contiguous layout for a particular custom op.
  auto Q_copy = Q.contiguous();
  auto K_copy = K.contiguous();
  auto V_copy = V.contiguous();

  at::Tensor O = at::empty_like(Q, Q.options().dtype(at::kBFloat16));

  auto Q_ptr = reinterpret_cast<const TypeQK *>(Q_copy.data_ptr());
  auto K_ptr = reinterpret_cast<const TypeQK *>(K_copy.data_ptr());
  auto V_ptr = reinterpret_cast<const TypeV *>(V_copy.data_ptr());
  auto scale_Q_ptr = reinterpret_cast<const TypeScale *>(scale_Q.data_ptr());
  auto scale_K_ptr = reinterpret_cast<const TypeScale *>(scale_K.data_ptr());
  auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 64;
  const int DIM = 128;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int Q_smem_size = BLOCK_Q * (DIM * sizeof(TypeQK) + sizeof(TypeScale));
  const int K_smem_size = 2 * BLOCK_KV * (DIM * sizeof(TypeQK) + sizeof(TypeScale));
  const int V_smem_size = BLOCK_KV * DIM * sizeof(TypeV);
  const int smem_size = max(Q_smem_size, K_smem_size + V_smem_size);

  auto kernel = sm80_attn_int8_qk_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, stream,
                Q_ptr, K_ptr, V_ptr, scale_Q_ptr, scale_K_ptr, O_ptr, bs, len_q, len_kv);

  return O;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m)
{
  m.impl("gn_kernels::sm80_attn_int8_qk", &sm80_attn_int8_qk);
}
