#include <hip/hip_bf16.h>

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int GROUP_M = 4;

constexpr int NUM_WARP_M = 2;
constexpr int NUM_WARP_N = 2;

constexpr int WARP_SIZE = 64;

__device__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// start of kernel
__global__
void matmul_kernel(
  const hip_bfloat16 *A_gmem,
  const hip_bfloat16 *B_gmem,
        hip_bfloat16 *C_gmem,
  int M, int N, int K
) {
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  // this is for MI300X
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  // https://github.com/ROCm/composable_kernel/blob/rocm-7.0.1/include/ck/utility/amd_xdlops.hpp
  // https://github.com/tile-ai/tilelang/blob/v0.1.6.post1/src/tl_templates/hip/gemm.h
  // another option is m32n32k8
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 16;
  constexpr int MMA_K = 16;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  const int bid = blockIdx.x;
  const int grid_m = cdiv(M, BLOCK_M);
  const int grid_n = cdiv(N, BLOCK_N);

  int bid_m, bid_n;
  if constexpr (GROUP_M == 1) {
    bid_m = bid / grid_n;
    bid_n = bid % grid_n;
  } else {
    // threadblock swizzling, from triton
    // improve L2 reuse when M is large.
    const int group_size = GROUP_M * grid_n;
    const int group_id = bid / group_size;
    const int first_bid_m = group_id * GROUP_M;
    const int group_size_m = min(grid_m - first_bid_m, GROUP_M);
    bid_m = first_bid_m + ((bid % group_size) % group_size_m);
    bid_n = (bid % group_size) / group_size_m;
  }

  // shared memory
  extern __shared__ hip_bfloat16 smem[];
  hip_bfloat16 *A_smem = smem;
  hip_bfloat16 *B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(hip_bfloat16);
}
