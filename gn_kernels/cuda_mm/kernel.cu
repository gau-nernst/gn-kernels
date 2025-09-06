#include "header.h"

// start of kernel
extern "C"
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel(
  const TypeAB *A_gmem,
  const TypeAB *B_gmem,
        TypeC *C_gmem,
  int M, int N, int K
) {
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int MMA_K = 32 / sizeof(TypeAB);

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

  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  // A and B are K-major. C is N-major
  A_gmem += offset_m * K;
  B_gmem += offset_n * K;
  C_gmem += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  // set up shared memory
  extern __shared__ char smem[];
  const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
  const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(TypeAB);
  constexpr int BUFFER_SIZE = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(TypeAB);

  // set up register memory
  int A_rmem[WARP_M / MMA_M][BLOCK_K / 16][4];
  int B_rmem[WARP_N / MMA_N][BLOCK_K / 16][2];
  int C_rmem[WARP_M / MMA_M][WARP_N / MMA_N][sizeof(TypeAcc)] = {};  // 4 for FP32/INT32, 2 for FP16

  // pre-compute ldmatrix address and swizzling
  int A_smem_thread, B_smem_thread;
  {
    const int m = (warp_id_m * WARP_M) + (lane_id % 16);
    const int k = (lane_id / 16) * 16;  // 16-byte
    A_smem_thread = swizzle<BLOCK_K * sizeof(TypeAB)>(A_smem + (m * BLOCK_K * sizeof(TypeAB)) + k);
  }
  {
    const int n = (warp_id_n * WARP_N) + (lane_id % 8);
    const int k = (lane_id / 8) * 16;
    B_smem_thread = swizzle<BLOCK_K * sizeof(TypeAB)>(B_smem + (n * BLOCK_K * sizeof(TypeAB)) + k);
  }

  const int num_k_iters = cdiv(K, BLOCK_K);
  auto load_AB = [&](int iter_k) {
    // select smem buffer
    if (iter_k < num_k_iters) {
        const int this_A_smem = A_smem + (iter_k % NUM_STAGES) * BUFFER_SIZE;
        const int this_B_smem = B_smem + (iter_k % NUM_STAGES) * BUFFER_SIZE;
        global_to_shared_swizzle<BLOCK_M, BLOCK_K, TB_SIZE>(this_A_smem, A_gmem, K, tid);
        global_to_shared_swizzle<BLOCK_N, BLOCK_K, TB_SIZE>(this_B_smem, B_gmem, K, tid);
        A_gmem += BLOCK_K;
        B_gmem += BLOCK_K;
    }
    asm volatile("cp.async.commit_group;\n");
  };

  // initiate prefetching
  for (int i = 0; i < NUM_STAGES - 1; i++)
    load_AB(i);

  for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
    // prefetch the next tile. wait for previous MMA to finish
    __syncthreads();
    load_AB(iter_k + NUM_STAGES - 1);

    // load smem->rmem
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
    __syncthreads();

    for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
        int addr = A_smem_thread + (iter_k % NUM_STAGES) * BUFFER_SIZE;
        addr += mma_id_m * MMA_M * BLOCK_K * sizeof(TypeAB);
        addr ^= mma_id_k * MMA_K * sizeof(TypeAB);
        ldmatrix<4>(A_rmem[mma_id_m][mma_id_k], addr);
      }

    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k += 2) {
        int addr = B_smem_thread + (iter_k % NUM_STAGES) * BUFFER_SIZE;
        addr += mma_id_n * MMA_N * BLOCK_K * sizeof(TypeAB);
        addr ^= mma_id_k * MMA_K * sizeof(TypeAB);
        ldmatrix<4>(B_rmem[mma_id_n][mma_id_k], addr);
      }

    // MMA
    for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
      for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
        for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++)
          mma<TypeAB, TypeAB, TypeAcc>(A_rmem[mma_id_m][mma_id_k],
                                       B_rmem[mma_id_n][mma_id_k],
                                       C_rmem[mma_id_m][mma_id_n]);
  }

  // write results to gmem. wait for the last MMA to finish.
  __syncthreads();
  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;

      const int *acc = C_rmem[mma_id_m][mma_id_n];

      if constexpr (cuda::std::is_same_v<TypeAcc, TypeC>) {
        // no conversion needed
        // either 4-byte (FP32/INT32) or 2-byte (FP16) per elem
        // which we pack 2 elems into 8-byte or 4-byte respectively
        using write_type = cuda::std::conditional_t<sizeof(TypeC) == 4, long, int>;
        reinterpret_cast<write_type *>(C_gmem + (row + 0) * N + col)[0] = reinterpret_cast<const write_type *>(acc)[0];
        reinterpret_cast<write_type *>(C_gmem + (row + 8) * N + col)[0] = reinterpret_cast<const write_type *>(acc)[1];
      }
      else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, nv_bfloat16>) {
        const float *acc_fp32 = reinterpret_cast<const float *>(acc);
        reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 0) * N + col)[0] = __float22bfloat162_rn({acc_fp32[0], acc_fp32[1]});
        reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 8) * N + col)[0] = __float22bfloat162_rn({acc_fp32[2], acc_fp32[3]});
      }
      else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, half>) {
        const float *acc_fp32 = reinterpret_cast<const float *>(acc);
        reinterpret_cast<half2 *>(C_gmem + (row + 0) * N + col)[0] = __float22half2_rn({acc_fp32[0], acc_fp32[1]});
        reinterpret_cast<half2 *>(C_gmem + (row + 8) * N + col)[0] = __float22half2_rn({acc_fp32[2], acc_fp32[3]});
      }
      // don't handle other cases yet...
    }
}
