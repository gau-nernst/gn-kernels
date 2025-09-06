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

  // TODO: threadblock swizzling
  const int bid = blockIdx.x;
  const int grid_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

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
  TypeAcc acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

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
          if constexpr (cuda::std::is_same_v<TypeAB, char>)
            mma_int8<TypeAB, TypeAB>(A_rmem[mma_id_m][mma_id_k],
                                     B_rmem[mma_id_n][mma_id_k],
                                     acc[mma_id_m][mma_id_n]);
          else
            mma_fp<TypeAB>(A_rmem[mma_id_m][mma_id_k],
                           B_rmem[mma_id_n][mma_id_k],
                           acc[mma_id_m][mma_id_n]);
  }

  // write results to gmem. wait for the last MMA to finish.
  __syncthreads();
  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 4);
      const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;
      TypeAcc *this_acc = acc[mma_id_m][mma_id_n];

      // TODO: maybe change this to some PTX
      if constexpr (cuda::std::is_same_v<TypeAcc, TypeC>) {
        reinterpret_cast<TypeAcc *>(C_gmem)[(row + 0) * N + (col + 0)] = this_acc[0];
        reinterpret_cast<TypeAcc *>(C_gmem)[(row + 0) * N + (col + 1)] = this_acc[1];
        reinterpret_cast<TypeAcc *>(C_gmem)[(row + 8) * N + (col + 0)] = this_acc[2];
        reinterpret_cast<TypeAcc *>(C_gmem)[(row + 8) * N + (col + 1)] = this_acc[3];
      }
      else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, nv_bfloat16>) {
        reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 0) * N + col)[0] = __float22bfloat162_rn({this_acc[0], this_acc[1]});
        reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 8) * N + col)[0] = __float22bfloat162_rn({this_acc[2], this_acc[3]});
      }
      else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, half>) {
        reinterpret_cast<half2 *>(C_gmem + (row + 0) * N + col)[0] = __float22half2_rn({this_acc[0], this_acc[1]});
        reinterpret_cast<half2 *>(C_gmem + (row + 8) * N + col)[0] = __float22half2_rn({this_acc[2], this_acc[3]});
      }
    }
}
