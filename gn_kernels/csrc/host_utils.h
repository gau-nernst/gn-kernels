#pragma once

#include <iostream>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  dim3 num_blocks,
  dim3 block_size,
  int smem_size,
  cudaStream_t stream,
  Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<num_blocks, block_size, smem_size, stream>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}
