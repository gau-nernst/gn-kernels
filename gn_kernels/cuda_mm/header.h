// this file only exists for code completion.
// we will generate this file when JIT-compile the kernel.

#include "common.h"
#include <cuda/std/type_traits>  // for std::is_same_v
#include <cuda_bf16.h>

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;

constexpr int NUM_WARP_M = 2;
constexpr int NUM_WARP_N = 2;

constexpr int NUM_STAGES = 1;

using TypeAB = nv_bfloat16;
using TypeC = nv_bfloat16;
using TypeAcc = float;
