#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/type_traits>  // for std::is_same_v

inline constexpr int WARP_SIZE = 32;
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__
int swizzle(int index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  int row_idx = (index / STRIDE) % 8;
  int bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void global_to_shared(int dst, const T *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(T);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int dst_addr = dst + (row * WIDTH + col) * sizeof(T);
    const T *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void global_to_shared_swizzle(int dst, const T *src, int src_stride, int tid) {
  static_assert(WIDTH * sizeof(T) >= 16);
  constexpr int num_elems = 16 / sizeof(T);

  auto load = [&](int idx) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int dst_addr = swizzle<WIDTH * sizeof(T)>(dst + (row * WIDTH + col) * sizeof(T));
    const T *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  };

  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++)
    load((iter * TB_SIZE + tid) * num_elems);

  // handle the case when tile size is not divisible by threadblock size
  if constexpr ((HEIGHT * WIDTH) % (TB_SIZE * num_elems) != 0) {
    const int idx = (num_iters * TB_SIZE + tid) * num_elems;
    if (idx < HEIGHT * WIDTH)
      load(idx);
  }
}

template <int num>
__device__ inline
void ldmatrix(int *regs, int addr) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                : "=r"(regs[0])
                : "r"(addr));
  else if constexpr (num == 2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                : "=r"(regs[0]), "=r"(regs[1])
                : "r"(addr));
  else if constexpr (num == 4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(addr));
}

template <int num>
__device__ inline
void ldmatrix_trans(int *regs, int addr) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                : "=r"(regs[0])
                : "r"(addr));
  else if constexpr (num == 2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                : "=r"(regs[0]), "=r"(regs[1])
                : "r"(addr));
  else if constexpr (num == 4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(addr));
}

struct int4x2 { char data; };
struct uint4x2 { char data; };

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
template <typename T>
struct Type_str;
template<> struct Type_str<float> { static constexpr const char value[] = "f32"; };
template<> struct Type_str<half> { static constexpr const char value[] = "f16"; };
template<> struct Type_str<nv_bfloat16> { static constexpr const char value[] = "bf16"; };
template<> struct Type_str<__nv_fp8_e4m3> { static constexpr const char value[] = "e4m3"; };
template<> struct Type_str<__nv_fp8_e5m2> { static constexpr const char value[] = "e5m2"; };
// NOTE: according to C/C++ spec, sign-ness of char is implementation-defined
template<> struct Type_str<signed char> { static constexpr const char value[] = "s8"; };
template<> struct Type_str<unsigned char> { static constexpr const char value[] = "u8"; };
// these types are defined by us
template<> struct Type_str<int4x2> { static constexpr const char value[] = "s4"; };
template<> struct Type_str<uint4x2> { static constexpr const char value[] = "u4"; };

template <int element_size>
struct MMA_shape_str;
template<> struct MMA_shape_str<2> { static constexpr const char value[] = "m16n8k16"; };
template<> struct MMA_shape_str<1> { static constexpr const char value[] = "m16n8k32"; };

template <typename atype, typename btype, typename ctype>
__device__ inline
void mma(int A[4], int B[2], void *C) {
  static_assert(cuda::std::is_same_v<ctype, float>
              || cuda::std::is_same_v<ctype, half>
              || cuda::std::is_same_v<ctype, int>);

  // use void * for input so that we can pass either float or int pointer
  int *D = reinterpret_cast<int *>(C);

  // m16n8k16 for FP16/BF16
  // m16n8k32 for FP8/INT8
  using shape = MMA_shape_str<sizeof(atype)>;

  if constexpr (cuda::std::is_same_v<ctype, float>)
    asm volatile("mma.sync.aligned.%14.row.col.f32.%15.%16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                  "C"(shape::value), "C"(Type_str<atype>::value), "C"(Type_str<btype>::value));

  else if constexpr (cuda::std::is_same_v<ctype, half>)
    asm volatile("mma.sync.aligned.%10.row.col.f16.%11.%12.f16 "
                "{%0, %1}, "
                "{%2, %3, %4, %5}, "
                "{%6, %7}, "
                "{%8, %9};"
                : "=r"(D[0]), "=r"(D[1])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]),
                  "C"(shape::value), "C"(Type_str<atype>::value), "C"(Type_str<btype>::value));

  // special case for INT4 MMA. override MMA shape
  else if constexpr (cuda::std::is_same_v<atype, int4x2> || cuda::std::is_same_v<atype, uint4x2>)
    asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.%14.%15.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                  "C"(Type_str<atype>::value), "C"(Type_str<btype>::value));

  // TODO: maybe we can include .satfinite in the 1st case as well?
  else if constexpr (cuda::std::is_same_v<ctype, int>)
    asm volatile("mma.sync.aligned.%14.row.col.satfinite.s32.%15.%16.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                  "C"(shape::value), "C"(Type_str<atype>::value), "C"(Type_str<btype>::value));
}

template <typename TypeAB>
__device__ inline
void mma_mxfp8(int A[4], int B[2], float C[4],
               int scale_A, short byte_id_A, short thread_id_A,
               int scale_B, short byte_id_B, short thread_id_B) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.%20.%20.f32.ue8m0 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13}, "
              "{%14}, {%15, %16}, "
              "{%17}, {%18, %19};"
              : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
                "r"(scale_A), "h"(byte_id_A), "h"(thread_id_A),
                "r"(scale_B), "h"(byte_id_B), "h"(thread_id_B),
                "C"(Type_str<TypeAB>::value));
}
