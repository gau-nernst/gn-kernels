#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


#define CUTLASS_CHECK(status) \
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass error: ", cutlassGetStatusString(status))


// define common params
using ElementA           = cutlass::int4b_t;
using ElementB           = cutlass::int4b_t;
using ElementAccumulator = int32_t;
using OpClass            = cutlass::arch::OpClassTensorOp;
using ArchTag            = cutlass::arch::Sm80;

// how many elements to load at a time -> load 128-bit = 32 x 4-bit
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;


// we will do input checks in python. A and B are stored as int8
at::Tensor int4_mm(at::Tensor A, at::Tensor B) {
  int M = A.size(0);
  int K = A.size(1) * 2;
  int N = B.size(1);
  at::Tensor C = at::empty({M, N}, A.options().dtype(at::kInt));

  // some configs for int4 mma
  // https://github.com/NVIDIA/cutlass/blob/v3.5.1/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cu
  // using default config
  // using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
  // using WarpShape        = GemmShape<64, 64, 128>;
  // using InstructionShape = GemmShape<16, 8, 64>;
  // static int const kStages = 3;
  using ElementC = int32_t;
  using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,    // A matrix
    ElementB, cutlass::layout::ColumnMajor, // B matrix
    ElementC, cutlass::layout::RowMajor,    // C matrix
    ElementAccumulator, OpClass, ArchTag
  >;
  Gemm::Arguments args {
    {M, N, K},
    {reinterpret_cast<ElementA *>(A.data_ptr<int8_t>()), K},
    {reinterpret_cast<ElementB *>(B.data_ptr<int8_t>()), K},
    {C.data_ptr<ElementC>(), N},
    {C.data_ptr<ElementC>(), N},
    {1, 0}  // epilogue
  };
  Gemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm_op(args, nullptr, stream));

  return C;
}

// we will do input checks in python. A and B are stored as int8
// this function is based on the following cutlass example
// https://github.com/NVIDIA/cutlass/blob/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu
// also with the help of emitted code from cutlass Python
at::Tensor scaled_int4_mm(at::Tensor A, at::Tensor B, at::Tensor row_scale, at::Tensor col_scale) {
  int M = A.size(0);
  int K = A.size(1) * 2;
  int N = B.size(1);
  at::Tensor C = at::empty({M, N}, row_scale.options());

  using ElementC        = cutlass::bfloat16_t;
  using ElementEpilogue = float;

  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

  // some configs for int4 mma
  // https://github.com/NVIDIA/cutlass/blob/v3.5.1/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cu
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  constexpr int numStages = 3;
  constexpr int numEpilogueStages = 1;

  // build epilogue visitor tree
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, numEpilogueStages
  >;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  constexpr auto RoundMode = cutlass::FloatRoundStyle::round_to_nearest;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementEpilogue, ElementEpilogue, RoundMode
  >;

  // (1, N)
  using ColScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<cute::_0, cute::_1, int32_t>  // MNL
  >;
  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accum, ColScale>;

  // (M, 1)
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<cute::_1, cute::_0, int32_t>  // MNL
  >;
  using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, EVTCompute0, RowScale>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC, RoundMode,
    cute::Stride<int64_t, cute::_1, int64_t>  // MNL
  >;
  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<Output, EVTCompute1>;

  // to make this work with GemmIdentityThreadblockSwizzle, requires the patch from
  // https://github.com/NVIDIA/cutlass/pull/1753
  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, cutlass::layout::RowMajor,    cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, cutlass::layout::RowMajor,                                      AlignmentC,
    ElementAccumulator, ElementEpilogue, OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVTOutput,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    numStages,
    cutlass::arch::OpMultiplyAddSaturate,  // OpMultiplyAdd does not work
    numEpilogueStages
  >::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  const ElementA *A_ptr         = reinterpret_cast<ElementA *>(A.data_ptr());
  const ElementB *B_ptr         = reinterpret_cast<ElementB *>(B.data_ptr());
  const ElementC *col_scale_ptr = reinterpret_cast<ElementC *>(col_scale.data_ptr());
  const ElementC *row_scale_ptr = reinterpret_cast<ElementC *>(row_scale.data_ptr());
  ElementC *C_ptr               = reinterpret_cast<ElementC *>(C.data_ptr());

  typename EVTOutput::Arguments callback_args{
    {
      {
        {},                                                                  // Accum
        {col_scale_ptr, ElementC(0), {cute::_0{}, cute::_1{}, int32_t(N)}},  // ColScale
        {}                                                                   // Multiply
      },                                                                     // EVTCompute0
      {row_scale_ptr, ElementC(0), {cute::_1{}, cute::_0{}, int32_t(M)}},    // RowScale
      {}                                                                     // Multiply
    },                                                                       // EVTCompute1
    {C_ptr, {int64_t{N}, cute::_1{}, int64_t{M*N}}}                          // EVTOutput
  };

  typename DeviceGemm::Arguments args(
    cutlass::gemm::GemmUniversalMode::kGemm,
    cutlass::gemm::GemmCoord{M, N, K},
    1,                              // batch_split
    callback_args,
    A_ptr, B_ptr, nullptr, nullptr, // unsued C_ptr and D_ptr
    M * K, N * K, 0, 0,             // batch_stride A, B, C, D
    K, K, 0, 0                      // stride A, B, C, D
  );

  DeviceGemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm_op.can_implement(args));
  CUTLASS_CHECK(gemm_op(args, nullptr, stream));

  return C;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m) {
  m.impl("gn_kernels::int4_mm", &int4_mm);
  m.impl("gn_kernels::scaled_int4_mm", &scaled_int4_mm);
}
