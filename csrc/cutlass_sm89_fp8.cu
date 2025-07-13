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
using ElementOutput      = cutlass::bfloat16_t;
using ElementScale       = float;
using ElementAccumulator = float;
using OpClass            = cutlass::arch::OpClassTensorOp;
using ArchTag            = cutlass::arch::Sm89;

constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

at::Tensor fp8_mm(at::Tensor A, at::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  at::Tensor C = at::empty({M, N}, A.options().dtype(at::kBFloat16));

  using ElementInput = cutlass::float_e4m3_t;

  // TODO: use better config
  // static int const kStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,    // A matrix
    ElementInput, cutlass::layout::ColumnMajor, // B matrix
    ElementOutput, cutlass::layout::RowMajor,   // C matrix
    ElementAccumulator, OpClass, ArchTag
  >;
  Gemm::Arguments args {
    {M, N, K},
    {reinterpret_cast<ElementInput *>(A.data_ptr()), K},
    {reinterpret_cast<ElementInput *>(B.data_ptr()), K},
    {reinterpret_cast<ElementOutput *>(C.data_ptr()), N},
    {reinterpret_cast<ElementOutput *>(C.data_ptr()), N},
    {1, 0}  // epilogue
  };
  Gemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm_op(args, nullptr, stream));

  return C;
}

// this function is based on the following cutlass example
// https://github.com/NVIDIA/cutlass/blob/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu
// also with the help of emitted code from cutlass Python
template <typename ElementInput>
void scaled_fp8_mm_kernel(
  const ElementInput* A_ptr,
  const ElementInput* B_ptr,
  const ElementScale* row_scale_ptr,
  const ElementScale* col_scale_ptr,
  ElementOutput* out_ptr,
  int M,
  int N,
  int K
) {
  using namespace cute;

  // https://github.com/NVIDIA/cutlass/blob/v3.9.2/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  constexpr int numStages = 3;
  constexpr int numEpilogueStages = 1;

  // build epilogue visitor tree
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentOutput, numEpilogueStages
  >;

  using ElementEpilogue = float;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  constexpr auto RoundMode = cutlass::FloatRoundStyle::round_to_nearest;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementEpilogue, ElementEpilogue, RoundMode
  >;

  // (1, N)
  using ColScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, ElementScale,
    cute::Stride<_0, _1, int32_t>  // MNL
  >;
  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accum, ColScale>;

  // (M, 1)
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, ElementScale,
    cute::Stride<_1, _0, int32_t>  // MNL
  >;
  using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, EVTCompute0, RowScale>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput, RoundMode,
    cute::Stride<int64_t, _1, int64_t>  // MNL
  >;
  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<Output, EVTCompute1>;

  constexpr int AlignmentInput = 128 / cutlass::sizeof_bits<ElementInput>::value;
  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementInput, cutlass::layout::RowMajor,    cutlass::ComplexTransform::kNone, AlignmentInput,
    ElementInput, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentInput,
    ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
    ElementAccumulator, ElementEpilogue, OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVTOutput,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    numStages,
    cutlass::arch::OpMultiplyAdd,
    numEpilogueStages
  >::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  typename EVTOutput::Arguments callback_args{
    {
      {
        {},                                                          // Accum
        {col_scale_ptr, ElementScale(0), {_0{}, _1{}, int32_t(N)}},  // ColScale
        {}                                                           // Multiply
      },                                                             // EVTCompute0
      {row_scale_ptr, ElementScale(0), {_1{}, _0{}, int32_t(M)}},    // RowScale
      {}                                                             // Multiply
    },                                                               // EVTCompute1
    {out_ptr, {int64_t{N}, _1{}, int64_t{M*N}}}                      // EVTOutput
  };

  typename Gemm::Arguments args(
    cutlass::gemm::GemmUniversalMode::kGemm,
    cutlass::gemm::GemmCoord{M, N, K},
    1,                              // batch_split
    callback_args,
    A_ptr, B_ptr, nullptr, nullptr, // unsued C_ptr and D_ptr
    M * K, N * K, 0, 0,             // batch_stride A, B, C, D
    K, K, 0, 0                      // stride A, B, C, D
  );

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(args));

  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm(args, nullptr, stream));
}

at::Tensor scaled_fp8_mm(at::Tensor A, at::Tensor B, at::Tensor row_scale, at::Tensor col_scale) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  at::Tensor out = at::empty({M, N}, row_scale.options().dtype(at::kBFloat16));

  const ElementScale *row_scale_ptr = reinterpret_cast<ElementScale *>(row_scale.data_ptr());
  const ElementScale *col_scale_ptr = reinterpret_cast<ElementScale *>(col_scale.data_ptr());
  ElementOutput *out_ptr            = reinterpret_cast<ElementOutput *>(out.data_ptr());

  if (A.dtype() == at::kFloat8_e4m3fn) {
    using ElementInput        = cutlass::float_e4m3_t;
    const ElementInput *A_ptr = reinterpret_cast<ElementInput *>(A.data_ptr());
    const ElementInput *B_ptr = reinterpret_cast<ElementInput *>(B.data_ptr());
    scaled_fp8_mm_kernel(A_ptr, B_ptr, row_scale_ptr, col_scale_ptr, out_ptr, M, N, K);
  } else if (A.dtype() == at::kFloat8_e5m2) {
    using ElementInput        = cutlass::float_e5m2_t;
    const ElementInput *A_ptr = reinterpret_cast<ElementInput *>(A.data_ptr());
    const ElementInput *B_ptr = reinterpret_cast<ElementInput *>(B.data_ptr());
    scaled_fp8_mm_kernel(A_ptr, B_ptr, row_scale_ptr, col_scale_ptr, out_ptr, M, N, K);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype");
  }

  return out;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m) {
  m.impl("gn_kernels::fp8_mm", &fp8_mm);
  m.impl("gn_kernels::scaled_fp8_mm", &scaled_fp8_mm);
}
