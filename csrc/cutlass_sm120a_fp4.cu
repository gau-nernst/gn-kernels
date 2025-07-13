// CUTLASS example 79

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

#define CUTLASS_CHECK(status) \
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass error: ", cutlassGetStatusString(status))

using namespace cute;
using namespace cutlass::epilogue::fusion;

using ElementOutput = cutlass::bfloat16_t;
using ElementBias   = cutlass::bfloat16_t;
using ElementAcc    = float;
using ElementScale  = float;

using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;
constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using Multiply = Sm90Compute<cutlass::multiplies, ElementAcc, ElementAcc, RoundStyle>;
using Add      = Sm90Compute<cutlass::plus, ElementAcc, ElementAcc, RoundStyle>;
using Cast     = Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementAcc, RoundStyle>;

at::Tensor mxfp4_mm(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& scales_A,
  const at::Tensor& scales_B,
  const std::optional<at::Tensor>& bias)
{
  int M = A.size(0);
  int K = A.size(1) * 2;
  int N = B.size(1);
  at::Tensor D = at::empty({M, N}, A.options().dtype(at::kBFloat16));

  using ElementInput = cutlass::mx_float4_t<cutlass::float_e2m1_t>;

  using EVT0 = Sm90EVT<Add, Sm90RowBroadcast<0, TileShape, ElementBias>, Sm90AccFetch>;  // bias
  using EVT1 = Sm90EVT<Cast, EVT0>;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
      ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
      EpilogueScheduleType, EVT1>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementInput, cutlass::layout::RowMajor, 32,
      ElementInput, cutlass::layout::ColumnMajor, 32,
      ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using DataType = typename ElementInput::DataType;
  using ScaleFactorType = typename ElementInput::ScaleFactorType;

  auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  auto *A_ptr = reinterpret_cast<const DataType *>(A.data_ptr());
  auto *B_ptr = reinterpret_cast<const DataType *>(B.data_ptr());
  auto *scales_A_ptr = reinterpret_cast<const ScaleFactorType *>(scales_A.data_ptr());
  auto *scales_B_ptr = reinterpret_cast<const ScaleFactorType *>(scales_B.data_ptr());
  auto *bias_ptr = bias.has_value() ? reinterpret_cast<const ElementBias *>(bias->data_ptr()) : nullptr;
  auto *D_ptr = reinterpret_cast<ElementOutput *>(D.data_ptr());

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          A_ptr, stride_A,
          B_ptr, stride_B,
          scales_A_ptr, layout_SFA,
          scales_B_ptr, layout_SFB,
      },
      {
          {  // cast
            {  // bias
              {bias_ptr},
              {},
              {}
            },
            {},
          },
          D_ptr, stride_D,
          D_ptr, stride_D,
      }};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));

  long workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({workspace_size}, A.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(stream));

  return D;
}

at::Tensor nvfp4_mm(
  const at::Tensor &A,
  const at::Tensor &B,
  const at::Tensor &scales_A,
  const at::Tensor &scales_B,
  const at::Tensor &output_scale,
  const std::optional<at::Tensor> &bias)
{
  int M = A.size(0);
  int K = A.size(1) * 2;
  int N = B.size(1);
  at::Tensor D = at::empty({M, N}, A.options().dtype(at::kBFloat16));

  using ElementInput = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  using EVT0 = Sm90EVT<Multiply, Sm90ScalarBroadcast<ElementScale>, Sm90AccFetch>;  // output scale
  using EVT1 = Sm90EVT<Add, Sm90RowBroadcast<0, TileShape, ElementBias>, EVT0>;  // bias
  using EVT2 = Sm90EVT<Cast, EVT1>;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
      ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
      EpilogueScheduleType, EVT2>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementInput, cutlass::layout::RowMajor, 32,
      ElementInput, cutlass::layout::ColumnMajor, 32,
      ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using DataType = typename ElementInput::DataType;
  using ScaleFactorType = typename ElementInput::ScaleFactorType;

  auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  auto *A_ptr = reinterpret_cast<const DataType *>(A.data_ptr());
  auto *B_ptr = reinterpret_cast<const DataType *>(B.data_ptr());
  auto *scales_A_ptr = reinterpret_cast<const ScaleFactorType *>(scales_A.data_ptr());
  auto *scales_B_ptr = reinterpret_cast<const ScaleFactorType *>(scales_B.data_ptr());
  auto *output_scale_ptr = reinterpret_cast<const ElementScale *>(output_scale.data_ptr());
  auto *bias_ptr = bias.has_value() ? reinterpret_cast<const ElementBias *>(bias->data_ptr()) : nullptr;
  auto *D_ptr = reinterpret_cast<ElementOutput *>(D.data_ptr());

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          A_ptr,
          stride_A,
          B_ptr,
          stride_B,
          scales_A_ptr,
          layout_SFA,
          scales_B_ptr,
          layout_SFB,
      },
      {
          {  // cast
            {  // bias
              {bias_ptr},
              {  // output scale
                {1.0f, output_scale_ptr},
                {},
                {},
              },
              {},
            },
            {},
          },
          D_ptr, stride_D,
          D_ptr, stride_D,
      }};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));

  long workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({workspace_size}, A.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(stream));

  return D;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m)
{
  m.impl("gn_kernels::mxfp4_mm", &mxfp4_mm);
  m.impl("gn_kernels::nvfp4_mm", &nvfp4_mm);
}
