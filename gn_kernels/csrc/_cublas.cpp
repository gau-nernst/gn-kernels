#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>
#include <cublasLt.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

namespace {

void check(cublasStatus_t status, const char* expression) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "cuBLASLt call failed (",
      expression,
      "): ",
      cublasLtGetStatusName(status));
}

#define CHECK_CUBLAS(expression) check((expression), #expression)

}  // namespace

at::Tensor cublas_nvfp4_mm(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
    double global_scale,
    const std::optional<at::Tensor>& bias) {
  at::cuda::CUDAGuard device_guard(A.device());
  const int64_t M = A.size(0);
  const int64_t N = B.size(0);
  const int64_t K = A.size(1) * 2;
  const int64_t ld_a = B.stride(0) * 2;
  const int64_t ld_b = A.stride(0) * 2;

  auto C = at::empty({M, N}, A.options().dtype(at::kBFloat16));
  const int64_t ld_c = C.stride(0);

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  auto cleanup = c10::make_scope_exit([&] {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (op) cublasLtMatmulDescDestroy(op);
  });

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_4F_E2M1, K, N, ld_a));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_4F_E2M1, K, M, ld_b));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16BF, N, M, ld_c));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, N, M, ld_c));
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

  auto set_desc = [&](auto attribute, const auto& value) {
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, attribute, &value, sizeof(value)));
  };

  const auto trans = CUBLAS_OP_T;
  const auto no_trans = CUBLAS_OP_N;
  const auto host_pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  const size_t workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  void* workspace = at::cuda::getCUDABlasLtWorkspace();
  const uint32_t alignment = 16;

  set_desc(CUBLASLT_MATMUL_DESC_TRANSA, trans);
  set_desc(CUBLASLT_MATMUL_DESC_TRANSB, no_trans);
  set_desc(CUBLASLT_MATMUL_DESC_POINTER_MODE, host_pointer_mode);
  set_desc(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scale_mode);
  set_desc(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scale_mode);
  set_desc(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, SFB.data_ptr());
  set_desc(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, SFA.data_ptr());
  const auto epilogue = bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
  set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
  if (bias) set_desc(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias->data_ptr());

  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  for (auto attribute : {
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, attribute, &alignment, sizeof(alignment)));
  }

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle,
      op,
      a_layout,
      b_layout,
      c_layout,
      d_layout,
      preference,
      1,
      &heuristic,
      &returned));
  TORCH_CHECK(returned > 0, "cuBLASLt found no NVFP4 matmul algorithm");

  const float alpha = static_cast<float>(global_scale);
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasLtMatmul(
      handle,
      op,
      &alpha,
      B.data_ptr(),
      a_layout,
      A.data_ptr(),
      b_layout,
      &beta,
      C.data_ptr(),
      c_layout,
      C.data_ptr(),
      d_layout,
      &heuristic.algo,
      workspace,
      workspace_size,
      at::cuda::getCurrentCUDAStream()));
  return C;
}

TORCH_LIBRARY_IMPL(gn_kernels, CUDA, m) {
  m.impl("gn_kernels::cublas_nvfp4_mm", &cublas_nvfp4_mm);
}
