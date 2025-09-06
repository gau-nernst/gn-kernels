import pytest
import torch

from gn_kernels.cuda_mm import MatmulKernel

_DTYPE_LOOKUP = dict(
    fp32=torch.float32,
    fp16=torch.float16,
    bf16=torch.bfloat16,
    fp8_e4m3=torch.float8_e4m3fn,
    fp8_e5m2=torch.float8_e5m2,
    int32=torch.int32,
    int8=torch.int8,
    uint8=torch.uint8,
)


# TODO:
# - FP8 and INT8
# - FP16 accumulation for FP16 and FP8
@pytest.mark.parametrize(
    "in_dtype,out_dtype,acc_dtype",
    [
        ("bf16", "bf16", "fp32"),
        ("fp16", "fp16", "fp32"),
    ],
)
def test_cuda_mm(in_dtype: str, out_dtype: str, acc_dtype: str):
    in_dtype = _DTYPE_LOOKUP[in_dtype]
    out_dtype = _DTYPE_LOOKUP[out_dtype]
    acc_dtype = _DTYPE_LOOKUP[acc_dtype]

    kernel = MatmulKernel(in_dtype, out_dtype, acc_dtype, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, dtype=in_dtype, device="cuda")
    B = torch.randn(N, K, dtype=in_dtype, device="cuda").T

    actual = kernel.run(A, B)
    expected = torch.mm(A, B)
    torch.testing.assert_close(actual, expected)
