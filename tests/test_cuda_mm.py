import pytest
import torch

from gn_kernels.cuda_mm import MatmulKernel


# TODO:
# - FP16 accumulation for FP16 and FP8
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cuda_mm_fp(dtype: str):
    dtype = dict(fp16=torch.float16, bf16=torch.bfloat16)[dtype]
    kernel = MatmulKernel(dtype, dtype, torch.float32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda").T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    expected = torch.mm(A, B)
    torch.testing.assert_close(actual, expected)


def test_cuda_mm_int8():
    kernel = MatmulKernel(torch.int8, torch.int32, torch.int32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda").T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    expected = torch._int_mm(A, B)
    torch.testing.assert_close(actual, expected)


def test_cuda_mm_fp8():
    dtype = torch.float8_e4m3fn
    kernel = MatmulKernel(dtype, torch.bfloat16, torch.float32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, device="cuda").to(dtype)
    B = torch.randn(N, K, device="cuda").to(dtype).T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    scale = torch.ones(1, device="cuda")
    expected = torch._scaled_mm(A, B, scale, scale, out_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)
