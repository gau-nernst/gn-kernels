import pytest
import torch

from gn_kernels.cuda_mm import MatmulKernel


@pytest.mark.parametrize("dtype_str", ["fp16", "bf16"])
def test_cuda_mm_fp(dtype_str: str):
    dtype = dict(fp16=torch.float16, bf16=torch.bfloat16)[dtype_str]
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
    print(actual)
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


@pytest.mark.parametrize("dtype_str", ["fp16", "fp8"])
def test_cuda_mm_fp16_acc(dtype_str: str):
    dtype = dict(fp16=torch.float16, fp8=torch.float8_e4m3fn)[dtype_str]
    kernel = MatmulKernel(dtype, torch.float16, torch.float16, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, device="cuda").to(dtype)
    B = torch.randn(N, K, device="cuda").to(dtype).T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    # simulate FP16 accumulation
    expected = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    mma_k = 32 // dtype.itemsize
    for offset_k in range(0, K, mma_k):
        A_tile = A[:, offset_k : offset_k + mma_k].float()
        B_tile = B[offset_k : offset_k + mma_k, :].float()
        expected.add_(torch.mm(A_tile, B_tile))

    # doesn't pass for FP16
    torch.testing.assert_close(actual, expected)
