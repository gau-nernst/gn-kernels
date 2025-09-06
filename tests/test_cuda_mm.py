import pytest
import torch

from gn_kernels.cuda_mm import MatmulKernel


@pytest.mark.parametrize(
    "in_dtype,out_dtype,acc_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float16, torch.float32),
    ],
)
def test_cuda_mm(in_dtype: torch.dtype, out_dtype: torch.dtype, acc_dtype: torch.dtype):
    kernel = MatmulKernel(in_dtype, out_dtype, acc_dtype, num_stages=2)

    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, dtype=in_dtype, device="cuda")
    B = torch.randn(N, K, dtype=in_dtype, device="cuda").T

    actual = kernel.run(A, B)
    expected = torch.mm(A.float(), B.float()).to(out_dtype)
    torch.testing.assert_close(actual, expected)
