import pytest
import torch

from gn_kernels.cutedsl.sm100 import sm100_mm_bf16, sm100_mm_mxfp8, sm100_mm_nvfp4
from gn_kernels.quant_utils import quantize_mx
from gn_kernels.torch_mm import mxfp8_mm, nvfp4_mm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires an SM10x GPU",
)


def test_mm_bf16():
    M, N, K = 128, 256, 512

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = sm100_mm_bf16.mm(A, B.T)
    ref = A @ B.T
    torch.testing.assert_close(out, ref)


def test_mm_mxfp8():
    M, N, K = 128, 256, 256

    def make_input(*shape):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (K**-0.5)
        return quantize_mx(x, torch.float8_e4m3fn)

    X, Xsf = make_input(M, K)
    W, Wsf = make_input(N, K)
    out = sm100_mm_mxfp8.mm(X, W, Xsf, Wsf)
    ref = mxfp8_mm(X, Xsf, W, Wsf)
    torch.testing.assert_close(out, ref)


def test_mm_nvfp4():
    M, N, K = 128, 256, 256

    def make_input(M: int, N: int):
        x = torch.randint(0, 255, size=(M, N // 2), dtype=torch.uint8, device="cuda")
        x = x.view(torch.float4_e2m1fn_x2)
        sf = torch.randn(M, N // 16, device="cuda").mul(10).to(torch.float8_e4m3fn)
        return x, sf

    X, Xsf = make_input(M, K)
    W, Wsf = make_input(N, K)
    s = torch.ones(1, device="cuda")
    out = sm100_mm_nvfp4.mm(X, W, Xsf, Wsf)
    ref = nvfp4_mm(X, Xsf, s, W, Wsf, s)
    torch.testing.assert_close(out, ref)
