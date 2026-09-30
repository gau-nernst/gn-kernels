import pytest
import torch
import torch.nn.functional as F

from gn_kernels.cutedsl.sm120 import sm120_gated_gemm_nvfp4, sm120_mm, sm120_mm_mxfp8, sm120_mm_nvfp4
from gn_kernels.quant_utils import quantize_mx, quantize_nvfp4_triton
from gn_kernels.torch_mm import mxfp8_mm, nvfp4_mm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires an SM120 GPU",
)


def test_mm_bf16():
    M, N, K = 128, 256, 512

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = sm120_mm.mm(A, B.T)
    ref = A @ B.T
    torch.testing.assert_close(out, ref)


def test_mm_int8():
    M, N, K = 128, 256, 512

    A = torch.randint(-127, 127, size=(M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-127, 127, size=(N, K), dtype=torch.int8, device="cuda")
    out = sm120_mm.mm(A, B.T)
    ref = torch._int_mm(A, B.T)
    torch.testing.assert_close(out, ref)


def test_mm_fp8():
    M, N, K = 128, 256, 512

    A = torch.randn(M, K, device="cuda").mul(10).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").mul(10).to(torch.float8_e4m3fn)
    s = torch.ones(1, device="cuda")
    out = sm120_mm.mm(A, B.T)
    ref = F.scaled_mm(A, B.T, s, F.ScalingType.TensorWise, s, F.ScalingType.TensorWise)
    torch.testing.assert_close(out, ref)


def test_mm_mxfp8():
    M, N, K = 128, 256, 256

    def make_input(*shape):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (K**-0.5)
        return quantize_mx(x, torch.float8_e4m3fn)

    X, Xsf = make_input(M, K)
    W, Wsf = make_input(N, K)
    out = sm120_mm_mxfp8.mm(X, W, Xsf, Wsf)
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
    out = sm120_mm_nvfp4.mm(X, W, Xsf, Wsf)
    ref = nvfp4_mm(X, Xsf, s, W, Wsf, s)
    torch.testing.assert_close(out, ref)


def test_gated_gemm_nvfp4():
    M, N, K = 128, 256, 256

    def make_input(*shape):
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (K**-0.5)
        sf2 = x.abs().amax().float()
        xq, sf = quantize_nvfp4_triton(x, sf2)
        return xq, sf, sf2

    X = make_input(M, K)
    W1 = make_input(N, K)
    W3 = make_input(N, K)

    # unquantized case
    out = sm120_gated_gemm_nvfp4.mm(*X, *W1, *W3)
    out_ref = F.silu(nvfp4_mm(*X, *W1)) * nvfp4_mm(*X, *W3)
    torch.testing.assert_close(out, out_ref)

    # quantized case
    out_sf2 = out.abs().amax().float()
    out_q_ref, out_sf_ref = quantize_nvfp4_triton(out, out_sf2)
    out_q, out_sf = sm120_gated_gemm_nvfp4.mm(*X, *W1, *W3, out_sf2)

    torch.testing.assert_close(out_q, out_q_ref)
    torch.testing.assert_close(out_sf, out_sf_ref)
