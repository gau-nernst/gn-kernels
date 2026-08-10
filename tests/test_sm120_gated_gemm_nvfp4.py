import pytest
import torch
import torch.nn.functional as F

from gn_kernels.cutedsl.sm120 import sm120_gated_gemm_nvfp4
from gn_kernels.quant_utils import quantize_nvfp4_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires an SM120 GPU",
)


def nvfp4_mm(x, x_sf, x_scale, w, w_sf, w_scale):
    recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    return F.scaled_mm(
        x,
        w.T,
        scale_a=[x_sf, x_scale],
        scale_recipe_a=recipe,
        scale_b=[w_sf, w_scale],
        scale_recipe_b=recipe,
        swizzle_a=swizzle,
        swizzle_b=swizzle,
    )


def test_gated_gemm_nvfp4():
    M, N, K = 256, 256, 128

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
