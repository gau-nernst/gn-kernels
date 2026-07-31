import pytest
import torch

from gn_kernels.quant_utils import quantize_nvfp4, quantize_nvfp4_triton, unpermute_nv_sf


@pytest.mark.parametrize("M,N", [(256, 128), (196, 256)])
def test_quantize_nvfp4_triton_correctness(M: int, N: int):
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    xscale = x.abs().amax()

    xq_ref, xsf_ref, _ = quantize_nvfp4(x, xscale)
    xq, xsf = quantize_nvfp4_triton(x, xscale)
    xsf = unpermute_nv_sf(xsf)[:M]

    torch.testing.assert_close(xq, xq_ref)
    torch.testing.assert_close(xsf, xsf_ref)
