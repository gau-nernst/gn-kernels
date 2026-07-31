import torch
from torch import Tensor

from ._lib import lib, lib_ops

lib.define(
    "cublas_nvfp4_mm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, float global_scale=1.0, Tensor? bias=None) -> Tensor"
)


def cublas_nvfp4_mm(
    A: Tensor,
    B: Tensor,
    SFA: Tensor,
    SFB: Tensor,
    global_scale: float = 1.0,
    bias: Tensor | None = None,
) -> Tensor:
    assert A.dtype == torch.float4_e2m1fn_x2 and B.dtype == torch.float4_e2m1fn_x2
    assert SFA.dtype == torch.float8_e4m3fn and SFB.dtype == torch.float8_e4m3fn
    assert A.stride(1) == 1 and B.stride(1) == 1
    if bias is not None:
        assert bias.is_contiguous
    return lib_ops.cublas_nvfp4_mm(A, B, SFA, SFB, global_scale, bias)


@torch.library.impl(lib, "cublas_nvfp4_mm", "Meta")
def _(A: Tensor, B: Tensor, SFA: Tensor, SFB: Tensor, global_scale: float = 1.0, bias: Tensor | None = None):
    return torch.empty((A.shape[0], B.shape[0]), device="meta", dtype=torch.bfloat16)
