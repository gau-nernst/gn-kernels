import torch
from torch import Tensor

from ._lib import lib, lib_ops

lib.define("sm120a_attn_mxfp8(Tensor Q, Tensor K, Tensor V, Tensor scale_Q, Tensor scale_V) -> Tensor")


def attn_mxfp8(Q: Tensor, K: Tensor, V: Tensor, scale_Q: Tensor, scale_K: Tensor):
    return lib_ops.sm120a_attn_mxfp8(
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        scale_Q.contiguous(),
        scale_K.contiguous(),
    )


@torch.library.impl(lib, "sm120a_attn_mxfp8", "Meta")
def _(Q: Tensor, K: Tensor, V: Tensor, scale_Q: Tensor, scale_K: Tensor):
    return torch.empty_like(Q, dtype=torch.bfloat16)
