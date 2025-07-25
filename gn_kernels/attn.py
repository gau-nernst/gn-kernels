import torch
from torch import Tensor

from ._lib import lib, lib_ops

lib.define("sm120a_attn_mxfp8_qk(Tensor Q, Tensor K, Tensor V, Tensor scale_Q, Tensor scale_K) -> Tensor")
lib.define("sm80_attn_int8_qk(Tensor Q, Tensor K, Tensor V, Tensor scale_Q, Tensor scale_K) -> Tensor")


def attn_mxfp8_qk(Q: Tensor, K: Tensor, V: Tensor, scale_Q: Tensor, scale_K: Tensor):
    assert Q.shape[-1] == 128
    # we need to .view(torch.uint8) due to inductor bug with e8m0
    # https://github.com/pytorch/pytorch/issues/158892
    return lib_ops.sm120a_attn_mxfp8_qk(
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        _permute_scale(scale_Q).view(torch.uint8),
        _permute_scale(scale_K).view(torch.uint8),
    )


def attn_int8_qk(Q: Tensor, K: Tensor, V: Tensor, scale_Q: Tensor, scale_K: Tensor):
    assert Q.shape[-1] == 128
    return lib_ops.sm80_attn_int8_qk(
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        scale_Q.contiguous(),
        scale_K.contiguous(),
    )


def _permute_scale(x: Tensor):
    x = x.flatten(0, -2)
    M, N = x.shape
    return x.view(M // 32, 4, 8, N // 4, 4).permute(0, 2, 3, 1, 4).flatten()


@torch.library.impl(lib, "sm120a_attn_mxfp8_qk", "Meta")
@torch.library.impl(lib, "sm80_attn_int8_qk", "Meta")
def _(Q: Tensor, K: Tensor, V: Tensor, *args):
    return torch.empty_like(Q, dtype=torch.bfloat16)
