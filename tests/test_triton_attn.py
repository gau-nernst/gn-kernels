import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.bias import causal_lower_right

from gn_kernels.triton_attn import triton_attn, triton_varlen_attn


# should the reference be computed in FP32?
def ref_attn(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=causal_lower_right(q_len, kv_len) if causal else None,
        enable_gqa=True,
    ).transpose(1, 2)

    return out


@pytest.mark.parametrize("q_len,kv_len", [(64, 256), (128, 32), (33, 123), (257, 39)])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_attn(q_len: int, kv_len: int, causal: bool):
    bs = 2
    q_heads = 8
    kv_heads = 4
    head_dim = 128

    q = torch.randn(bs, q_len, q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    out = triton_attn(q, k, v, causal=causal)
    assert not out.isnan().any()

    out_ref = ref_attn(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, rtol=1.6e-2, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_triton_varlen_attn(causal: bool):
    bs = 4
    q_heads = 8
    kv_heads = 4
    head_dim = 128

    q_list = []
    k_list = []
    v_list = []
    q_offsets_list = [0]
    kv_offsets_list = [0]
    max_q_len = 0

    for _ in range(bs):
        q_len = torch.randint(16, 256, (1,)).item()
        kv_len = torch.randint(16, 256, (1,)).item()
        q_list.append(torch.randn(q_len, q_heads, head_dim, dtype=torch.bfloat16, device="cuda"))
        k_list.append(torch.randn(kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"))
        v_list.append(torch.randn(kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"))
        q_offsets_list.append(q_offsets_list[-1] + q_len)
        kv_offsets_list.append(kv_offsets_list[-1] + kv_len)
        max_q_len = max(max_q_len, q_len)

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    q_offsets = torch.tensor(q_offsets_list, device="cuda")
    kv_offsets = torch.tensor(kv_offsets_list, device="cuda")

    out = triton_varlen_attn(q, k, v, q_offsets, kv_offsets, max_q_len, causal=causal)

    out_ref_list = []
    for q_, k_, v_ in zip(q_list, k_list, v_list):
        out_ref_list.append(ref_attn(q_[None], k_[None], v_[None], causal=causal)[0])
    out_ref = torch.cat(out_ref_list, dim=0)

    torch.testing.assert_close(out, out_ref)
