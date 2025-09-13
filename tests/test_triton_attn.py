import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from gn_kernels.triton_attn import triton_attn


@pytest.mark.parametrize("q_len,kv_len", [(64, 256), (33, 123)])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_attn(q_len: int, kv_len: int, causal: bool):
    bs = 2
    num_heads_q = 8
    num_heads_kv = 4
    head_dim = 128

    q = torch.randn(bs, q_len, num_heads_q, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(bs, kv_len, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(bs, kv_len, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    out = triton_attn(q, k, v, causal=causal)

    out_ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=causal_lower_right(q_len, kv_len) if causal else None,
        enable_gqa=True,
    ).transpose(1, 2)

    torch.testing.assert_close(out, out_ref, rtol=1.6e-2, atol=1e-3)
