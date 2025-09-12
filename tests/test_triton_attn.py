import torch
import torch.nn.functional as F

from gn_kernels.triton_attn import triton_attn


def test_triton_attn():
    bs = 2
    num_heads_q = 8
    num_heads_kv = 4
    head_dim = 128
    len_q = 64
    len_kv = 256

    q = torch.randn(bs, len_q, num_heads_q, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(bs, len_kv, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(bs, len_kv, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    out = triton_attn(q, k, v)

    out_ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)

    torch.testing.assert_close(out, out_ref, rtol=1.6e-2, atol=1e-3)
