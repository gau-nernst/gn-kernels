import pytest
import torch
import torch.nn.functional as F

from gn_kernels.cuda_attn import AttnKernel


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
def test_cuda_attn(q_heads: int, kv_heads: int, dtype_str: int):
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16)[dtype_str]
    kernel = AttnKernel(dtype)

    bs = 2
    q_len = 512
    kv_len = 256
    head_dim = 128
    q = torch.randn(bs, q_len, q_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=dtype, device="cuda") + 0.6

    actual = kernel.run(q, k, v)
    torch.cuda.synchronize()

    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
