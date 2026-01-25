import pytest
import torch
from torch import Tensor

from gn_kernels.cuda_attn import AttnSm80Kernel


def ref_attn(q: Tensor, k: Tensor, v: Tensor):
    """Compute reference in FP64"""
    # GQA
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    n = num_q_heads // num_kv_heads

    q_f64 = q.transpose(1, 2).to(torch.float64)  # [B, L, nH, D] -> [B, nH, L, D]
    k_f64 = k.repeat_interleave(n, dim=2).transpose(1, 2).to(torch.float64)
    v_f64 = v.repeat_interleave(n, dim=2).transpose(1, 2).to(torch.float64)

    scale = q.shape[-1] ** -0.5
    s = torch.matmul(q_f64, k_f64.transpose(-1, -2))
    p = torch.softmax(s * scale, dim=-1)
    o = torch.matmul(p, v_f64)

    return o.to(q.dtype).transpose(1, 2)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
def test_cuda_attn(q_heads: int, kv_heads: int, dtype_str: int):
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16)[dtype_str]
    kernel = AttnSm80Kernel(dtype)

    bs = 2
    q_len = 512
    kv_len = 256
    head_dim = 128
    q = torch.randn(bs, q_len, q_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(bs, kv_len, kv_heads, head_dim, dtype=dtype, device="cuda") + 1.0

    actual = kernel.run(q, k, v)
    torch.cuda.synchronize()

    expected = ref_attn(q, k, v)
    torch.testing.assert_close(actual, expected)
