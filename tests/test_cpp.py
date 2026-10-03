import pytest
import torch
from torch import Tensor

from gn_kernels.cpp import Sm80AttnKernel, Sm80MatmulKernel
from gn_kernels.cpp.nvrtc_utils import int4x2, uint4x2


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
    kernel = Sm80AttnKernel(dtype)

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


@pytest.mark.parametrize("dtype_str", ["fp16", "bf16"])
def test_cuda_mm_fp(dtype_str: str):
    dtype = dict(fp16=torch.float16, bf16=torch.bfloat16)[dtype_str]
    kernel = Sm80MatmulKernel(dtype, dtype, torch.float32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda").T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    expected = torch.mm(A, B)
    torch.testing.assert_close(actual, expected)


def test_cuda_mm_int8():
    kernel = Sm80MatmulKernel(torch.int8, torch.int32, torch.int32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda").T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    expected = torch._int_mm(A, B)
    torch.testing.assert_close(actual, expected)


def test_cuda_mm_fp8():
    dtype = torch.float8_e4m3fn
    kernel = Sm80MatmulKernel(dtype, torch.bfloat16, torch.float32, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, device="cuda").to(dtype)
    B = torch.randn(N, K, device="cuda").to(dtype).T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    scale = torch.ones(1, device="cuda")
    expected = torch._scaled_mm(A, B, scale, scale, out_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype_str", ["fp16", "fp8"])
def test_cuda_mm_fp16_acc(dtype_str: str):
    dtype = dict(fp16=torch.float16, fp8=torch.float8_e4m3fn)[dtype_str]
    kernel = Sm80MatmulKernel(dtype, torch.float16, torch.float16, num_stages=2)

    M, N, K = 512, 768, 1024
    A = torch.randn(M, K, device="cuda").to(dtype)
    B = torch.randn(N, K, device="cuda").to(dtype).T

    actual = kernel.run(A, B)
    torch.cuda.synchronize()

    # simulate FP16 accumulation
    expected = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    mma_k = 32 // dtype.itemsize
    for offset_k in range(0, K, mma_k):
        A_tile = A[:, offset_k : offset_k + mma_k].float()
        B_tile = B[offset_k : offset_k + mma_k, :].float()
        expected.add_(torch.mm(A_tile, B_tile))

    # doesn't pass for FP16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype_str", ["int4", "uint4"])
def test_cuda_mm_int4(dtype_str: str):
    dtype, low, high = dict(
        int4=(int4x2, -8, 7),
        uint4=(uint4x2, 0, 15),
    )[dtype_str]
    kernel = Sm80MatmulKernel(dtype, torch.int32, torch.int32, num_stages=2)

    def pack_int4(x: torch.Tensor):
        return (x[:, 1::2] << 4) | (x[:, ::2] & 0xF)

    M, N, K = 512, 768, 1024
    A = torch.randint(low, high, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(low, high, (N, K), dtype=torch.int8, device="cuda").T

    A_packed = pack_int4(A)
    B_packed = pack_int4(B.T).T

    actual = kernel.run(A_packed, B_packed)
    torch.cuda.synchronize()

    expected = torch._int_mm(A, B)
    torch.testing.assert_close(actual, expected)
