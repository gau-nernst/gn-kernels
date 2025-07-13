# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import torch
import triton
import triton.language as tl
from torch import Tensor

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
]


@triton.autotune(configs, key=["LEN_Q", "LEN_KV", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q_ptr,  # [BS, LEN_Q, HEAD_DIM]
    K_ptr,  # [BS, LEN_KV, HEAD_DIM]
    V_ptr,  # [BS, LEN_KV, HEAD_DIM]
    O_ptr,  # [BS, LEN_Q, HEAD_DIM]
    sm_scale,
    LEN_Q,
    LEN_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1)

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_bs * (LEN_Q * HEAD_DIM),
        shape=(LEN_Q, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_bs * (LEN_KV * HEAD_DIM),
        shape=(HEAD_DIM, LEN_KV),
        strides=(1, HEAD_DIM),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_bs * (LEN_KV * HEAD_DIM),
        shape=(LEN_KV, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + pid_bs * (LEN_Q * HEAD_DIM),
        shape=(LEN_Q, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # loop over k, v and update accumulator
    for _ in range(0, LEN_KV, BLOCK_N):
        # 1st matmul - q @ k.t
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)  # rescale factor
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # 2nd matmul - p @ v
        p = p.to(V_block_ptr.type.element_ty)
        v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O_ptr.type.element_ty))


def triton_attn(q: Tensor, k: Tensor, v: Tensor):
    BS, nH, LEN_Q, HEAD_DIM = q.shape
    LEN_KV = k.shape[2]
    assert k.shape[-1] == v.shape[-1] == HEAD_DIM
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    sm_scale = HEAD_DIM**-0.5
    o = torch.empty_like(q)

    def grid(args):
        return (triton.cdiv(LEN_Q, args["BLOCK_M"]), BS * nH, 1)

    _attn_fwd[grid](q, k, v, o, sm_scale, LEN_Q, LEN_KV, HEAD_DIM)

    return o
