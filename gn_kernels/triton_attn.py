# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    pid_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    LEN_KV: tl.constexpr,
):
    # range of KV
    # NOTE: causal case doesn't handle LEN_Q != LEN_KV
    if STAGE == 1:
        # causal = True. no masking within blocks
        lo = 0
        hi = pid_m * BLOCK_M
    elif STAGE == 2:
        # causal = False. masking within blocks
        lo = tl.multiple_of(pid_m * BLOCK_M, BLOCK_M)
        hi = (pid_m + 1) * BLOCK_M
    else:
        # causal = False. attend to full KV. no masking
        lo = 0
        hi = LEN_KV

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 1st matmul - q @ k.t
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)
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

    return acc, l_i, m_i


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
    M_ptr,  # [BS, LEN_Q]
    O_ptr,  # [BS, LEN_Q, HEAD_DIM]
    sm_scale,
    LEN_Q,
    LEN_KV,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
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

    # initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

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

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            pid_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            LEN_KV,
        )

    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            pid_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            LEN_KV,
        )

    # epilogue
    m_i += tl.math.log2(l_i)  # logsumexp + max
    acc = acc / l_i[:, None]
    m_ptrs = M_ptr + pid_bs * LEN_Q + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(O_ptr.type.element_ty))


def triton_attn(q: Tensor, k: Tensor, v: Tensor, causal: bool = False):
    BS, nH, LEN_Q, HEAD_DIM = q.shape
    LEN_KV = k.shape[2]
    assert HEAD_DIM in (16, 32, 64, 128, 256)
    assert k.shape[-1] == v.shape[-1] == HEAD_DIM
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    stage = 3 if causal else 1
    sm_scale = HEAD_DIM**-0.5

    M = torch.empty((BS, nH, LEN_Q), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    def grid(args):
        return (triton.cdiv(LEN_Q, args["BLOCK_M"]), BS * nH, 1)

    _attn_fwd[grid](q, k, v, M, o, sm_scale, LEN_Q, LEN_KV, HEAD_DIM, stage)

    return o
