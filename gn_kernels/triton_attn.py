# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _triton_attn_kernel(
    Q_ptr,  # [BS, LEN_Q, NUM_HEADS, HEAD_DIM]
    K_ptr,  # [BS, LEN_KV, NUM_HEADS, HEAD_DIM]
    V_ptr,  # [BS, LEN_KV, NUM_HEADS, HEAD_DIM]
    O_ptr,  # [BS, LEN_Q, NUM_HEADS, HEAD_DIM]
    scale_Q_ptr,  # [BS, LEN_Q, NUM_HEADS]
    scale_K_ptr,  # [BS, LEN_KV, NUM_HEADS]
    LEN_Q: int,
    LEN_KV: int,
    NUM_HEADS: int,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float | None = None,
    SCALED_QK: tl.constexpr = False,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid_m = tl.program_id(0)  # better L2 reuse for K and V
    pid_head = tl.program_id(1)
    pid_bs = tl.program_id(2)

    Q_ptr += (
        (pid_bs * LEN_Q * NUM_HEADS * HEAD_DIM) + ((pid_m * BLOCK_M) * NUM_HEADS * HEAD_DIM) + (pid_head * HEAD_DIM)
    )
    K_ptr += (pid_bs * LEN_KV * NUM_HEADS * HEAD_DIM) + (pid_head * HEAD_DIM)
    V_ptr += (pid_bs * LEN_KV * NUM_HEADS * HEAD_DIM) + (pid_head * HEAD_DIM)
    O_ptr += (
        (pid_bs * LEN_Q * NUM_HEADS * HEAD_DIM) + ((pid_m * BLOCK_M) * NUM_HEADS * HEAD_DIM) + (pid_head * HEAD_DIM)
    )
    if SCALED_QK:
        scale_Q_ptr += (pid_bs * LEN_Q * NUM_HEADS) + ((pid_m * BLOCK_M) * NUM_HEADS) + pid_head
        scale_K_ptr += (pid_bs * LEN_KV * NUM_HEADS) + pid_head

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr,
        shape=(LEN_Q, HEAD_DIM),
        strides=(NUM_HEADS * HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr,
        shape=(HEAD_DIM, LEN_KV),
        strides=(1, NUM_HEADS * HEAD_DIM),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr,
        shape=(LEN_KV, HEAD_DIM),
        strides=(NUM_HEADS * HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    # initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    if sm_scale is None:
        sm_scale = HEAD_DIM**-0.5
    sm_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SMEM throughout
    q = tl.load(Q_block_ptr)  # [BLOCK_M, HEAD_DIM]

    if SCALED_QK:
        scale_q_ptrs = scale_Q_ptr + tl.arange(0, BLOCK_M)[:, None] * NUM_HEADS  # [BLOCK_M, 1]
        scale_k_ptrs = scale_K_ptr + tl.arange(0, BLOCK_N)[None, :] * NUM_HEADS  # [1, BLOCK_N]

        # fused softmax scale to scale_Q
        sm_scale = sm_scale * tl.load(scale_q_ptrs)

    # loop over k, v and update accumulator
    for _ in tl.range(0, LEN_KV, BLOCK_N):
        # 1st matmul: S = Q @ K.T
        k = tl.load(K_block_ptr)  # [HEAD_DIM, BLOCK_N]

        out_dtype: tl.constexpr = tl.int32 if q.type == tl.int8 else tl.float32
        qk = tl.dot(q, k, out_dtype=out_dtype)  # [BLOCK_M, BLOCK_N]
        qk = qk.to(tl.float32) * sm_scale

        if SCALED_QK:
            # NOTE: this is not pipelined
            scale_k = tl.load(scale_k_ptrs)  # [1, BLOCK_N]
            qk *= scale_k
            scale_k_ptrs += BLOCK_N * NUM_HEADS

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)  # rescale factor
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # 2nd matmul: O += P @ V
        p = p.to(V_block_ptr.type.element_ty)
        v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # epilogue
    acc = acc / l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr,
        shape=(LEN_Q, HEAD_DIM),
        strides=(NUM_HEADS * HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(O_ptr.type.element_ty))


triton_attn_kernel = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN), num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["LEN_Q", "LEN_KV", "HEAD_DIM"],
)(_triton_attn_kernel)


triton_scaled_qk_attn_kernel = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, SCALED_QK=True), num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["LEN_Q", "LEN_KV", "HEAD_DIM"],
)(_triton_attn_kernel)


def triton_attn(q: Tensor, k: Tensor, v: Tensor, scale_q: Tensor | None = None, scale_k: Tensor | None = None):
    # TODO: support GQA
    BS, LEN_Q, NUM_HEADS, HEAD_DIM = q.shape
    LEN_KV = k.shape[1]
    assert k.shape == (BS, LEN_KV, NUM_HEADS, HEAD_DIM)
    assert v.shape == (BS, LEN_KV, NUM_HEADS, HEAD_DIM)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    def grid(args):
        return (triton.cdiv(LEN_Q, args["BLOCK_M"]), NUM_HEADS, BS)

    o = torch.empty_like(q)

    if scale_q is not None and scale_k is not None:
        assert scale_q.shape == (BS, LEN_Q, NUM_HEADS)
        assert scale_k.shape == (BS, LEN_KV, NUM_HEADS)
        assert scale_q.is_contiguous()
        assert scale_k.is_contiguous()

        triton_scaled_qk_attn_kernel[grid](q, k, v, o, scale_q, scale_k, LEN_Q, LEN_KV, NUM_HEADS, HEAD_DIM)

    else:
        triton_attn_kernel[grid](q, k, v, o, scale_q, scale_k, LEN_Q, LEN_KV, NUM_HEADS, HEAD_DIM)

    return o
