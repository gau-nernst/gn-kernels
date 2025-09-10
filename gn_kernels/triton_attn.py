# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _triton_attn_kernel(
    Q_ptr,  # [BS, LEN_Q, NUM_HEADS, DIM_QK]
    K_ptr,  # [BS, LEN_KV, NUM_HEADS, DIM_QK]
    V_ptr,  # [BS, LEN_KV, NUM_HEADS, DIM_V]
    O_ptr,  # [BS, LEN_Q, NUM_HEADS, DIM_V]
    scale_Q_ptr,  # [BS, NUM_HEADS, LEN_Q]
    scale_K_ptr,  # [BS, NUM_HEADS, LEN_KV]
    stride_Q,
    stride_K,
    stride_V,
    stride_O,
    LEN_Q: int,
    LEN_KV: int,
    NUM_HEADS: int,
    DIM_QK: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float | None = None,
    SCALED_QK: tl.constexpr = False,
    FP16_ACC: tl.constexpr = False,
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid_m = tl.program_id(0)  # better L2 reuse for K and V
    head_id = tl.program_id(1)
    bs_id = tl.program_id(2)

    Q_ptr += (bs_id * stride_Q[0]) + (pid_m * BLOCK_M * stride_Q[1]) + (head_id * stride_Q[2])
    K_ptr += (bs_id * stride_K[0]) + (head_id * stride_K[2])
    V_ptr += (bs_id * stride_V[0]) + (head_id * stride_V[2])
    O_ptr += (bs_id * stride_O[0]) + (pid_m * BLOCK_M * stride_O[1]) + (head_id * stride_O[2])
    if SCALED_QK:
        scale_Q_ptr += (bs_id * NUM_HEADS * LEN_Q) + (head_id * LEN_Q) + (pid_m * BLOCK_M)
        scale_K_ptr += (bs_id * NUM_HEADS * LEN_KV) + (head_id * LEN_KV)

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr,
        shape=(LEN_Q, DIM_QK),
        strides=(stride_Q[1], 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, DIM_QK),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr,
        shape=(DIM_QK, LEN_KV),
        strides=(1, stride_K[1]),
        offsets=(0, 0),
        block_shape=(DIM_QK, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr,
        shape=(LEN_KV, DIM_V),
        strides=(stride_V[1], 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM_V),
        order=(1, 0),
    )

    # initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp
    acc = tl.zeros([BLOCK_M, DIM_V], dtype=tl.float32)

    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    if sm_scale is None:
        sm_scale = DIM_QK**-0.5
    sm_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SMEM throughout
    q = tl.load(Q_block_ptr)  # [BLOCK_M, DIM_QK]

    if SCALED_QK:
        scale_q_ptrs = scale_Q_ptr + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
        scale_k_ptrs = scale_K_ptr + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]

        # fused softmax scale to scale_Q
        sm_scale = sm_scale * tl.load(scale_q_ptrs)

    # loop over k, v and update accumulator
    for _ in tl.range(0, LEN_KV, BLOCK_N):
        # 1st matmul: S = Q @ K.T
        k = tl.load(K_block_ptr)  # [DIM_QK, BLOCK_N]

        QK_DTYPE: tl.constexpr = (
            tl.int32 if (q.type == tl.int32 and k.type == tl.int32) else tl.float16 if FP16_ACC else tl.float32
        )
        s = tl.dot(q, k, out_dtype=QK_DTYPE)  # [BLOCK_M, BLOCK_N]
        s = s.to(tl.float32) * sm_scale

        if SCALED_QK:
            # NOTE: this is not pipelined
            scale_k = tl.load(scale_k_ptrs)  # [1, BLOCK_N]
            s *= scale_k
            scale_k_ptrs += BLOCK_N

        # softmax
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        s -= m_ij[:, None]
        p = tl.math.exp2(s)
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
        shape=(LEN_Q, DIM_V),
        strides=(stride_O[1], 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, DIM_V),
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


def triton_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale_q: Tensor | None = None,
    scale_k: Tensor | None = None,
    *,
    fp16_acc: bool = False,
):
    # TODO: support GQA
    BS, LEN_Q, NUM_HEADS, DIM_QK = q.shape
    _, LEN_KV, _, DIM_V = v.shape
    assert k.shape == (BS, LEN_KV, NUM_HEADS, DIM_QK)
    assert v.shape == (BS, LEN_KV, NUM_HEADS, DIM_V)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1

    def grid(args):
        return (triton.cdiv(LEN_Q, args["BLOCK_M"]), NUM_HEADS, BS)

    o = q.new_empty(BS, LEN_Q, NUM_HEADS, DIM_V, dtype=v.dtype)

    kwargs = dict(
        Q_ptr=q,
        K_ptr=k,
        V_ptr=v,
        O_ptr=o,
        scale_Q_ptr=scale_q,
        scale_K_ptr=scale_k,
        stride_Q=q.stride(),
        stride_K=k.stride(),
        stride_V=v.stride(),
        stride_O=o.stride(),
        LEN_Q=LEN_Q,
        LEN_KV=LEN_KV,
        NUM_HEADS=NUM_HEADS,
        DIM_QK=DIM_QK,
        DIM_V=DIM_V,
        FP16_ACC=fp16_acc,
    )

    if scale_q is not None and scale_k is not None:
        assert scale_q.shape == (BS, NUM_HEADS, LEN_Q)
        assert scale_k.shape == (BS, NUM_HEADS, LEN_KV)
        assert scale_q.is_contiguous()
        assert scale_k.is_contiguous()

        triton_scaled_qk_attn_kernel[grid](**kwargs)

    else:
        triton_attn_kernel[grid](**kwargs)

    return o
