# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _forward_block(
    # attention state
    acc,
    m_i,
    l_i,
    # inputs
    q,
    K_block_ptr,
    V_block_ptr,
    mask,
    scale_k_ptrs,
    sm_scale,
    SCALED_QK: tl.constexpr,
    QK_ACC_DTYPE: tl.constexpr,
    PV_ACC_DTYPE: tl.constexpr,
    BOUNDS_CHECK: tl.constexpr,
):
    # 1st matmul: S = Q @ K.T
    k = tl.load(K_block_ptr, mask[None, :] if BOUNDS_CHECK else None)
    s = tl.dot(q, k, out_dtype=QK_ACC_DTYPE)  # [BLOCK_M, BLOCK_N]
    s = s.to(tl.float32) * sm_scale

    if SCALED_QK:
        # NOTE: this is not pipelined
        scale_k = tl.load(scale_k_ptrs)  # [1, BLOCK_N]
        s *= scale_k

    if BOUNDS_CHECK:
        s = tl.where(mask[None, :], s, float("-inf"))

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
    v = tl.load(V_block_ptr, mask[:, None] if BOUNDS_CHECK else None)  # [BLOCK_N, DIM_V]
    p = p.to(v.type)

    if PV_ACC_DTYPE == tl.float32:
        acc = tl.dot(p, v, acc)
    else:
        # += will do addition with CUDA cores
        acc += tl.dot(p, v, out_dtype=PV_ACC_DTYPE).to(tl.float32)

    return acc, m_i, l_i


@triton.heuristics(dict(BOUNDS_CHECK=lambda args: args["LEN_KV"] % args["BLOCK_N"] > 0))
@triton.jit
def triton_attn_kernel(
    Q_ptr,  # [BS, LEN_Q, NUM_HEADS_Q, DIM_QK]
    K_ptr,  # [BS, LEN_KV, NUM_HEADS_KV, DIM_QK]
    V_ptr,  # [BS, LEN_KV, NUM_HEADS_KV, DIM_V]
    O_ptr,  # [BS, LEN_Q, NUM_HEADS_Q, DIM_V]
    scale_Q_ptr,  # [BS, NUM_HEADS_Q, LEN_Q]
    scale_K_ptr,  # [BS, NUM_HEADS_KV, LEN_KV]
    stride_Q,
    stride_K,
    stride_V,
    stride_O,
    # problem shape
    LEN_Q: int,
    LEN_KV: int,
    NUM_HEADS_Q: int,
    NUM_HEADS_KV: int,
    DIM_QK: tl.constexpr,
    DIM_V: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # others
    sm_scale: float | None = None,
    SCALED_QK: tl.constexpr = False,
    QK_ACC_DTYPE: tl.constexpr = tl.float32,
    PV_ACC_DTYPE: tl.constexpr = tl.float32,
    BOUNDS_CHECK: tl.constexpr = False,
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    head_id_q = tl.program_id(0)  # L2 reuse of KV across heads
    head_id_kv = head_id_q // (NUM_HEADS_Q // NUM_HEADS_KV)
    pid_m = tl.program_id(1)
    bs_id = tl.program_id(2)

    Q_ptr += (bs_id * stride_Q[0]) + (pid_m * BLOCK_M * stride_Q[1]) + (head_id_q * stride_Q[2])
    O_ptr += (bs_id * stride_O[0]) + (pid_m * BLOCK_M * stride_O[1]) + (head_id_q * stride_O[2])
    K_ptr += (bs_id * stride_K[0]) + (head_id_kv * stride_K[2])
    V_ptr += (bs_id * stride_V[0]) + (head_id_kv * stride_V[2])
    if SCALED_QK:
        scale_Q_ptr += (bs_id * NUM_HEADS_Q * LEN_Q) + (head_id_q * LEN_Q) + (pid_m * BLOCK_M)
        scale_K_ptr += (bs_id * NUM_HEADS_KV * LEN_KV) + (head_id_kv * LEN_KV)

    # initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp
    acc = tl.zeros([BLOCK_M, DIM_V], dtype=tl.float32)

    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    if sm_scale is None:
        sm_scale = DIM_QK**-0.5
    sm_scale *= 1.44269504  # 1/log(2)

    # load q. it will stay in rmem throughout
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dim_qk = tl.arange(0, DIM_QK)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_Q[1] + offs_dim_qk[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < LEN_Q)  # [BLOCK_M, DIM_QK]

    k_ptrs = K_ptr + offs_dim_qk[:, None] + offs_n[None, :] * stride_K[1]
    v_ptrs = V_ptr + offs_n[:, None] * stride_V[1] + tl.arange(0, DIM_V)[None, :]

    if SCALED_QK:
        scale_q_ptrs = scale_Q_ptr + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
        scale_k_ptrs = scale_K_ptr + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]

        # fused softmax scale to scale_Q
        sm_scale = sm_scale * tl.load(scale_q_ptrs)

    else:
        scale_k_ptrs = None

    # for causal, we align the last q with the last kv i.e.
    # - q[0]       will attend to kv[0:LEN_KV-LEN_Q]
    # - q[LEN_Q-1] will attend to kv[0:LEN_KV] (full)
    # if LEN_Q > LEN_KV, some q will attend to nothing -> result will be zeros

    # loop over k, v and update accumulator
    for kv_id in range(0, tl.cdiv(LEN_KV, BLOCK_N)):
        acc, m_i, l_i = _forward_block(
            acc,
            m_i,
            l_i,
            q,
            k_ptrs,
            v_ptrs,
            offs_n < LEN_KV - kv_id * BLOCK_N,
            scale_k_ptrs,
            sm_scale,
            SCALED_QK,
            QK_ACC_DTYPE,
            PV_ACC_DTYPE,
            BOUNDS_CHECK=BOUNDS_CHECK,
        )
        k_ptrs += BLOCK_N * stride_K[1]
        v_ptrs += BLOCK_N * stride_V[1]
        if SCALED_QK:
            scale_k_ptrs += BLOCK_N

    # epilogue
    acc = acc / l_i[:, None]
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    o_ptrs = O_ptr + offs_m * stride_O[1] + tl.arange(0, DIM_V)
    tl.store(o_ptrs, acc, mask=offs_m < LEN_Q - pid_m * BLOCK_M)


triton_attn_kernel_autotune = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN), num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["LEN_Q", "LEN_KV", "HEAD_DIM"],
)(triton_attn_kernel)


triton_scaled_qk_attn_kernel_autotune = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, SCALED_QK=True), num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["LEN_Q", "LEN_KV", "HEAD_DIM"],
)(triton_attn_kernel)


def triton_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale_q: Tensor | None = None,
    scale_k: Tensor | None = None,
    *,
    causal: bool = False,
    qk_fp16_acc: bool = False,
    pv_fp16_acc: bool = False,
):
    BS, LEN_Q, NUM_HEADS_Q, DIM_QK = q.shape
    _, LEN_KV, NUM_HEADS_KV, DIM_V = v.shape
    assert NUM_HEADS_Q % NUM_HEADS_KV == 0
    assert k.shape == (BS, LEN_KV, NUM_HEADS_KV, DIM_QK)
    assert v.shape == (BS, LEN_KV, NUM_HEADS_KV, DIM_V)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1

    def grid(args):
        return (NUM_HEADS_Q, triton.cdiv(LEN_Q, args["BLOCK_M"]), BS)

    o = q.new_empty(BS, LEN_Q, NUM_HEADS_Q, DIM_V, dtype=v.dtype)

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
        NUM_HEADS_Q=NUM_HEADS_Q,
        NUM_HEADS_KV=NUM_HEADS_KV,
        DIM_QK=DIM_QK,
        DIM_V=DIM_V,
        QK_ACC_DTYPE=tl.float16 if qk_fp16_acc else tl.float32 if q.is_floating_point() else tl.int32,
        PV_ACC_DTYPE=tl.float16 if pv_fp16_acc else tl.float32,
    )

    if scale_q is not None and scale_k is not None:
        assert scale_q.shape == (BS, NUM_HEADS_Q, LEN_Q)
        assert scale_k.shape == (BS, NUM_HEADS_KV, LEN_KV)
        assert scale_q.is_contiguous()
        assert scale_k.is_contiguous()
        triton_scaled_qk_attn_kernel_autotune[grid](**kwargs)

    else:
        triton_attn_kernel_autotune[grid](**kwargs)

    return o
