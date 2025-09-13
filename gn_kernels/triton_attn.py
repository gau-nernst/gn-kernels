# https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _forward_loop(
    attn_state,
    limits,
    q,
    k_ptrs,
    v_ptrs,
    scale_k_ptrs,
    k_stride,
    v_stride,
    offs_m,
    sm_scale,
    CONSTEXPR_ARGS: tl.constexpr,
):
    acc, m_i, l_i = attn_state
    start, end, KV_LEN = limits
    BLOCK_N, CAUSAL, SCALED_QK, QK_ACC_DTYPE, PV_ACC_DTYPE, BOUNDS_CHECK = CONSTEXPR_ARGS

    offs_n = start + tl.arange(0, BLOCK_N)

    for _ in range(tl.cdiv(end - start, BLOCK_N)):
        kv_mask = offs_n < KV_LEN

        # 1st matmul: S = Q @ K.T
        k = tl.load(k_ptrs, kv_mask[None, :] if BOUNDS_CHECK else None)
        s = tl.dot(q, k, out_dtype=QK_ACC_DTYPE)  # [BLOCK_M, BLOCK_N]
        s = s.to(tl.float32) * sm_scale

        if SCALED_QK:  # NOTE: this is not pipelined
            s *= tl.load(scale_k_ptrs)  # [1, BLOCK_N]

        # TODO: merge the masks?
        if BOUNDS_CHECK:
            s = tl.where(kv_mask[None, :], s, float("-inf"))

        # FLexAttention's mask_mod
        if CAUSAL:
            mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask, s, float("-inf"))

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
        v = tl.load(v_ptrs, kv_mask[:, None] if BOUNDS_CHECK else None)  # [BLOCK_N, DIM_V]
        p = p.to(v.type)

        if PV_ACC_DTYPE == tl.float32:
            acc = tl.dot(p, v, acc)
        else:
            # += will do addition with CUDA cores
            acc += tl.dot(p, v, out_dtype=PV_ACC_DTYPE).to(tl.float32)

        k_ptrs += BLOCK_N * k_stride
        v_ptrs += BLOCK_N * v_stride
        offs_n += BLOCK_N
        if SCALED_QK:
            scale_k_ptrs += BLOCK_N

    return acc, m_i, l_i


# TODO: change this to unify with varlen kernel
@triton.heuristics(dict(BOUNDS_CHECK=lambda args: args["KV_LEN"] % args["BLOCK_N"] > 0))
@triton.jit
def triton_attn_kernel(
    Q_ptr,  # [BS, Q_LEN, Q_HEADS, QK_DIM]
    K_ptr,  # [BS, KV_LEN, KV_HEADS, QK_DIM]
    V_ptr,  # [BS, KV_LEN, KV_HEADS, V_DIM]
    O_ptr,  # [BS, Q_LEN, Q_HEADS, V_DIM]
    scale_Q_ptr,  # [BS, Q_HEADS, Q_LEN]
    scale_K_ptr,  # [BS, KV_HEADS, LEN_KV]
    Q_stride,
    K_stride,
    V_stride,
    O_stride,
    # problem shape
    Q_LEN: int,
    KV_LEN: int,
    Q_HEADS: int,
    KV_HEADS: int,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # others
    sm_scale: float | None = None,
    CAUSAL: tl.constexpr = False,
    SCALED_QK: tl.constexpr = False,
    QK_ACC_DTYPE: tl.constexpr = tl.float32,
    PV_ACC_DTYPE: tl.constexpr = tl.float32,
    BOUNDS_CHECK: tl.constexpr = False,
):
    head_id_q = tl.program_id(0)  # L2 reuse of KV across heads
    head_id_kv = head_id_q // (Q_HEADS // KV_HEADS)
    pid_m = tl.program_id(1)
    bs_id = tl.program_id(2)

    # select the input/output
    Q_ptr += bs_id * Q_stride[0] + head_id_q * Q_stride[2]
    O_ptr += bs_id * O_stride[0] + head_id_q * O_stride[2]
    K_ptr += bs_id * K_stride[0] + head_id_kv * K_stride[2]
    V_ptr += bs_id * V_stride[0] + head_id_kv * V_stride[2]
    if SCALED_QK:
        scale_Q_ptr += (bs_id * Q_HEADS + head_id_q) * Q_LEN
        scale_K_ptr += (bs_id * KV_HEADS + head_id_kv) * KV_LEN

    # initialize attention state
    # don't use -inf for initial value of max to avoid (-inf) - (-inf) = nan
    acc = tl.zeros([BLOCK_M, V_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e38  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp

    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    if sm_scale is None:
        sm_scale = QK_DIM**-0.5
    sm_scale *= 1.44269504  # 1/log(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dim_qk = tl.arange(0, QK_DIM)
    offs_dim_v = tl.arange(0, V_DIM)

    q_ptrs = Q_ptr + offs_m[:, None] * Q_stride[1] + offs_dim_qk[None, :]  # [BLOCK_M, DIM_QK]
    k_ptrs = K_ptr + offs_n[None, :] * K_stride[1] + offs_dim_qk[:, None]  # [QK_DIM, BLOCK_N]
    v_ptrs = V_ptr + offs_n[:, None] * V_stride[1] + offs_dim_v[None, :]  # [BLOCK_N, V_DIM]

    # load q. it will stay in rmem throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    if SCALED_QK:
        scale_q_ptrs = scale_Q_ptr + offs_m[:, None]  # [BLOCK_M, 1]
        scale_k_ptrs = scale_K_ptr + offs_n[None, :]  # [1, BLOCK_N]

        # fused softmax scale to scale_Q
        sm_scale = sm_scale * tl.load(scale_q_ptrs)

    else:
        scale_k_ptrs = None

    if not CAUSAL:
        # for non-causal, we need to do bounds check for the last block to avoid out-of-bounds read.
        # just do bounds check for all blocks for now, since separating out the last block may cause
        # problems with pipelining.
        end = KV_LEN

    else:
        # for causal, we align the last q with the last kv i.e. causal lower right. this is FA's
        # default behavior to make decode easier. in other words:
        # - q[0]       will attend to kv[0 : KV_LEN-Q_LEN+1]
        # - q[Q_LEN-1] will attend to kv[0 : KV_LEN] (full)
        #
        # we will split KV loop into 2:
        # 1. full KV blocks: don't need to apply causal mask within a block
        # 2. partial KV block: need to apply causal mask within a block
        #
        # this threadblock is responsible for q[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M]
        # we have full KV blocks for kv[0 : KV_LEN-Q_LEN + pid_m * BLOCK_M]
        # and partial KV blocks for the remaining.
        # we also enable bounds check for the partial blocks loop, since the partial blocks are
        # guaranteed to contain the last block.

        # end is rounded down to a multiple of BLOCK_N
        end = (KV_LEN - Q_LEN + pid_m * BLOCK_M) // BLOCK_N * BLOCK_N

    # full blocks
    # always set CAUSAL=False and only set BOUNDS_CHECK for non-causal
    FULL_ARGS: tl.constexpr = (BLOCK_N, False, SCALED_QK, QK_ACC_DTYPE, PV_ACC_DTYPE, False if CAUSAL else BOUNDS_CHECK)
    acc, m_i, l_i = _forward_loop(
        (acc, m_i, l_i),
        (0, end, KV_LEN),
        q,
        k_ptrs,
        v_ptrs,
        scale_k_ptrs,
        K_stride[1],
        V_stride[1],
        None,  # offs_m
        sm_scale,
        FULL_ARGS,
    )

    if CAUSAL:
        # handle partial blocks in causal attention
        start = end
        end = KV_LEN - Q_LEN + (pid_m + 1) * BLOCK_M

        k_ptrs += start * K_stride[1]
        v_ptrs += start * V_stride[1]
        if SCALED_QK:
            scale_k_ptrs += start

        # this is only used for causal mask within a block
        # to use causal upper left, remove this line
        offs_m += KV_LEN - Q_LEN

        PARTIAL_ARGS: tl.constexpr = (BLOCK_N, CAUSAL, SCALED_QK, QK_ACC_DTYPE, PV_ACC_DTYPE, BOUNDS_CHECK)
        acc, m_i, l_i = _forward_loop(
            (acc, m_i, l_i),
            (start, end, KV_LEN),
            q,
            k_ptrs,
            v_ptrs,
            scale_k_ptrs,
            K_stride[1],
            V_stride[1],
            offs_m,
            sm_scale,
            PARTIAL_ARGS,
        )

    # epilogue
    acc = acc / l_i[:, None]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    # mask out the elements that don't attend to anything.
    # i think if the code above is implemented correctly, i don't
    # actually need this. but for some reasons, i do...
    if CAUSAL:
        acc = tl.where(KV_LEN - Q_LEN + offs_m >= 0, acc, 0.0)

    offs_dim_v = tl.arange(0, V_DIM)[None, :]
    o_ptrs = O_ptr + offs_m * O_stride[1] + offs_dim_v
    tl.store(o_ptrs, acc, mask=offs_m < Q_LEN)


triton_attn_kernel_autotune = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN), num_stages=s, num_warps=w)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["Q_LEN", "KV_LEN", "QK_DIM", "V_DIM", "CAUSAL", "QK_ACC_DTYPE", "PV_ACC_DTYPE"],
)(triton_attn_kernel)


triton_scaled_qk_attn_kernel_autotune = triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, SCALED_QK=True), num_stages=s, num_warps=w)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["Q_LEN", "KV_LEN", "QK_DIM", "V_DIM", "CAUSAL", "QK_ACC_DTYPE", "PV_ACC_DTYPE"],
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
    BS, Q_LEN, Q_HEADS, QK_DIM = q.shape
    _, KV_LEN, KV_HEADS, V_DIM = v.shape
    assert Q_HEADS % KV_HEADS == 0
    assert k.shape == (BS, KV_LEN, KV_HEADS, QK_DIM)
    assert v.shape == (BS, KV_LEN, KV_HEADS, V_DIM)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1

    def grid(args):
        return (Q_HEADS, triton.cdiv(Q_LEN, args["BLOCK_M"]), BS)

    o = q.new_empty(BS, Q_LEN, Q_HEADS, V_DIM, dtype=v.dtype)

    kwargs = dict(
        Q_ptr=q,
        K_ptr=k,
        V_ptr=v,
        O_ptr=o,
        scale_Q_ptr=scale_q,
        scale_K_ptr=scale_k,
        Q_stride=q.stride(),
        K_stride=k.stride(),
        V_stride=v.stride(),
        O_stride=o.stride(),
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        Q_HEADS=Q_HEADS,
        KV_HEADS=KV_HEADS,
        QK_DIM=QK_DIM,
        V_DIM=V_DIM,
        CAUSAL=causal,
        QK_ACC_DTYPE=tl.float16 if qk_fp16_acc else tl.float32 if q.is_floating_point() else tl.int32,
        PV_ACC_DTYPE=tl.float16 if pv_fp16_acc else tl.float32,
    )

    if scale_q is not None and scale_k is not None:
        assert scale_q.shape == (BS, Q_HEADS, Q_LEN)
        assert scale_k.shape == (BS, KV_HEADS, KV_LEN)
        assert scale_q.is_contiguous()
        assert scale_k.is_contiguous()
        triton_scaled_qk_attn_kernel_autotune[grid](**kwargs)

    else:
        triton_attn_kernel_autotune[grid](**kwargs)

    return o


@triton.jit
def triton_varlen_attn_kernel(
    Q_ptr,  # [Q_LEN_TOTAL, Q_HEADS, QK_DIM]
    K_ptr,  # [KV_LEN_TOTAL, KV_HEADS, QK_DIM]
    V_ptr,  # [KV_LEN_TOTAL, KV_HEADS, V_DIM]
    O_ptr,  # [Q_LEN_TOTAL, Q_HEADS, V_DIM]
    Q_stride,
    K_stride,
    V_stride,
    O_stride,
    Q_offsets_ptr,  # [BS+1]
    KV_offsets_ptr,  # [BS+1]
    # problem shape
    Q_HEADS: int,
    KV_HEADS: int,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # others
    sm_scale: float | None = None,
    CAUSAL: tl.constexpr = False,
    QK_ACC_DTYPE: tl.constexpr = tl.float32,
    PV_ACC_DTYPE: tl.constexpr = tl.float32,
):
    SCALED_QK: tl.constexpr = False
    BOUNDS_CHECK: tl.constexpr = True

    head_id_q = tl.program_id(0)  # L2 reuse of KV across heads
    head_id_kv = head_id_q // (Q_HEADS // KV_HEADS)
    pid_m = tl.program_id(1)
    bs_id = tl.program_id(2)

    Q_start = tl.load(Q_offsets_ptr + bs_id)
    Q_end = tl.load(Q_offsets_ptr + bs_id + 1)
    Q_LEN = Q_end - Q_start

    KV_start = tl.load(KV_offsets_ptr + bs_id)
    KV_end = tl.load(KV_offsets_ptr + bs_id + 1)
    KV_LEN = KV_end - KV_start

    # early exit
    if pid_m * BLOCK_M > Q_end:
        return

    # select the input/output
    Q_ptr += Q_start * Q_stride[0] + head_id_q * Q_stride[1]
    O_ptr += Q_start * O_stride[0] + head_id_q * O_stride[1]
    K_ptr += KV_start * K_stride[0] + head_id_kv * K_stride[1]
    V_ptr += KV_start * V_stride[0] + head_id_kv * V_stride[1]

    # initialize attention state
    acc = tl.zeros([BLOCK_M, V_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # sumexp

    # e^x = e^(log(2) * x / log(2)) = 2^(x/log(2))
    if sm_scale is None:
        sm_scale = QK_DIM**-0.5
    sm_scale *= 1.44269504  # 1/log(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dim_qk = tl.arange(0, QK_DIM)
    offs_dim_v = tl.arange(0, V_DIM)

    q_ptrs = Q_ptr + offs_m[:, None] * Q_stride[0] + offs_dim_qk[None, :]  # [BLOCK_M, DIM_QK]
    k_ptrs = K_ptr + offs_n[None, :] * K_stride[0] + offs_dim_qk[:, None]  # [QK_DIM, BLOCK_N]
    v_ptrs = V_ptr + offs_n[:, None] * V_stride[0] + offs_dim_v[None, :]  # [BLOCK_N, V_DIM]

    # load q. it will stay in rmem throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    if not CAUSAL:
        end = KV_LEN
    else:
        end = (KV_LEN - Q_LEN + pid_m * BLOCK_M) // BLOCK_N * BLOCK_N

    # full blocks
    FULL_ARGS: tl.constexpr = (BLOCK_N, False, SCALED_QK, QK_ACC_DTYPE, PV_ACC_DTYPE, False if CAUSAL else BOUNDS_CHECK)
    acc, m_i, l_i = _forward_loop(
        (acc, m_i, l_i),
        (0, end, KV_LEN),
        q,
        k_ptrs,
        v_ptrs,
        None,  # scale_k_ptrs
        K_stride[0],
        V_stride[0],
        None,  # offs_m
        sm_scale,
        FULL_ARGS,
    )

    if CAUSAL:
        # handle partial blocks in causal attention
        start = end
        end = KV_LEN - Q_LEN + (pid_m + 1) * BLOCK_M

        k_ptrs += start * K_stride[0]
        v_ptrs += start * V_stride[0]

        # this is only used for causal mask within a block
        # to use causal upper left, remove this line
        offs_m += KV_LEN - Q_LEN

        PARTIAL_ARGS: tl.constexpr = (BLOCK_N, CAUSAL, SCALED_QK, QK_ACC_DTYPE, PV_ACC_DTYPE, BOUNDS_CHECK)
        acc, m_i, l_i = _forward_loop(
            (acc, m_i, l_i),
            (start, end, KV_LEN),
            q,
            k_ptrs,
            v_ptrs,
            None,  # scale_k_ptrs
            K_stride[0],
            V_stride[0],
            offs_m,
            sm_scale,
            PARTIAL_ARGS,
        )

    # epilogue
    acc = acc / l_i[:, None]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    if CAUSAL:
        acc = tl.where(KV_LEN - Q_LEN + offs_m >= 0, acc, 0.0)

    offs_dim_v = tl.arange(0, V_DIM)[None, :]
    o_ptrs = O_ptr + offs_m * O_stride[0] + offs_dim_v
    tl.store(o_ptrs, acc, mask=offs_m < Q_LEN)


def triton_varlen_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_offsets: Tensor,
    kv_offsets: Tensor,
    max_q_len: int,
    *,
    causal: bool = False,
    qk_fp16_acc: bool = False,
    pv_fp16_acc: bool = False,
):
    Q_LEN_TOTAL, Q_HEADS, QK_DIM = q.shape
    KV_LEN_TOTAL, KV_HEADS, V_DIM = v.shape
    BS = q_offsets.shape[0] - 1
    assert Q_HEADS % KV_HEADS == 0
    assert k.shape == (KV_LEN_TOTAL, KV_HEADS, QK_DIM)
    assert kv_offsets.shape[0] == BS + 1
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert q_offsets.is_contiguous()
    assert kv_offsets.is_contiguous()

    def grid(args):
        return (Q_HEADS, triton.cdiv(max_q_len, args["BLOCK_M"]), BS)

    o = q.new_empty(Q_LEN_TOTAL, Q_HEADS, V_DIM, dtype=v.dtype)

    kwargs = dict(
        Q_ptr=q,
        K_ptr=k,
        V_ptr=v,
        O_ptr=o,
        Q_stride=q.stride(),
        K_stride=k.stride(),
        V_stride=v.stride(),
        O_stride=o.stride(),
        Q_offsets_ptr=q_offsets,
        KV_offsets_ptr=kv_offsets,
        Q_HEADS=Q_HEADS,
        KV_HEADS=KV_HEADS,
        QK_DIM=QK_DIM,
        V_DIM=V_DIM,
        BLOCK_M=64,
        BLOCK_N=64,
        CAUSAL=causal,
        QK_ACC_DTYPE=tl.float16 if qk_fp16_acc else tl.float32 if q.is_floating_point() else tl.int32,
        PV_ACC_DTYPE=tl.float16 if pv_fp16_acc else tl.float32,
        num_warps=4,
        num_stages=3,
    )

    triton_varlen_attn_kernel[grid](**kwargs)

    return o
