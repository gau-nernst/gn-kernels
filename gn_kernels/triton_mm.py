import torch
import triton
import triton.language as tl
from torch import Tensor

from ._lib import lib, lib_ops

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
configs = [
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
]

configs = [
    triton.Config(dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K), num_stages=num_stages, num_warps=num_warps)
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


def _grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)


# templated matmul from pytorch
# https://github.com/pytorch/pytorch/blob/c2e2602ecdc2ec1f120e19198dfc18fc39f7bd09/torch/_inductor/kernel/mm.py
# re-tune when stride changes i.e. transpose configuration
@triton.autotune(
    configs=configs,
    key=["M", "N", "K", "stride_A", "stride_B"],
)
@triton.jit
def triton_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    bias_ptr,
    scale_A_ptr,
    scale_B_ptr,
    M,
    N,
    K,
    stride_A,
    stride_B,
    stride_C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    ACC_DTYPE: tl.constexpr = None,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_A[0] + rk[None, :] * stride_A[1])
    B = B_ptr + (rk[:, None] * stride_B[0] + rbn[None, :] * stride_B[1])

    if ACC_DTYPE is None:
        ACC_DTYPE = tl.float32 if A_ptr.type.element_ty.is_floating() else tl.int32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A, mask=rk[None, :] < k, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE)
        A += BLOCK_K * stride_A[1]
        B += BLOCK_K * stride_B[0]

    # rematerialize rm and rn to save registers
    idx_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    if scale_A_ptr is not None:
        scale_A = tl.load(scale_A_ptr + idx_m, mask=idx_m < M)
        acc = acc.to(tl.float32) * scale_A.to(tl.float32)

    if scale_B_ptr is not None:
        scale_B = tl.load(scale_B_ptr + idx_n, mask=idx_n < N)
        acc = acc.to(tl.float32) * scale_B.to(tl.float32)

    # NOTE: this doesn't consider the case i8xi8=i32 +i32
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + idx_n, mask=idx_n < N)
        acc = acc.to(tl.float32) + bias.to(tl.float32)

    # inductor generates a suffix
    xindex = idx_m * stride_C[0] + idx_n * stride_C[1]
    tl.store(C_ptr + xindex, acc, mask=(idx_m < M) & (idx_n < N))


def triton_mm(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    *,
    scale_A: Tensor | None = None,
    scale_B: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    acc_dtype: torch.dtype | None = None,
):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape

    # row-scaled matmul
    if scale_A is not None:
        assert scale_A.shape == (M, 1)
        assert scale_A.is_contiguous()

    if scale_B is not None:
        assert scale_B.shape == (1, N)
        assert scale_B.is_contiguous()

    if bias is not None:
        assert bias.shape == (N,)
        assert bias.is_contiguous()

    if out_dtype is None:
        if A.dtype in (torch.float32, torch.float16, torch.bfloat16):
            out_dtype = A.dtype
        elif A.is_floating_point() or scale_A is not None or scale_B is not None:  # FP8 or row-scaled mm
            out_dtype = torch.bfloat16
        else:
            out_dtype = torch.int32

    # map PyTorch dtype to Triton dtype
    if acc_dtype is not None:
        acc_dtype = {torch.float32: tl.float32, torch.float16: tl.float16, torch.int32: tl.int32}[acc_dtype]

    C = A.new_empty(M, N, dtype=out_dtype)
    triton_mm_kernel[_grid](
        A, B, C, bias, scale_A, scale_B, M, N, K, A.stride(), B.stride(), C.stride(), ACC_DTYPE=acc_dtype
    )
    return C


@triton.autotune(
    # need to find more performant configs...
    configs=[
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=2, num_warps=8),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K", "stride_ak", "stride_bk"],
)
@triton.jit
def block2d_scaled_mm_kernel(
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    C_ptr,  # (M, N)
    scale_A_ptr,  # (M // QUANT_BLOCK_M, K // QUANT_BLOCK_K)
    scale_B_ptr,  # (K // QUANT_BLOCK_K, N // QUANT_BLOCK_N)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bk,
    stride_scale_bn,
    QUANT_BLOCK_M: tl.constexpr,
    QUANT_BLOCK_N: tl.constexpr,
    QUANT_BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
):
    # NOTE: most of the time, it's most performant with BLOCK_K == QUANT_BLOCK_K
    tl.static_assert(QUANT_BLOCK_K % BLOCK_K == 0)

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    # NOTE: it seems like we can afford to have QUANT_BLOCK_M and QUANT_BLOCK_N not be constexpr
    A_scale = scale_A_ptr + ((rm // QUANT_BLOCK_M)[:, None] * stride_scale_am)
    B_scale = scale_B_ptr + ((rn // QUANT_BLOCK_N)[None, :] * stride_scale_bn)

    # we use 2 accumulators. acc is the final result. mma_acc is accumulator for MMA before
    # scaling. for every QUANT_BLOCK_K, we will scale mma_acc and accumulate it to acc.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # v1
    # mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    # for k in range(K, 0, -BLOCK_K):
    #     if EVEN_K:
    #         a = tl.load(A)
    #         b = tl.load(B)
    #     else:
    #         a = tl.load(A, mask=rk[None, :] < k, other=0.0)
    #         b = tl.load(B, mask=rk[:, None] < k, other=0.0)
    #     mma_acc += tl.dot(a, b)
    #     A += BLOCK_K * stride_ak
    #     B += BLOCK_K * stride_bk

    #     if (k - BLOCK_K) % QUANT_BLOCK_K == 0:
    #         a_scale = tl.load(A_scale).to(tl.float32)
    #         b_scale = tl.load(B_scale).to(tl.float32)
    #         acc += mma_acc.to(tl.float32) * a_scale * b_scale
    #         mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    #         A_scale += stride_scale_ak
    #         B_scale += stride_scale_bk

    # v2
    for k in range(K, 0, -QUANT_BLOCK_K):
        mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for _ in tl.static_range(QUANT_BLOCK_K // BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            mma_acc += tl.dot(a, b)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        a_scale = tl.load(A_scale).to(tl.float32)
        b_scale = tl.load(B_scale).to(tl.float32)
        acc += mma_acc.to(tl.float32) * a_scale * b_scale
        A_scale += stride_scale_ak
        B_scale += stride_scale_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


# TODO: check if we still need custom op for triton kernels
lib.define(
    "triton_block2d_scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B, ScalarType out_dtype) -> Tensor"
)


def triton_block2d_scaled_mm(
    A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """Matmul for tile-wise quantized A and B. `A` and `B` are both INT8 or FP8 to utilize
    INT8/FP8 tensor cores. `scale_A` and `scaled_B` are quantization scales for A and B
    respectively with appropriate shapes.

    E.g.
      - if `A` is quantized with tile shape (128, 64), `scale_A`'s shape will be
    `(A.shape[0] / 128, A.shape[1] / 64)`.
      - if `A` is row-wise quantized, `scale_A`'s shape will be `(A.shape[0], 1)`.
    """
    _f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    assert (A.dtype == B.dtype == torch.int8) or (A.dtype in _f8 and B.dtype in _f8)
    assert A.ndim == B.ndim == scale_A.ndim == scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]

    assert scale_A.shape[1] == scale_B.shape[0]
    return lib_ops.triton_block2d_scaled_mm(A, B, scale_A, scale_B, out_dtype)


@torch.library.impl(lib, "triton_block2d_scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, out_dtype: torch.dtype):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=out_dtype)


@torch.library.impl(lib, "triton_block2d_scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, out_dtype: torch.dtype):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=out_dtype)

    QUANT_BLOCK_K = A.shape[1] // scale_A.shape[1]
    block2d_scaled_mm_kernel[_grid](
        A,
        B,
        C,
        scale_A,
        scale_B,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        *scale_A.stride(),
        *scale_B.stride(),
        QUANT_BLOCK_M=A.shape[0] // scale_A.shape[0],
        QUANT_BLOCK_N=B.shape[1] // scale_B.shape[1],
        QUANT_BLOCK_K=QUANT_BLOCK_K,
        BLOCK_K=QUANT_BLOCK_K,
        ACC_DTYPE=tl.int32 if A.dtype == torch.int8 else tl.float32,
        EVEN_K=K % 2 == 0,
    )
    return C
