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
# also re-tune when stride changes i.e. transpose configuration
@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACC_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):  # fmt: skip
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("int8_mm(Tensor A, Tensor B) -> Tensor")


def int8_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A.shape[1] == B.shape[0]
    return lib_ops.int8_mm(A, B)


@torch.library.impl(lib, "int8_mm", "Meta")
def _(a: Tensor, b: Tensor):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)


@torch.library.impl(lib, "int8_mm", "CUDA")
def _(A: Tensor, B: Tensor):
    return _triton_mm(A, B, torch.int32, torch.int32)


def _triton_mm(A: Tensor, B: Tensor, out_dtype: torch.dtype | None = None, acc_dtype: torch.dtype | None = None):
    out_dtype = out_dtype or A.dtype
    if acc_dtype is None:
        ACC_DTYPE_TRITON = tl.float32 if A.dtype.is_floating_point else tl.int32
    else:
        ACC_DTYPE_TRITON = {torch.float32: tl.float32, torch.float16: tl.float16, torch.int32: tl.int32}[acc_dtype]
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    EVEN_K = K % 2 == 0
    C = torch.empty(M, N, dtype=out_dtype, device=A.device)
    _matmul_kernel[_grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(), ACC_DTYPE_TRITON, EVEN_K)
    return C


@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    row_scale_ptr,
    col_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    COL_SCALE_SCALAR: tl.constexpr = False,
):
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    row_scale = tl.load(row_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    if COL_SCALE_SCALAR:
        # hack to support BitNet. col_scale is now a scalar
        col_scale = tl.load(col_scale_ptr).to(tl.float32)
    else:
        col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


@triton.autotune(
    # need to find more performant configs...
    configs=[
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=2, num_warps=8),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K", "stride_ak", "stride_bk"],
)
@triton.jit
def _tile_scaled_mm_kernel(
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


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("tile_scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")


def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
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
    assert scale_A.dtype == scale_B.dtype
    assert A.ndim == B.ndim == scale_A.ndim == scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]

    # row-scale + col-scale or row-scale + tensor-scale
    if scale_A.shape == (A.shape[0], 1) and scale_B.shape in ((1, B.shape[1]), (1, 1)):
        assert scale_A.is_contiguous()
        assert scale_B.is_contiguous()
        return lib_ops.scaled_mm(A, B, scale_A, scale_B)

    # generic tile-wise scaling
    assert scale_A.shape[1] == scale_B.shape[0]
    return lib_ops.tile_scaled_mm(A, B, scale_A, scale_B)


@torch.library.impl(lib, "scaled_mm", "Meta")
@torch.library.impl(lib, "tile_scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale_A.dtype)


@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale.dtype)

    def grid(meta):
        return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    _scaled_mm_kernel[grid](
        A,
        B,
        C,
        row_scale,
        col_scale,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        ACC_DTYPE=tl.int32 if A.dtype == torch.int8 else tl.float32,
        EVEN_K=K % 2 == 0,
        COL_SCALE_SCALAR=col_scale.numel() == 1,
    )
    return C


@torch.library.impl(lib, "tile_scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=scale_A.dtype)

    def grid(meta):
        return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    QUANT_BLOCK_K = A.shape[1] // scale_A.shape[1]
    _tile_scaled_mm_kernel[grid](
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
