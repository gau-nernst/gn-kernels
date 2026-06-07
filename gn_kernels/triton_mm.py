import torch
import triton
import triton.language as tl
from torch import Tensor

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
