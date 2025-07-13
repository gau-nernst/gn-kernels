from pathlib import Path

import torch
from torch import Tensor

from ._lib import lib, lib_ops
from .utils import FP4_DTYPE

CURRENT_DIR = Path(__file__).parent


for shared_lib in CURRENT_DIR.glob("*.so"):
    torch.ops.load_library(shared_lib)

# sm80
lib.define("cutlass_sm80_int4_mm(Tensor A, Tensor B) -> Tensor")
lib.define("cutlass_sm80_row_scaled_int4_mm(Tensor A, Tensor B, Tensor row_scale, Tensor col_scale) -> Tensor")

# sm89
lib.define("cutlass_sm89_fp8_mm(Tensor A, Tensor B) -> Tensor")
lib.define("cutlass_sm89_row_scaled_fp8_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")

# sm120a
lib.define("cutlass_sm120a_fp8_mm(Tensor A, Tensor B) -> Tensor")
lib.define("cutlass_sm120a_row_scaled_fp8_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("cutlass_sm120a_mxfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B, Tensor? bias) -> Tensor")
lib.define(
    "cutlass_sm120a_nvfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B, Tensor output_scale, Tensor? bias) -> Tensor"
)


def cutlass_int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    return lib_ops.cutlass_sm80_int4_mm(A, B)


@torch.library.impl(lib, "cutlass_sm80_int4_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.int32)


def cutlass_row_scaled_int4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    assert scale_A.dtype == scale_B.dtype == torch.float32  # only support float32 for now
    assert scale_A.shape == (A.shape[0], 1)
    assert scale_B.shape == (1, B.shape[1])
    return lib_ops.cutlass_sm80_row_scaled_int4_mm(A, B, scale_A, scale_B)


def cutlass_fp8_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.float8_e4m3fn and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.float8_e4m3fn and B.T.is_contiguous()
    if torch.cuda.get_device_capability()[0] == 12:
        return lib_ops.cutlass_sm120a_fp8_mm(A, B)
    else:
        return lib_ops.cutlass_sm89_fp8_mm(A, B)


@torch.library.impl(lib, "cutlass_sm80_row_scaled_int4_mm", "Meta")
@torch.library.impl(lib, "cutlass_sm89_fp8_mm", "Meta")
@torch.library.impl(lib, "cutlass_sm89_row_scaled_fp8_mm", "Meta")
@torch.library.impl(lib, "cutlass_sm120a_fp8_mm", "Meta")
@torch.library.impl(lib, "cutlass_sm120a_row_scaled_fp8_mm", "Meta")
def _(A: Tensor, B: Tensor, *args) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.bfloat16)


def cutlass_row_scaled_fp8_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.float8_e4m3fn and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.float8_e4m3fn and B.T.is_contiguous()
    assert scale_A.dtype == scale_B.dtype == torch.float32  # only support float32 for now
    assert scale_A.shape == (A.shape[0], 1)
    assert scale_B.shape == (1, B.shape[1])

    if torch.cuda.get_device_capability()[0] == 12:
        return lib_ops.cutlass_sm120a_row_scaled_fp8_mm(A, B, scale_A, scale_B)
    else:
        return lib_ops.cutlass_sm89_row_scaled_fp8_mm(A, B, scale_A, scale_B)


def cutlass_mxfp4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, bias: Tensor | None = None) -> Tensor:
    assert A.ndim == 2 and A.dtype is FP4_DTYPE and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is FP4_DTYPE and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    assert scale_A.dtype == torch.float8_e8m0fnu
    assert scale_B.dtype == torch.float8_e8m0fnu
    return lib_ops.cutlass_sm120a_mxfp4_mm(A, B, scale_A, scale_B, bias)


@torch.library.impl(lib, "cutlass_sm120a_mxfp4_mm", "Meta")
@torch.library.impl(lib, "cutlass_sm120a_nvfp4_mm", "Meta")
def _(A: Tensor, B: Tensor, *args) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.bfloat16)


def cutlass_nvfp4_mm(
    A: Tensor,
    B: Tensor,
    scale_A: Tensor,
    scale_B: Tensor,
    output_scale: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    assert A.ndim == 2 and A.dtype is FP4_DTYPE and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is FP4_DTYPE and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    assert scale_A.dtype == torch.float8_e4m3fn
    assert scale_B.dtype == torch.float8_e4m3fn
    return lib_ops.cutlass_sm120a_nvfp4_mm(A, B, scale_A, scale_B, output_scale, bias)
