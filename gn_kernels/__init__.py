from pathlib import Path

import torch

from .attn import attn_mxfp8
from .cutlass_mm import (
    cutlass_fp8_mm,
    cutlass_int4_mm,
    cutlass_mxfp4_mm,
    cutlass_nvfp4_mm,
    cutlass_row_scaled_fp8_mm,
    cutlass_row_scaled_int4_mm,
)
from .triton_attn import triton_attn, triton_scaled_qk_attn
from .triton_mm import triton_mm, triton_scaled_mm
from .utils import FP4_DTYPE, dequantize_mx, pack_block_scales_nv, quantize_mx, quantize_nvfp4, quantize_nvfp4_triton

__all__ = [
    "cutlass_int4_mm",
    "cutlass_row_scaled_int4_mm",
    "cutlass_row_scaled_fp8_mm",
    "cutlass_fp8_mm",
    "cutlass_mxfp4_mm",
    "cutlass_nvfp4_mm",
    "triton_mm",
    "triton_scaled_mm",
    "triton_attn",
    "triton_scaled_qk_attn",
    "attn_mxfp8",
    "quantize_mx",
    "quantize_nvfp4",
    "quantize_nvfp4_triton",
    "dequantize_mx",
    "pack_block_scales_nv",
    "FP4_DTYPE",
]

CURRENT_DIR = Path(__file__).parent

for shared_lib in CURRENT_DIR.glob("*.so"):
    torch.ops.load_library(shared_lib)
