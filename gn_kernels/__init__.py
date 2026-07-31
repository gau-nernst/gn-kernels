from pathlib import Path

import torch

from . import cutedsl
from .attn import attn_int8, attn_int8_qk, attn_mxfp8_qk
from .cublas_nvfp4 import cublas_nvfp4_mm
from .quant_utils import (
    dequantize_mx,
    permute_nv_sf,
    quantize_mx,
    quantize_nvfp4,
    quantize_nvfp4_triton,
    unpermute_nv_sf,
)
from .triton_attn import triton_attn
from .triton_mm import triton_mm

__all__ = [
    "triton_mm",
    "triton_attn",
    "triton_scaled_qk_attn",
    "attn_int8",
    "attn_int8_qk",
    "attn_mxfp8_qk",
    "cublas_nvfp4_mm",
    "quantize_mx",
    "quantize_nvfp4",
    "quantize_nvfp4_triton",
    "dequantize_mx",
    "permute_nv_sf",
    "unpermute_nv_sf",
    "cutedsl",
]

CURRENT_DIR = Path(__file__).parent

for shared_lib in CURRENT_DIR.glob("*.so"):
    torch.ops.load_library(shared_lib)
