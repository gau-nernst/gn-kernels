from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

CURRENT_DIR = Path(__file__).parent
load(
    "gn_kernels_hip",
    [str(CURRENT_DIR / "kernel.cu")],
    is_python_module=False,
)


def hip_mm(A: Tensor, B: Tensor):
    assert A.stride(1) == 1
    assert B.stride(0) == 1
    assert A.shape[1] == B.shape[0]

    C = A.new_empty(A.shape[0], B.shape[1])
    torch.ops.gn_kernels_hip.matmul(A, B, C)
    return C
