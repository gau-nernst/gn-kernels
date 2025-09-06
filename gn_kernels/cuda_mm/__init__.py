import dataclasses
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import include_paths

_TYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.int8: "char",
    torch.uint8: "unsigned char",
    torch.int32: "int",
}

CURRENT_DIR = Path(__file__).parent
MARKER = "// start of kernel"
_, KERNEL = open(CURRENT_DIR / "kernel.cu").read().split(MARKER)

HEADER_TEMPLATE = """
#include "common.h"
#include <cuda_bf16.h>

constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};

constexpr int NUM_WARP_M = {};
constexpr int NUM_WARP_N = {};

constexpr int NUM_STAGES = {};

using TypeAB = {};
using TypeC = {};
using TypeAcc = {};
"""


def cdiv(a, b):
    return (a + b - 1) // b


@dataclasses.dataclass
class MatmulKernel:
    type_ab: torch.dtype = torch.bfloat16
    type_c: torch.dtype = torch.bfloat16
    type_acc: torch.dtype = torch.float
    block_mnk: tuple[int, int, int] = (128, 128, 64)
    warp_mn: tuple[int, int] = (2, 2)
    num_stages: int = 1

    def __post_init__(self) -> None:
        header = HEADER_TEMPLATE.format(
            *self.block_mnk,
            *self.warp_mn,
            self.num_stages,
            _TYPE_MAP[self.type_ab],
            _TYPE_MAP[self.type_c],
            _TYPE_MAP[self.type_acc],
        )
        self.kernel = torch.cuda._compile_kernel(
            kernel_source=KERNEL,
            kernel_name="matmul_kernel",
            header_code=header,
            cuda_include_dirs=[
                *include_paths("cuda"),
                str(CURRENT_DIR.parent / "csrc"),
            ],
        )
        self.tb_size = (self.warp_mn[0] * self.warp_mn[1] * 32, 1, 1)
        BM, BN, BK = self.block_mnk
        self.smem_size = (BM + BN) * BK * self.type_ab.itemsize * self.num_stages

    def run(self, A: Tensor, B: Tensor):
        assert A.stride(1) == 1
        assert B.stride(0) == 1
        assert A.shape[1] == B.shape[0]

        M, K = A.shape
        _, N = B.shape
        C = A.new_zeros(M, N, dtype=self.type_c)

        grid = cdiv(M, self.block_mnk[0]) * cdiv(N, self.block_mnk[1])
        self.kernel((grid, 1, 1), self.tb_size, (A, B, C, M, N, K), self.smem_size)
        return C
