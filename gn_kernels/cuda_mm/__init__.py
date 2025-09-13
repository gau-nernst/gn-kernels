import dataclasses
from pathlib import Path

import torch
from torch import Tensor

from ..nvrtc_utils import _TYPE_MAP, _compile_kernel, cdiv

CURRENT_DIR = Path(__file__).parent
MARKER = "// start of kernel"
_, KERNEL = open(CURRENT_DIR / "kernel.cu").read().split(MARKER)

HEADER_TEMPLATE = """
#include "common.h"

constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};
constexpr int GROUP_M = {};

constexpr int NUM_WARP_M = {};
constexpr int NUM_WARP_N = {};

constexpr int NUM_STAGES = {};

using TypeAB = {};
using TypeC = {};
using TypeAcc = {};
"""


@dataclasses.dataclass
class MatmulKernel:
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    acc_dtype: torch.dtype = torch.float32
    block_mnk: tuple[int, int, int] = (128, 128, 64)
    group_m: int = 8
    warp_mn: tuple[int, int] = (2, 2)
    num_stages: int = 1

    def __post_init__(self) -> None:
        self.tb_size = (self.warp_mn[0] * self.warp_mn[1] * 32, 1, 1)
        BM, BN, BK = self.block_mnk
        self.smem_size = (BM + BN) * BK * self.in_dtype.itemsize * self.num_stages

        header = HEADER_TEMPLATE.format(
            *self.block_mnk,
            self.group_m,
            *self.warp_mn,
            self.num_stages,
            _TYPE_MAP[self.in_dtype],
            _TYPE_MAP[self.out_dtype],
            _TYPE_MAP[self.acc_dtype],
        )
        self.kernel = _compile_kernel(KERNEL, "matmul_kernel", header, self.smem_size)

    def run(self, A: Tensor, B: Tensor):
        assert A.stride(1) == 1
        assert B.stride(0) == 1
        assert A.shape[1] == B.shape[0]

        M, K = A.shape
        _, N = B.shape
        C = A.new_empty(M, N, dtype=self.out_dtype)

        grid = cdiv(M, self.block_mnk[0]) * cdiv(N, self.block_mnk[1])
        self.kernel((grid, 1, 1), self.tb_size, (A, B, C, M, N, K), self.smem_size)
        return C
