import dataclasses
from pathlib import Path

import torch
from torch import Tensor

from ..nvrtc_utils import _compile_kernel, cdiv

CURRENT_DIR = Path(__file__).parent
MARKER = "// start of kernel"
_, KERNEL = open(CURRENT_DIR / "kernel.cu").read().split(MARKER)

HEADER_TEMPLATE = """
#include <hip/hip_bf16.h>

constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};
constexpr int GROUP_M = {};

constexpr int NUM_WARP_M = {};
constexpr int NUM_WARP_N = {};

constexpr int WARP_SIZE = 64;
"""


@dataclasses.dataclass
class MatmulKernel:
    block_mnk: tuple[int, int, int] = (128, 128, 64)
    group_m: int = 4
    warp_mn: tuple[int, int] = (2, 2)

    def __post_init__(self) -> None:
        self.tb_size = (self.warp_mn[0] * self.warp_mn[1] * 32, 1, 1)
        BM, BN, BK = self.block_mnk
        self.smem_size = (BM + BN) * BK * torch.bfloat16.itemsize

        header = HEADER_TEMPLATE.format(
            *self.block_mnk,
            self.group_m,
            *self.warp_mn,
        )
        header += "__device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }"
        self.kernel = _compile_kernel(KERNEL, "matmul_kernel", header, self.smem_size)

    def run(self, A: Tensor, B: Tensor):
        assert A.stride(1) == 1
        assert B.stride(0) == 1
        assert A.shape[1] == B.shape[0]

        M, K = A.shape
        _, N = B.shape
        C = A.new_empty(M, N)

        grid = cdiv(M, self.block_mnk[0]) * cdiv(N, self.block_mnk[1])
        self.kernel((grid, 1, 1), self.tb_size, (A, B, C, M, N, K), self.smem_size)
        return C
