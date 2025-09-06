import dataclasses
from pathlib import Path

import torch
from torch import Tensor
from torch.cuda._utils import _check_cuda, _get_cuda_library
from torch.utils.cpp_extension import include_paths

_TYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
    torch.int32: "int",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
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
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    acc_dtype: torch.dtype = torch.float32
    block_mnk: tuple[int, int, int] = (128, 128, 64)
    warp_mn: tuple[int, int] = (2, 2)
    num_stages: int = 1

    def __post_init__(self) -> None:
        self.tb_size = (self.warp_mn[0] * self.warp_mn[1] * 32, 1, 1)
        BM, BN, BK = self.block_mnk
        self.smem_size = (BM + BN) * BK * self.in_dtype.itemsize * self.num_stages

        max_smem_size = torch.cuda.get_device_properties().shared_memory_per_block_optin
        if self.smem_size >= max_smem_size:
            msg = (
                f"Shared memory {self.smem_size / 1e3} MB exceeds the maxmimum"
                f" value allowed on the current GPU ({max_smem_size / 1e3} MB)"
            )
            raise ValueError(msg)

        header = HEADER_TEMPLATE.format(
            *self.block_mnk,
            *self.warp_mn,
            self.num_stages,
            _TYPE_MAP[self.in_dtype],
            _TYPE_MAP[self.out_dtype],
            _TYPE_MAP[self.acc_dtype],
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

        # need to do this manually until patch is merged upstream
        if self.smem_size >= 48 * 1024:
            libcuda = _get_cuda_library()
            _check_cuda(libcuda.cuFuncSetAttribute(self.kernel.func, 8, self.smem_size))

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
