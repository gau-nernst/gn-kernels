from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


# sentinel object to represent int4 types
class FakeType:
    itemsize = 1


int4x2 = FakeType()
uint4x2 = FakeType()

_TYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
    torch.int32: "int",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    int4x2: "int4x2",
    uint4x2: "uint4x2",
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# not sure why we need to manually add CCCL include. new changes in CUDA 13?
_include_dirs = (
    str(Path(CUDA_HOME) / "include" / "cccl"),
    str(Path(__file__).parent / "csrc"),
)


def _compile_kernel(kernel_source: str, kernel_name: str, smem_size: int):
    kernel = torch.cuda._compile_kernel(
        kernel_source=kernel_source,
        kernel_name=kernel_name,
        cuda_include_dirs=_include_dirs,
    )
    kernel.set_shared_memory_config(smem_size)
    return kernel
