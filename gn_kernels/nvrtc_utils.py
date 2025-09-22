import inspect
from pathlib import Path

import torch
from torch.cuda._utils import _check_cuda

try:
    from torch.cuda._utils import _get_cuda_library as _get_gpu_runtime_library
except ImportError:
    from torch.cuda._utils import _get_gpu_runtime_library

from torch.utils.cpp_extension import include_paths


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


def _compile_kernel(kernel_source: str, kernel_name: str, header_code: str, smem_size: int):
    # HIP doesn't have shared_memory_per_block_optin. skip
    if not torch.version.hip:
        max_smem_size = torch.cuda.get_device_properties().shared_memory_per_block_optin
        if smem_size >= max_smem_size:
            msg = (
                f"Shared memory {smem_size / 1e3} MB exceeds the maxmimum"
                f" value allowed on the current GPU ({max_smem_size / 1e3} MB)"
            )
            raise ValueError(msg)

    include_dirs = include_paths("cuda") + [str(Path(__file__).parent / "csrc")]
    sig = inspect.signature(torch.cuda._compile_kernel)

    if "header_code" in sig.parameters:  # before 2.10
        kernel = torch.cuda._compile_kernel(
            kernel_source=kernel_source,
            kernel_name=kernel_name,
            header_code=header_code,
            cuda_include_dirs=include_dirs,
        )
    else:
        kernel = torch.cuda._compile_kernel(
            kernel_source=f"{header_code}\n{kernel_source}",
            kernel_name=kernel_name,
            cuda_include_dirs=include_dirs,
        )

    if smem_size >= 48 * 1024:
        libcuda = _get_gpu_runtime_library()
        _check_cuda(libcuda.cuFuncSetAttribute(kernel.func, 8, smem_size))
    return kernel
