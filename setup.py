import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CURRENT_DIR = Path(__file__).parent
NAME = "gn_kernels"


def get_extension(arch: str):
    if arch.endswith("a"):
        gencode = f"arch=compute_{arch},code=sm_{arch}"  # compile to SASS
    else:
        gencode = f"arch=compute_{arch},code=compute_{arch}"  # compile to PTX

    nvcc_flags = [
        f"-I{CURRENT_DIR / 'cutlass/include'}",
        f"-I{CURRENT_DIR / 'cutlass/tools/util/include'}",
        f"-gencode={gencode}",
        # "-DCUTLASS_DEBUG_TRACE_LEVEL=1",
        # "-Xptxas=-v",
    ]

    return CUDAExtension(
        name=f"{NAME}.sm{arch}",
        sources=[str(x.relative_to(CURRENT_DIR)) for x in CURRENT_DIR.glob(f"gn_kernels/csrc/sm{arch}/*.cu")],
        py_limited_api=True,
        extra_compile_args=dict(nvcc=nvcc_flags),
    )


def get_ext_modules():
    if os.getenv("NO_EXT"):
        return []

    return [
        get_extension("80"),
        get_extension("89"),
        get_extension("120a"),
    ]


setup(
    name=NAME,
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
