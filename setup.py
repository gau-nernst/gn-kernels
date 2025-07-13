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

    return CUDAExtension(
        name=f"{NAME}.cutlass_sm{arch}",
        sources=[str(x) for x in CURRENT_DIR.glob(f"csrc/cutlass_sm{arch}_*.cu")],
        py_limited_api=True,
        extra_compile_args=dict(
            nvcc=[
                f"-I{CURRENT_DIR / 'cutlass/include'}",
                f"-I{CURRENT_DIR / 'cutlass/tools/util/include'}",
                f"-gencode={gencode}",
                # "-DCUTLASS_DEBUG_TRACE_LEVEL=1",
            ]
        ),
        define_macros=[("Py_LIMITED_API", "0x03090000")],
    )


setup(
    name=NAME,
    packages=find_packages(),
    version="0.1",
    ext_modules=[
        get_extension("80"),
        get_extension("89"),
        get_extension("120a"),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
