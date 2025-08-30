import argparse
import time

import pandas as pd
import torch
import torch._inductor.config
import torch._inductor.utils
from torch import Tensor
from triton.testing import do_bench

from gn_kernels import (
    FP4_DTYPE,
    cutlass_fp8_mm,
    cutlass_int4_mm,
    cutlass_mxfp4_mm,
    cutlass_nvfp4_mm,
    cutlass_row_scaled_fp8_mm,
    cutlass_row_scaled_int4_mm,
    triton_block2d_scaled_mm,
    triton_mm,
)


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    return (x[:, ::2] << 4) | (x[:, 1::2] & 0xF)


def unpack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    # NOTE: do this way to handle sign-extension correctly
    return torch.stack([x >> 4, x << 4 >> 4], dim=1).view(x.shape[0], -1)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def scaled_mm_inductor(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    if A.dtype == torch.int8:
        # TODO: check codegen of this
        return (torch._int_mm(A, B) * scale_B * scale_A).bfloat16()
    else:
        return torch._scaled_mm(A, B, scale_A, scale_B, out_dtype=torch.bfloat16)


def scaled_mm_ref(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    for dim in [0, 1]:
        scale_A = scale_A.repeat_interleave(A.shape[dim] // scale_A.shape[dim], dim)
        scale_B = scale_B.repeat_interleave(B.shape[dim] // scale_B.shape[dim], dim)
    return ((A.float() * scale_A) @ (B.float() * scale_B)).bfloat16()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(2025 * 2024 + 2023)
    COMPUTE_CAPABILITY = torch.cuda.get_device_capability()

    # we need to do this to force inductor to use triton's implementation of torch._scaled_mm() on devices with num_sms<68
    # TODO: try inductor Aten and Cutlass as well?
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    torch._inductor.utils.is_big_gpu = lambda _: True
    torch._inductor.config.force_fuse_int_mm_with_mul = True

    dims = list(args.dims)
    data = []
    for dim in args.dims:
        M = N = K = dim
        print(f"{dim=}")

        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16)
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16).T
        A_f16 = torch.randn(M, K, dtype=torch.float16)
        B_f16 = torch.randn(N, K, dtype=torch.float16).T
        A_i8 = torch.randint(-128, 127, size=(M, K), dtype=torch.int8)
        B_i8 = torch.randint(-128, 127, size=(N, K), dtype=torch.int8).T

        scale_A = torch.randn(M, 1).div(128)
        scale_B = torch.randn(N, 1).div(128).T
        # DeepSeek style. (1,128) for act, (128,128) for weight
        block2d_scale_A = torch.randn(M, K // 128).div(128)
        block2d_scale_B = torch.randn(N // 128, K // 128).div(128).T

        def bench_tflops(f, ref, *args, atol=None, rtol=None, **kwargs):
            if callable(ref):
                ref = ref(*args, **kwargs)
            out = f(*args, **kwargs)
            if ref is not None:
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

            time.sleep(1)  # stabilize thermal
            latency_ms = do_bench(lambda: f(*args, **kwargs), return_mode="median")
            return (2 * M * N * K) / (latency_ms * 1e-3) * 1e-12

        bf16_tflops = bench_tflops(torch.mm, None, A_bf16, B_bf16)
        f16_acc_f16_triton_tflops = bench_tflops(triton_mm, None, A_f16, B_f16, acc_dtype=torch.float16)
        i8_pt_tflops = bench_tflops(torch._int_mm, None, A_i8, B_i8)
        i8_triton_tflops = bench_tflops(triton_mm, torch._int_mm, A_i8, B_i8)
        scaled_i8_inductor_tflops = bench_tflops(
            scaled_mm_inductor, scaled_mm_ref, A_i8, B_i8, scale_A, scale_B, atol=4e-4, rtol=1e-2
        )
        scaled_i8_triton_tflops = bench_tflops(
            triton_mm, scaled_mm_ref, A_i8, B_i8, scale_A, scale_B, atol=4e-4, rtol=1e-2
        )
        block2d_scaled_i8_triton_tflops = bench_tflops(
            triton_block2d_scaled_mm, scaled_mm_ref, A_i8, B_i8, block2d_scale_A, block2d_scale_B, atol=1e-4, rtol=1e-2
        )

        # FP8
        if COMPUTE_CAPABILITY >= (8, 9):
            A_f8 = torch.randn(M, K).to(torch.float8_e4m3fn)
            B_f8 = torch.randn(N, K).to(torch.float8_e4m3fn).T

            f8_mm_output = (A_f8.float() @ B_f8.float()).bfloat16()
            f8_triton_tflops = bench_tflops(triton_mm, f8_mm_output, A_f8, B_f8, out_dtype=torch.bfloat16)
            f8_cutlass_tflops = bench_tflops(cutlass_fp8_mm, f8_mm_output, A_f8, B_f8)
            scaled_f8_inductor_tflops = bench_tflops(scaled_mm_inductor, scaled_mm_ref, A_f8, B_f8, scale_A, scale_B)
            scaled_f8_cutlass_tflops = bench_tflops(
                cutlass_row_scaled_fp8_mm, scaled_mm_ref, A_f8, B_f8, scale_A.float(), scale_B.float()
            )
            scaled_f8_triton_tflops = bench_tflops(triton_mm, scaled_mm_ref, A_f8, B_f8, scale_A, scale_B)
            block2d_scaled_f8_triton_tflops = bench_tflops(
                triton_block2d_scaled_mm, scaled_mm_ref, A_f8, B_f8, block2d_scale_A, block2d_scale_B
            )

        else:
            f8_triton_tflops = 0
            f8_cutlass_tflops = 0
            scaled_f8_inductor_tflops = 0
            scaled_f8_cutlass_tflops = 0

        # INT4
        A_i8_ref = torch.randint(-8, 7, size=(M, K), dtype=torch.int8)
        B_i8_ref_t = torch.randint(-8, 7, size=(N, K), dtype=torch.int8).T
        A_i4 = pack_int4(A_i8_ref)
        B_i4 = pack_int4(B_i8_ref_t).T

        i4_cutlass_tflops = bench_tflops(cutlass_int4_mm, torch._int_mm(A_i8_ref, B_i8_ref_t), A_i4, B_i4)
        scaled_i4_cutlass_tflops = bench_tflops(
            cutlass_row_scaled_int4_mm,
            scaled_mm_ref(A_i8_ref, B_i8_ref_t, scale_A, scale_B),
            A_i4,
            B_i4,
            scale_A,
            scale_B,
        )

        # FP4
        if COMPUTE_CAPABILITY == (12, 0):
            A_fp4 = torch.randint(255, size=(M, K // 2), dtype=torch.uint8).view(FP4_DTYPE)
            B_fp4 = torch.randint(255, size=(N, K // 2), dtype=torch.uint8).view(FP4_DTYPE).T

            scale_A_mx = torch.randn(M, K // 32).to(torch.float8_e8m0fnu)
            scale_B_mx = torch.randn(N, K // 32).to(torch.float8_e8m0fnu)
            mxfp4_cutlass_tflops = bench_tflops(cutlass_mxfp4_mm, None, A_fp4, B_fp4, scale_A_mx, scale_B_mx)

            scale_A_nv = torch.randn(M, K // 16).to(torch.float8_e4m3fn)
            scale_B_nv = torch.randn(N, K // 16).to(torch.float8_e4m3fn)
            output_scale = torch.tensor(1.0)
            nvfp4_cutlass_tflops = bench_tflops(
                cutlass_nvfp4_mm, None, A_fp4, B_fp4, scale_A_nv, scale_B_nv, output_scale
            )

        else:
            mxfp4_cutlass_tflops = 0
            nvfp4_cutlass_tflops = 0

        data_point = {
            "PyTorch (CuBLAS) BF16": bf16_tflops,
            "Triton FP16 w/ FP16 accumulate": f16_acc_f16_triton_tflops,
            "Triton FP8": f8_triton_tflops,
            "Cutlass FP8": f8_cutlass_tflops,
            "PyTorch (CuBLAS) INT8": i8_pt_tflops,
            "Triton INT8": i8_triton_tflops,
            "Cutlass INT4": i4_cutlass_tflops,
            "Inductor (Triton) row-scaled FP8": scaled_f8_inductor_tflops,
            "Triton row-scaled FP8": scaled_f8_triton_tflops,
            "Cutlass row-scaled FP8": scaled_f8_cutlass_tflops,
            "Triton block2d-scaled FP8": block2d_scaled_f8_triton_tflops,
            "Inductor (Triton) row-scaled INT8": scaled_i8_inductor_tflops,
            "Triton row-scaled INT8": scaled_i8_triton_tflops,
            "Triton block2d-scaled INT8": block2d_scaled_i8_triton_tflops,
            "Cutlass row-scaled INT4": scaled_i4_cutlass_tflops,
            "Cutlass MXFP4": mxfp4_cutlass_tflops,
            "Cutlass NVFP4": nvfp4_cutlass_tflops,
        }
        data.append(data_point)

    gpu_name = torch.cuda.get_device_name()
    if gpu_name == "NVIDIA GeForce RTX 5090":
        # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
        bf16_tflops = 209.5
        fp16_acc_fp16_tflops = bf16_tflops * 2
        f8_tflops = bf16_tflops * 2
        i8_tflops = bf16_tflops * 4
        i4_tflops = 0
        f4_tflops = bf16_tflops * 8
        dims.append("Theoretical")
        data_point = {
            "PyTorch (CuBLAS) BF16": bf16_tflops,
            "Triton FP16 w/ FP16 accumulate": fp16_acc_fp16_tflops,
            "Triton FP8": f8_tflops,
            "Cutlass FP8": f8_tflops,
            "PyTorch (CuBLAS) INT8": i8_tflops,
            "Triton INT8": i8_tflops,
            "Cutlass INT4": i4_tflops,
            "Inductor (Triton) row-scaled FP8": f8_tflops,
            "Triton row-scaled FP8": f8_tflops,
            "Cutlass row-scaled FP8": f8_tflops,
            "Triton block2d-scaled FP8": f8_tflops,
            "Inductor (Triton) row-scaled INT8": i8_tflops,
            "Triton row-scaled INT8": i8_tflops,
            "Triton block2d-scaled INT8": i8_tflops,
            "Cutlass row-scaled INT4": i4_tflops,
            "Cutlass MXFP4": f4_tflops,
            "Cutlass NVFP4": f4_tflops,
        }
        data.append(data_point)

    df = pd.DataFrame(data, index=dims)
    print(df.round(2).T.to_markdown())
