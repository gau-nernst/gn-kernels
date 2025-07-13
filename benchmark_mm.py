import argparse
import time

import pandas as pd
import torch
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
    triton_mm,
    triton_scaled_mm,
)


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    return (x[:, ::2] << 4) | (x[:, 1::2] & 0xF)


def unpack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    # NOTE: do this way to handle sign-extension correctly
    return torch.stack([x >> 4, x << 4 >> 4], dim=1).view(x.shape[0], -1)


def to_layout(x: torch.Tensor, column_major: bool):
    return x.T.contiguous().T if column_major else x.contiguous()


def triton_int8_mm(A: Tensor, B: Tensor):
    return triton_mm(A, B, out_dtype=torch.int32, acc_dtype=torch.int32)


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
    parser.add_argument("--a_column_major", action="store_true")
    parser.add_argument("--b_column_major", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(2025 * 2024)
    COMPUTE_CAPABILITY = torch.cuda.get_device_capability()

    # we need to do this to force inductor to use triton's implementation of torch._scaled_mm() on devices with num_sms<68
    # TODO: try inductor Aten and Cutlass as well?
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    torch._inductor.utils.is_big_gpu = lambda _: True
    torch._inductor.config.force_fuse_int_mm_with_mul = True

    data = []
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    for M, N, K in shapes:
        print(f"{M=}, {N=}, {K=}")

        A_bf16 = torch.randn(M, K).bfloat16()
        B_bf16 = torch.randn(K, N).bfloat16()
        A_f16 = torch.randn(M, K).half()
        B_f16 = torch.randn(K, N).half()
        A_i8 = torch.randint(-128, 127, size=(M, K), dtype=torch.int8)
        B_i8 = torch.randint(-128, 127, size=(K, N), dtype=torch.int8)

        scale_A = torch.randn(M, 1).div(128)
        scale_B = torch.randn(1, N).div(128)
        # DeepSeek style. (1,128) for act, (128,128) for weight
        block2d_scale_A = torch.randn(M, K // 128).div(128)
        block2d_scale_B = torch.randn(K // 128, N // 128).div(128)

        A_bf16, A_f16, A_i8, block2d_scale_A = [
            to_layout(x, args.a_column_major) for x in [A_bf16, A_f16, A_i8, block2d_scale_A]
        ]
        B_bf16, B_f16, B_i8, block2d_scale_B = [
            to_layout(x, args.b_column_major) for x in [B_bf16, B_f16, B_i8, block2d_scale_B]
        ]

        def bench_tflops(f, ref, *args, atol=None, rtol=None, **kwargs):
            if callable(ref):
                ref = ref(*args, **kwargs)
            if ref is not None:
                torch.testing.assert_close(f(*args, **kwargs), ref, atol=atol, rtol=rtol)
            f(*args, **kwargs)

            time.sleep(1)  # stabilize thermal
            latency_ms = do_bench(lambda: f(*args, **kwargs), return_mode="median")
            return (2 * M * N * K) / (latency_ms * 1e-3) * 1e-12

        bf16_tflops = bench_tflops(torch.mm, None, A_bf16, B_bf16)
        f16_acc_f16_triton_tflops = bench_tflops(triton_mm, None, A_f16, B_f16, acc_dtype=torch.float16)
        i8_pt_tflops = bench_tflops(torch._int_mm, None, A_i8, B_i8)
        i8_triton_tflops = bench_tflops(triton_int8_mm, torch._int_mm, A_i8, B_i8)
        scaled_i8_inductor_tflops = bench_tflops(
            scaled_mm_inductor, scaled_mm_ref, A_i8, B_i8, scale_A, scale_B, atol=4e-4, rtol=1e-2
        )
        scaled_i8_triton_tflops = bench_tflops(
            triton_scaled_mm, scaled_mm_ref, A_i8, B_i8, scale_A, scale_B, atol=4e-4, rtol=1e-2
        )
        block2d_scaled_i8_triton_tflops = bench_tflops(
            triton_scaled_mm, scaled_mm_ref, A_i8, B_i8, block2d_scale_A, block2d_scale_B, atol=1e-4, rtol=1e-2
        )

        # FP8
        if COMPUTE_CAPABILITY >= (8, 9):
            A_f8 = to_layout(torch.randn(M, K).to(torch.float8_e4m3fn), args.a_column_major)
            B_f8 = to_layout(torch.randn(K, N).to(torch.float8_e4m3fn), args.b_column_major)

            f8_mm_output = (A_f8.float() @ B_f8.float()).bfloat16()
            f8_triton_tflops = bench_tflops(triton_mm, f8_mm_output, A_f8, B_f8, out_dtype=torch.bfloat16)
            if not args.a_column_major and args.b_column_major:
                f8_cutlass_tflops = bench_tflops(cutlass_fp8_mm, f8_mm_output, A_f8, B_f8)
                scaled_f8_inductor_tflops = bench_tflops(
                    scaled_mm_inductor, scaled_mm_ref, A_f8, B_f8, scale_A, scale_B
                )
                scaled_f8_cutlass_tflops = bench_tflops(
                    cutlass_row_scaled_fp8_mm, scaled_mm_ref, A_f8, B_f8, scale_A.float(), scale_B.float()
                )
            else:
                f8_cutlass_tflops = 0
                scaled_f8_inductor_tflops = 0
                scaled_f8_cutlass_tflops = 0
            scaled_f8_triton_tflops = bench_tflops(triton_scaled_mm, scaled_mm_ref, A_f8, B_f8, scale_A, scale_B)
            block2d_scaled_f8_triton_tflops = bench_tflops(
                triton_scaled_mm, scaled_mm_ref, A_f8, B_f8, block2d_scale_A, block2d_scale_B
            )

        else:
            f8_triton_tflops = 0
            f8_cutlass_tflops = 0
            scaled_f8_inductor_tflops = 0
            scaled_f8_cutlass_tflops = 0

        # INT4
        if not args.a_column_major and args.b_column_major:
            A_i8_ref = torch.randint(-8, 7, size=(M, K), dtype=torch.int8)
            B_i8_ref_t = torch.randint(-8, 7, size=(N, K), dtype=torch.int8)
            A_i4 = pack_int4(A_i8_ref)
            B_i4 = pack_int4(B_i8_ref_t).T

            i4_cutlass_tflops = bench_tflops(cutlass_int4_mm, torch._int_mm(A_i8_ref, B_i8_ref_t.T), A_i4, B_i4)
            scaled_i4_cutlass_tflops = bench_tflops(
                cutlass_row_scaled_int4_mm,
                scaled_mm_ref(A_i8_ref, B_i8_ref_t.T, scale_A, scale_B),
                A_i4,
                B_i4,
                scale_A,
                scale_B,
            )

        else:
            i4_cutlass_tflops = 0
            scaled_i4_cutlass_tflops = 0

        # FP4
        if COMPUTE_CAPABILITY == (12, 0) and not args.a_column_major and args.b_column_major:
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

        data.append(
            [
                bf16_tflops,
                f16_acc_f16_triton_tflops,
                f8_triton_tflops,
                f8_cutlass_tflops,
                i8_pt_tflops,
                i8_triton_tflops,
                i4_cutlass_tflops,
                scaled_f8_inductor_tflops,
                scaled_f8_triton_tflops,
                scaled_f8_cutlass_tflops,
                block2d_scaled_f8_triton_tflops,
                scaled_i8_inductor_tflops,
                scaled_i8_triton_tflops,
                block2d_scaled_i8_triton_tflops,
                scaled_i4_cutlass_tflops,
                mxfp4_cutlass_tflops,
                nvfp4_cutlass_tflops,
            ]
        )

    gpu_name = torch.cuda.get_device_name()
    if gpu_name == "NVIDIA GeForce RTX 5090":
        # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
        bf16_tflops = 209.5
        fp16_acc_fp16_tflops = bf16_tflops * 2
        f8_tflops = bf16_tflops * 2
        i8_tflops = bf16_tflops * 4
        i4_tflops = 0
        f4_tflops = bf16_tflops * 8
        shapes.append("Theoretical")
        data.append(
            [
                bf16_tflops,
                fp16_acc_fp16_tflops,
                f8_tflops,  # triton
                f8_tflops,  # cutlass
                i8_tflops,  # pt
                i8_tflops,  # triton
                i4_tflops,  # cutlass
                f8_tflops,  # scaled inductor
                f8_tflops,  # scaled triton
                f8_tflops,  # scaled cutlass
                f8_tflops,  # block2d-scaled triton
                i8_tflops,  # scaled inductor
                i8_tflops,  # scaled triton
                i8_tflops,  # block2d-scaled triton
                i4_tflops,  # scaled cutlass
                f4_tflops,  # mxfp4 cutlass
                f4_tflops,  # nvfp4 cutlass
            ]
        )

    df = pd.DataFrame(
        data,
        index=shapes,
        columns=[
            "PyTorch (CuBLAS) BF16",
            "Triton FP16 w/ FP16 accumulate",
            "Triton FP8",
            "Cutlass FP8",
            "PyTorch (CuBLAS) INT8",
            "Triton INT8",
            "Cutlass INT4",
            "Inductor (Triton) row-scaled FP8",
            "Triton row-scaled FP8",
            "Cutlass row-scaled FP8",
            "Triton block2d-scaled FP8",
            "Inductor (Triton) row-scaled INT8",
            "Triton row-scaled INT8",
            "Triton block2d-scaled INT8",
            "Cutlass row-scaled INT4",
            "Cutlass MXFP4",
            "Cutlass NVFP4",
        ],
    )
    print(df.round(2).T.to_markdown())
