import time

import torch
from triton.testing import do_bench

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()
torch.set_default_device("cuda")

best = {k: (0, 0) for k in ("bf16", "int8", "fp8", "mxfp8", "nvfp4")}

dims = list(range(4096, 8192, 512)) + list(range(8192, 16384, 1024))
for dim in dims:
    M = N = K = dim
    print(f"{M=}, {N=}, {K=}")

    def bench_tflops(f, *args, **kwargs):
        time.sleep(0.5)
        latency = do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e-3
        return (2 * M * N * K) / latency * 1e-12

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16).T
    tflops = bench_tflops(torch.mm, A, B)
    best["bf16"] = max(best["bf16"], (tflops, dim))

    A = torch.randint(-128, 127, size=(M, K), dtype=torch.int8)
    B = torch.randint(-128, 127, size=(N, K), dtype=torch.int8).T
    tflops = bench_tflops(torch._int_mm, A, B)
    best["int8"] = max(best["int8"], (tflops, dim))

    if COMPUTE_CAPABILITY >= (8, 9):
        A = torch.randn(M, K).to(torch.float8_e4m3fn)
        B = torch.randn(N, K).to(torch.float8_e4m3fn).T
        scale_A = torch.randn(M, 1)
        scale_B = torch.randn(N, 1).T
        tflops = bench_tflops(torch._scaled_mm, A, B, scale_A, scale_B, out_dtype=torch.bfloat16)
        best["fp8"] = max(best["fp8"], (tflops, dim))

    if COMPUTE_CAPABILITY >= (10, 0):
        A = torch.randn(M, K).to(torch.float8_e4m3fn)
        B = torch.randn(N, K).to(torch.float8_e4m3fn).T
        scale_A = torch.randn(M, K // 32).to(torch.float8_e8m0fnu)
        scale_B = torch.randn(N, K // 32).to(torch.float8_e8m0fnu)
        tflops = bench_tflops(torch._scaled_mm, A, B, scale_A, scale_B, out_dtype=torch.bfloat16)
        best["mxfp8"] = max(best["mxfp8"], (tflops, dim))

        A = torch.randint(255, size=(M, K // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        B = torch.randint(255, size=(N, K // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2).T
        scale_A = torch.randn(M, K // 16).to(torch.float8_e4m3fn)
        scale_B = torch.randn(N, K // 16).to(torch.float8_e4m3fn)
        tflops = bench_tflops(torch._scaled_mm, A, B, scale_A, scale_B, out_dtype=torch.bfloat16)
        best["nvfp4"] = max(best["nvfp4"], (tflops, dim))

print(torch.cuda.get_device_name())
for k, (tflops, dim) in best.items():
    print(f"{k}: {tflops:.2f} TFLOPS, {dim=}")
