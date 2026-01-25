import time

import torch
from triton.testing import do_bench

from gn_kernels.cuda_mm import MatmulSm80Kernel


def main(M: int, N: int, K: int):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T

    kernel = MatmulSm80Kernel(block_mnk=(128, 64, 64), num_stages=2)
    torch.testing.assert_close(kernel.run(A, B), torch.mm(A, B))

    def bench(f, name: str):
        time.sleep(0.2)
        latency_ms = do_bench(lambda: f(A, B), warmup=100, rep=100, return_mode="median")
        tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
        print(f"{name}: {latency_ms * 1e3:.2f} us, {tflops:.2f} TFLOPS")

    bench(kernel.run, "Ours")
    bench(torch.mm, "CuBLAS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()

    main(args.m, args.n, args.k)
