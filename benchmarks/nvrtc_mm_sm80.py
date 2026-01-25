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


def modal_main(M: int, N: int, K: int, gpu: str):
    import subprocess

    import modal

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()

    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("git", "clang")
        .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
        .uv_pip_install("ninja", "wheel")
        .uv_pip_install(
            f"git+https://github.com/gau-nernst/gn-kernels.git@{commit_hash}",
            extra_options="--no-build-isolation",
        )
    )
    app = modal.App("benchmark-cuda-sm80")
    func = app.function(image=image, gpu=gpu, serialized=True)(main)

    with modal.enable_output(), app.run():
        func.remote(M, N, K)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--modal")
    args = parser.parse_args()

    if args.modal is not None:
        modal_main(args.m, args.n, args.k, args.modal)
    else:
        main(args.m, args.n, args.k)
