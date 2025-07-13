import argparse
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func

    def flash_attn(q: Tensor, k: Tensor, v: Tensor):
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

except ImportError:
    flash_attn = None

from gn_kernels import triton_attn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--nh", type=int, default=8)
    parser.add_argument("--lq", type=int, default=4096)
    parser.add_argument("--lkv", type=int, default=8192)
    args = parser.parse_args()

    bs = args.bs
    nh = args.nh
    lq = args.lq
    lkv = args.lkv
    head_dim = 128

    torch.set_default_device("cuda")
    torch.manual_seed(2025 * 2026)
    COMPUTE_CAPABILITY = torch.cuda.get_device_capability()

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    def generate_input(*shape):
        return torch.randn(shape).add(0.5).bfloat16().cuda()

    Q = generate_input(bs, nh, lq, head_dim)
    K = generate_input(bs, nh, lkv, head_dim)
    V = generate_input(bs, nh, lkv, head_dim)

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    results = []

    out_ref = F.scaled_dot_product_attention(Q, K, V)

    def bench(f, name):
        print(f"Benching {name}")
        out = f(Q, K, V)
        torch.testing.assert_close(out, out_ref)

        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(Q, K, V), return_mode="median")
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        results.append([name, round(latency_ms, 4), round(tflops, 2), round(pct_sol, 2)])

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        bench(F.scaled_dot_product_attention, "F.sdpa() - FA")

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        bench(F.scaled_dot_product_attention, "F.sdpa() - CuDNN")

    compiled_flex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=False)
    bench(compiled_flex_attention, "FlexAttention")

    if flash_attn is not None:
        bench(flash_attn, "flash-attn")

    bench(triton_attn, "triton")

    df = pd.DataFrame(results, columns=["Kernel", "Latency (ms)", "TFLOPS", "% SOL"])
    print(df.to_markdown(index=False))
