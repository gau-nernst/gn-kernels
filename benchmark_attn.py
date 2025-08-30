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
    import flash_attn

except ImportError:
    flash_attn = None

from gn_kernels import triton_attn


# add a small offset so that output does not have a mean of zero,
# which will result in large relative error
def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()


def sdpa(q: Tensor, k: Tensor, v: Tensor):
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def flex_attn(q: Tensor, k: Tensor, v: Tensor):
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    return flex_attention(q, k, v).transpose(1, 2)


def quantize(x: Tensor, dtype: torch.dtype):
    info_fn = torch.finfo if dtype.is_floating_point else torch.iinfo
    dtype_min = info_fn(dtype).min
    dtype_max = info_fn(dtype).max

    x = x.float()
    amax = x.abs().amax(dim=-1)
    scale = amax / min(-dtype_min, dtype_max)
    x = x / scale.clip(1e-6).unsqueeze(-1)
    if not dtype.is_floating_point:
        x = x.round()
    xq = x.clip(dtype_min, dtype_max).to(dtype)
    return xq, scale


def dequantize(xq: Tensor, scale: Tensor):
    return xq.float() * scale.float().unsqueeze(-1)


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

    Q = generate_input(bs, lq, nh, head_dim)
    K = generate_input(bs, lkv, nh, head_dim)
    V = generate_input(bs, lkv, nh, head_dim)

    Q_i8, scale_Q_i8 = quantize(Q, torch.int8)
    K_i8, scale_K_i8 = quantize(K, torch.int8)
    scale_Q_i8 = scale_Q_i8.transpose(1, 2).contiguous()
    scale_K_i8 = scale_K_i8.transpose(1, 2).contiguous()

    Q_f8, scale_Q_f8 = quantize(Q, torch.float8_e4m3fn)
    K_f8, scale_K_f8 = quantize(K, torch.float8_e4m3fn)
    scale_Q_f8 = scale_Q_f8.transpose(1, 2).contiguous()
    scale_K_f8 = scale_K_f8.transpose(1, 2).contiguous()

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    results = []

    def bench(f, name, *args, out_ref):
        print(f"Benching {name}")
        out = f(*args)
        torch.testing.assert_close(out, out_ref)

        time.sleep(1)  # stabilize thermal
        latency_ms = do_bench(lambda: f(*args), return_mode="median")
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol

        data_point = {
            "Kernel": name,
            "Latency (ms)": f"{latency_ms:.2f}",
            "TFLOPS": f"{tflops:.2f}",
            "%SOL": f"{pct_sol * 100:.2f}%",
        }
        results.append(data_point)

    bf16_ref = sdpa(Q, K, V)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        bench(sdpa, "F.sdpa() - FA", Q, K, V, out_ref=bf16_ref)

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        bench(sdpa, "F.sdpa() - CuDNN", Q, K, V, out_ref=bf16_ref)

    bench(flex_attn, "FlexAttention", Q, K, V, out_ref=bf16_ref)

    if flash_attn is not None:
        bench(flash_attn.flash_attn_func, "flash-attn", Q, K, V, out_ref=bf16_ref)

    bench(triton_attn, "Triton", Q, K, V, out_ref=bf16_ref)

    Q_i8_dq = dequantize(Q_i8, scale_Q_i8.transpose(1, 2))
    K_i8_dq = dequantize(K_i8, scale_K_i8.transpose(1, 2))
    qk_i8_ref = sdpa(Q_i8_dq, K_i8_dq, V.float()).bfloat16()
    bench(triton_attn, "Triton qk-int8", Q_i8, K_i8, V, scale_Q_i8, scale_K_i8, out_ref=qk_i8_ref)

    Q_f8_dq = dequantize(Q_f8, scale_Q_f8.transpose(1, 2))
    K_f8_dq = dequantize(K_f8, scale_K_f8.transpose(1, 2))
    qk_f8_ref = sdpa(Q_f8_dq, K_f8_dq, V.float()).bfloat16()
    bench(triton_attn, "Triton qk-fp8", Q_f8, K_f8, V, scale_Q_f8, scale_K_f8, out_ref=qk_f8_ref)

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
