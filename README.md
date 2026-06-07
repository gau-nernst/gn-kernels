# My kernels collection

Kernels in this folder can be installed as a package to provide code sharing for my projects.

```bash
# CUDA version must match your nvcc version
uv pip install torch>=2.10 --index-url https://download.pytorch.org/whl/cu130

uv pip install git+https://github.com/gau-nernst/gn_kernels --no-build-isolation
```

Available kernels

- Triton
  - Matmul with various input dtypes, FP16 accumulation, row-scaled and 2d-block-scaled (DeepSeek)
  - Attention:
    - GQA, causal
    - Optional QK with INT8/FP8 MMA
    - Optional QK and PV with FP16 accumulation
    - TODO: varlen, paged
- CUDA
  - (NVRTC) Matmul with various input dtypes and FP16 accumulation
  - Attention: optional QK with INT8/FP8 MMA
- CuteDSL:
  - SM80/SM89 matmul (`cp.async` + `mma.sync`): BF16, INT8, FP8
  - SM100 matmul (TMA + `tcgen05`): BF16, MXFP8, NVFP4
  - SM120 matmul (TMA + `mma.sync`): BF16, INT8, FP8, MXFP8

## Speed benchmarks

### Realistic SOL

Sweep over large matmul shapes and get the max achieved TFLOPS.

Device name       | memcpy BW | BF16    | INT8    | FP8     | MXFP8   | NVFP4
------------------|-----------|---------|---------|---------|---------|--------
PRO 6000 @ 600W   |           |  427.62 |  795.87 |  789.28 |  741.56 | 1425.95
5090 @ 600W       |   1509.43 |  245.80 |  726.34 |  480.37 |  659.95 | 1306.48
5090 @ 400W       |   1509.05 |  229.55 |  602.79 |  447.50 |  552.20 | 1165.04
A100-80GB (Modal) |   1700.37 |  266.90 |  464.50
H200 (Modal)      |   4055.45 |  763.41 | 1160.57 | 1085.03
B200 (Modal)      |   5981.99 | 1722.53 | 3378.12 | 2668.98 | 3171.94 | 6241.55
MI300X @ 750W     |           |  660.48 |  887.66 | 1197.41

Note: INT8 matmul implementation I'm using on MI300x is probably not good.

### Matmul

RTX 5090 TFLOPS @ 400W. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16)
- `torch==2.12.0`
- `triton==3.7.0`

Note:
- Row-major x Column-major

|                                   |   1024 |   2048 |    4096 |    SOL |
|:----------------------------------|-------:|-------:|--------:|-------:|
| PyTorch (CuBLAS) BF16             | 130.82 | 167.77 |  203.05 |  209.5 |
| Triton FP16 w/ FP16 accumulate    | 149.46 | 279.47 |  267.37 |  419   |
| Triton FP8                        | 173.86 | 320.33 |  379.08 |  419   |
| PyTorch (Cutlass) row-scaled FP8  | 174.76 | 310.69 |  390.17 |  419   |
| Inductor (Triton) row-scaled FP8  | 178.96 | 309.08 |  392.09 |  419   |
| Triton row-scaled FP8             | 170.33 | 310.69 |  376.78 |  419   |
| PyTorch (CuBLAS) INT8             | 209.72 | 466.03 |  583.56 |  838   |
| Triton INT8                       | 175.22 | 470.53 |  530.47 |  838   |
| Inductor (Triton) row-scaled INT8 | 147.17 | 419.43 |  520.22 |  838   |
| Triton row-scaled INT8            | 174.76 | 493.45 |  542.57 |  838   |
| PyTorch (CuBLAS) MXFP8            | 171.2  | 462.82 |  531.52 |  838   |
| PyTorch (CuBLAS) NVFP4            | 215.09 | 762.6  | 1139.85 | 1676   |
