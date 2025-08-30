# My kernels collection

Kernels in this folder can be installed as a package to provide code sharing for my projects.

```bash
uv pip install git+https://github.com/gau-nernst/gn_kernels --no-build-isolation
```

TODO:
- Separate matmul and block-scaled matmul (mx and nvfp4)
- Attention: triton
- Triton persistent matmul kernel

Available kernels

- SM80:
  - Cutlass INT4 + rowwise-scaled INT4
  - INT8 attention (QK only and QK+PV)
- SM89: Cutlass FP8 + rowwise-scaled FP8
- SM120:
  - Cutlass FP8 + rowwise-scaled FP8
  - Cutlass FP4 + rowwise-scaled FP4
  - MXFP8 attention (QK only)
- Triton:
  - Matmul with configurable input dtype, accumulate dtype e.g. FP16 MMA with FP16 accumulate
  - Rowwise-scaled matmul
  - Tile-scaled matmul (i.e. DeepSeek style)

## Speed benchmarks

### Realistic SOL

Sweep over large matmul shapes and get the max achieved TFLOPS.

Device name | BF16 | INT8 | FP8 | MXFP8 | NVFP4
------------|------|------|-----|-------|------
5090 @ 600W | 245.80 | 726.34 | 480.37 | 659.95 | 1306.48
5090 @ 400W | 229.55 | 602.79 | 447.50 | 552.20 | 1165.04

### Matmul

RTX 5090 TFLOPS @ 400W. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16. Use default Cutlass INT4 GEMM)
- `torch==2.8.0`
- `pytorch-triton==3.4.0+git11ec6354`

Note:
- Bad FP8 perf on Triton is fixed in https://github.com/triton-lang/triton/pull/7409
- Row-major x Column-major

|                                   |   1024 |   2048 |    4096 |   Theoretical |
|:----------------------------------|-------:|-------:|--------:|--------------:|
| PyTorch (CuBLAS) BF16             |  87.38 | 167.77 |  176.99 |         209.5 |
| Triton FP16 w/ FP16 accumulate    | 149.8  | 288.95 |  273.77 |         419   |
| Triton FP8                        | 116.71 | 190.58 |  217.19 |         419   |
| Cutlass FP8                       | 116.11 | 310.33 |  383.41 |         419   |
| PyTorch (CuBLAS) INT8             | 209.72 | 466.03 |  593.8  |         838   |
| Triton INT8                       | 173.41 | 466.03 |  524.29 |         838   |
| Cutlass INT4                      |  18.08 |  73.58 |   74.73 |           0   |
| Inductor (Triton) row-scaled FP8  | 116.51 | 189.77 |  214.27 |         419   |
| Triton row-scaled FP8             | 116.11 | 190.45 |  216.83 |         419   |
| Cutlass row-scaled FP8            | 116.51 | 310.15 |  387.6  |         419   |
| Triton block2d-scaled FP8         |  69.91 | 161.22 |  192.85 |         419   |
| Inductor (Triton) row-scaled INT8 | 149.46 | 400.35 |  520.22 |         838   |
| Triton row-scaled INT8            | 173.41 | 493.45 |  541.41 |         838   |
| Triton block2d-scaled INT8        | 148.8  | 381.03 |  453.44 |         838   |
| Cutlass row-scaled INT4           |  17.72 |  72.91 |   73.86 |           0   |
| Cutlass MXFP4                     | 209.72 | 709.21 | 1099.58 |        1676   |
| Cutlass NVFP4                     | 209.72 | 699.05 | 1100.43 |        1676   |
