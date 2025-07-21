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
  - INT8 attention (QK only)
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

### Matmul

RTX 5090 TFLOPS @ 400W. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16. Use default Cutlass INT4 GEMM)
- `torch==2.9.0.dev20250712+cu129`
- `pytorch-triton==3.4.0+gitae848267`

Note:
- Bad FP8 perf on Triton is fixed in https://github.com/triton-lang/triton/pull/7409

Row-major x Column-major (`A @ B.T`)

|                                   |   1024 |   2048 |    4096 |   Theoretical |
|:----------------------------------|-------:|-------:|--------:|--------------:|
| PyTorch (CuBLAS) BF16             |  87.38 | 167.77 |  176.98 |         209.5 |
| Triton FP16 w/ FP16 accumulate    | 149.8  | 271.97 |  265.06 |         419   |
| Triton FP8                        | 118.78 | 190.58 |  217.89 |         419   |
| Cutlass FP8                       | 116.11 | 308.81 |  383.41 |         419   |
| PyTorch (CuBLAS) INT8             | 209.72 | 466.03 |  591.84 |         838   |
| Triton INT8                       | 172.96 | 466.03 |  523.71 |         838   |
| Cutlass INT4                      |  18.08 |  73.58 |   74.77 |           0   |
| Inductor (Triton) row-scaled FP8  |   3.31 |  26.76 |  212.75 |         419   |
| Triton row-scaled FP8             | 116.51 | 190.65 |  217.55 |         419   |
| Cutlass row-scaled FP8            | 116.31 | 309.61 |  383.58 |         419   |
| Triton block2d-scaled FP8         |  69.91 | 161.22 |  192.85 |         419   |
| Inductor (Triton) row-scaled INT8 |   3.58 |  22.18 |  176.47 |         838   |
| Triton row-scaled INT8            | 173.41 | 493.45 |  540.32 |         838   |
| Triton block2d-scaled INT8        | 148.8  | 381.03 |  453.44 |         838   |
| Cutlass row-scaled INT4           |  17.82 |  72.52 |   73.87 |           0   |
| Cutlass MXFP4                     | 209.06 | 701.79 | 1099.86 |        1676   |
| Cutlass NVFP4                     | 209.72 | 699.05 | 1100.99 |        1676   |

Row-major x Row-major (`A @ B`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.77 | 177.54 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 270.74 | 241.36 |       419   |
| Triton FP8                     | 116.51 | 171.2  | 196.3  |       419   |
| PyTorch (CuBLAS) INT8          |  61.74 | 167.77 | 185.9  |       838   |
| Triton INT8                    | 152.52 | 363.98 | 360.8  |       838   |
| Triton scaled FP8              | 115.9  | 167.77 | 193.4  |       419   |
| Triton tile-scaled FP8         |  66.05 | 149.8  | 177.54 |       419   |
| Inductor (Triton) scaled INT8  | 131.07 | 335.54 | 413.81 |       838   |
| Triton scaled INT8             | 173.41 | 349.53 | 324.17 |       838   |
| Triton tile-scaled INT8        | 116.51 | 271.97 | 299.59 |       838   |

Column-major x Row-major (`A.T @ B`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.77 | 176.83 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 278.17 | 244.37 |       419   |
| Triton FP8                     | 116.51 | 164.43 | 184.94 |       419   |
| PyTorch (CuBLAS) INT8          |  69.91 | 209.72 | 219.67 |       838   |
| Triton INT8                    | 147.17 | 364.72 | 362.25 |       838   |
| Triton scaled FP8              | 116.51 | 161.71 | 184.9  |       419   |
| Triton tile-scaled FP8         |  58.25 | 127.34 | 154.33 |       419   |
| Inductor (Triton) scaled INT8  | 118.15 | 226.72 | 289.1  |       838   |
| Triton scaled INT8             | 149.8  | 380.49 | 370.66 |       838   |
| Triton tile-scaled INT8        |  95.33 | 233.02 | 257.12 |       838   |
