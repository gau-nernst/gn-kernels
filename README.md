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

- SM80: Cutlass INT4 + rowwise-scaled INT4
- SM89: Cutlass FP8 + rowwise-scaled FP8
- SM120:
  - Cutlass FP8 + rowwise-scaled FP8
  - Cutlass FP4 + rowwise-scaled FP4
- Triton:
  - Matmul with configurable input dtype, accumulate dtype e.g. FP16 MMA with FP16 accumulate
  - Rowwise-scaled matmul
  - Tile-scaled matmul (i.e. DeepSeek style)

## Speed benchmarks

### Matmul

RTX 5090 TFLOPS @ 400W. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16. Use default Cutlass INT4 GEMM)
- `torch==2.7.0+cu128`
- `triton==3.3.1`

Row-major x Column-major (`A @ B.T`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.72 | 176.37 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 270.6  | 234.85 |       419   |
| Triton FP8                     | 116.51 | 188.51 | 208.41 |       419   |
| PyTorch (CuBLAS) INT8          | 210.37 | 466.03 | 479.3  |       838   |
| Triton INT8                    | 173.63 | 466.03 | 489.68 |       838   |
| Cutlass INT4                   |  17.77 |  72.42 |  74.1  |         0   |
| Inductor (Triton) scaled FP8   |  95.33 | 181.81 | 215.87 |       419   |
| Triton scaled FP8              | 116.51 | 186.41 | 207.24 |       419   |
| Triton tile-scaled FP8         |  69.91 | 158.28 | 189.57 |       419   |
| Inductor (Triton) scaled INT8  | 149.8  | 381.3  | 512.28 |       838   |
| Triton scaled INT8             | 174.76 | 493.45 | 480.56 |       838   |
| Triton tile-scaled INT8        | 149.8  | 399.46 | 399.42 |       838   |
| Cutlass scaled INT4            |  18.08 |  74.24 |  75.23 |         0   |

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
