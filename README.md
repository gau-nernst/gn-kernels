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
