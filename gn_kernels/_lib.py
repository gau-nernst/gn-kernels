import torch

lib = torch.library.Library("gn_kernels", "DEF")
lib_ops = torch.ops.gn_kernels
