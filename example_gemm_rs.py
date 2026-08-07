# torchrun --standalone --nproc-per-node=4 example_gemm_rs.py

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from gn_kernels.cutedsl.sm100.sm100_gemm_rs import gemm_rs

if __name__ == "__main__":
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    symm_mem.set_backend("NCCL")

    M, N, K = 2048, 7168, 1536

    # allocate symmetric memory
    partial = symm_mem.empty(M, N, dtype=torch.bfloat16, device=device)
    partial_handle = symm_mem.rendezvous(partial, dist.group.WORLD)
    output = torch.empty(M // dist.get_world_size(), N, dtype=torch.bfloat16, device=device)
    num_flags = (M // 128) * (N // 128) + torch.cuda.get_device_properties().multi_processor_count
    flags = symm_mem.empty(num_flags, dtype=torch.int32, device=device)
    flags.zero_()
    flags_handle = symm_mem.rendezvous(flags, dist.group.WORLD)

    # issue gemm-rs
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    gemm_rs(x, w, partial, partial_handle, output, flags, flags_handle)

    torch.cuda.synchronize()
    dist.destroy_process_group()
