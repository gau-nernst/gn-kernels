import dataclasses
from pathlib import Path

import torch
from torch import Tensor

from ..cuda_utils import _TYPE_MAP, _compile_kernel, cdiv

CURRENT_DIR = Path(__file__).parent
MARKER = "// start of kernel"
_, KERNEL = open(CURRENT_DIR / "kernel.cu").read().split(MARKER)

HEADER_TEMPLATE = """
#include "common.h"

constexpr int QK_DIM = {};
constexpr int V_DIM = {};
using Type = {};

constexpr int BLOCK_Q = {};
constexpr int BLOCK_KV = {};
constexpr int NUM_WARPS = {};
"""


@dataclasses.dataclass
class AttnKernel:
    dtype: torch.dtype = torch.bfloat16
    qk_dim: int = 128
    v_dim: int = 128
    # hparams
    block_q: int = 64
    block_kv: int = 64
    num_warps: int = 4

    def __post_init__(self) -> None:
        self.tb_size = (self.num_warps * 32, 1, 1)

        q_size = self.block_q * self.qk_dim
        k_size = self.block_kv * self.qk_dim * 2
        v_size = self.block_kv * self.v_dim
        self.smem_size = max(q_size, k_size + v_size) * self.dtype.itemsize

        header = HEADER_TEMPLATE.format(
            self.qk_dim,
            self.v_dim,
            _TYPE_MAP[self.dtype],
            self.block_q,
            self.block_kv,
            self.num_warps,
        )
        self.kernel = _compile_kernel(KERNEL, "attn_kernel", header, self.smem_size)

    def run(self, q: Tensor, k: Tensor, v: Tensor):
        assert q.stride(-1) == 1
        assert k.stride(-1) == 1
        assert v.stride(-1) == 1

        bs, len_q, num_heads, _ = q.shape
        _, len_kv, _, _ = k.shape
        o = q.new_empty(bs, len_q, num_heads, self.v_dim)

        grid = (cdiv(len_q, self.block_q), num_heads, bs)
        args = (
            q,
            k,
            v,
            o,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            *o.stride()[:3],
            bs,
            len_q,
            len_kv,
            num_heads,
        )
        self.kernel(grid, self.tb_size, args, self.smem_size)
        return o
