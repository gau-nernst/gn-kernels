import dataclasses
from pathlib import Path

import torch
from torch import Tensor

from ..nvrtc_utils import _TYPE_MAP, _compile_kernel, cdiv

CURRENT_DIR = Path(__file__).parent
KERNEL = open(CURRENT_DIR / "kernel_sm80.cu").read()


@dataclasses.dataclass
class AttnSm80Kernel:
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

        template_args = [self.block_q, self.block_kv, self.num_warps, self.qk_dim, self.v_dim, _TYPE_MAP[self.dtype]]
        kernel_name = f"attn_sm80_kernel<{', '.join(map(str, template_args))}>"
        self.kernel = _compile_kernel(KERNEL, kernel_name, self.smem_size)

    def run(self, q: Tensor, k: Tensor, v: Tensor):
        assert q.stride(-1) == 1
        assert k.stride(-1) == 1
        assert v.stride(-1) == 1

        bs, len_q, q_heads, _ = q.shape
        _, len_kv, kv_heads, _ = k.shape
        o = q.new_empty(bs, len_q, q_heads, self.v_dim)

        grid = (q_heads, cdiv(len_q, self.block_q), bs)
        args = (
            q,
            k,
            v,
            o,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            *o.stride()[:3],
            len_q,
            len_kv,
            q_heads,
            kv_heads,
        )
        self.kernel(grid, self.tb_size, args, self.smem_size)
        return o
