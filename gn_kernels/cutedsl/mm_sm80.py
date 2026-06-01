import math
from functools import cache
from typing import NamedTuple

import torch
from cuda.bindings.driver import CUstream
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from torch import Tensor

import cutlass
from cutlass import Float32, Int8, Int32, cute

from .utils import TORCH_TO_CUTE_DTYPE, mma_sync


class MatmulSm80(NamedTuple):
    dtype: cutlass.Numeric
    acc_dtype: cutlass.Numeric
    warp_layout: tuple[int, int] = (2, 2)
    cta_tile: tuple[int, int, int] = (128, 128, 32)
    num_stages: int = 2

    @cute.jit
    def make_AB_layout(self, BM: int, BK: int, num_stages: int):
        # only go up to 128B i.e. 64 for BF16
        elem_width = self.dtype.width // 8  # in byte
        swizzle_width = min(BK, 128 // elem_width)

        # canonical 128B swizzling is (3,4,3) on raw smem address.
        # for BF16, each elem = 2 bytes -> 128B swizzling is (3, 3, 3).
        B = int(math.log2(swizzle_width * 2 // 16))
        M = int(math.log2(16 // elem_width))
        S = 3
        swizzle = cute.make_swizzle(B, M, S)

        slayout = cute.make_layout(
            (BM, (swizzle_width, BK // swizzle_width), num_stages),
            stride=(swizzle_width, (1, BM * swizzle_width), BM * BK),
        )
        return cute.make_composed_layout(swizzle, 0, slayout)

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, stream: CUstream):
        M, N = gC.shape
        BM, BN, BK = self.cta_tile

        sA_layout = self.make_AB_layout(BM, BK, self.num_stages)
        sB_layout = self.make_AB_layout(BN, BK, self.num_stages)

        grid = (cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1)
        block = (math.prod(self.warp_layout) * 32, 1, 1)

        self.kernel(
            gA,
            gB,
            gC,
            sA_layout,
            sB_layout,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        _, K = gA.shape
        dtype = self.dtype
        acc_dtype = self.acc_dtype
        BM, BN, BK = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages

        WM = BM // num_warp_m
        WN = BN // num_warp_n
        warp_id_m = warp_id // num_warp_n
        warp_id_n = warp_id % num_warp_n

        # select input/output tiles
        gA_tiles = cute.local_tile(gA, tiler=(BM, BK), coord=(bid_m, None))  # (BM, BK, K/BK)
        gB_tiles = cute.local_tile(gB, tiler=(BN, BK), coord=(bid_n, None))  # (BN, BK, K/BK)

        # allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(dtype, sA_layout, 16)
        sB = smem.allocate_tensor(dtype, sB_layout, 16)

        # warp partition
        # shape: (WM, BK, num_stages)
        # (well, the 2nd mode, is actually (width,BK/width). but to simplify the expresion...)
        sA_warp = cute.logical_divide(sA, (WM, None, None))[(None, warp_id_m), None, None]
        sB_warp = cute.logical_divide(sB, (WN, None, None))[(None, warp_id_n), None, None]

        # pre-compute ldmatrix address (16x16 tile)
        # bring ldsm elems in front
        # (8, (WM,BK/8,num_stages))
        sA_ldsm = cute.zipped_divide(sA_warp, (1, 8, 1))[(0, None, 0), None]
        sB_ldsm = cute.zipped_divide(sB_warp, (1, 8, 1))[(0, None, 0), None]

        # partition to (16,16) tiles (appear as (16,2) lane tiles)
        # (8, ((16,WM/16), (2,BK/16), num_stages))
        sA_ldsm = cute.logical_divide(sA_ldsm, (None, (16, 2, None)))
        sB_ldsm = cute.logical_divide(sB_ldsm, (None, (16, 2, None)))

        # select the address
        # (8, WM/16, BK/16, num_stages)
        sA_ldsm = sA_ldsm[None, ((lane_id % 16, None), (lane_id // 16, None), None)]
        sB_ldsm = sB_ldsm[None, (((lane_id // 16) * 8 + (lane_id % 8), None), ((lane_id // 8) % 2, None), None)]

        # ldmatrix.x4
        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

        # registers
        # let ptxas decides register reuse for rA and rB
        rA = cute.make_rmem_tensor((8, WM // 16, BK // 16), dtype)
        rB = cute.make_rmem_tensor(((4, 2), WN // 16, BK // 16), dtype)
        rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), acc_dtype)
        rC.fill(0.0)

        # prefetch
        for i in cutlass.range_constexpr(num_stages - 1):
            self.load_g2s(gA_tiles[None, None, i], sA[None, None, i], tid)
            self.load_g2s(gB_tiles[None, None, i], sB[None, None, i], tid)
            cute.arch.cp_async_commit_group()

        load_stage = num_stages - 1
        load_k = num_stages - 1
        compute_stage = 0

        num_iters = cute.ceil_div(K, BK)
        for iter_k in range(num_iters):
            load_k = iter_k + (num_stages - 1)
            if load_k < num_iters:
                cute.arch.sync_threads()
                self.load_g2s(gA_tiles[None, None, load_k], sA[None, None, load_stage], tid)
                self.load_g2s(gB_tiles[None, None, load_k], sB[None, None, load_stage], tid)
                load_stage = (load_stage + 1) % num_stages
            cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(num_stages - 1)
            cute.arch.sync_threads()

            for k in cutlass.range_constexpr(BK // 16):
                cute.copy(ldsm_atom, sA_ldsm[None, None, k, compute_stage], rA[None, None, k])
                cute.copy(ldsm_atom, sB_ldsm[None, None, k, compute_stage], rB[None, None, k])

                for m in cutlass.range_constexpr(WM // 16):
                    for n in cutlass.range_constexpr(WN // 16):
                        rC[None, n * 2 + 0, m] = mma_sync(rA[None, m, k], rB[(None, 0), n, k], rC[None, n * 2 + 0, m])
                        rC[None, n * 2 + 1, m] = mma_sync(rA[None, m, k], rB[(None, 1), n, k], rC[None, n * 2 + 1, m])

            compute_stage = (compute_stage + 1) % num_stages

        # epilogue
        cp_op = nvgpu.CopyUniversalOp()
        cp_atom = cute.make_copy_atom(cp_op, dtype, num_bits_per_copy=2 * gC.element_type.width)

        # create view into C gmem
        gC_cta = cute.local_tile(gC, tiler=(BM, BN), coord=(bid_m, bid_n))
        gC_warp = cute.local_tile(gC_cta, tiler=(WM, WN), coord=(warp_id_m, warp_id_n))

        # (2, (WM, WN/2))
        gC_warp = cute.zipped_divide(gC_warp, (1, 2))[(0, None), None]

        # (2, (((8,2), WM/16), (4,WN/8)))
        gC_view = cute.logical_divide(gC_warp, (None, (cute.make_layout((8, 2)), 4)))

        # (2, 2, WM/16, WN/8)
        gC_thr = gC_view[None, (((lane_id // 4, None), None), (lane_id % 4, None))]

        # explicit for loop to interleave cvt with st.global
        for m in cutlass.range_constexpr(WM // 16):
            for n in cutlass.range_constexpr(WN // 8):
                if cutlass.const_expr(acc_dtype == gC.element_type):
                    cute.copy(cp_atom, rC[None, n, m], gC_thr[None, None, m, n])
                else:
                    rC_bf16 = cute.make_rmem_tensor((2, 2), dtype)
                    rC_bf16.store(rC[None, n, m].load().to(dtype))
                    cute.copy(cp_atom, rC_bf16, gC_thr[None, None, m, n])

    @cute.jit
    def load_g2s(self, gA: cute.Tensor, sA: cute.Tensor, tid: Int32):
        # recast as bytes
        gA = cute.recast_tensor(gA, Int8)
        sA = cute.recast_tensor(sA, Int8)

        # cp.async.cg
        op = cpasync.CopyG2SOp(nvgpu.LoadCacheMode.GLOBAL)
        atom = cute.make_copy_atom(op, Int8, num_bits_per_copy=128)

        tb_size = math.prod(self.warp_layout) * 32
        _, _, BK = self.cta_tile
        BK_ = BK * (self.dtype.width // 8)

        num_cols = BK_ // 16
        num_rows = tb_size // num_cols

        # (num_rows, 16B, BM/num_rows)
        gA_view = cute.local_tile(gA, (num_rows, 16), (None, tid % num_cols))
        sA_view = cute.local_tile(sA, (num_rows, 16), (None, tid % num_cols))

        # (16B, BM/num_rows)
        gA_view = gA_view[tid // num_cols, None, None]
        sA_view = sA_view[tid // num_cols, None, None]
        cute.copy(atom, gA_view, sA_view)

    @cache
    @staticmethod
    def compile(dtype: torch.dtype):
        cute_dtype: cutlass.Numeric = TORCH_TO_CUTE_DTYPE[dtype]
        acc_dtype = Float32 if dtype.is_floating_point else Int32

        divisibility = 128 // cute_dtype.width  # 16B
        acc_divisibility = 128 // acc_dtype.width
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()

        A = cute.runtime.make_fake_tensor(cute_dtype, (M, K), (cute.sym_int64(divisibility), 1), assumed_align=16)
        B = cute.runtime.make_fake_tensor(cute_dtype, (N, K), (cute.sym_int64(divisibility), 1), assumed_align=16)
        C = cute.runtime.make_fake_tensor(cute_dtype, (M, N), (cute.sym_int64(acc_divisibility), 1), assumed_align=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel = MatmulSm80(cute_dtype, acc_dtype)
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def mm(A: Tensor, B: Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulSm80.compile(A.dtype)(A, B.T, C)
    return C
