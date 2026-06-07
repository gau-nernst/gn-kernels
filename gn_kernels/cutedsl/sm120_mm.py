import math
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int8, Int32, Int64, cute
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp

from .utils import TORCH_TO_CUTE_DTYPE, mma_sync, simple_tma_g2s


class Sm120MatmulBF16:
    """Supports BF16, INT8, FP8"""

    warp_layout = (2, 2)
    num_stages = 2

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, stream: CUstream):
        BM, BN = 128, 128
        BK = 128 // (gA.element_type.width // 8)  # 128B
        cta_tile = (BM, BN, BK)

        A_args = self.prepare_AB(gA, BM, BK)
        B_args = self.prepare_AB(gB, BN, BK)

        M, N = gC.shape
        grid = (cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1)
        num_warps = math.prod(self.warp_layout) + 1
        block = (num_warps * 32, 1, 1)
        self.kernel(A_args, B_args, gC, cta_tile).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        B_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        gC: cute.Tensor,
        cta_tile: cutlass.Constexpr[tuple[int, int, int]],
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, BK = cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages

        A_tma_atom, A_tma_tensor, sA_layout = A_args
        B_tma_atom, B_tma_tensor, sB_layout = B_args

        _, K = A_tma_tensor.shape
        dtype = A_tma_atom.value_type

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(dtype, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(dtype, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if warp_id == 0:
            for i in cutlass.range_constexpr(num_stages):
                cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
            cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma_atom)
            cpasync.prefetch_descriptor(B_tma_atom)
        cute.arch.sync_threads()

        if warp_id == 4:
            # TMA warp
            tma_stage = 0
            parity = 1

            # select gmem tile
            gA_tiles = cute.local_tile(A_tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
            gB_tiles = cute.local_tile(B_tma_tensor, (BN, BK), (bid_n, None))

            for iter_k in range(K // BK):
                mbar = tma_full_mbar + tma_stage

                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = (BM + BN) * BK * (dtype.width // 8)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_g2s(A_tma_atom, gA_tiles[None, None, iter_k], sA[None, None, tma_stage], mbar)
                simple_tma_g2s(B_tma_atom, gB_tiles[None, None, iter_k], sB[None, None, tma_stage], mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

        else:
            # MMA warps
            tma_stage = 0
            parity = 0

            WM = BM // num_warp_m
            WN = BN // num_warp_n
            warp_id_m = warp_id // num_warp_n
            warp_id_n = warp_id % num_warp_n

            # warp partition
            # shape: (WM, BK, num_stages)
            sA_warp = cute.local_tile(sA, (WM, BK, num_stages), (warp_id_m, 0, 0))
            sB_warp = cute.local_tile(sB, (WN, BK, num_stages), (warp_id_n, 0, 0))

            # pre-compute ldmatrix address (16x16 tile)
            # ((16, (16B, 2), 1), (WM/16, BK/32B, num_stages))
            elems = 128 // dtype.width  # elements to cover 16B row
            sA_ldsm = cute.zipped_divide(sA_warp, (16, cute.make_layout((elems, 2)), 1))
            sB_ldsm = cute.zipped_divide(sB_warp, (16, cute.make_layout((elems, 2)), 1))

            # select the address
            # (16B, (WM/16, BK/32B, num_stages))
            sA_ldsm = sA_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
            sB_ldsm = sB_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2), 0), None]

            # ldmatrix.x4
            ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
            ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

            # registers
            # let ptxas decides register reuse for rA and rB
            acc_dtype = Int32 if dtype is Int8 else Float32
            rA = cute.make_rmem_tensor((elems, WM // 16, BK // 16), dtype)
            rB = cute.make_rmem_tensor(((elems // 2, 2), WN // 16, BK // 16), dtype)
            rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), acc_dtype)
            rC.fill(0.0)

            for iter_k in range(K // BK):
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                MMA_K = 32 // (dtype.width // 8)  # 32B
                for k in cutlass.range_constexpr(BK // MMA_K):
                    cute.copy(ldsm_atom, sA_ldsm[None, (None, k, tma_stage)], rA[None, None, k])
                    cute.copy(ldsm_atom, sB_ldsm[None, (None, k, tma_stage)], rB[None, None, k])

                    for m in cutlass.range_constexpr(WM // 16):
                        for n in cutlass.range_constexpr(WN // 16):
                            rC[None, n * 2 + 0, m] = mma_sync(
                                rA[None, m, k], rB[(None, 0), n, k], rC[None, n * 2 + 0, m]
                            )
                            rC[None, n * 2 + 1, m] = mma_sync(
                                rA[None, m, k], rB[(None, 1), n, k], rC[None, n * 2 + 1, m]
                            )

                cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

            # epilogue
            C_dtype = gC.element_type
            cp_op = nvgpu.CopyUniversalOp()
            cp_atom = cute.make_copy_atom(cp_op, C_dtype, num_bits_per_copy=C_dtype.width * 2)

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
                    rC_bf16 = cute.make_rmem_tensor((2, 2), C_dtype)
                    rC_bf16.store(rC[None, n, m].load().to(C_dtype))
                    cute.copy(cp_atom, rC_bf16, gC_thr[None, None, m, n])

    @cache
    @staticmethod
    def compile(ab_dtype, c_dtype):
        AB_dtype = TORCH_TO_CUTE_DTYPE[ab_dtype]
        C_dtype = TORCH_TO_CUTE_DTYPE[c_dtype]

        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()

        A = cute.runtime.make_fake_tensor(AB_dtype, (M, K), (cute.sym_int64(16), 1), assumed_align=16)
        B = cute.runtime.make_fake_tensor(AB_dtype, (N, K), (cute.sym_int64(16), 1), assumed_align=16)
        C = cute.runtime.make_fake_tensor(C_dtype, (M, N), (cute.sym_int64(8), 1), assumed_align=16)

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120MatmulBF16()
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def mm(A: torch.Tensor, B: torch.Tensor):
    out_dtype = torch.bfloat16 if A.is_floating_point() else torch.int32
    C = A.new_empty(A.shape[0], B.shape[1], dtype=out_dtype)
    Sm120MatmulBF16.compile(A.dtype, out_dtype)(A, B.T, C)
    return C
