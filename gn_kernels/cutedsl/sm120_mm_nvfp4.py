import math
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int16, Int32, Int64, cute
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_tensor

from .utils import mma_sync_nvfp4, permute, simple_tma_g2s, tma_g2s


class Sm120MatmulNVFP4:
    cta_tile = (128, 128, 256)
    warp_layout = (2, 2)
    num_stages = 2

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))

    @cute.jit
    def prepare_SF(self, SF: cute.Tensor, M: Int32, K: Int32):
        # NVIDIA SF layout in gmem
        # if SF has shape [M, Ksf], it's permuted as [M/128, Ksf/4, 32, 4, 4]
        g_layout = cute.make_layout((2048, K // 256, M // 128))
        return cute.make_tensor(SF.iterator, g_layout)

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gSFA: cute.Tensor,
        gSFB: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ):
        M, K = gA.shape
        N, _ = gB.shape
        BM, BN, BK = self.cta_tile

        A_tma = self.prepare_AB(gA, BM, BK)
        B_tma = self.prepare_AB(gB, BN, BK)
        gSFA = self.prepare_SF(gSFA, M, K)
        gSFB = self.prepare_SF(gSFB, N, K)

        grid = (cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1)
        num_warps = math.prod(self.warp_layout) + 1
        block = (num_warps * 32, 1, 1)
        self.kernel(A_tma, B_tma, gSFA, gSFB, gC).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        gSFA: cute.Tensor,
        gSFB: cute.Tensor,
        gC: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, BK = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages

        _, K = A_tma.tma_tensor.shape
        sA_layout = A_tma.smem_layout
        sB_layout = B_tma.smem_layout

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(Float4E2M1FN, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(Float4E2M1FN, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)

        sf_slayout = cute.make_layout(((4, 4, 32, 4), num_stages))
        sSFA = smem.allocate_tensor(Float8E4M3FN, sf_slayout, byte_alignment=128)
        sSFB = smem.allocate_tensor(Float8E4M3FN, sf_slayout, byte_alignment=128)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
        cute.arch.sync_threads()

        if warp_id == 4:
            # TMA warp
            tma_stage = 0
            parity = 1

            # select gmem tile
            gA_tiles = cute.local_tile(A_tma.tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
            gB_tiles = cute.local_tile(B_tma.tma_tensor, (BN, BK), (bid_n, None))
            gSFA_tiles = gSFA[None, None, bid_m]
            gSFB_tiles = gSFB[None, None, bid_n]

            for iter_k in range(K // BK):
                mbar = tma_full_mbar + tma_stage

                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = (BM + BN) * (BK // 2 + BK // 16)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_g2s(A_tma.atom, gA_tiles[None, None, iter_k], sA[None, None, tma_stage], mbar)
                simple_tma_g2s(B_tma.atom, gB_tiles[None, None, iter_k], sB[None, None, tma_stage], mbar)

                # cpasync.CopyBulkG2SOp() generates mapa + cp.async.bulk.shared::cluster.global,
                # which is unnecessary.
                tma_g2s(sSFA[None, tma_stage], gSFA_tiles[None, iter_k], Int32(2048), mbar)
                tma_g2s(sSFB[None, tma_stage], gSFB_tiles[None, iter_k], Int32(2048), mbar)

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
            elems = 32  # 16B
            sA_ldsm = cute.zipped_divide(sA_warp, (16, cute.make_layout((elems, 2)), 1))
            sB_ldsm = cute.zipped_divide(sB_warp, (16, cute.make_layout((elems, 2)), 1))

            # select the address
            # (16B, (WM/16, BK/32B, num_stages))
            sA_ldsm = sA_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
            sB_ldsm = sB_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2), 0), None]

            # ldmatrix.x4
            ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
            ldsm_atom = cute.make_copy_atom(ldsm_op, Float4E2M1FN)

            # select SF smem
            # logically, it looks like [32x4][32x4][32x4][32x4], times 4
            # original shape: UE8M0 ((4, 4, 32, 4), num_stages)
            # new shape: Int32 (4, 4, num_stages)
            # why (lane_id % 4) * 8 + (lane_id // 4)? just stare at PTX doc
            sSFA_view = cute.recast_tensor(sSFA, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4), None), None]
            sSFB_view = cute.recast_tensor(sSFB, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4), None), None]

            # shape: Int32 (2, 4, num_stages)
            sSFA_view = cute.local_tile(sSFA_view, (2, 4, num_stages), (warp_id_m, 0, 0))
            sSFB_view = cute.local_tile(sSFB_view, (2, 4, num_stages), (warp_id_n, 0, 0))

            sf_s2r_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=64)

            # registers
            # let ptxas decides register reuse for rA and rB
            MMA_K = 64  # 32B
            rA = cute.make_rmem_tensor((32, WM // 16, BK // MMA_K), Float4E2M1FN)
            rB = cute.make_rmem_tensor(((16, 2), WN // 16, BK // MMA_K), Float4E2M1FN)
            rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
            rC.fill(0.0)

            rSFA = cute.make_rmem_tensor((2, BK // MMA_K), Int32)
            rSFB = cute.make_rmem_tensor((2, BK // MMA_K), Int32)

            for iter_k in range(K // BK):
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                for k in cutlass.range_constexpr(BK // MMA_K):
                    # TODO: check bank conflicts
                    cute.copy(sf_s2r_atom, sSFA_view[None, k, tma_stage], rSFA[None, k])
                    cute.copy(sf_s2r_atom, sSFB_view[None, k, tma_stage], rSFB[None, k])

                    cute.copy(ldsm_atom, sA_ldsm[None, (None, k, tma_stage)], rA[None, None, k])
                    cute.copy(ldsm_atom, sB_ldsm[None, (None, k, tma_stage)], rB[None, None, k])

                    for m in cutlass.range_constexpr(WM // 16):
                        for n in cutlass.range_constexpr(WN // 8):
                            rC[None, n, m] = mma_sync_nvfp4(
                                rA[None, m, k],
                                rB[(None, n % 2), n // 2, k],
                                rC[None, n, m],
                                rSFA[m // 2, k],
                                Int16(0),
                                Int16(m % 2),
                                rSFB[n // 4, k],
                                Int16(0),
                                Int16(n % 4),
                            )

                cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

            # epilogue
            cp_op = nvgpu.CopyUniversalOp()
            cp_atom = cute.make_copy_atom(cp_op, BFloat16, num_bits_per_copy=32)

            # create view into C gmem
            gC_cta = cute.local_tile(gC, tiler=(BM, BN), coord=(bid_m, bid_n))
            gC_warp = cute.local_tile(gC_cta, tiler=(WM, WN), coord=(warp_id_m, warp_id_n))

            # (((8,2),(2,4)), (WM/16,WN/8))
            gC_view = cute.zipped_divide(gC_warp, (cute.make_layout((8, 2)), cute.make_layout((2, 4))))

            # (2, 2, (WM/16,WN/8))
            gC_view = gC_view[((lane_id // 4, None), (None, lane_id % 4)), None]
            gC_view = permute(gC_view, (1, 0, 2))

            # explicit for loop to interleave cvt with st.global
            for m in cutlass.range_constexpr(WM // 16):
                for n in cutlass.range_constexpr(WN // 8):
                    rC_bf16 = cute.make_rmem_tensor((2, 2), BFloat16)
                    rC_bf16.store(rC[None, n, m].load().to(BFloat16))
                    cute.copy(cp_atom, rC_bf16, gC_view[None, None, (m, n)])

    @cache
    @staticmethod
    def compile():
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int(divisibility=2)

        A = make_fake_tensor(Float4E2M1FN, (M, K), (cute.sym_int64(32), 1), assumed_align=16)
        B = make_fake_tensor(Float4E2M1FN, (N, K), (cute.sym_int64(32), 1), assumed_align=16)
        SFA = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFB = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        C = make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(8), 1), assumed_align=16)

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120MatmulNVFP4()
        return cute.compile(kernel, A, B, SFA, SFB, C, stream, options="--enable-tvm-ffi")


def mm(A: torch.Tensor, B: torch.Tensor, SFA: torch.Tensor, SFB: torch.Tensor):
    C = A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)
    Sm120MatmulNVFP4.compile()(A, B, SFA.view(-1), SFB.view(-1), C)
    return C
