from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Int32, Int64, Uint16, Uint64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.utils import get_smem_capacity_in_bytes

from .utils import _tcgen05, mbarrier, simple_tma_g2s, to_cta0_smem


class Sm100MatmulMXFP8:
    def __init__(self, BN: int, cta_group: int) -> None:
        BM, BK = 128, 128
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + (BN // cta_group)) * BK + (BM + BN) * (BK // 32)
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp(
            cta_group=tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def prepare_SF(self, SF: cute.Tensor, M: Int32, Ksf: Int32, BM: cutlass.Constexpr, BKsf: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp(
            cta_group=tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )

        # NVIDIA SF layout in gmem
        # if SF has shape [M, Ksf], it's permuted as [M/128, Ksf/4, 32, 4, 4]
        # the [32,4,4] atom corresponds to the layout required by tcgen05 exactly.
        # hence, we can treat it as an opaque 512B.
        g_layout = cute.make_layout((Ksf * 128, M // 128))
        SF = cute.make_tensor(SF.iterator, g_layout)

        # due to TMA restriction of boxDim <= 256, we have to cast SF dtype to Int64
        # CuteDSL doesn't warn or tell us how it handles boxDim > 256
        # with Int64, we can support (128 * BKsf) up to 2048
        SF_i64 = cute.recast_tensor(SF, Int64)
        tiler = (128 * BKsf // 8, BM // 128)
        s_layout = cute.make_layout((*tiler, self.num_stages))

        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, SF_i64, s_layout, tiler)
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        SFA: cute.Tensor,
        SFB: cute.Tensor,
        C: cute.Tensor,
        stream: CUstream,
    ):
        M, K = A.shape
        N, _ = B.shape
        BM, BN, BK = self.cta_tile

        A_args = self.prepare_AB(A, BM, BK)
        B_args = self.prepare_AB(B, BN // self.cta_group, BK)
        SFA_args = self.prepare_SF(SFA, M, K // 32, BM, BK // 32)
        SFB_args = self.prepare_SF(SFB, N, K // 32, BN, BK // 32)

        self.kernel(A_args, B_args, SFA_args, SFB_args, C).launch(
            grid=(128, 1, 1),
            block=(6 * 32, 1, 1),
            cluster=(self.cta_group, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        A_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        B_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        SFA_args: tuple[cute.CopyAtom, cute.Tensor, cute.Layout],
        SFB_args: tuple[cute.CopyAtom, cute.Tensor, cute.Layout],
        C_tensor: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM, BN, BK = self.cta_tile
        cta_group = self.cta_group
        num_stages = self.num_stages

        is_2cta = cta_group == 2
        cta_rank = raw_bid % cta_group

        A_tma_atom, A_tma_tensor, sA_layout = A_args
        B_tma_atom, B_tma_tensor, sB_layout = B_args
        SFA_tma_atom, SFA_tma_tensor, sSFA_layout = SFA_args
        SFB_tma_atom, SFB_tma_tensor, sSFB_layout = SFB_args

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(Float8E4M3FN, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(Float8E4M3FN, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)
        sSFA = smem.allocate_tensor(Int64, sSFA_layout, byte_alignment=128)
        sSFB = smem.allocate_tensor(Int64, sSFB_layout, byte_alignment=128)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        if cutlass.const_expr(BN == 256):
            partial_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        M, K = A_tma_tensor.shape
        N, _ = B_tma_tensor.shape
        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
                if cutlass.const_expr(BN == 256):
                    cute.arch.mbarrier_init(partial_mbar, 128 * cta_group)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma_atom)
            cpasync.prefetch_descriptor(B_tma_atom)

        if cutlass.const_expr(is_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        # TMA warp
        if warp_id == 5:
            tma_stage = 0
            parity = 1

            if cutlass.const_expr(is_2cta):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
            else:
                tma_full_mbar_ = tma_full_mbar

            # select gmem tile
            gA_tiles = cute.zipped_divide(A_tma_tensor, (BM, BK))  # [(BM, BK), (M/BM, K/BK)]
            gB_tiles = cute.zipped_divide(B_tma_tensor, (BN // cta_group, BK))
            gSFA_tiles = cute.zipped_divide(SFA_tma_tensor, (128 * (BK // 32) // 8, BM // 128))
            gSFB_tiles = cute.zipped_divide(SFB_tma_tensor, (128 * (BK // 32) // 8, BN // 128))

            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                bid_m = bid // (grid_n * 2) * 2 + bid % 2
                bid_n = (bid // 2) % grid_n
                bid_n_ = bid_n * cta_group + cta_rank

                for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                    mbar = tma_full_mbar_ + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    simple_tma_g2s(A_tma_atom, gA_tiles[None, (bid_m, iter_k)], sA[None, None, tma_stage], mbar)
                    simple_tma_g2s(B_tma_atom, gB_tiles[None, (bid_n_, iter_k)], sB[None, None, tma_stage], mbar)
                    simple_tma_g2s(SFA_tma_atom, gSFA_tiles[None, (iter_k, bid_m)], sSFA[None, None, tma_stage], mbar)
                    simple_tma_g2s(SFB_tma_atom, gSFB_tiles[None, (iter_k, bid_n)], sSFB[None, None, tma_stage], mbar)

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

        # MMA warp
        elif warp_id == 4:
            _tcgen05.alloc(taddr, cta_group)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1
                partial_parity = 0

                sdesc = _tcgen05.make_sdesc_128B()
                sdesc_sf = Uint64(((8 * 16) >> 4 << 32) | (1 << 46))  # SBO=8*16
                multicast_mask = Uint16((1 << cta_group) - 1)

                for bid in range(raw_bid, grid_m * grid_n, num_bids):
                    # we only have 512 tmem columns. when BN=256, d_tmem across
                    # 2 stages need to overlap 16 columns, leaving the remaning
                    # 16 columns for SFA+SFB.
                    #   d_tmem (stage0): [  0, 256]
                    #   d_tmem (stage1): [240, 496]
                    #   sf_tmem        : [496, 512]
                    #
                    # for a [128,128] A or B tile, its corresponding SF tile is [128,4]
                    # NVIDIA layout permutes it as [32,4][32,4][32,4][32,4]
                    # hence, a single tcgen05.cp.32x128b is enough to cover 1 [128,128] tile.
                    d_tmem = tmem_stage * 240
                    sfa_tmem = 496
                    sfb_tmem = sfa_tmem + 4

                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_empty_parity)
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                        a_desc = sdesc | (sA[None, None, tma_stage].iterator.toint() >> 4)
                        b_desc = sdesc | (sB[None, None, tma_stage].iterator.toint() >> 4)
                        sfa_desc = sdesc_sf | (sSFA[None, None, tma_stage].iterator.toint() >> 4)
                        sfb_desc = sdesc_sf | (sSFB[None, None, tma_stage].iterator.toint() >> 4)

                        MMA_M = BM * cta_group
                        MMA_N = BN
                        idesc = _tcgen05.make_idesc_mxfp8(MMA_M, MMA_N)

                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_full_parity)
                        _tcgen05.fence_after_thread_sync()

                        _tcgen05.cp(sfa_tmem, sfa_desc, "32x128b", "warpx4", cta_group)
                        for j in cutlass.range_constexpr(BN // 128):
                            _tcgen05.cp(sfb_tmem + j * 4, sfb_desc, "32x128b", "warpx4", cta_group)
                            sfb_desc += (128 * 4) >> 4

                        for k in cutlass.range_constexpr(BK // 32):
                            enable_input_d = iter_k > 0 or k > 0
                            _tcgen05.mma_mxfp8(
                                d_tmem, a_desc, b_desc, idesc, sfa_tmem, sfb_tmem, enable_input_d, cta_group
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                            idesc += (1 << 4) | (1 << 29)  # increment SF ID
                        _tcgen05.commit(tma_empty_mbar + tma_stage, multicast_mask, cta_group)

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, multicast_mask, cta_group)

                    # wait for partial tmem load to finish
                    if cutlass.const_expr(BN == 256):
                        cute.arch.mbarrier_wait(partial_mbar, partial_parity)
                        partial_parity ^= 1

                    tmem_stage = (tmem_stage + 1) % 2
                    if tmem_stage == 0:
                        tmem_empty_parity ^= 1

        # epilogue warps
        else:
            # (WIDTH, (M, N/WIDTH))
            WIDTH = cutlass.const_expr(16)
            C_ = cute.zipped_divide(C_tensor, (1, WIDTH))[(0, None), None]

            bf16x16_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                BFloat16,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            tmem_stage = 0
            parity = 0

            if cutlass.const_expr(is_2cta):
                tmem_empty_mbar_ = to_cta0_smem(tmem_empty_mbar)
                if cutlass.const_expr(BN == 256):
                    partial_mbar_ = to_cta0_smem(partial_mbar)
            else:
                tmem_empty_mbar_ = tmem_empty_mbar
                if cutlass.const_expr(BN == 256):
                    partial_mbar_ = partial_mbar

            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                bid_m = bid // (grid_n * 2) * 2 + bid % 2
                bid_n = (bid // 2) % grid_n

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                if cutlass.const_expr(BN == 128):
                    for i in cutlass.range_constexpr(BN // WIDTH):
                        tcol = tmem_stage * 240 + i * WIDTH
                        regs = _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                        _tcgen05.wait_ld()

                        if cutlass.const_expr(i == BN // WIDTH - 1):
                            _tcgen05.fence_before_thread_sync()
                            mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, "cluster")

                        tmp = cute.make_rmem_tensor(16, BFloat16)
                        tmp.store(regs.to(BFloat16))

                        # C_ shape: (WIDTH, (M, N/WIDTH))
                        coord = (bid_m * BM + tid, bid_n * (BN // WIDTH) + i)
                        cute.copy(bf16x16_atom, tmp, C_[None, coord])

                elif cutlass.const_expr(BN == 256):
                    # always load [240,256] tmem columns first
                    regs = _tcgen05.ld(warp_id * 32, 240, "32x32b", 16)
                    _tcgen05.wait_ld()
                    _tcgen05.fence_before_thread_sync()
                    mbarrier.arrive(partial_mbar_, "cluster")

                    tmp = cute.make_rmem_tensor(16, BFloat16)
                    tmp.store(regs.to(BFloat16))

                    # C_ shape: (WIDTH, (M, N/WIDTH))
                    coord = (bid_m * BM + tid, bid_n * (BN // WIDTH) + (BN // WIDTH - 1) * (1 - tmem_stage))
                    cute.copy(bf16x16_atom, tmp, C_[None, coord])

                    # load the remaining
                    #   tmem_stage=0: [  0,240]
                    #   tmem_stage=1: [256,496]
                    for i in cutlass.range_constexpr(BN // WIDTH - 1):
                        tcol = tmem_stage * 256 + i * WIDTH
                        regs = _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                        _tcgen05.wait_ld()

                        if cutlass.const_expr(i == BN // WIDTH - 2):
                            _tcgen05.fence_before_thread_sync()
                            mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, "cluster")

                        tmp = cute.make_rmem_tensor(16, BFloat16)
                        tmp.store(regs.to(BFloat16))

                        # C_ shape: (WIDTH, (M, N/WIDTH))
                        coord = (bid_m * BM + tid, bid_n * (BN // WIDTH) + tmem_stage + i)
                        cute.copy(bf16x16_atom, tmp, C_[None, coord])

                tmem_stage = (tmem_stage + 1) % 2
                if tmem_stage == 0:
                    parity ^= 1

            if cutlass.const_expr(is_2cta):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp_id == 0:
                _tcgen05.dealloc(cta_group)

    @cache
    @staticmethod
    def compile(BN: int, cta_group: int):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()

        A = make_fake_tensor(Float8E4M3FN, (M, K), (cute.sym_int64(divisibility=16), 1), assumed_align=16)
        B = make_fake_tensor(Float8E4M3FN, (N, K), (cute.sym_int64(divisibility=16), 1), assumed_align=16)
        SFA = make_fake_tensor(Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFB = make_fake_tensor(Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        C = make_fake_tensor(BFloat16, (M, N), (cute.sym_int(divisibility=16), 1), assumed_align=32)

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100MatmulMXFP8(BN, cta_group)
        return cute.compile(kernel, A, B, SFA, SFB, C, stream, options="--enable-tvm-ffi")


def mm(A: torch.Tensor, B: torch.Tensor, SFA: torch.Tensor, SFB: torch.Tensor):
    C = A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)
    Sm100MatmulMXFP8.compile(256, 2)(A, B, SFA.view(-1), SFB.view(-1), C)
    return C
