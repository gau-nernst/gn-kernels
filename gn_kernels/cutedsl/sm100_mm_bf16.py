from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Int64, Uint16, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import get_smem_capacity_in_bytes

from .utils import _tcgen05, mbarrier, simple_tma_g2s, to_cta0_smem


class MatmulSm100:
    def __init__(self, BN: int = 128, cta_group: int = 1) -> None:
        BM = 128
        BK = 64
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + (BN // cta_group)) * BK * 2
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
    def __call__(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream: CUstream):
        BM, BN, BK = self.cta_tile
        A_args = self.prepare_AB(A, BM, BK)
        B_args = self.prepare_AB(B, BN // self.cta_group, BK)
        self.kernel(A_args, B_args, C).launch(
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

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(BFloat16, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        taddr = smem.allocate(Int32, 4)

        M, K = A_tma_tensor.shape
        N, _ = B_tma_tensor.shape
        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)

        if warp_id == 0:
            for i in cutlass.range_constexpr(num_stages):
                cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
            cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma_atom)
            cpasync.prefetch_descriptor(B_tma_atom)

        if cutlass.const_expr(is_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        if warp_id == 5:
            # TMA warp
            tma_stage = 0
            parity = 1

            if cutlass.const_expr(is_2cta):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
            else:
                tma_full_mbar_ = tma_full_mbar

            # select gmem tile
            gA_tiles = cute.zipped_divide(A_tma_tensor, (BM, BK))  # [(BM, BK), (M/BM, K/BK)]
            gB_tiles = cute.zipped_divide(B_tma_tensor, (BN // cta_group, BK))

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

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

        elif warp_id == 4:
            # MMA warp
            _tcgen05.alloc(taddr, cta_group)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1

                # BF16 MMA
                MMA_M = BM * cta_group
                MMA_N = BN
                idesc = _tcgen05.make_idesc_bf16(MMA_M, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B()
                multicast_mask = Uint16((1 << cta_group) - 1)

                for bid in range(raw_bid, grid_m * grid_n, num_bids):
                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_empty_parity)
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                        d_tmem = BN * tmem_stage
                        a_desc = sdesc | (sA[None, None, tma_stage].iterator.toint() >> 4)
                        b_desc = sdesc | (sB[None, None, tma_stage].iterator.toint() >> 4)

                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_full_parity)
                        _tcgen05.fence_after_thread_sync()

                        with cute.arch.elect_one():
                            for k in cutlass.range_constexpr(BK // 16):
                                _tcgen05.mma_f16(d_tmem, a_desc, b_desc, idesc, iter_k > 0 or k > 0, cta_group)
                                a_desc += 32 >> 4
                                b_desc += 32 >> 4
                            _tcgen05.commit(tma_empty_mbar + tma_stage, multicast_mask, cta_group)

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    with cute.arch.elect_one():
                        _tcgen05.commit(tmem_full_mbar + tmem_stage, multicast_mask, cta_group)

                    tmem_stage = (tmem_stage + 1) % 2
                    if tmem_stage == 0:
                        tmem_empty_parity ^= 1

        else:
            # epilogue warps
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
            else:
                tmem_empty_mbar_ = tmem_empty_mbar

            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                bid_m = bid // (grid_n * 2) * 2 + bid % 2
                bid_n = (bid // 2) % grid_n

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(BN // WIDTH):
                    tcol = tmem_stage * BN + i * WIDTH
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
    def compile(BN: int = 128, cta_group: int = 2):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        A = cute.runtime.make_fake_tensor(BFloat16, (M, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        B = cute.runtime.make_fake_tensor(BFloat16, (N, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        C = cute.runtime.make_fake_tensor(BFloat16, (M, N), (cute.sym_int(divisibility=16), 1), assumed_align=32)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = MatmulSm100(BN, cta_group)
        return cute.compile(kernel, A, B, C, stream, options="--enable-tvm-ffi")


def mm(A: torch.Tensor, B: torch.Tensor):
    C = A.new_empty(A.shape[0], B.shape[1])
    MatmulSm100.compile(256, 2)(A, B.T, C)
    return C
