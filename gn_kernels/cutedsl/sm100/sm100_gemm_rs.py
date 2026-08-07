# refer to https://github.com/NVIDIA/cutlass/issues/3117 for a more detailed
# explanation on memory semantics used here.

from functools import cache

import cutlass
import torch
import torch.distributed as dist
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Int64, Uint16, cute, utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor, make_ptr, nullptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import get_smem_capacity_in_bytes

from ..utils import mbarrier, permute, simple_tma_g2s, to_cta0_smem
from . import _tcgen05


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    nvvm.nanosleep(Int32(ns).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def multimem_ld_reduce_16B(
    x: cute.Tensor,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    # NOTE: assume x is contiguous
    if x.element_type == BFloat16:
        vec_type = ".v4.bf16x2"
    else:
        raise ValueError

    ptr = x.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    struct = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [ptr],
        f"multimem.ld_reduce.relaxed.gpu.global.add.acc::f32{vec_type} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], Int32.mlir_type, loc=loc),
        [llvm.extractvalue(Int32.mlir_type, struct, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    ssa = cute.TensorSSA(vec, 4, Int32)

    y = cute.make_rmem_tensor(4, Int32)
    y.store(ssa)
    return cute.recast_tensor(y, x.element_type)


class Sm100GemmRsBF16:
    def __init__(self, rank: int, num_ranks: int, BN: int = 128, cta_group: int = 1) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        BM, BK = 128, 64
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + (BN // cta_group)) * BK * 2
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr) -> cpasync.TmaInfo:
        tma_op = cpasync.CopyBulkTensorTileG2SOp(
            cta_group=tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        swizzle_128b = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        layout = cute.make_composed_layout(swizzle_128b, 0, layout)
        return cpasync.make_tiled_tma_atom(tma_op, tensor, layout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        partial_uc: cute.Tensor,
        partial_mc_ptr: cute.Pointer,
        output: cute.Tensor,
        flags_uc: cute.Tensor,
        flags_mc_ptr: cute.Pointer,
        peer_flags,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        M = A.shape[0]
        N = B.shape[0]
        BM, BN, BK = self.cta_tile
        A_tma = self.prepare_tma(A, BM, BK)
        B_tma = self.prepare_tma(B, BN // self.cta_group, BK)
        partial_mc = cute.make_tensor(partial_mc_ptr, cute.make_layout((M, N), stride=(N, 1)))

        grid = (grid_size, 1, 1)
        block = (10 * 32, 1, 1)
        cluster = (self.cta_group, 1, 1)
        self.kernel(
            A_tma,
            B_tma,
            partial_uc,
            partial_mc,
            output,
            flags_uc.iterator,
            flags_mc_ptr,
            peer_flags,
        ).launch(grid=grid, block=block, cluster=cluster, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        partial_uc: cute.Tensor,
        partial_mc: cute.Tensor,
        output: cute.Tensor,
        flags_uc_ptr: cute.Pointer,
        flags_mc_ptr: cute.Pointer,
        peer_flags,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM, BN, BK = self.cta_tile
        cta_group = self.cta_group
        num_stages = self.num_stages
        num_ranks = self.num_ranks

        is_2cta = cta_group == 2
        cta_rank = raw_bid % self.cta_group
        num_tmem_stages = 512 // BN

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            BFloat16,
            A_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=A_tma.smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            BFloat16,
            B_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=B_tma.smem_layout.inner,
        )
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, num_tmem_stages)
        tmem_empty_mbar = smem.allocate_array(Int64, num_tmem_stages)
        taddr = smem.allocate(Int32, 4)

        # named barriers
        BAR_TMEM_ALLOC = 1
        BAR_EPILOGUE = 2
        BAR_COMM = 3

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape
        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(num_tmem_stages):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)

        if cutlass.const_expr(is_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        local_grid_m = grid_m // num_ranks
        total_tiles = grid_m * grid_n

        if warp_id == 9:
            # TMA warp
            tma_stage = 0
            parity = 1

            if cutlass.const_expr(is_2cta):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
            else:
                tma_full_mbar_ = tma_full_mbar

            # select gmem tile
            # [(BM, BK), (M/BM, K/BK)]
            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // cta_group, BK))

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m
                if cutlass.const_expr(cta_group == 2):
                    bid_n = bid_n * cta_group + cta_rank

                for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    simple_tma_g2s(A_tma.atom, gA_tiles[None, (bid_m, iter_k)], sA[None, None, tma_stage], mbar)
                    simple_tma_g2s(B_tma.atom, gB_tiles[None, (bid_n, iter_k)], sB[None, None, tma_stage], mbar)

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

        elif warp_id == 8:
            # MMA warp
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1

                MMA_M = BM * cta_group
                MMA_N = BN
                idesc = _tcgen05.make_idesc_bf16(MMA_M, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B()
                multicast_mask = Uint16((1 << self.cta_group) - 1)

                for bid in range(raw_bid, total_tiles, num_bids):
                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_empty_parity)
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                        d_tmem = BN * tmem_stage
                        a_desc = sdesc | (sA[None, None, tma_stage].iterator.toint() >> 4)
                        b_desc = sdesc | (sB[None, None, tma_stage].iterator.toint() >> 4)

                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_full_parity)
                        _tcgen05.fence_after_thread_sync()

                        for mma_k in cutlass.range_constexpr(BK // 16):
                            _tcgen05.mma_f16(d_tmem, a_desc, b_desc, idesc, iter_k > 0 or mma_k > 0, cta_group)
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                        _tcgen05.commit(tma_empty_mbar + tma_stage, multicast_mask, cta_group)

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, multicast_mask, cta_group)

                    tmem_stage = (tmem_stage + 1) % num_tmem_stages
                    if tmem_stage == 0:
                        tmem_empty_parity ^= 1

        elif warp_id >= 4:
            # comm warps
            # NOTE: it's important that warp0-3 are epilogue, and 4-7 are comms
            # if we swap the warpgroup order, the kernel will hang. not really sure
            # why, possibly due to warp scheduling.
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            # each thread issues multimem.ld_reduce for BF16x8
            cols = BN // 8
            rows = 128 // cols

            # ((8,cols),(rows,local_BM/rows), (N/BN,M/local_BM))
            local_BM = BM // num_ranks
            tiler = (cute.make_layout((8, cols)), cute.make_layout((rows, local_BM // rows)))
            input_view = cute.zipped_divide(permute(partial_mc, (1, 0)), tiler)
            output_view = cute.zipped_divide(permute(output, (1, 0)), tiler)

            # (8,local_BM/rows, (N/BN,M/local_BM))
            idx = (((None, tid_ % cols), (tid_ // cols, None)), None)
            input_view = input_view[idx]
            output_view = output_view[idx]

            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m_rs = bid % grid_m
                bid_n = bid // grid_m

                # map to GEMM tile coordinate
                bid_m_gemm = self.rank * local_grid_m + bid_m_rs // num_ranks

                # (8,local_BM/rows)
                input_tile = input_view[None, None, (bid_n, self.rank * grid_m + bid_m_rs)]
                output_tile = output_view[None, None, (bid_n, bid_m_rs)]

                # relaxed poll with gpu scope since we are polling local L2.
                # acquire fence is not needed because multimem.ld_reduce will
                # read each rank's local L2 (don't need to invalidate L1).
                if tid_ == 0:
                    flag_ptr = flags_uc_ptr + bid_n * grid_m + bid_m_gemm
                    flag_value = cute.arch.load(flag_ptr, Int32, sem="relaxed", scope="gpu")
                    while flag_value < num_ranks:
                        nanosleep(64)
                        flag_value = cute.arch.load(flag_ptr, Int32, sem="relaxed", scope="gpu")
                cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                results = []
                for i in cutlass.range_constexpr(local_BM // rows):
                    results.append(multimem_ld_reduce_16B(input_tile[None, i]))

                for i in cutlass.range_constexpr(local_BM // rows):
                    cute.copy(st_atom, results[i], output_tile[None, i])
                cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                # finish using the GEMM tile
                # last consumer reset the flag
                if tid_ == 0:
                    flag_ptr = flags_uc_ptr + bid_n * grid_m + bid_m_gemm
                    old = cute.arch.atomic_add(flag_ptr, Int32(1), sem="relaxed", scope="gpu")
                    if old == num_ranks * 2 - 1:
                        cute.arch.store(flag_ptr, Int32(0), sem="relaxed", scope="gpu")

            # exit barrier
            # gpu scope is enough since we write to local gmem -> only need to flush to local L2.
            cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)
            if tid_ == 0:
                final_flag = total_tiles + raw_bid
                utils.distributed.multimem_red_add1(flags_mc_ptr + final_flag, order="release", scope="gpu")
                utils.distributed.spin_lock_atom_cas_acquire_wait(
                    flags_uc_ptr + final_flag, expected_val=num_ranks, reset_val=0, scope="gpu"
                )

        else:
            # epilogue warps
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            if warp_id_ == 0:
                _tcgen05.alloc(taddr, cta_group)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            WIDTH = cutlass.const_expr(16)
            partial_vecs = cute.zipped_divide(partial_uc, (1, WIDTH))[(0, None), None]

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

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m

                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(BN // WIDTH):
                    tcol = tmem_stage * BN + i * WIDTH
                    regs = _tcgen05.ld(warp_id_ * 32, tcol, "32x32b", WIDTH)
                    _tcgen05.wait_ld()

                    if cutlass.const_expr(i == BN // WIDTH - 1):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, "cluster")

                    tmp = cute.make_rmem_tensor(WIDTH, BFloat16)
                    tmp.store(regs.to(BFloat16))

                    coord = (bid_m * BM + tid_, bid_n * (BN // WIDTH) + i)
                    cute.copy(bf16x16_atom, tmp, partial_vecs[None, coord])

                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)

                # signal done
                # gpu scope since we only need to flush to local L2.
                if tid_ == 0:
                    owner_rank = bid_m // local_grid_m
                    for rank in cutlass.range_constexpr(num_ranks):
                        if owner_rank == rank:
                            ptr = peer_flags[rank] + bid_m + bid_n * grid_m
                            utils.distributed.red_add1(ptr, order="release", scope="gpu")

                tmem_stage = (tmem_stage + 1) % num_tmem_stages
                if tmem_stage == 0:
                    parity ^= 1

            if cutlass.const_expr(is_2cta):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
            if warp_id_ == 0:
                _tcgen05.dealloc(cta_group)

    @cache
    @staticmethod
    def compile(rank: int, num_ranks: int, BN: int, cta_group: int):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        local_M = cute.sym_int()
        num_flags = cute.sym_int()

        A = make_fake_tensor(BFloat16, (M, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        B = make_fake_tensor(BFloat16, (N, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16)
        partial = make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(divisibility=16), 1), assumed_align=32)
        partial_mc_ptr = nullptr(BFloat16, cute.AddressSpace.gmem, assumed_align=32)
        output = make_fake_tensor(BFloat16, (local_M, N), (cute.sym_int64(divisibility=16), 1), assumed_align=32)
        flags = make_fake_tensor(Int32, (num_flags,), (1,), assumed_align=16)
        flags_mc_ptr = nullptr(Int32, cute.AddressSpace.gmem, assumed_align=16)
        peer_flags = tuple(nullptr(Int32, cute.AddressSpace.gmem, assumed_align=16) for _ in range(8))

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100GemmRsBF16(rank, num_ranks, BN, cta_group)
        return cute.compile(
            kernel,
            A,
            B,
            partial,
            partial_mc_ptr,
            output,
            flags,
            flags_mc_ptr,
            peer_flags,
            128,
            stream,
            options="--enable-tvm-ffi",
        )


def gemm_rs(
    x: torch.Tensor,
    w: torch.Tensor,
    partial: torch.Tensor,
    partial_handle,
    output: torch.Tensor,
    flags: torch.Tensor,
    flags_handle,
    *,
    BN: int | None = None,
) -> torch.Tensor:
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size <= 8

    M, K = x.shape
    N, _ = w.shape

    # heuristic
    cta_group = 2
    if BN is None:
        BN = 256 if M * K >= 24 * 1024 * 1024 else 128

    # NOTE: cta_group=2 is not handling correctly here
    num_tiles = (M // 128) * (N // BN)
    num_sms = torch.cuda.get_device_properties().multi_processor_count
    num_ctas = min(num_tiles, num_sms)

    partial_mc_ptr = make_ptr(BFloat16, partial_handle.multicast_ptr, cute.AddressSpace.gmem, assumed_align=32)
    flags_mc_ptr = make_ptr(Int32, flags_handle.multicast_ptr, cute.AddressSpace.gmem, assumed_align=16)
    peer_flags = tuple(
        make_ptr(Int32, peer_ptr, cute.AddressSpace.gmem, assumed_align=16) for peer_ptr in flags_handle.buffer_ptrs
    )
    peer_flags += (peer_flags[0],) * (8 - world_size)

    compiled = Sm100GemmRsBF16.compile(rank, world_size, BN, cta_group)
    compiled(x, w, partial, partial_mc_ptr, output, flags, flags_mc_ptr, peer_flags, num_ctas)
    return output
