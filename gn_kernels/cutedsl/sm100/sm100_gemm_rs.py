from functools import cache

import cutlass
import torch
import torch.distributed as dist
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Int64, Uint16, cute, utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.torch import as_tensor
from cutlass.utils import get_smem_capacity_in_bytes

from ..utils import mbarrier, simple_tma_g2s, to_cta0_smem
from . import _tcgen05


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    nvvm.nanosleep(Int32(ns).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


class Sm100GemmReduceScatterBF16:
    """Persistent BF16 GEMM with dedicated LDMC reduce-scatter warps."""

    def __init__(self, rank: int, num_ranks: int, BN: int = 128, cta_group: int = 2) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        BM, BK = 128, 64
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group
        self.num_tmem_stages = 4

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + (BN // cta_group)) * BK * 2
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_ab(self, tensor: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr) -> cpasync.TmaInfo:
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
        partial_mc: cute.Tensor,
        output: cute.Tensor,
        flags_uc: cute.Tensor,
        flags_mc: cute.Tensor,
        peer_flags,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        BM, BN, BK = self.cta_tile
        A_tma = self.prepare_ab(A, BM, BK)
        B_tma = self.prepare_ab(B, BN // self.cta_group, BK)
        self.kernel(
            A_tma,
            B_tma,
            partial_uc,
            partial_mc,
            output,
            flags_uc,
            flags_mc,
            peer_flags,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(10 * 32, 1, 1),
            cluster=(self.cta_group, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        partial_uc: cute.Tensor,
        partial_mc: cute.Tensor,
        output: cute.Tensor,
        flags_uc: cute.Tensor,
        flags_mc: cute.Tensor,
        peer_flags,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM, BN, BK = self.cta_tile
        cta_group = self.cta_group
        num_stages = self.num_stages
        num_tmem_stages = self.num_tmem_stages
        num_ranks = self.num_ranks

        is_2cta = cta_group == 2
        cta_rank = raw_bid % self.cta_group

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
        cluster_grid_m = grid_m // cta_group

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

            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                cluster_bid = bid // cta_group
                bid_m = cluster_bid % cluster_grid_m * cta_group + cta_rank
                bid_n = cluster_bid // cluster_grid_m
                bid_n_rank = bid_n * self.cta_group + cta_rank

                for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    simple_tma_g2s(A_tma.atom, gA_tiles[None, (bid_m, iter_k)], sA[None, None, tma_stage], mbar)
                    simple_tma_g2s(B_tma.atom, gB_tiles[None, (bid_n_rank, iter_k)], sB[None, None, tma_stage], mbar)

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

                for bid in range(raw_bid, grid_m * grid_n, num_bids):
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
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            # comm warps
            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                cluster_bid = bid // cta_group
                bid_m = cluster_bid % cluster_grid_m * cta_group + cta_rank
                bid_n = cluster_bid // cluster_grid_m

                output_tile_m = bid_m % local_grid_m
                chunk_id = bid_m // local_grid_m
                target_tile_m = output_tile_m + self.rank * local_grid_m

                if tid_ == 0:
                    flag_ptr = flags_uc.iterator + target_tile_m + bid_n * grid_m
                    flag_value = cute.arch.load(flag_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    while flag_value < num_ranks:
                        nanosleep(64)
                        flag_value = cute.arch.load(flag_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
                cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)
                cute.arch.fence_proxy("alias")

                stripe_m = BM // num_ranks
                input_tile = cute.local_tile(partial_mc, (BM, BN), (target_tile_m, bid_n))
                output_tile = cute.local_tile(output, (BM, BN), (output_tile_m, bid_n))
                input_chunks = cute.zipped_divide(input_tile, (stripe_m, BN))
                output_chunks = cute.zipped_divide(output_tile, (stripe_m, BN))
                input_chunk = cute.slice_(input_chunks, ((None, None), (chunk_id, 0)))
                output_chunk = cute.slice_(output_chunks, ((None, None), (chunk_id, 0)))

                atom_elements = 8
                total_comm_threads = 4 * cute.arch.WARP_SIZE
                threads_n = BN // atom_elements
                threads_m = total_comm_threads // threads_n
                thread_layout = cute.make_layout((threads_m, threads_n), stride=(threads_n, 1))
                value_layout = cute.make_layout((1, atom_elements), stride=(atom_elements, 1))
                copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)
                tiled_copy = cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)
                thread_copy = tiled_copy.get_slice(tid_)
                thread_input = thread_copy.partition_S(input_chunk)
                thread_output = thread_copy.partition_S(output_chunk)
                _, loop_m, loop_n = thread_input.shape

                results = []
                for i in cutlass.range_constexpr(loop_m):
                    for j in cutlass.range_constexpr(loop_n):
                        results.append(
                            utils.distributed.multimem_ld_reduce(
                                thread_input[None, i, j].iterator,
                                dtype=BFloat16,
                                num_elements=atom_elements,
                            )
                        )

                for i in cutlass.range_constexpr(loop_m):
                    for j in cutlass.range_constexpr(loop_n):
                        x, y, z, w = results[i * loop_n + j]
                        vec_type = ir.VectorType.get([4], Int32.mlir_type)
                        value = vector.from_elements(vec_type, [x, y, z, w])
                        cute.arch.store(thread_output[None, i, j].iterator.llvm_ptr, value)
                cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                if tid_ == 0:
                    flag_ptr = flags_uc.iterator + target_tile_m + bid_n * grid_m
                    old = cute.arch.atomic_add(ptr=flag_ptr.llvm_ptr, val=Int32(1), sem="relaxed", scope="gpu")
                    if old == num_ranks * 2 - 1:
                        cute.arch.store(flag_ptr.llvm_ptr, Int32(0), sem="relaxed", scope="gpu")

            cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)
            if tid_ == 0:
                final_flag = total_tiles + raw_bid
                utils.distributed.multimem_red_add1(
                    flags_mc.iterator + final_flag,
                    order="release",
                    scope="gpu",
                )
                utils.distributed.spin_lock_atom_cas_acquire_wait(
                    flags_uc.iterator + final_flag,
                    expected_val=num_ranks,
                    reset_val=0,
                    scope="gpu",
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

            for bid in range(raw_bid, grid_m * grid_n, num_bids):
                cluster_bid = bid // cta_group
                bid_m = cluster_bid % cluster_grid_m * cta_group + cta_rank
                bid_n = cluster_bid // cluster_grid_m

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
                if tid_ == 0:
                    owner_rank = bid_m // local_grid_m
                    for rank in cutlass.range_constexpr(num_ranks):
                        if owner_rank == rank:
                            utils.distributed.red_add1(
                                peer_flags[rank].iterator + bid_m + bid_n * grid_m,
                                order="release",
                                scope="gpu",
                            )

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
    def compile(rank: int, num_ranks: int):
        m = cute.sym_int()
        n = cute.sym_int()
        k = cute.sym_int()
        local_m = cute.sym_int()
        num_flags = cute.sym_int()

        def fake_bf16_matrix(
            rows: cutlass.Int32,
            assumed_align: int = 16,
            memspace: cute.AddressSpace = cute.AddressSpace.gmem,
        ) -> cute.Tensor:
            return make_fake_tensor(
                BFloat16,
                (rows, n),
                (cute.sym_int64(divisibility=16), 1),
                memspace=memspace,
                assumed_align=assumed_align,
            )

        a = make_fake_tensor(
            BFloat16,
            (m, k),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        b = make_fake_tensor(
            BFloat16,
            (n, k),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        partial = fake_bf16_matrix(m, assumed_align=32)
        partial_mc = fake_bf16_matrix(m, memspace=cute.AddressSpace.generic)
        output = fake_bf16_matrix(local_m)
        flags = make_fake_tensor(Int32, (num_flags,), (1,), assumed_align=16)
        flags_mc = make_fake_tensor(
            Int32,
            (num_flags,),
            (1,),
            memspace=cute.AddressSpace.generic,
            assumed_align=16,
        )
        peer_flags = tuple(
            make_fake_tensor(
                Int32,
                (num_flags,),
                (1,),
                memspace=cute.AddressSpace.generic,
                assumed_align=16,
            )
            for _ in range(8)
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100GemmReduceScatterBF16(rank, num_ranks)
        kernel.__call__.set_name_prefix(f"gn_gemm_rs_r{rank}_w{num_ranks}")
        return cute.compile(
            kernel,
            a,
            b,
            partial,
            partial_mc,
            output,
            flags,
            flags_mc,
            peer_flags,
            128,
            stream,
            options="--enable-tvm-ffi",
        )


def gemm_reduce_scatter(
    activations: torch.Tensor,
    weight: torch.Tensor,
    partial: torch.Tensor,
    partial_handle,
    output: torch.Tensor,
    flags: torch.Tensor,
    flags_handle,
) -> torch.Tensor:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size > 8:
        raise ValueError("at most 8 ranks are supported")

    grid_size = 148

    m, k = activations.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError("weight must have shape [N, K]")
    if m % world_size or 128 % world_size:
        raise ValueError("M and the 128-row CTA tile must divide world_size")
    if partial.shape != (m, n):
        raise ValueError(f"partial must have shape {(m, n)}")
    if output.shape != (m // world_size, n):
        raise ValueError(f"output must have shape {(m // world_size, n)}")
    expected_flags = (m // 128) * (n // 128) + grid_size
    if flags.shape != (expected_flags,) or flags.dtype != torch.int32:
        raise ValueError(f"flags must be an int32 tensor with shape {(expected_flags,)}")
    tensors = (activations, weight, partial, output, flags)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous CUDA tensors")
    if any(tensor.device != activations.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(tensor.dtype != torch.bfloat16 for tensor in (activations, weight, partial, output)):
        raise TypeError("only contiguous BF16 inputs and outputs are supported")
    if not partial_handle.multicast_ptr:
        raise RuntimeError("partial has no multicast mapping")
    if not flags_handle.multicast_ptr:
        raise RuntimeError("flags have no multicast mapping")

    partial_mc = as_tensor(partial_handle.multicast_ptr, partial.shape, partial.dtype)
    flags_mc = as_tensor(flags_handle.multicast_ptr, flags.shape, flags.dtype)
    peer_flags = tuple(as_tensor(peer_ptr, flags.shape, flags.dtype) for peer_ptr in flags_handle.buffer_ptrs)
    peer_flags += (peer_flags[0],) * (8 - world_size)

    compiled = Sm100GemmReduceScatterBF16.compile(rank, world_size)
    compiled(activations, weight, partial, partial_mc, output, flags, flags_mc, peer_flags, grid_size)
    return output
