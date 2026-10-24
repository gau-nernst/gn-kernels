import math
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float32, Int64, cute
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..utils import mma_sync_fp8, permute, simple_tma_g2s


class Sm120Fp8RowwiseMatmul:
    """C = (A @ B.T) * SA[:, None] * SB[None, :], with FP8 A/B and FP32 row-wise scales"""

    def __init__(
        self,
        cta_tile: tuple[int, int, int] = (128, 128, 128),
        warp_layout: tuple[int, int] = (4, 2),
        num_stages: int = 2,
    ) -> None:
        self.cta_tile = cta_tile
        self.warp_layout = warp_layout
        self.num_stages = num_stages
        # FP8 mma with FP32 accumulate is half-rate on GeForce. use MXFP8 mma with unit scales instead.
        self.use_mxfp8_mma = "GeForce RTX 50" in torch.cuda.get_device_name()

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: int, BK: int):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()

        # compute swizzle param
        width = BK * (A.dtype.width // 8)
        B = int(math.log2(width)) - 4
        assert B <= 3  # 128B
        swizzle = cute.make_swizzle(B, 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gSA: cute.Tensor,
        gB: cute.Tensor,
        gSB: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ):
        BM, BN, BK = self.cta_tile
        A_tma = self.prepare_AB(gA, BM, BK)
        B_tma = self.prepare_AB(gB, BN, BK)

        M, N = gC.shape
        grid = (cute.ceil_div(M, BM) * cute.ceil_div(N, BN), 1, 1)
        num_warps = math.prod(self.warp_layout) + 1
        block = (num_warps * 32, 1, 1)
        self.kernel(A_tma, gSA, B_tma, gSB, gC).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        gSA: cute.Tensor,
        B_tma: cpasync.TmaInfo,
        gSB: cute.Tensor,
        gC: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, BK = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages
        NUM_MMA_THREADS = num_warp_m * num_warp_n * 32

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape
        sA_layout = A_tma.smem_layout
        sB_layout = B_tma.smem_layout

        grid_m = cute.ceil_div(M, BM)
        grid_n = cute.ceil_div(N, BN)
        GROUP_M = 8
        if cutlass.const_expr(GROUP_M == 1):
            bid_n = bid % grid_n
            bid_m = bid // grid_n
        else:
            group_size = GROUP_M * grid_n
            group_id = bid // group_size
            excess = bid % group_size
            group_m = min(grid_m - group_id * GROUP_M, GROUP_M)
            bid_m = group_id * GROUP_M + excess % group_m
            bid_n = excess // group_m

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(Float8E4M3FN, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(Float8E4M3FN, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, NUM_MMA_THREADS)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
        cute.arch.sync_threads()

        if warp_id == num_warp_m * num_warp_n:
            # TMA warp
            tma_stage = 0
            parity = 1

            # select gmem tile
            gA_tiles = cute.local_tile(A_tma.tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
            gB_tiles = cute.local_tile(B_tma.tma_tensor, (BN, BK), (bid_n, None))

            for iter_k in range(K // BK):
                mbar = tma_full_mbar + tma_stage
                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = (BM + BN) * BK
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_g2s(A_tma.atom, gA_tiles[None, None, iter_k], sA[None, None, tma_stage], mbar)
                simple_tma_g2s(B_tma.atom, gB_tiles[None, None, iter_k], sB[None, None, tma_stage], mbar)

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
            # ((16, (16B, 2)), (WM/16, BK/32B, num_stages))
            sA_ldsm = cute.zipped_divide(sA_warp, (16, cute.make_layout((16, 2))))
            sB_ldsm = cute.zipped_divide(sB_warp, (16, cute.make_layout((16, 2))))

            # select the address
            # (16B, (WM/16, BK/32B, num_stages))
            sA_ldsm = sA_ldsm[(lane_id % 16, (None, lane_id // 16)), None]
            sB_ldsm = sB_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2)), None]

            # ldmatrix.x4
            ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
            ldsm_atom = cute.make_copy_atom(ldsm_op, Float8E4M3FN)

            # registers
            # let ptxas decides register reuse for rA and rB
            MMA_K = 32
            rA = cute.make_rmem_tensor((16, WM // 16, BK // MMA_K), Float8E4M3FN)
            rB = cute.make_rmem_tensor(((8, 2), WN // 16, BK // MMA_K), Float8E4M3FN)
            rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
            rC.fill(0.0)

            for iter_k in range(K // BK):
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=NUM_MMA_THREADS)

                for k in cutlass.range_constexpr(BK // MMA_K):
                    cute.copy(ldsm_atom, sA_ldsm[None, (None, k, tma_stage)], rA[None, None, k])
                    cute.copy(ldsm_atom, sB_ldsm[None, (None, k, tma_stage)], rB[None, None, k])

                    for m in cutlass.range_constexpr(WM // 16):
                        for n in cutlass.range_constexpr(WN // 8):
                            rC[None, n, m] = mma_sync_fp8(
                                rA[None, m, k], rB[(None, n % 2), n // 2, k], rC[None, n, m], self.use_mxfp8_mma
                            )

                # proxy-fence is required for TMA to observe ldmatrix
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

            # epilogue
            # row scales. clamp out-of-bounds rows: C is over-allocated, so their outputs are discarded.
            rSA = cute.make_rmem_tensor((2, WM // 16), Float32)
            for m in cutlass.range_constexpr(WM // 16):
                off_m = bid_m * BM + warp_id_m * WM + m * 16 + (lane_id // 4)
                rSA[0, m] = gSA[min(off_m + 0, M - 1)]
                rSA[1, m] = gSA[min(off_m + 8, M - 1)]

            # column scales. N is a multiple of BN, no bounds check.
            rSB = cute.make_rmem_tensor((2, WN // 8), Float32)
            for n in cutlass.range_constexpr(WN // 8):
                off_n = bid_n * BN + warp_id_n * WN + n * 8 + (lane_id % 4) * 2
                rSB[0, n] = gSB[off_n + 0]
                rSB[1, n] = gSB[off_n + 1]

            for m in cutlass.range_constexpr(WM // 16):
                for n in cutlass.range_constexpr(WN // 8):
                    rC[0, n, m] *= rSA[0, m] * rSB[0, n]
                    rC[1, n, m] *= rSA[0, m] * rSB[1, n]
                    rC[2, n, m] *= rSA[1, m] * rSB[0, n]
                    rC[3, n, m] *= rSA[1, m] * rSB[1, n]

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
    def compile(cta_tile: tuple[int, int, int], warp_layout: tuple[int, int], num_stages: int):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()

        A = make_fake_tensor(Float8E4M3FN, (M, K), (cute.sym_int64(16), 1), assumed_align=16)
        B = make_fake_tensor(Float8E4M3FN, (N, K), (cute.sym_int64(16), 1), assumed_align=16)
        SA = make_fake_tensor(Float32, (M,), (1,), assumed_align=4)
        SB = make_fake_tensor(Float32, (N,), (1,), assumed_align=4)
        C = make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(8), 1), assumed_align=16)

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120Fp8RowwiseMatmul(cta_tile, warp_layout, num_stages)
        return cute.compile(kernel, A, SA, B, SB, C, stream, options="--enable-tvm-ffi")


def mm(
    A: torch.Tensor,
    SA: torch.Tensor,
    B: torch.Tensor,
    SB: torch.Tensor,
    *,
    cta_tile: tuple[int, int, int] = (128, 128, 128),
    warp_layout: tuple[int, int] = (4, 2),
    num_stages: int = 2,
):
    # A: [M, K] FP8, SA: [M] or [M, 1] FP32
    # B: [N, K] FP8, SB: [N] or [1, N] FP32
    M, K = A.shape
    N = B.shape[0]
    BM, BN, BK = cta_tile
    assert K % BK == 0 and N % BN == 0

    SA = SA.reshape(M)
    SB = SB.reshape(N)
    assert SA.dtype == SB.dtype == torch.float32
    assert SA.is_contiguous() and SB.is_contiguous()

    # over-allocate to avoid predicate stores
    grid_m = (M + BM - 1) // BM
    C = A.new_empty(grid_m * BM, N, dtype=torch.bfloat16)[:M]
    kernel = Sm120Fp8RowwiseMatmul.compile(tuple(cta_tile), tuple(warp_layout), num_stages)
    kernel(A, SA, B, SB, C)
    return C
