import math
from functools import cache
from typing import Literal

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int16, Int32, Int64, Uint8, cute
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_tensor

from ..utils import (
    CUTE_TO_PTX_DTYPE,
    PTX_TO_CUTE_DTYPE,
    TORCH_TO_CUTE_DTYPE,
    LowBitType,
    mma_sync_mxf8f6f4,
    permute,
    simple_tma_g2s,
    tma_g2s,
)


class Sm120MatmulMXF8F6F4:
    cta_tile = (128, 128, 128)
    warp_layout = (2, 2)
    num_stages = 2

    def __init__(self, a_type: LowBitType, b_type: LowBitType):
        self.a_type = a_type
        self.b_type = b_type

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: int, BK: int):
        """NOTE: BK is in unit of bytes, not elements"""
        # <3,4,3> for 128B, <2,4,3> for 64B
        swizzle = cute.make_swizzle(int(math.log2(BK // 16)), 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle, 0, s_layout)

        # it's not obvious but when we use FP4/FP6 with internal_type=U8, CuteDSL
        # will use CU_TENSOR_MAP_DATA_TYPE_16U4/U6_ALIGN16B for TMA
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        return cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK), internal_type=Uint8)

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
        M, _ = gA.shape
        N, _ = gB.shape
        BM, BN, BK = self.cta_tile

        A_tma = self.prepare_AB(cute.recast_tensor(gA, PTX_TO_CUTE_DTYPE[self.a_type]), BM, BK)
        B_tma = self.prepare_AB(cute.recast_tensor(gB, PTX_TO_CUTE_DTYPE[self.b_type]), BN, BK)

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

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape
        sA_layout = A_tma.smem_layout
        sB_layout = B_tma.smem_layout

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(Uint8, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(Uint8, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)

        sf_slayout = cute.make_layout(((4, 4, 32), num_stages))
        sSFA = smem.allocate_tensor(Float8E8M0FNU, sf_slayout, byte_alignment=128)
        sSFB = smem.allocate_tensor(Float8E8M0FNU, sf_slayout, byte_alignment=128)

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

            SF_size = Int32(32 * 4 * 4)
            gSFA_ = cute.make_tensor(gSFA.iterator, cute.make_layout((SF_size, K // BK, M // BM)))
            gSFB_ = cute.make_tensor(gSFB.iterator, cute.make_layout((SF_size, K // BK, N // BN)))
            gSFA_tiles = gSFA_[None, None, bid_m]
            gSFB_tiles = gSFB_[None, None, bid_n]

            for iter_k in range(K // BK):
                mbar = tma_full_mbar + tma_stage

                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                with cute.arch.elect_one():
                    BK_A = BK // 8 * PTX_TO_CUTE_DTYPE[self.a_type].width
                    BK_B = BK // 8 * PTX_TO_CUTE_DTYPE[self.b_type].width
                    STAGE_SIZE = BM * BK_A + BN * BK_B + (BM + BN) * (BK // 32)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_g2s(A_tma.atom, gA_tiles[None, None, iter_k], sA[None, None, tma_stage], mbar)
                simple_tma_g2s(B_tma.atom, gB_tiles[None, None, iter_k], sB[None, None, tma_stage], mbar)

                # cpasync.CopyBulkG2SOp() generates mapa + cp.async.bulk.shared::cluster.global,
                # which is unnecessary.
                tma_g2s(sSFA[None, tma_stage], gSFA_tiles[None, iter_k], SF_size, mbar)
                tma_g2s(sSFB[None, tma_stage], gSFB_tiles[None, iter_k], SF_size, mbar)

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
            # custom stride for fp6 and fp4
            elems = 16  # 16B
            sA_ldsm = cute.zipped_divide(sA_warp, (16, cute.make_layout((elems, 2))))
            sB_ldsm = cute.zipped_divide(sB_warp, (16, cute.make_layout((elems, 2))))

            # select the address
            # (16B, (WM/16, BK/32B, num_stages))
            sA_ldsm = sA_ldsm[(lane_id % 16, (None, lane_id // 16)), None]
            sB_ldsm = sB_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2)), None]

            # ldmatrix
            def get_s2r_atom(dtype: LowBitType):
                if cutlass.const_expr(dtype in ("e5m2", "e4m3")):
                    op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                else:
                    op = warp.LdMatrix8x16x8bOp(num_matrices=4, unpack_bits=PTX_TO_CUTE_DTYPE[dtype].width)
                return cute.make_copy_atom(op, Uint8)

            A_s2r_atom = get_s2r_atom(self.a_type)
            B_s2r_atom = get_s2r_atom(self.b_type)

            # select SF smem
            # logically, it looks like [32x4][32x4][32x4][32x4]
            # original shape: UE8M0 ((4, 4, 32), num_stages)
            # new shape: Int32 (4, num_stages)
            # why (lane_id % 4) * 8 + (lane_id // 4)? just stare at PTX doc
            sSFA_view = cute.recast_tensor(sSFA, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4)), None]
            sSFB_view = cute.recast_tensor(sSFB, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4)), None]

            # shape: Int32 (2, num_stages)
            sSFA_view = cute.local_tile(sSFA_view, (2, num_stages), (warp_id_m, 0))
            sSFB_view = cute.local_tile(sSFB_view, (2, num_stages), (warp_id_n, 0))

            sf_s2r_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=64)

            # registers
            # let ptxas decides register reuse for rA and rB
            MMA_K = 32  # 32B
            rA = cute.make_rmem_tensor((16, WM // 16, BK // MMA_K), Uint8)
            rB = cute.make_rmem_tensor(((8, 2), WN // 16, BK // MMA_K), Uint8)
            rC = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
            rC.fill(0.0)

            rSFA = cute.make_rmem_tensor(2, Int32)
            rSFB = cute.make_rmem_tensor(2, Int32)

            for iter_k in range(K // BK):
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                # TODO: check bank conflicts
                cute.copy(sf_s2r_atom, sSFA_view[None, tma_stage], rSFA)
                cute.copy(sf_s2r_atom, sSFB_view[None, tma_stage], rSFB)

                for k in cutlass.range_constexpr(BK // MMA_K):
                    cute.copy(A_s2r_atom, sA_ldsm[None, (None, k, tma_stage)], rA[None, None, k])
                    cute.copy(B_s2r_atom, sB_ldsm[None, (None, k, tma_stage)], rB[None, None, k])

                    # ldmatrix loads fp4 as 0000xxxx, but mma wants 00xxxx00
                    if cutlass.const_expr(self.a_type == "e2m1"):
                        rA_view = cute.recast_tensor(rA[None, None, k], Int32)
                        for i in cutlass.range_constexpr(cute.size(rA_view)):
                            rA_view[i] <<= 2

                    if cutlass.const_expr(self.b_type == "e2m1"):
                        rB_view = cute.recast_tensor(rB[None, None, k], Int32)
                        for i in cutlass.range_constexpr(cute.size(rB_view)):
                            rB_view[i] <<= 2

                    for m in cutlass.range_constexpr(WM // 16):
                        for n in cutlass.range_constexpr(WN // 8):
                            rC[None, n, m] = mma_sync_mxf8f6f4(
                                rA[None, m, k],
                                rB[(None, n % 2), n // 2, k],
                                rC[None, n, m],
                                rSFA[m // 2],
                                Int16(k),
                                Int16(m % 2),
                                rSFB[n // 4],
                                Int16(k),
                                Int16(n % 4),
                                a_type=self.a_type,
                                b_type=self.b_type,
                            )

                cute.arch.fence_proxy("async.shared", space="cta")
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
    def compile(a_type: LowBitType, b_type: LowBitType):
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int(divisibility=2)

        K_A = K // 8 * PTX_TO_CUTE_DTYPE[a_type].width
        K_B = K // 8 * PTX_TO_CUTE_DTYPE[b_type].width

        A = make_fake_tensor(Uint8, (M, K_A), (cute.sym_int64(16), 1), assumed_align=16)
        B = make_fake_tensor(Uint8, (N, K_B), (cute.sym_int64(16), 1), assumed_align=16)
        SFA = make_fake_tensor(Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFB = make_fake_tensor(Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        C = make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(8), 1), assumed_align=16)

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120MatmulMXF8F6F4(a_type, b_type)
        return cute.compile(kernel, A, B, SFA, SFB, C, stream, options="--enable-tvm-ffi")


def mm(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    *,
    a_type: LowBitType | None = None,
    b_type: LowBitType | None = None,
):
    # autodetect
    ALLOWED_TORCH_TYPES = (torch.float8_e5m2, torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
    if a_type is None:
        assert A.dtype in ALLOWED_TORCH_TYPES
        a_type = CUTE_TO_PTX_DTYPE[TORCH_TO_CUTE_DTYPE[A.dtype]]
    if b_type is None:
        assert B.dtype in ALLOWED_TORCH_TYPES
        b_type = CUTE_TO_PTX_DTYPE[TORCH_TO_CUTE_DTYPE[B.dtype]]

    C = A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)
    kernel = Sm120MatmulMXF8F6F4.compile(a_type, b_type)
    kernel(A.view(torch.uint8), B.view(torch.uint8), SFA.view(-1), SFB.view(-1), C)
    return C
