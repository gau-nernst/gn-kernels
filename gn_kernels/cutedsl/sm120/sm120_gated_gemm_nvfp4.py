import math
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int16, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from ..utils import mma_sync_nvfp4, permute, simple_tma_g2s, tma_g2s


@dsl_user_op
def _f32x8_to_f4x8(x: cute.Tensor, *, loc=None, ip=None) -> Uint32:
    # NOTE: lower elem is stored as higher nibble
    out = llvm.inline_asm(
        T.i32(),
        [x[i].ir_value(loc=loc, ip=ip) for i in range(8)],
        "{\n"
        ".reg .b8 x0, x1, x2, x3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 x0, $1, $2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 x1, $3, $4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 x2, $5, $6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 x3, $7, $8;\n"
        "mov.b32 $0, {x0, x1, x2, x3};\n"
        "}",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


class Sm120GatedGemmNVFP4:
    # the current code supports the following
    # - BM: 128 (fixed)
    # - BN: 64 or 128
    # - BK: 256 or 128 (128B or 64B)
    # - WN: >=32

    cta_tile = (128, 128, 128)
    warp_layout = (2, 4)
    num_stages = 2

    @cute.jit
    def prepare_tma(
        self, gX: cute.Tensor, BM: cutlass.Constexpr[int], BK: cutlass.Constexpr[int], tma_op: cpasync.TmaCopyOp
    ):
        assert BK <= 256
        # <3,4,3> for 128B, <2,4,3> for 64B
        swizzle = cute.make_swizzle(int(math.log2(BK // 32)), 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, gX, s_layout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gSFX: cute.Tensor,
        gSFX_tensor: cute.Tensor,
        gW1: cute.Tensor,
        gSFW1: cute.Tensor,
        gSFW1_tensor: cute.Tensor,
        gW3: cute.Tensor,
        gSFW3: cute.Tensor,
        gSFW3_tensor: cute.Tensor,
        gC: cute.Tensor,
        gSFC: cute.Tensor,
        gSFC_tensor: cute.Tensor,
        stream: CUstream,
    ):
        M, _ = gX.shape
        N, _ = gW1.shape
        BM, BN, BK = self.cta_tile

        tma_op = cpasync.CopyBulkTensorTileG2SOp()

        grid = (cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1)
        num_warps = math.prod(self.warp_layout) + 1
        block = (num_warps * 32, 1, 1)
        self.kernel(
            self.prepare_tma(gX, BM, BK, tma_op),
            gSFX,
            gSFX_tensor,
            self.prepare_tma(gW1, BN, BK, tma_op),
            gSFW1,
            gSFW1_tensor,
            self.prepare_tma(gW3, BN, BK, tma_op),
            gSFW3,
            gSFW3_tensor,
            gC,
            gSFC,
            gSFC_tensor,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        X_tma: cpasync.TmaInfo,
        gSFX: cute.Tensor,
        gSFX_tensor: cute.Tensor,
        W1_tma: cpasync.TmaInfo,
        gSFW1: cute.Tensor,
        gSFW1_tensor: cute.Tensor,
        W3_tma: cpasync.TmaInfo,
        gSFW3: cute.Tensor,
        gSFW3_tensor: cute.Tensor,
        gC: cute.Tensor,
        gSFC,
        gSFC_tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, BK = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        num_stages = self.num_stages
        MMA_K = 64  # 32B
        NUM_MMA_THREADS = math.prod(self.warp_layout) * 32

        M, K = X_tma.tma_tensor.shape
        N, K = W1_tma.tma_tensor.shape

        # allocate smem
        def allocate_tensor(smem, layout):
            return smem.allocate_tensor(Float4E2M1FN, layout.outer, byte_alignment=128, swizzle=layout.inner)

        smem = cutlass.utils.SmemAllocator()
        sX = allocate_tensor(smem, X_tma.smem_layout)
        sW1 = allocate_tensor(smem, W1_tma.smem_layout)
        sW3 = allocate_tensor(smem, W3_tma.smem_layout)

        sf_slayout = cute.make_layout(((4, 4, 32, BK // MMA_K), num_stages))
        sSFX = smem.allocate_tensor(Float8E4M3FN, sf_slayout, byte_alignment=128)
        sSFW1 = smem.allocate_tensor(Float8E4M3FN, sf_slayout, byte_alignment=128)
        sSFW3 = smem.allocate_tensor(Float8E4M3FN, sf_slayout, byte_alignment=128)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, NUM_MMA_THREADS)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(X_tma.atom)
            cpasync.prefetch_descriptor(W1_tma.atom)
            cpasync.prefetch_descriptor(W3_tma.atom)
        cute.arch.sync_threads()

        if warp_id == math.prod(self.warp_layout):
            # TMA warp
            tma_stage = 0
            parity = 1

            # select gmem tile
            gX_tiles = cute.local_tile(X_tma.tma_tensor, (BM, BK), (bid_m, None))  # [BM, BK, K/BK]
            gW1_tiles = cute.local_tile(W1_tma.tma_tensor, (BN, BK), (bid_n, None))  # [BN, BK, K/BK]
            gW3_tiles = cute.local_tile(W3_tma.tma_tensor, (BN, BK), (bid_n, None))  # [BN, BK, K/BK]

            SF_SIZE = Int32(32 * 4 * 4 * (BK // MMA_K))
            gSFX_ = cute.make_tensor(gSFX.iterator, cute.make_layout((SF_SIZE, K // BK, M // 128)))
            gSFW1_ = cute.make_tensor(gSFW1.iterator, cute.make_layout((SF_SIZE, K // BK, N // 128)))
            gSFW3_ = cute.make_tensor(gSFW3.iterator, cute.make_layout((SF_SIZE, K // BK, N // 128)))
            gSFX_tiles = gSFX_[None, None, bid_m]
            gSFW1_tiles = gSFW1_[None, None, bid_n * BN // 128]
            gSFW3_tiles = gSFW3_[None, None, bid_n * BN // 128]

            for iter_k in range(K // BK):
                mbar = tma_full_mbar + tma_stage

                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = (BM + BN * 2) * (BK // 2) + SF_SIZE * 3
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_g2s(X_tma.atom, gX_tiles[None, None, iter_k], sX[None, None, tma_stage], mbar)
                simple_tma_g2s(W1_tma.atom, gW1_tiles[None, None, iter_k], sW1[None, None, tma_stage], mbar)
                simple_tma_g2s(W3_tma.atom, gW3_tiles[None, None, iter_k], sW3[None, None, tma_stage], mbar)

                # cpasync.CopyBulkG2SOp() generates mapa + cp.async.bulk.shared::cluster.global,
                # which is unnecessary.
                tma_g2s(sSFX[None, tma_stage], gSFX_tiles[None, iter_k], SF_SIZE, mbar)
                tma_g2s(sSFW1[None, tma_stage], gSFW1_tiles[None, iter_k], SF_SIZE, mbar)
                tma_g2s(sSFW3[None, tma_stage], gSFW3_tiles[None, iter_k], SF_SIZE, mbar)

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
            assert WN >= 32

            # warp partition
            # shape: (WM, BK, num_stages)
            sX_warp = cute.local_tile(sX, (WM, BK, num_stages), (warp_id_m, 0, 0))
            sW1_warp = cute.local_tile(sW1, (WN, BK, num_stages), (warp_id_n, 0, 0))
            sW3_warp = cute.local_tile(sW3, (WN, BK, num_stages), (warp_id_n, 0, 0))

            # pre-compute ldmatrix address (16x16 tile)
            # ((16, (16B, 2), 1), (WM/16, BK/32B, num_stages))
            elems = 32  # 16B
            sX_ldsm = cute.zipped_divide(sX_warp, (16, cute.make_layout((elems, 2))))
            sW1_ldsm = cute.zipped_divide(sW1_warp, (16, cute.make_layout((elems, 2))))
            sW3_ldsm = cute.zipped_divide(sW3_warp, (16, cute.make_layout((elems, 2))))

            # select the address
            # (16B, (WM/16, BK/32B, num_stages))
            sX_ldsm = sX_ldsm[(lane_id % 16, (None, lane_id // 16)), None]
            sW1_ldsm = sW1_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2)), None]
            sW3_ldsm = sW3_ldsm[((lane_id // 16) * 8 + (lane_id % 8), (None, (lane_id // 8) % 2)), None]

            # ldmatrix.x4
            ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
            ldsm_atom = cute.make_copy_atom(ldsm_op, Float4E2M1FN)

            # select SF smem
            # logically, it looks like [32x4][32x4][32x4][32x4], times 4
            # original shape: UE4M3 ((4, 4, 32, 4), num_stages)
            # new shape: Int32 (4, 4, num_stages)
            # why (lane_id % 4) * 8 + (lane_id // 4)? just stare at PTX doc
            sSFX_view = cute.recast_tensor(sSFX, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4), None), None]
            sSFW1_view = cute.recast_tensor(sSFW1, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4), None), None]
            sSFW3_view = cute.recast_tensor(sSFW3, Int32)[(0, None, (lane_id % 4) * 8 + (lane_id // 4), None), None]

            # shape: Int32 (2, 4, num_stages)
            sSFX_view = cute.local_tile(sSFX_view, (2, 4, num_stages), (warp_id_m, 0, 0))

            # select the correct half
            if cutlass.const_expr(BN == 64):
                sSFW1_view = cute.local_tile(sSFW1_view, (2, 4, num_stages), (bid_n % 2, 0, 0))
                sSFW3_view = cute.local_tile(sSFW3_view, (2, 4, num_stages), (bid_n % 2, 0, 0))

            sSFW1_view = cute.local_tile(sSFW1_view, (WN // 32, 4, num_stages), (warp_id_n, 0, 0))
            sSFW3_view = cute.local_tile(sSFW3_view, (WN // 32, 4, num_stages), (warp_id_n, 0, 0))

            sfx_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=64)
            sfw_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=32 * (WN // 32))

            # registers
            # let ptxas decides register reuse for rA and rB
            rX = cute.make_rmem_tensor((32, WM // 16, BK // MMA_K), Float4E2M1FN)
            rW1 = cute.make_rmem_tensor(((16, 2), WN // 16, BK // MMA_K), Float4E2M1FN)
            rW3 = cute.make_rmem_tensor(((16, 2), WN // 16, BK // MMA_K), Float4E2M1FN)
            rO1 = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
            rO3 = cute.make_rmem_tensor((4, WN // 8, WM // 16), Float32)
            rO1.fill(0.0)
            rO3.fill(0.0)

            rSFX = cute.make_rmem_tensor((2, BK // MMA_K), Int32)
            rSFW1 = cute.make_rmem_tensor((WN // 32, BK // MMA_K), Int32)
            rSFW3 = cute.make_rmem_tensor((WN // 32, BK // MMA_K), Int32)

            for iter_k in range(K // BK):
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=NUM_MMA_THREADS)

                for k in cutlass.range_constexpr(BK // MMA_K):
                    # TODO: check bank conflicts
                    cute.copy(sfx_atom, sSFX_view[None, k, tma_stage], rSFX[None, k])
                    cute.copy(sfw_atom, sSFW1_view[None, k, tma_stage], rSFW1[None, k])
                    cute.copy(sfw_atom, sSFW3_view[None, k, tma_stage], rSFW3[None, k])

                    cute.copy(ldsm_atom, sX_ldsm[None, (None, k, tma_stage)], rX[None, None, k])
                    cute.copy(ldsm_atom, sW1_ldsm[None, (None, k, tma_stage)], rW1[None, None, k])
                    cute.copy(ldsm_atom, sW3_ldsm[None, (None, k, tma_stage)], rW3[None, None, k])

                    for m in cutlass.range_constexpr(WM // 16):
                        for n in cutlass.range_constexpr(WN // 8):
                            rO1[None, n, m] = mma_sync_nvfp4(
                                rX[None, m, k],
                                rW1[(None, n % 2), n // 2, k],
                                rO1[None, n, m],
                                rSFX[m // 2, k],
                                Int16(m % 2),
                                rSFW1[n // 4, k],
                                Int16(n % 4),
                            )
                            rO3[None, n, m] = mma_sync_nvfp4(
                                rX[None, m, k],
                                rW3[(None, n % 2), n // 2, k],
                                rO3[None, n, m],
                                rSFX[m // 2, k],
                                Int16(m % 2),
                                rSFW3[n // 4, k],
                                Int16(n % 4),
                            )

                cute.arch.barrier(barrier_id=1, number_of_threads=NUM_MMA_THREADS)
                cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

            # epilogue
            X_scale = gSFX_tensor[0]
            W1_scale = gSFW1_tensor[0] * X_scale
            W3_scale = gSFW3_tensor[0] * X_scale

            if cutlass.const_expr(gSFC_tensor is not None):
                gSFC_ = cute.make_tensor(
                    gSFC.iterator,
                    cute.make_layout(
                        ((32, 4, M // 128), (4, N // 64)),
                        stride=((16, 4, N * 8), (1, 512)),
                    ),
                )
                self.quantize_epilogue(rO1, rO3, W1_scale, W3_scale, gC, gSFC_, gSFC_tensor, sX.iterator)

            else:
                self.epilogue(rO1, rO3, W1_scale, W3_scale, gC)

    @cute.jit
    def epilogue(self, rO1: cute.Tensor, rO3: cute.Tensor, W1_scale: Float32, W3_scale: Float32, gC: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, _ = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        WM = BM // num_warp_m
        WN = BN // num_warp_n
        warp_id_m = warp_id // num_warp_n
        warp_id_n = warp_id % num_warp_n

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
                tmp_f32 = cute.make_rmem_tensor(4, Float32)
                for i in cutlass.range_constexpr(4):
                    o1 = rO1[i, n, m] * W1_scale
                    o3 = rO3[i, n, m] * W3_scale
                    sigmoid = cute.arch.rcp_approx(1.0 + cute.exp(-o1))
                    tmp_f32[i] = o1 * o3 * sigmoid

                tmp = cute.make_rmem_tensor((2, 2), BFloat16)
                tmp.store(tmp_f32.load().to(BFloat16))
                stg_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32)
                cute.copy(stg_atom, tmp, gC_view[None, None, (m, n)])

    @cute.jit
    def quantize_epilogue(
        self,
        rO1: cute.Tensor,
        rO3: cute.Tensor,
        W1_scale: Float32,
        W3_scale: Float32,
        gC: cute.Tensor,
        gSFC: cute.Tensor,
        gSFC_tensor: cute.Tensor,
        smem_ptr: cute.Pointer,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BM, BN, _ = self.cta_tile
        num_warp_m, num_warp_n = self.warp_layout
        WM = BM // num_warp_m
        WN = BN // num_warp_n
        NUM_MMA_THREADS = math.prod(self.warp_layout) * 32
        warp_id_m = warp_id // num_warp_n
        warp_id_n = warp_id % num_warp_n

        # swiglu then store to smem
        stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4)
        stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)

        sTmp = cute.make_tensor(
            cute.recast_ptr(smem_ptr, dtype=BFloat16),
            cute.make_layout(((64, BN // 64), BM), stride=((1, BM * 64), 64)),
        )
        sTmp_warp = cute.local_tile(sTmp, (WN, WM), (warp_id_n, warp_id_m))
        sTmp_stsm = cute.zipped_divide(sTmp_warp, (cute.make_layout((8, 2)), 16))  # (((8,2),16), (WN/16,WM/16))
        sTmp_stsm = sTmp_stsm[((None, lane_id // 16), lane_id % 16), None]  # (8, (WN/16,WM/16))

        for m in cutlass.range_constexpr(WM // 16):
            tmp = cute.make_rmem_tensor(((4, 2), WN // 16), Float32)

            for n in cutlass.range_constexpr(WN // 8):
                for i in cutlass.range_constexpr(4):
                    o1 = rO1[i, n, m] * W1_scale
                    o3 = rO3[i, n, m] * W3_scale
                    sigmoid = cute.arch.rcp_approx(1.0 + cute.exp(-o1))
                    tmp[(i, n % 2), n // 2] = o1 * o3 * sigmoid

            tmp_bf16 = cute.make_rmem_tensor((8, WN // 16), BFloat16)
            tmp_bf16.store(tmp.load().to(BFloat16))
            cute.copy(stsm_atom, tmp_bf16, sTmp_stsm[None, (None, m)])

        tensor_scale = gSFC_tensor[0]
        cute.arch.barrier(barrier_id=1, number_of_threads=NUM_MMA_THREADS)

        # reload. each thread loads 8 BF16 elems
        num_cols = BN // 8
        num_rows = NUM_MMA_THREADS // num_cols

        sTmp_view = cute.zipped_divide(sTmp, (8, num_rows))  # ((8,rows), (BN/8,BM/rows))
        sTmp_view = sTmp_view[(None, tid // num_cols), (tid % num_cols, None)]  # (8, BM/rows)

        # create view into C gmem
        gC_cta = cute.local_tile(permute(gC, (1, 0)), tiler=(BN, BM), coord=(bid_n, bid_m))
        gC_view = cute.zipped_divide(gC_cta, (8, num_rows))
        gC_view = gC_view[(None, tid // num_cols), (tid % num_cols, None)]

        gSFC_cta = cute.local_tile(gSFC, (BM, BN // 16), (bid_m, bid_n))

        for i in cutlass.range_constexpr(BM // num_rows):
            lds_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
            tmp = cute.make_rmem_tensor(8, BFloat16)
            cute.copy(lds_atom, sTmp_view[None, i], tmp)

            tmp_f32 = cute.make_rmem_tensor(8, Float32)
            tmp_f32.store(tmp.load().to(Float32))

            # tried using bf16x2 math here, but the compiler has correctness issue (eliminate code).
            amax_f32 = Float32(0)
            for j in cutlass.range_constexpr(8):
                amax_f32 = cute.arch.fmax(amax_f32, cute.abs(tmp_f32[j]))
            amax_f32 = cute.arch.fmax(amax_f32, cute.arch.shuffle_sync_bfly(amax_f32, 1))

            scale_f32 = amax_f32 * cute.rcp(cute.arch.fmax(6.0 * tensor_scale, 1e-12), approx=True)
            scale_f32 = cute.arch.fmin(cute.arch.fmax(scale_f32, -448.0), 448.0)
            scale_f8 = scale_f32.to(Float8E4M3FN)
            if tid % 2 == 0:
                gSFC_cta[i * num_rows + tid // num_cols, (tid % num_cols) // 2] = scale_f8

            inv_scale = cute.rcp(cute.arch.fmax(scale_f8.to(Float32) * tensor_scale, 1e-12), approx=True)
            out = _f32x8_to_f4x8(tmp_f32.load() * inv_scale)
            cute.arch.store(gC_view[None, i].iterator, out)

    @cache
    @staticmethod
    def compile(quantize_epilogue: bool):
        M = cute.sym_int()
        N = cute.sym_int(divisibility=2)
        K = cute.sym_int(divisibility=2)

        X = make_fake_tensor(Float4E2M1FN, (M, K), (cute.sym_int64(32), 1), assumed_align=16)
        W1 = make_fake_tensor(Float4E2M1FN, (N, K), (cute.sym_int64(32), 1), assumed_align=16)
        W3 = make_fake_tensor(Float4E2M1FN, (N, K), (cute.sym_int64(32), 1), assumed_align=16)
        SFX = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFW1 = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFW3 = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
        SFX_tensor = make_fake_tensor(Float32, (1,), (0,), assumed_align=16)
        SFW1_tensor = make_fake_tensor(Float32, (1,), (0,), assumed_align=16)
        SFW3_tensor = make_fake_tensor(Float32, (1,), (0,), assumed_align=16)

        if quantize_epilogue:
            C = make_fake_tensor(Float4E2M1FN, (M, N), (cute.sym_int64(32), 1), assumed_align=16)
            SFC = make_fake_tensor(Float8E4M3FN, (cute.sym_int(divisibility=512),), (1,), assumed_align=16)
            SFC_tensor = make_fake_tensor(Float32, (1,), (0,), assumed_align=16)
        else:
            C = make_fake_tensor(BFloat16, (M, N), (cute.sym_int64(8), 1), assumed_align=16)
            SFC = None
            SFC_tensor = None

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm120GatedGemmNVFP4()
        return cute.compile(
            kernel,
            X,
            SFX,
            SFX_tensor,
            W1,
            SFW1,
            SFW1_tensor,
            W3,
            SFW3,
            SFW3_tensor,
            C,
            SFC,
            SFC_tensor,
            stream,
            options="--enable-tvm-ffi",
        )


def mm(
    X: torch.Tensor,
    SFX: torch.Tensor,
    SFX_tensor: torch.Tensor,
    W1: torch.Tensor,
    SFW1: torch.Tensor,
    SFW1_tensor: torch.Tensor,
    W3: torch.Tensor,
    SFW3: torch.Tensor,
    SFW3_tensor: torch.Tensor,
    SFC_tensor: torch.Tensor | None = None,
):
    M, _ = X.shape
    N, _ = W1.shape

    quantize_epilogue = SFC_tensor is not None
    kernel = Sm120GatedGemmNVFP4.compile(quantize_epilogue)

    if quantize_epilogue:
        # over-allocate C so i don't need to handle out-of-bounds writes lol
        pad_M = (M + 128 - 1) // 128 * 128
        C = X.new_empty(pad_M, N // 2, dtype=torch.float4_e2m1fn_x2)[:M]
        SFC = X.new_empty(pad_M, (N // 16), dtype=torch.float8_e4m3fn)
        SFC_tensor = SFC_tensor.view(-1)
    else:
        C = X.new_empty(M, N, dtype=torch.bfloat16)
        SFC = None

    kernel(
        X,
        SFX.view(-1),
        SFX_tensor.view(-1),
        W1,
        SFW1.view(-1),
        SFW1_tensor.view(-1),
        W3,
        SFW3.view(-1),
        SFW3_tensor.view(-1),
        C,
        SFC.view(-1) if SFC is not None else None,
        SFC_tensor,
    )
    return (C, SFC) if quantize_epilogue else C
