import cutlass
from cutlass import Boolean, Float32, Int32, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op

CTA_GROUP_MAP = [
    None,
    nvvm.Tcgen05GroupKind.CTA_1,
    nvvm.Tcgen05GroupKind.CTA_2,
]

LDST_MAP = {
    "32x32b": (1, nvvm.Tcgen05LdStShape.SHAPE_32X32B),
    "16x128b": (2, nvvm.Tcgen05LdStShape.SHAPE_16X128B),
    "16x256b": (4, nvvm.Tcgen05LdStShape.SHAPE_16X256B),
}


def _make_tmem_llvm_ptr(taddr, *, loc=None, ip=None):
    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(tmem_ptr_ty, Int32(taddr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def make_idesc_bf16(MMA_M: int, MMA_N: int, *, loc=None, ip=None):
    return Uint32((1 << 4) | (1 << 7) | (1 << 10) | (MMA_N >> 3 << 17) | (MMA_M >> 4 << 24))


@dsl_user_op
def make_idesc_mxfp8(MMA_M: int, MMA_N: int, *, loc=None, ip=None):
    return Uint32((MMA_N >> 3 << 17) | (1 << 23) | (MMA_M >> 7 << 27))


@dsl_user_op
def make_idesc_nvfp4(MMA_M: int, MMA_N: int, *, loc=None, ip=None):
    return Uint32((1 << 7) | (1 << 10) | (MMA_N >> 3 << 17) | (MMA_M >> 7 << 27))


@dsl_user_op
def make_sdesc_128B(*, loc=None, ip=None):
    return Uint64(((8 * 128) >> 4 << 32) | (1 << 46) | (2 << 61))


@dsl_user_op
def alloc(taddr: cute.Pointer, cta_group: int = 1, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Int32(512).ir_value(loc=loc, ip=ip),
        group=CTA_GROUP_MAP[cta_group],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def dealloc(cta_group: int = 1, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_dealloc(
        _make_tmem_llvm_ptr(0, loc=loc, ip=ip),
        Int32(512).ir_value(loc=loc, ip=ip),
        group=CTA_GROUP_MAP[cta_group],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_f16(
    d_tmem,
    a_desc,
    b_desc,
    idesc,
    enable_input_d,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            CTA_GROUP_MAP[cta_group],
            _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Uint32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def mma_mxfp8(
    d_tmem,
    a_desc,
    b_desc,
    idesc,
    sfa_tmem,
    sfb_tmem,
    enable_input_d,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    with cute.arch.elect_one():
        nvvm.tcgen05_mma_block_scale(
            nvvm.Tcgen05MMAKind.MXF8F6F4,
            CTA_GROUP_MAP[cta_group],
            _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Uint32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            _make_tmem_llvm_ptr(sfa_tmem, loc=loc, ip=ip),
            _make_tmem_llvm_ptr(sfb_tmem, loc=loc, ip=ip),
            scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.X1,  # BLOCK32 doesn't work
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def mma_nvfp4(
    d_tmem,
    a_desc,
    b_desc,
    idesc,
    sfa_tmem,
    sfb_tmem,
    enable_input_d,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    with cute.arch.elect_one():
        nvvm.tcgen05_mma_block_scale(
            nvvm.Tcgen05MMAKind.MXF4NVF4,
            CTA_GROUP_MAP[cta_group],
            _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Uint32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            _make_tmem_llvm_ptr(sfa_tmem, loc=loc, ip=ip),
            _make_tmem_llvm_ptr(sfb_tmem, loc=loc, ip=ip),
            scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.X4,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def commit(mbar, cta_mask=None, cta_group: int = 1, *, loc=None, ip=None) -> None:
    mbar_llvm = mbar.to_llvm_ptr(loc=loc, ip=ip)
    group = CTA_GROUP_MAP[cta_group]
    with cute.arch.elect_one():
        if cutlass.const_expr(cta_mask is not None):
            mask = cta_mask.ir_value(loc=loc, ip=ip)
            nvvm.tcgen05_commit_arrive(mbar_llvm, multicast_mask=mask, group=group, loc=loc, ip=ip)
        else:
            nvvm.tcgen05_commit_arrive(mbar_llvm, group=group, loc=loc, ip=ip)


@dsl_user_op
def cp(
    tmem,
    sdesc,
    shape: str,
    mcast: str,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
):
    SHAPE_MAP = {
        "128x256b": nvvm.Tcgen05CpShape.SHAPE_128x256b,
        "4x256b": nvvm.Tcgen05CpShape.SHAPE_4x256b,
        "128x128b": nvvm.Tcgen05CpShape.SHAPE_128x128b,
        "64x128b": nvvm.Tcgen05CpShape.SHAPE_64x128b,
        "32x128b": nvvm.Tcgen05CpShape.SHAPE_32x128b,
    }
    MCAST_MAP = {
        "none": nvvm.Tcgen05CpMulticast.NONE,
        "warpx2::02_13": nvvm.Tcgen05CpMulticast.WARPX2_02_13,
        "warpx2::01_23": nvvm.Tcgen05CpMulticast.WARPX2_01_23,
        "warpx4": nvvm.Tcgen05CpMulticast.WARPX4,
    }
    with cute.arch.elect_one():
        nvvm.tcgen05_cp(
            SHAPE_MAP[shape],
            _make_tmem_llvm_ptr(tmem, ip=ip, loc=loc),
            Uint64(sdesc).ir_value(loc=loc, ip=ip),
            group=CTA_GROUP_MAP[cta_group],
            multicast=MCAST_MAP[mcast],
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def ld(row, col, shape: str, num: int, *, loc=None, ip=None):
    mult, nvvm_shape = LDST_MAP[shape]
    num_regs = num * mult

    tmem_ptr = _make_tmem_llvm_ptr((row << 16) | col, loc=loc, ip=ip)

    if num_regs == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, nvvm_shape, tmem_ptr, loc=loc, ip=ip)
        reg_f32 = llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip)
        return Float32(reg_f32)

    else:
        vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
        vec_f32_ty = ir.VectorType.get([num_regs], Float32.mlir_type, loc=loc)
        regs = nvvm.tcgen05_ld(vec_i32_ty, nvvm_shape, tmem_ptr, loc=loc, ip=ip)
        regs_f32 = llvm.bitcast(vec_f32_ty, regs, loc=loc, ip=ip)
        return cute.TensorSSA(regs_f32, (num_regs,), Float32)


@dsl_user_op
def fence_after_thread_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def fence_before_thread_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def wait_ld(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)


@dsl_user_op
def wait_st(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.STORE, loc=loc, ip=ip)
