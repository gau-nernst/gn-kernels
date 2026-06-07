import cutlass
import torch
from cutlass import Int16, Int32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

TORCH_TO_CUTE_DTYPE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.int8: cutlass.Int8,
    torch.uint8: cutlass.Uint8,
    torch.int32: cutlass.Int32,
    torch.uint32: cutlass.Uint32,
}

CUTE_TO_PTX_DTYPE = {
    cutlass.Float32: "f32",
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int8: "s8",
    cutlass.Uint8: "u8",
    cutlass.Int32: "s32",
    cutlass.Uint32: "u32",
}


def simple_tma_g2s(atom, src, dst, mbar):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    dst = cute.group_modes(dst, 0)
    src = cute.group_modes(src, 0)
    s_part, g_part = cpasync.tma_partition(atom, 0, cute.make_layout(1), dst, src)
    cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)


@dsl_user_op
def to_cta0_smem(ptr: cute.Pointer, *, loc=None, ip=None):
    return cute.make_ptr(
        ptr.dtype,
        ptr.toint(loc=loc, ip=ip) & 0xFEFF_FFFF,
        cute.AddressSpace.smem,
        assumed_align=8,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def permute(x: cute.Tensor, dims: tuple[int, ...], *, loc=None, ip=None):
    layout = cute.select(x.layout, mode=dims, loc=loc, ip=ip)
    return cute.make_tensor(x.iterator, layout, loc=loc, ip=ip)


@dsl_user_op
def mma_sync(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, *, loc=None, ip=None):
    # CuteDSL has nvvm.mma_sync(), but it doesn't export nvvm's MMA shape.
    # hence we have to use PTX.
    a_ty = CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)

    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


@dsl_user_op
def mma_sync_mxfp8(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    SFA: Int32,
    byte_id_A: Int16,
    thread_id_A: Int16,
    SFB: Int32,
    byte_id_B: Int16,
    thread_id_B: Int16,
    *,
    loc=None,
    ip=None,
):
    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)

    mlir_ty = cutlass.Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [SFA.ir_value(loc=loc, ip=ip), byte_id_A.ir_value(loc=loc, ip=ip), thread_id_A.ir_value(loc=loc, ip=ip)]
        + [SFB.ir_value(loc=loc, ip=ip), byte_id_B.ir_value(loc=loc, ip=ip), thread_id_B.ir_value(loc=loc, ip=ip)],
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13}, "
        "$14, {$15, $16}, "
        "$17, {$18, $19};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,h,h,r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


@dsl_user_op
def mma_sync_nvfp4(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    SFA: Int32,
    byte_id_A: Int16,
    thread_id_A: Int16,
    SFB: Int32,
    byte_id_B: Int16,
    thread_id_B: Int16,
    *,
    loc=None,
    ip=None,
):
    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)

    mlir_ty = cutlass.Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [SFA.ir_value(loc=loc, ip=ip), byte_id_A.ir_value(loc=loc, ip=ip), thread_id_A.ir_value(loc=loc, ip=ip)]
        + [SFB.ir_value(loc=loc, ip=ip), byte_id_B.ir_value(loc=loc, ip=ip), thread_id_B.ir_value(loc=loc, ip=ip)],
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e4m3.e4m3.f32.ue4m3 "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13}, "
        "$14, {$15, $16}, "
        "$17, {$18, $19};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,h,h,r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)
