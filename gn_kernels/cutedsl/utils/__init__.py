from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

from cutlass import BFloat16, Float32, Uint32, cute


def simple_tma_g2s(atom, src, dst, mbar):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(dst, 0),
        cute.group_modes(src, 0),
    )
    cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)


@dsl_user_op
def recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


@dsl_user_op
def permute(x: cute.Tensor, dims: tuple[int, ...], *, loc=None, ip=None):
    layout = cute.select(x.layout, mode=dims, loc=loc, ip=ip)
    return cute.make_tensor(x.iterator, layout, loc=loc, ip=ip)


@dsl_user_op
def mma_bf16(a: cute.TensorSSA, b: cute.TensorSSA, c: cute.TensorSSA, *, loc=None, ip=None):
    if a.element_type == BFloat16:
        a = cute.recast_tensor(a, Uint32)
    if b.element_type == BFloat16:
        b = cute.recast_tensor(b, Uint32)

    mlir_ty = Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
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
    return cute.TensorSSA(vec, 4, Float32)
