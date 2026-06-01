from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

from cutlass import Int32

SPACE_MAP = dict(
    cta=nvvm.MBarrierSpaceKind.CTA,
    cluster=nvvm.MBarrierSpaceKind.CLUSTER,
)

MEMORY_ORDER_MAP = dict(
    weak=nvvm.MemOrderKind.WEAK,
    relaxed=nvvm.MemOrderKind.RELAXED,
    acquire=nvvm.MemOrderKind.ACQUIRE,
    release=nvvm.MemOrderKind.RELEASE,
    acq_rel=nvvm.MemOrderKind.ACQ_REL,
)


@dsl_user_op
def arrive(mbar, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None):
    nvvm.mbarrier_txn(
        mbar.to_llvm_ptr(),
        Int32(1).ir_value(),
        kind=nvvm.MBarrierTxnKind.ARRIVE,
        space=SPACE_MAP[space],
        order=MEMORY_ORDER_MAP[order],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_expect_tx(mbar, size, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None):
    nvvm.mbarrier_txn(
        mbar.to_llvm_ptr(),
        Int32(size).ir_value(),
        kind=nvvm.MBarrierTxnKind.ARRIVE_EXPECT_TX,
        space=SPACE_MAP[space],
        order=MEMORY_ORDER_MAP[order],
        loc=loc,
        ip=ip,
    )
