# GPU Memory Ordering: PTX and SASS

This note separates the formal PTX memory model from the hardware behavior used
to optimize distributed kernels.

## PTX correctness model

PTX defines ordering in terms of memory semantics, proxies, and thread scopes:

- **Release** orders earlier memory operations before a publishing operation.
- **Acquire** orders later memory operations after observing a publication.
- **Relaxed** participates in scoped coherence but does not order other memory
  operations.
- **Weak** does not participate in the PTX memory-consistency model.

A formal producer-consumer proof requires a compatible release/acquire relation.
Observing a flag with a relaxed load is not, by itself, an acquire.

Each thread scope has an associated point of coherency:

| PTX scope | Participating threads | Point of coherency |
| --- | --- | --- |
| `.cta` | one thread block | L1 |
| `.cluster` | one cluster | L2 |
| `.gpu` | one GPU | L2 |
| `.sys` | CPUs and GPUs in the system | L2 plus connected caches |

Formally, scope describes which threads may synchronize. Physically, the point
of coherency describes how far writes must propagate. These are related but not
the same as the physical home of an allocation.

## Observed SASS behavior

SASS exposes the mechanisms used to implement PTX ordering and their cost:

| PTX | Relevant SASS |
| --- | --- |
| `st.release.gpu.global` | `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; STG.E.STRONG.GPU` |
| `st.release.sys.global` | `MEMBAR.ALL.SYS; ERRBAR; CGAERRBAR; STG.E.STRONG.SYS` |
| `ld.relaxed.{gpu,sys}.global` | scoped `LDG.E.STRONG` |
| `ld.acquire.{gpu,sys}.global` | scoped `LDG.E.STRONG; CCTL.IVALL` |
| `fence.acquire.{gpu,sys}` | `CCTL.IVALL` |
| `fence.sc.{gpu,sys}` | scoped `MEMBAR.SC; ERRBAR; CGAERRBAR; CCTL.IVALL` |
| `fence.proxy.alias` | runtime-selected scoped `MEMBAR.SC; CCTL.IVALL` |
| `multimem.ld_reduce.{weak,relaxed.gpu}` | `LDGMC.E.*.STRONG.SYS` |

`MEMBAR.ALL` waits for earlier operations to reach the point of coherency
required by the scope before publication. `CCTL.IVALL` invalidates
requester-local global-data L1 state; it does not invalidate L2.

Thus a relaxed PTX load is still a coherent `STRONG` hardware load. Relaxed
means no acquire ordering. Conversely, identical SASS does not make PTX
`.weak` and `.relaxed` formally equivalent.

## Hardware-oriented scope selection

For optimized kernels, ask where the **data writes** must land:

- Data written to memory homed on the producer GPU needs to reach that GPU's
  L2. A GPU-scope release provides the required flush, even if another GPU later
  reads the data through the owner GPU's memory system.
- Data written to memory homed on a peer GPU must reach the peer's L2. This
  requires system scope on the publishing side.
- A consumer polling a local flag and reading local memory needs only GPU-scope
  acquire. Symmetric system scope on both sides is often stronger than the
  minimum hardware requirement.

This physical reasoning can justify faster protocols than a conservative,
general-purpose PTX proof. Keep the distinction explicit.

## References

- [PTX memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)
- [CUDA thread scopes and points of coherency](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scopes)
- [CUTLASS issue #3117: spin-loop synchronization](https://github.com/NVIDIA/cutlass/issues/3117)
