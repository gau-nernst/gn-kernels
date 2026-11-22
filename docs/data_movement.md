# Data Movement: SASS Notes

## Global → shared

### LDGSTS (`cp.async`)

#### Cache mode, descriptor, L2 cache hint

CUDA 13.4, sm_80–sm_121a.

sm_80–sm_89, plain form:

```
LDGSTS.E.BYPASS.128 [R7], [R2.64]   // cp.async.cg ..., 16
LDGSTS.E.128        [R7], [R2.64]   // cp.async.ca ..., 16
LDGSTS.E.64         [R7], [R2.64]   // cp.async.ca ..., 8
LDGSTS.E            [R7], [R2.64]   // cp.async.ca ..., 4

// [R7]    = dst smem address (32-bit)
// [R2.64] = src gmem address (64-bit, R2:R3)
// BYPASS  = L1 bypass (one flag bit); 4/8-byte copies are .ca only
```

sm_90a–sm_121a, plain form:

```
LDCU.64 UR6, c[0x0][0x358]                   // default descriptor from constant bank
LDGSTS.E.BYPASS.128 [R7], desc[UR6][R2.64]   // use UR6:UR7 as descriptor
```

- `c[bank][offset]` is constant memory; bank 0 is filled by the driver at launch.
- `desc[URx]` is undocumented. Its only observed use is the L2 cache policy; the default observed is `0x0`, presumably "no policy".

`.L2::cache_hint` (all targets) passes the 64-bit policy as the descriptor (sm_90a+: in place of the default); otherwise the encoding is unchanged:

```
// set policy (EVICT_FIRST), sm_80–sm_110a: two 32-bit moves
UMOV UR8, 0x0                                // policy low half
UMOV UR9, 0x12f00000                         // policy high half

// set policy, sm_120a–sm_121a: one 64-bit move
UMOV.64 UR8, 0x12f0000000000000

LDGSTS.E.BYPASS.128 [R7], desc[UR8][R2.64]   // use UR8:UR9 (policy) as descriptor
LDGSTS.E.BYPASS.128 [R7], desc[URZ][R2.64]   // policy 0: URZ is zero (see below)
```

ptxas emits `[R+URn]` when the smem address is a warp-uniform base (`URn`, e.g. the `__shared__` array or stage address) plus a per-thread offset (`R`). The single UR field now holds the base, so the descriptor is implicitly the next register, printed as `desc[URn+1]`. sm_120a:

```
UMOV.64 UR14, 0x12f0000000000000   // policy in UR14:UR15
UMOV UR14, UR7                     // UR14 reused for the smem base

// descriptor = UR15 only (inferred: high half; policy low half is 0)
LDGSTS.E.BYPASS.128 [R4+UR14], desc[UR15][R64.64]
```

Where the policy comes from:

- Observed: `createpolicy` lowers to plain integer ops with a deterministic result. A constexpr policy is rematerialized at each use; a runtime value (`createpolicy` result, kernel argument) stays in a register. The compiler usually hoists `createpolicy` out of loops, but not always.
- Recommended: create the policy once, outside the loop, as a runtime value.
- ptxas 13.3/13.4 crashes (C7907) on sm_90a+ for `.L2::cache_hint` with a constexpr policy `0`; CUDA 13.0 compiles it.
- CUTLASS `EVICT_NORMAL = 0x1000000000000000` differs from `createpolicy ... evict_normal` (`0x16f0000000000000`).

#### Masked loads

CUDA 13.4, sm_80–sm_121a, unless noted. Listings are sm_120a.

##### PTX forms

`cp.async.cg.shared.global [dst], [src], 16, src-size` or `..., 16, ignore-src`, with or without `.L2::cache_hint`:

```
// src-size 16 (constexpr): plain form
LDGSTS.E.BYPASS.128 [R9], desc[UR6][R2.64]

// src-size 0 (constexpr): predicate operand !PT (PT is always true)
// false predicate operand: zero-fill smem, no gmem read
LDGSTS.E.BYPASS.128 [R9], desc[UR6][R2.64], !PT

// runtime predicate: ignore-src = row >= M, or src-size = row < M ? 16 : 0
// predicate operand !P0, from the runtime predicate
ISETP.GE.U32.AND P0, PT, R4, UR8, PT              // P0 = row >= M
LDGSTS.E.BYPASS.128 [R9], desc[UR6][R2.64], !P0   // zero-fill if P0
```

Any other src-size uses `.ZFILL`. The src address is 16-byte aligned, so its low 4 bits are free; `.ZFILL` stores the bytes to zero, `(16 - size) & 15`, there. Value 0 means a full copy, so size 0 takes a separate predicate:

```
// src-size 8 (constexpr): low bits set by the immediate, no extra instructions
LDGSTS.E.BYPASS.128.ZFILL [R9], desc[UR6][R2.64+0x8]  // copy 8 bytes, zero 8

// src-size runtime: ~5 extra instructions for the low bits and P3 = size != 0
LDGSTS.E.BYPASS.128.ZFILL [R9], desc[UR6][R6.64], P3  // R6:R7 = src | low bits
```

Copies with the same runtime size share this work: the low bits are set once on the base src address, and the other copies' addresses derive from it as in the unmasked case.

ptxas pattern-matches src-size `p ? 16 : 0` (still a `selp` and a register in PTX) to the predicate operand, but not always: in one kernel with several copies, the `.L2::cache_hint` form took the runtime `.ZFILL` sequence on sm_100a+. `ignore-src` maps to the predicate operand directly. Check the SASS to be sure.

##### Row tail (e.g. M < BM)

Each 16-byte copy lies within one row, so it loads all 16 bytes or nothing. Three approaches:

```
// predicate form: ignore-src = row >= M, or src-size = row < M ? 16 : 0
ISETP.GE.U32.AND P0, PT, R4, UR8, PT              // P0 = row >= M
LDGSTS.E.BYPASS.128 [R9], desc[UR6][R2.64], !P0   // row >= M: zero-fill smem

// guard predicate: @p cp.async, p = row < M
ISETP.GE.U32.AND P0, PT, R4, UR8, PT              // P0 = row >= M
@!P0 LDGSTS.E.BYPASS.128 [R9], desc[UR6][R2.64]   // row >= M: smem keeps stale data

// remap in bounds: load row min(row, M - 1), no predicate
VIMNMX.S32 R0, R0, UR4, PT                        // R0 = min(row, UR4 = M - 1)
LDGSTS.E.BYPASS.128 [R7], desc[UR6][R4.64]        // address from remapped row
```

- The guard and remap leave stale or duplicated rows in smem, which is fine when the epilogue masks C rows >= M.
- Each masked row holds one predicate per thread, hoisted out of the k-loop. There are 7 (`P0`–`P6`); beyond that, ptxas keeps masks in GPRs (`P2R`) and re-derives them with `ISETP.NE` every iteration. The remap uses none.

Benchmark (RTX 5090, µs per launch): copy-only `cp.async` kernel, no math. Each CTA streams 128 x 4096 bf16 as 128x64 tiles (4 stages, 128 threads, 8 copies per thread per tile):

| valid rows | unmasked (all rows) | predicate form | guard predicate | remap |
|---|---|---|---|---|
| 1 | 418.9 | 27.9 | 15.4 | 132.5 |
| 64 | 418.9 | 212.6 | 212.5 | 233.6 |
| 128 | 418.8 | 418.9 | 418.8 | 418.9 |
| k-loop instructions / tile (8 `LDGSTS`) | 65 | 69 (+4) | 71 (+6) | 65 (+0) |

- Full M: no mask costs anything.
- Partial M: skipping beats zero-filling; the remap's L2 re-reads cost the most.
- Unmasked is DRAM-bound (1.70 TB/s). At 1 valid row the in-bounds data stays in L2, so that row measures issue cost and L2 traffic.
- Masks may still change the surrounding codegen in a real kernel (extra instructions, unrolling). If a masked build regresses, compare its k-loop SASS against an unmasked build.
