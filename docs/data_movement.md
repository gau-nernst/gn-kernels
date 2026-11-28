# Data Movement: SASS Notes

## LDG and STG

CUDA 13.4, sm_80–sm_121a. Listings are sm_100a.

`LDG`/`STG` cover two instructions:
- Legacy opcode `0x981`/`0x986`, 1–16 bytes.
- 256-bit access opcode `0x97e`/`0x97f`, sm_100+ only.

### Legacy `0x981`/`0x986` (1–16 bytes)

```
// < 4 bytes: always lands in the low bits of a 32-bit register
// zero- (.U*) or sign-extended (.S*) into the upper bits
// (no lane select; STG reads low bits)
LDG.E.U8  / .S8    R3, desc[UR4][R2.64]    // 1 byte  (bits [7:0])
LDG.E.U16 / .S16   R3, desc[UR4][R2.64]    // 2 bytes (bits [15:0])

// >= 4 bytes: contiguous, naturally aligned register tuple
// only the base register is printed
LDG.E              R3, desc[UR4][R2.64]    // 4 bytes
LDG.E.64           R2, desc[UR4][R2.64]    // 8 bytes  (R2..R3, 2-aligned)
LDG.E.128          R8, desc[UR4][R2.64]    // 16 bytes (R8..R11, 4-aligned)

// STG has the same width suffixes with operands reversed
STG.E desc[UR4][R2.64], R3

// [R2.64]   = gmem address (64-bit, R2..R3)
// desc[UR4] = undocumented 64-bit descriptor (UR4..UR5)
//             only observed use is L2 cache policy
```

**Global address** `[R2.64+imm]` takes a 64-bit register pair plus a signed 24-bit immediate (±8 MiB); larger constant offsets need a separate 64-bit add, and there is no uniform-register address mode even for warp-uniform pointers. For indexing, a 32-bit index folds `base + i * stride` into one `IMAD.WIDE`, whereas a 64-bit index (`int64_t`, `size_t`) takes a 64-bit shift plus add (two `IADD3`s, or one `IADD.64` on sm_120a+).

**Descriptor** The 64-bit descriptor in `desc[URx]` encodes an L2 eviction policy and, optionally, an address window — the hardware checks each access's address against that window at runtime. On sm_80–sm_89, plain accesses still emit the bank-0 load (`ULDC.64`) and encode the descriptor register, though `nvdisasm` omits `desc[...]` (it prints `desc[...]` with `.L2::cache_hint`).

The descriptor comes from one of two sources:

**1. Launch default (driver → constant bank 0)** — used by every plain access without `.L2::cache_hint`:

```
// default descriptor from constant bank 0 (offset varies by arch)
LDCU.64 UR4, c[0x0][0x358]
LDG.E R3, desc[UR4][R2.64]
```

A launch or stream can attach at most one access policy window (`cudaLaunchAttributeAccessPolicyWindow` / `cudaStreamAttributeAccessPolicyWindow`). At launch, the driver packs that window (base address, size, hit/miss policies, hitRatio) into `c[0x0][0x358]`, or writes `0` if none is set; the compiled binary just passes `c[0x0][0x358]` on every plain access and lets hardware match addresses against the window.

B300, driver 595.112, for window `[buf, buf + 1 MiB)` at `buf = 0x10adfb000400`:

| Access policy window | `c[0x0][0x358]` |
|---|---|
| none | `0x0000000000000000` |
| persisting / streaming, `hitRatio = 1.0` | `0x05f020010adfb000` (low 36 bits = `buf >> 12`) |
| streaming / normal, `hitRatio = 0.5` | `0x028020010adfb000` |

Other L2/cache settings (persisting-L2 limit, cache config, carveout, memsync domain) do not change it. Not probed on sm_80–sm_90a.

**2. Per-instruction override (PTX `.L2::cache_hint`)** — skips constant bank 0 and passes an explicit 64-bit policy in `URx..URx+1` (or `URZ` when `0`). This is how to set per-instruction hints or multiple windows in one kernel:

```
// EVICT_FIRST, sm_80–sm_110a: two 32-bit moves
UMOV UR4, 0x0
UMOV UR5, 0x12f00000

// EVICT_FIRST, sm_120a–sm_121a: one 64-bit move
UMOV.64 UR4, 0x12f0000000000000

LDG.E.128 R4, desc[UR4][R2.64]
LDG.E.128 R4, desc[URZ][R2.64]   // policy 0: URZ is zero
```

Producing the 64-bit policy value:

- `createpolicy` lowers to plain integer bit-packing ops with a deterministic result, in two forms: `createpolicy.fractional` (no address window; applies unconditionally to that instruction) and `createpolicy.range` (encodes an address window + primary/secondary policies).
- `ptxas` rematerializes constant policies at each use and doesn't always hoist `createpolicy` out of loops; keep hot-loop policies in a runtime register created outside the loop.
- CUTLASS `EVICT_NORMAL = 0x1000000000000000` differs from `createpolicy ... evict_normal` (`0x16f0000000000000`).

#### Memory ordering

`LDG`/`STG` itself only encodes weak (default) vs `.STRONG.{SM,GPU,SYS}`:

- `.STRONG.*` is scoped coherence, not acquire/release (`.cta` → `.SM`, `.cluster`/`.gpu` → `.GPU`, `.sys`/`.volatile` → `.SYS`).
- Acquire/release ordering is emitted as separate instructions around `LDG`/`STG`:

```
// ld.acquire.gpu.global.u32
LDG.E.STRONG.GPU R3, desc[UR4][R2.64]
CCTL.IVALL                             // invalidate local L1 global cache (not L2)

// st.release.gpu.global.u32
MEMBAR.ALL.GPU                         // drain prior ops to scope's PoC (L2)
ERRBAR
CGAERRBAR
STG.E.STRONG.GPU desc[UR4][R4.64], R3
```

See [memory_ordering.md](memory_ordering.md) and the [PTX memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).

#### Cache modifiers

Legacy PTX cache operators are formally weak in the PTX spec, but across `sm_80`–`sm_120a` `ptxas` lowers them to the same `.STRONG.*` coherence bits in SASS:
- `.ca`/`.wb` → `.STRONG.SM`
- `.cg` → `.STRONG.GPU`
- `.cv`/`.wt` → `.STRONG.SYS`

The remaining cache modifiers are orthogonal to `.STRONG.*` and combine freely on one instruction (e.g. `LDG.E.EF.LTC256B.CONSTANT`, `LDG.E.NA.STRONG.GPU`):

- L1 eviction/allocation: `.L1::evict_normal` (default, none), `.L1::evict_first` / `.cs` (`__ldcs`/`__stcs`) → `.EF`, `.L1::evict_last` → `.EL`, `.L1::evict_unchanged` → `.EU`, `.L1::no_allocate` → `.NA`, `.lu` (`ld` only) → `.LU`.
- L2 prefetch (`ld` only): `.L2::64B` / `128B` / `256B` → `.LTC64B` / `.LTC128B` / `.LTC256B`.
- Read-only path (`ld` only): `.nc` (`__ldg`, `const __restrict__`) → `.CONSTANT`.

### 256-bit access `0x97e`/`0x97f` (sm_100+)

Emitted for `.v8.b32` / `.v4.b64` on sm_100+ (including sm_120a/sm_121a). Differences from legacy:

```
// two 128-bit quads, need not be adjacent
// LDG lists hi quad first (R8 = r0..r3, R12 = r4..r7); STG lists lo quad first
LDG.E.ENL2.256 R12, R8, desc[UR4][R2.64]
STG.E.ENL2.256 desc[UR4][R4.64], R8, R12

// immediate offset in 32-byte units: LDG ±2 MiB, STG ±8 MiB
LDG.E.ENL2.256 R12, R8, desc[UR4][R2.64+-0x20]
STG.E.ENL2.256 desc[UR4][R2.64+0x7fffe0], R4, R8
```

- L2 eviction priority: only 256-bit `LDG`/`STG` have an opcode-level field — `.ENL2` (default, `.L2::evict_normal`), `.EFL2` (`.L2::evict_first`), `.ELL2` (`.L2::evict_last`).
- Otherwise (descriptor, memory ordering, cache modifiers) as legacy.

#### Register write mask (`LDG` only)

- `LDG.256` takes an 8-bit mask (one bit per 32-bit word, omitted when `0xff`), stealing bits from its immediate field (±2 MiB vs `STG`'s ±8 MiB, which has no mask).
- Each nibble packs its selected words contiguously into that half's dest register, leaving other registers untouched (alignment relaxes to active count: 1 = any, 2 = 2-aligned, 3–4 = 4-aligned; 0 = `RZ`).
- Emitted by `ptxas` for `_` bit-bucket or dead outputs of `ld.global.v8.b32` / `.v4.b64` (plain C++ partial struct loads split into narrower `LDG`s).

```
// ld.global.v8.b32 {%r0, %r1, _, %r2, _, _, _, %r3}, [%rd1]
// 0x8b (1000_1011): lo -> R8..R10 (w0,w1,w3); hi -> R2 (w7)
LDG.E.ENL2.256 R2, R8, desc[UR4][R2.64], 0x8b

// ld.global.v4.b64 {%rd3, _, _, %rd4}, [%rd1]
// 0xc3 (1100_0011): lo -> R2..R3 (w0); hi -> R4..R5 (w3)
LDG.E.ENL2.256 R4, R2, desc[UR4][R2.64], 0xc3

// ld.global.v8.b32 {%r0, %r1, %r2, %r3, _, _, _, _}, [%rd1]
// 0x0f (0000_1111): lo quad only; unused upper dest is RZ
LDG.E.ENL2.256 RZ, R8, desc[UR4][R2.64], 0xf
```

## LDGSTS (`cp.async`)

### Cache mode and L2 cache hint

CUDA 13.4, sm_80–sm_121a.

```
LDGSTS.E.BYPASS.128 [R7], desc[UR6][R2.64]   // cp.async.cg ..., 16
LDGSTS.E.128        [R7], desc[UR6][R2.64]   // cp.async.ca ..., 16
LDGSTS.E.64         [R7], desc[UR6][R2.64]   // cp.async.ca ..., 8
LDGSTS.E            [R7], desc[UR6][R2.64]   // cp.async.ca ..., 4

// [R7]      = dst smem address (32-bit)
// [R2.64]   = src gmem address (64-bit, R2..R3)
// desc[UR6] = descriptor (see LDG and STG)
// BYPASS    = L1 bypass (one flag bit); 4/8-byte copies are .ca only
```

The descriptor and `.L2::cache_hint` work as for [LDG and STG](#ldg-and-stg).

ptxas emits `[R+URn]` when the smem address is a warp-uniform base (`URn`, e.g. the `__shared__` array or stage address) plus a per-thread offset (`R`). The single UR field now holds the base, so the descriptor is implicitly the next register, printed as `desc[URn+1]`. sm_120a:

```
UMOV.64 UR14, 0x12f0000000000000   // policy in UR14..UR15
UMOV UR14, UR7                     // UR14 reused for the smem base

// descriptor = UR15 only (inferred: high half; policy low half is 0)
LDGSTS.E.BYPASS.128 [R4+UR14], desc[UR15][R64.64]
```

ptxas 13.3/13.4 crashes (C7907) on sm_90a+ for `.L2::cache_hint` with a constexpr policy `0`; CUDA 13.0 compiles it. LDG/STG compile fine.

### Masked loads

CUDA 13.4, sm_80–sm_121a, unless noted. Listings are sm_120a.

#### PTX forms

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
// src-size 8 (constexpr): low bits set by immediate; copy 8 B, zero 8 B
LDGSTS.E.BYPASS.128.ZFILL [R9], desc[UR6][R2.64+0x8]

// src-size runtime: ~5 extra ops for R6..R7 = src | low bits, P3 = size != 0
LDGSTS.E.BYPASS.128.ZFILL [R9], desc[UR6][R6.64], P3
```

Copies with the same runtime size share this work: the low bits are set once on the base src address, and the other copies' addresses derive from it as in the unmasked case.

ptxas pattern-matches src-size `p ? 16 : 0` (still a `selp` and a register in PTX) to the predicate operand, but not always: in one kernel with several copies, the `.L2::cache_hint` form took the runtime `.ZFILL` sequence on sm_100a+. `ignore-src` maps to the predicate operand directly. Check the SASS to be sure.

#### Row tail (e.g. M < BM)

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
