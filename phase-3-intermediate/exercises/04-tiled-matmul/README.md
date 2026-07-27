<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 04 - Tiled matrix multiplication

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~3 h | Prerequisites: [02](../02-shared-memory-basics/), [03](../03-bank-conflicts/), [Phase 2/07](../../../phase-2-cuda-fundamentals/exercises/07-matmul-naive/) | **Requires a GPU**

## Goal

Apply shared-memory tiling to the matmul from Phase 2 — and discover that **the
textbook answer buys almost nothing**, while a second, less-discussed technique
buys 3×. Understanding *why* is worth more than the kernel.

## Background

Phase 2/07 left the naive kernel at ~6% of the GPU's FP32 peak. Its arithmetic
intensity is fixed at

```
2N FLOPs / (2N × 4 bytes) = 0.25 FLOP per byte
```

no matter how large `N` gets. Against a ridge point near 37, that is hopeless.

**Tiling should fix it.** Load a `TILE×TILE` block of `A` and `B` into shared
memory and every loaded value serves `TILE` threads instead of 1:

```
intensity = 2 × TILE / (2 × 4 bytes) = TILE/4 = 8 FLOP/byte for TILE = 32
```

A 32× improvement on paper. The loop is Phase 1/02's cache blocking, with
`__shared__` as the cache — except you place the data yourself.

**Both barriers are necessary.** One after loading the tile, one after consuming
it. Dropping the second is a classic bug: fast threads begin loading the next tile
while slow threads are still reading the current one. It usually still produces the
right answer on small inputs, which is what makes it dangerous.

**No padding needed here.** In the inner loop, for a warp (`tx` varies, `ty`
fixed): `As[ty][k]` is the same address for all 32 threads — a broadcast, free —
and `Bs[k][tx]` spans 32 consecutive words — 32 distinct banks, free. Exercise 05
is where padding becomes essential.

## Your task

Open `main.cu` and write three kernels: `matmulNaive`, `matmulTiled<TILE>`, and
`matmulTiledRegister<TILE, TM>` where each thread keeps `TM` accumulators in
registers and computes `TM` output rows.

**Predict before measuring.** Plain tiling cuts global loads per output by 32×; how
much speedup do you expect? Register tiling does **not** reduce global traffic at
all; how much do you expect from that?

One of those two predictions is probably badly wrong. Finding out which is the
exercise.

## Build and run

```bash
cmake --build build --target p3_04_tiled_matmul -j
./build/bin/p3/p3_04_tiled_matmul
```

## Expected output

RTX 3060, `n = 1024`:

```
Variant                           Time (ms)    GFLOP/s   Speedup
----------------------------------------------------------------
CPU ikj (1 thread)                  152.269      14.10     1.00x
GPU naive                             2.688     798.92    56.65x
GPU tiled (shared)                    2.235     961.02    68.14x
GPU tiled + register                  0.847    2535.47   179.78x

  variant                       loads/output   FLOP per byte   % of peak
  ----------------------------------------------------------------------
  naive                                 2048            0.25          6%
  tiled (TILE=32)                         64            8.00          7%
  tiled + register (TM=4)                 64            8.00         19%
```

### The result the textbook does not prepare you for

**Plain tiling bought 1.20×**, despite a 32× reduction in global loads per output.

The reason is the one you already met in Phase 2/07: the naive kernel's 2048 loads
per output were never really reaching DRAM. **L1 and L2 were already capturing
almost all of that reuse.** Writing the tiling by hand mostly replaced an automatic
cache with a manual one.

**Register tiling bought 3.0×** — and it is a *different kind of change*. Look at
the table: `loads/output` is identical for both tiled variants. It does not move
less data. What it does is give each thread more independent work:

- each value read from **shared** memory now serves 4 multiply-adds
- 4 accumulators live in registers, so the FMA pipeline has 4 independent chains
  and stops stalling on its own latency — the same effect you measured on the CPU
  in [Phase 1/07](../../../phase-1-foundation/exercises/07-roofline-model/)
- the block shrinks from 1024 to 256 threads, freeing registers and scheduler slots

> **The lesson to carry forward:** on a modern GPU, shared memory is rarely valuable
> because it is "faster than global" — the cache already gave you that. It is
> valuable because it lets you **restructure the computation** so each thread does
> more work per byte it touches. The restructuring is the point; the shared memory
> is just what makes it possible.

## Key takeaways

- **A 32× reduction in loads produced a 1.2× speedup.** Always check whether the
  cache was already solving your problem.
- **Work per thread matters as much as traffic per thread.** Register tiling adds
  ILP, and ILP is what fills a latency-bound pipeline.
- **Two barriers per tile iteration**, and the missing-second-barrier bug is silent
  on small inputs.
- 19% of peak, versus cuBLAS's 80–90%. The rest is more of the same idea: wider
  register tiles, `float4` loads, double buffering, per-architecture tuning.

## Going further

- Sweep `TM` = 1, 2, 4, 8. Where does it stop helping, and what stops it?
  (`-DCL_PTXAS_VERBOSE=ON` shows register usage.)
- Try `TILE` = 16 and 64. What limits the tile size?
- Add `float4` loads for the tile fill.
- Double-buffer the tiles: load tile `t+1` while computing on tile `t`.
- **Next:** [05 transpose-optimized](../05-transpose-optimized/) uses shared memory
  for *reordering* rather than reuse — and there the padding from exercise 03 is
  essential. [Phase 5/03](../../../phase-5-expert/exercises/03-cublas-gemm/)
  measures the gap to cuBLAS.
