<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 02 - Cache blocking and loop order

> **Phase 1 · Foundation** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [01](../01-matmul-naive-cpu/)

## Goal

Make the exact same arithmetic run several times faster without changing a single
formula — only the order in which memory is touched. This is the single most
important lesson in the curriculum, because *the identical idea* reappears as
shared-memory tiling in Phase 3.

## Background

The naive `ijk` loop reads `B[k][j]` with the `k` loop innermost, so it walks
**down a column** of `B` — stride `N`. Each of those accesses pulls a fresh
64-byte cache line from which exactly 4 bytes are used. The other 60 bytes are
evicted before the loop comes back for them.

**Fix 1 — reorder to `ikj`.** Move `k` to the middle:

```
for i:
  for k:
    a = A[i][k]                  // invariant in j, hoist it
    for j:
      C[i][j] += a * B[k][j]     // B and C both stride 1
```

Now the innermost loop is a scaled vector add over contiguous memory. Two things
improve at once: every cache line is fully consumed, and the loop is in a shape
the compiler can auto-vectorise into AVX FMA instructions.

**Fix 2 — block (tile) the loops.** `ikj` is still sweeping all of `B` once per
row of `A`. For `N = 768`, `B` is 2.25 MB, larger than a typical L2 slice, so
each sweep starts from DRAM again. Blocking bounds the working set: while
computing one `T×T` tile of `C`, only a `T×T` tile of `A` and of `B` are live.
Three 64×64 float tiles are 48 KB — they stay in cache, and each element of `B`
is read `N/T` times instead of `N` times.

That last sentence is worth rereading. **Blocking converts memory traffic into
reuse.**

### An honest warning about the result

On a modern desktop CPU, blocking often buys you **almost nothing** — sometimes it
is even slightly slower. That is not a bug in your code, and the exercise prints a
size sweep so you can see exactly where the crossover is. The reason:

- The CPU already has 4–32 MB of L3 and an aggressive hardware prefetcher. For
  `n ≤ 1024` it is doing the blocking *for you*, automatically.
- The blocked version pays extra loop overhead and gives the vectoriser a shorter
  inner loop, which cancels much of the gain.
- Only once the matrices clearly exceed L3 (`n = 2048`, `B` = 16 MB) does manual
  blocking start to win — and even then, modestly.

**This is exactly why the same idea is worth 5–10× on a GPU.** A GPU has no large
automatic cache to fall back on: an SM has ~128 KB of L1/shared *shared between
1536 threads*, and nothing prefetches for you. There, the choice is manual tiling
or DRAM. Phase 3 exercise 04 rewrites this same loop nest with `__shared__`
memory, and there the speedup is unmissable.

So the takeaway from this exercise is really two things: **loop order is a huge,
free win everywhere**, and **manual blocking is only worth it when the hardware
stops caching for you** — which on a GPU is always.

## Your task

Open `main.cpp`:

1. **`matmulIKJ()`** — the reordered loop. Zero `C` first; this version
   accumulates.
2. **`matmulBlocked()`** — six loops: `ii, kk, jj` over tiles, then `i, k, j`
   inside a tile. Use `std::min(ii + tile, n)` for the bounds so non-multiple
   sizes work.
3. **Time both** and add them to the `ResultTable`.

## Build and run

```bash
cmake --build build --target p1_02_matmul_cache_blocking -j
./build/bin/p1/p1_02_matmul_cache_blocking
```

## Expected output

Measured on a 12-thread desktop CPU at `n = 1024`:

```
--- Performance ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
naive ijk (baseline)               1619.395         -       1.33     1.00x
ikj order                           154.051         -      13.94    10.51x
blocked ikj                         200.687         -      10.70     8.07x
blocked + OpenMP (12 threads)        42.458         -      50.58    38.14x

--- When blocking starts to matter ---
  n              B size     ikj (ms) blocked (ms)    speedup
  --------------------------------------------------------------
  256            0.2 MB          2.3          2.4      0.94x
  512            1.0 MB         19.3         19.6      0.98x
  1024           4.0 MB        155.7        208.6      0.75x
  2048          16.0 MB       2201.1       2067.0      1.06x
```

Read that second table carefully — it is the real content of this exercise.
Blocking is a *loss* until the working set exceeds the cache, then it turns into
a win. The exact crossover point is a property of **your** machine.

## Key takeaways

- **Loop order changes performance by 10× with identical arithmetic.** Memory
  layout, not FLOP count, is the bottleneck.
- The innermost loop should be **stride 1** — for the cache *and* for the
  vectoriser. Most of the `ikj` win is actually auto-vectorisation.
- **Manual blocking only pays when the hardware stops caching for you.** On a
  big-L3 CPU that means large `n`; on a GPU it means *always*.
- There is an optimum tile size, and it is a property of the *cache*, not of the
  algorithm. Measure it, do not guess it.
- Multi-core is another ~4× on top — and still 100× short of what the GPU will do
  in Phase 3.

## Going further

- Rerun with `n = 2048`. The gap between naive and blocked widens — why?
- Add `-march=native` and compare. How much of the `ikj` win was vectorisation?
  (`g++ -O3 -march=native -fopt-info-vec`)
- Compare your best result against OpenBLAS `sgemm`. Expect to be 5–20× off; the
  remaining gap is register blocking, packing and hand-written microkernels.
- Read Simon Boehm's ["How to optimize a CUDA matmul kernel"](https://siboehm.com/articles/22/CUDA-MMM)
  — the same progression, on a GPU.
