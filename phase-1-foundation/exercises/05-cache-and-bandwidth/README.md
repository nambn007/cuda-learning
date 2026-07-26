<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 05 - Cache lines and memory bandwidth

> **Phase 1 · Foundation** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [01](../01-matmul-naive-cpu/), [02](../02-matmul-cache-blocking/)

## Goal

Measure — not read about — the three numbers that decide how fast almost any real
program runs: the **cache line size**, the **cache level boundaries**, and the
**sustained DRAM bandwidth** of your machine. This is the CPU rehearsal for
[Phase 3 exercise 01 (memory coalescing)](../../../phase-3-intermediate/exercises/01-memory-coalescing/).

## Background

The single fact behind this exercise:

> **Memory is not moved in bytes, it is moved in blocks.**

x86 CPUs move 64-byte **cache lines**. NVIDIA GPUs move 32-byte **sectors**,
grouped into 128-byte transactions. Ask for one `float` and the hardware fetches
the whole block regardless. If you then use only that one float, you have thrown
away 15/16 of your bandwidth — and bandwidth, not arithmetic, is what limits most
kernels.

**Experiment 1 — the stride sweep.** Sum an array touching every `stride`-th
element. Report two bandwidths: *useful* (bytes the program asked for) and *DRAM*
(whole lines the bus actually moved). As stride grows from 1 to 16 floats, useful
bandwidth collapses by ~16× while DRAM bandwidth barely changes. The bus is just
as busy; it is moving data you discard. Past stride 16 nothing more is lost —
you were already wasting an entire line per access.

**Experiment 2 — the working-set sweep.** Re-read arrays of growing size. Small
arrays stay in L1, then L2, then L3, then fall out to DRAM. The steps in the GB/s
column *are* your cache sizes. You can read them off the graph without ever
looking up your CPU's spec sheet.

**Experiment 3 — STREAM triad.** `a[i] = b[i] + s*c[i]`: three arrays, one pass,
zero reuse. The standard way to measure sustained bandwidth. Compare your result
to the GPU figure printed by any Phase 2 exercise — that ratio is the reason this
curriculum exists.

## Your task

Open `main.cpp`:

1. **`strideSum()`** — sum `data[0], data[stride], data[2*stride], ...`
   **Return** the sum. A benchmark whose result is unused gets deleted by the
   optimiser and you measure an empty loop.
2. **`triad()`** — `a[i] = b[i] + scalar * c[i]`.
3. **Stride sweep** — time each stride in `kStrides` and print both bandwidth
   columns using `usefulBytes()` / `dramBytes()` from `reference.h`.
4. **Triad benchmark** — 3 arrays × 4 bytes = 12 bytes of traffic per element.

## Build and run

```bash
cmake --build build --target p1_05_cache_and_bandwidth -j
./build/bin/p1/p1_05_cache_and_bandwidth
```

## Expected output

```
--- Stride sweep ---
  stride     time (ms)    useful GB/s      DRAM GB/s  efficiency
  ------------------------------------------------------------------
  1              12.50           21.5           21.5        100%
  2               9.80           13.7           27.4         50%
  4               8.90            7.5           30.2         25%
  8               8.60            3.9           31.2         13%
  16              8.50            2.0           31.5          6%
  32              4.30            2.0           31.2          6%
```

Note how the **useful** column halves at every step while **DRAM** stays flat.
Your absolute numbers will differ; the *shape* is the result.

## Key takeaways

- The hardware moves **fixed-size blocks**. Partial use of a block is wasted
  bandwidth that no amount of clever arithmetic recovers.
- Effective bandwidth is `useful bytes / time`, and it is the number that matters
  — a kernel can saturate the bus while delivering 6% useful throughput.
- **Cache levels are measurable**, not theoretical. Sweep the working set and read
  them off.
- On a GPU the block is a 32-byte sector and the "stride" is the gap between
  addresses touched by neighbouring threads in a warp. Same graph, same lesson.

## Going further

- Add a random-access variant (shuffle the indices). How much worse is it than
  stride 128, and why?
- Repeat the triad with 2, 4 and 8 threads. Bandwidth saturates well before you
  run out of cores — the memory controller, not the CPU, is the limit.
- Read Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007) —
  still the definitive treatment.
