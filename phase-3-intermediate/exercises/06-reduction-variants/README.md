<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 06 - Parallel reduction, seven ways

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~3 h | Prerequisites: [02](../02-shared-memory-basics/), [03](../03-bank-conflicts/), [Phase 1/07](../../../phase-1-foundation/exercises/07-roofline-model/) | **Requires a GPU**

## Goal

Turn *n* values into one, seven times. Every version is correct; each is faster
than the last for a **different** reason. Naming the reason is the exercise — this
is the best optimisation drill on a GPU.

## Background

A reduction reads every input once and writes almost nothing, so it is purely
memory bound. Its ceiling is a memory copy: ~330 GB/s on an RTX 3060, or **0.373 ms
for 128 MB**. Every variant is scored against that floor, not against the previous
variant.

| Version | Fixes |
|---|---|
| v1 interleaved, `tid % (2*s) == 0` | — (baseline) |
| v2 interleaved, `index = 2*s*tid` | **warp divergence** |
| v3 sequential addressing | **bank conflicts** |
| v4 first add during load | **idle threads** |
| v5 unroll last warp (`__shfl_down_sync`) | **barriers** |
| v6 grid-stride sweep | changes the **shape** |
| v7 grid-stride, 4 accumulators | **the dependency chain** |

**On the last warp:** the classic trick was to drop `__syncthreads()` and mark the
shared array `volatile`, relying on warp lockstep. That has been **unsafe since
Volta**, where threads in a warp diverge independently. Use `__shfl_down_sync`,
which moves values directly between lanes' registers.

## Your task

Open `main.cu` and write all seven. **Predict before each measurement** — and in
particular, before v6: it looks strictly better than v5. Is it?

Only attempt v7 after you have measured v6 and been surprised.

## Build and run

```bash
cmake --build build --target p3_06_reduction_variants -j
./build/bin/p3/p3_06_reduction_variants
```

## Expected output

RTX 3060, 32M floats:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
v1 interleaved, divergent             1.805     74.35     1.00x
v2 interleaved, no divergence         1.291    103.95     1.40x
v3 sequential addressing              1.245    107.79     1.45x
v4 first add during load              0.643    208.71     2.81x
v5 unrolled last warp                 0.398    336.95     4.53x
v6 grid-stride + shuffle              0.483    277.69     3.74x     ← slower!
v7 grid-stride, 4 accumulators        0.396    338.69     4.56x

  Best variant : v7   0.396 ms  (338.7 GB/s, 94% of peak)
  Floor        :      0.373 ms  (reading 128 MB once at peak)
```

### The surprise: v6 is 1.21× *slower* than v5

This is the most useful result in the exercise, and the cause is one you have
already measured — on a **CPU**, in
[Phase 1/07](../../../phase-1-foundation/exercises/07-roofline-model/).

v6's sweep accumulates into **one variable**. Each thread performs ~580 adds and
every one waits for the previous result. An FMA issues every cycle but takes
several to complete, so a single dependency chain runs at a fraction of peak. **The
kernel is latency bound, not bandwidth bound**, and no amount of memory tuning
fixes it.

v7 uses **four independent accumulators** — the identical fix — and reaches 94% of
peak.

## Key takeaways

- **Divergence, bank conflicts, idle threads, barriers and dependency chains are
  five separate problems.** Fixing one does not fix the others, and each has its
  own signature.
- **Score against the floor, not the previous version.** "1.21× faster" is
  meaningless; "94% of peak" tells you to stop.
- **Latency-boundedness looks like a memory problem and is not.** The same
  single-accumulator mistake costs you on a CPU and on a GPU.
- **Warps are not implicitly synchronised since Volta.** `volatile` + no barrier is
  a bug, not an optimisation.
- **Every variant produces a slightly different sum.** Float addition is not
  associative. If you need reproducibility, you need a fixed order or Kahan
  summation, and both cost performance.

## Going further

- Sweep the block size 64→1024 for v7. Where is the optimum?
- Try 2 and 8 accumulators. Where does adding more stop helping, and why?
- Implement a single-kernel reduction using an atomic for the final combine.
  Faster, or does the contention cost more? (Exercise 08 covers atomics.)
- Replace v7 with `cub::BlockReduce` (Phase 5/02) and compare. You should be close
  — CUB emits essentially this, tuned per architecture. **Write it once by hand to
  understand it, then never again.**
