<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 01 - Naive matrix multiplication on the CPU

> **Phase 1 · Foundation** | Difficulty: ⭐ | Time: ~45 min | Prerequisites: none

## Goal

Write the reference implementation that every later exercise is compared against,
and — more importantly — learn how to **measure** it honestly. By the end you can
state your result in GFLOP/s and explain why a single `clock()` call around a
function is not a benchmark.

## Background

Matrix multiplication is the running example of this whole curriculum. It appears
again as a naive CUDA kernel (Phase 2), a shared-memory tiled kernel (Phase 3),
a multi-GPU kernel (Phase 4) and a cuBLAS/Tensor-Core comparison (Phase 5). Having
one trustworthy CPU baseline makes all those numbers meaningful.

Two ideas from this exercise carry all the way to the GPU:

**Row-major layout.** `C[r][c]` lives at `C[r * N + c]`. Elements of a *row* are
adjacent in memory; elements of a *column* are `N` floats apart. Hardware moves
memory in cache lines (64 bytes = 16 floats on x86), so a stride-1 walk gets 16
useful floats per line while a stride-`N` walk gets 1. On a GPU the exact same
principle appears under the name *memory coalescing*.

**FLOP counting.** The inner statement `acc += A[i][k] * B[k][j]` is one multiply
and one add: 2 floating point operations. It runs `N³` times, so the algorithm
performs `2N³` FLOPs regardless of how you write it. Dividing by elapsed time
gives a rate you can compare against the hardware's peak — and against the GPU
later on.

Why the median of several runs, not a single measurement? The first run pays for
page faults on the freshly allocated output buffer and for the CPU still being at
its idle clock frequency. Later runs may be interrupted by the scheduler. Warmup
iterations remove the first effect; taking the median removes the second.

## Your task

Open `main.cpp` and complete three TODOs:

1. **`matmulNaive()`** — the triple loop `i, j, k`. Accumulate into a local
   `float acc` inside the `j` loop, not directly into `C[i * n + j]`.
2. **Measure it** with `timeCpuMs(iterations, callable)` from `common/timer.h`.
3. **Report it** — the `ResultTable` turns your time plus `matmulFlops(n)` into
   GFLOP/s.

`reference.h` holds the golden result used for checking. For this first exercise
it is the same algorithm you are writing, so try it yourself before peeking.

## Build and run

```bash
cmake --build build --target p1_01_matmul_naive_cpu -j
./build/bin/p1/p1_01_matmul_naive_cpu

# Reference solution
cmake --build build --target p1_01_matmul_naive_cpu_sol -j
./build/bin/p1/p1_01_matmul_naive_cpu_sol
```

## Expected output

```
============================================================
  Phase 1 / 01 - Naive matrix multiplication
============================================================
  Matrix size: 512 x 512 (1.0 MB per matrix)

--- Correctness ---
  [PASS] C == A * B (n=262144, max abs err ..., max rel err ...)

--- Performance ---
CPU matrix multiplication
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
naive ijk                           ~100        ~0.03      ~2.7      1.00x
```

Absolute numbers depend on your CPU; anything in the 1–5 GFLOP/s range is normal
for this loop order.

## Key takeaways

- Row-major means **stride-1 along rows**; a column walk touches a new cache line
  every element.
- Report **rates** (GFLOP/s, GB/s), not raw milliseconds — rates are comparable
  across problem sizes and machines.
- A benchmark needs **warmup + repetition + median**, or you are measuring noise.
- Check correctness *before* performance, and never compare floats with `==`.

## Going further

- Try `n = 1024` and `n = 2048`. Time grows as `N³`, so 2× the size is 8× the
  work — does your measured time match?
- Look at the generated assembly: `g++ -O3 -S -march=native`. Did the compiler
  vectorise the inner loop? Why not?
- Read *Computer Organization and Design* (Patterson & Hennessy), chapter 5, on
  the memory hierarchy.
