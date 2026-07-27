<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 02 - cuBLAS GEMM, and the gap to it

> **Phase 5 · Expert** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [Phase 3/04](../../../phase-3-intermediate/exercises/04-tiled-matmul/) | **Requires a GPU**
>
> ⚠️ **Not yet verified on the reference machine** — no measured numbers below.

## Goal

Phase 3/04 got a tiled matmul to ~19% of the card's FP32 peak. Measure the rest of
the distance, and learn the one API convention that trips up everybody.

## Background

### The column-major trap

cuBLAS inherits Fortran's convention: element `(i, j)` lives at `A[i + j*ld]`, not
`A[i*ld + j]`. Hand it a row-major matrix and it computes **the transpose of what
you wanted — silently, with no error**.

The fix moves no data at all:

```
C   = A * B        (row-major)
  is the same bytes as
C^T = B^T * A^T    (column-major)
```

So pass **B first and A second**, with `CUBLAS_OP_N` for both, and cuBLAS writes
exactly the row-major `C` you wanted. Only the argument order changes.

> If your result looks like the transpose of the expected answer, you have swapped
> exactly one thing.

### What the remaining gap is made of

Roughly in order of effort to close it: wider register tiles (8×8 per thread, not
4×1) · vectorised `float4` loads · double buffering · a second shared-memory
staging level · per-architecture tuning of every tile size · Tensor Cores where the
data type allows ([exercise 03](../03-tensor-core-wmma/)).

**Reaching 80% of cuBLAS by hand is an excellent result and a multi-week project.
Reaching 100% is somebody's full-time job.** Knowing the number is what lets you
decide whether to try.

## Your task

1. Bring your tiled + register kernel across from Phase 3/04.
2. Call `cublasSgemm` on row-major data using the identity above. Work out `m`,
   `n`, `k` and the three `ld` values by reasoning, not by permutation.
3. Compare as GFLOP/s, as a percentage of FP32 peak, and as a percentage of cuBLAS.
   Use cuBLAS as the correctness reference — there is no CPU golden at n = 2048.

## Build and run

```bash
cmake --build build --target p5_02_cublas_gemm -j
./build/bin/p5/p5_02_cublas_gemm
```

## Key takeaways

- **cuBLAS is column-major.** The swap trick costs nothing and is the standard way
  to use it from row-major C++.
- **Know your percentage of the library**, not just of peak. It tells you whether
  further work is worth it.
- Libraries win through many levels of the same ideas you already know, tuned per
  architecture.
