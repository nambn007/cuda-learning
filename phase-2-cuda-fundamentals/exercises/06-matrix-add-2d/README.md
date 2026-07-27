<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 06 - 2D matrix addition, and the axis order

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐ | Time: ~1 h | Prerequisites: [02](../02-thread-indexing/), [04](../04-saxpy/) | **Requires a GPU**

## Goal

Write the same kernel three ways — all correct, all producing identical numbers —
and discover that **swapping two lines costs 70% of your performance**. This is
your first encounter with the effect that Phase 3 spends five exercises on.

## Background

Adding two matrices is arithmetically identical to adding two vectors: row-major
storage is one contiguous array either way. So why launch a 2D grid at all?

**Convenience, mostly.** A stencil, a convolution or a transpose needs the
`(row, col)` of each element, and computing that from a flat index costs a
division. Use a 2D grid when the *kernel* needs 2D coordinates — not because the
data is drawn as a rectangle.

**What is not cosmetic is which axis you map to `x`.** Threads are linearised with
`x` fastest, so a warp of 32 threads covers `threadIdx.x = 0..15` for two values of
`threadIdx.y` (with a 16×16 block):

| Mapping | What one warp touches | Result |
|---|---|---|
| `x` = column | 16 consecutive columns × 2 adjacent rows — two runs of 64 contiguous bytes | a few sectors |
| `x` = row | 16 addresses `cols` floats apart | **16 separate 32-byte sectors**, 8 bytes used from each |

The swapped version requests roughly **4× the memory traffic** for exactly the same
result.

## Your task

Open `main.cu`:

1. **`matrixAddRowMajor()`** — `x` = column. Check both bounds.
2. **`matrixAddColumnMajor()`** — `x` = row, everything else identical.
   ⚠️ The swapped kernel needs its **grid** swapped too, or it will not cover the
   matrix.
3. **`matrixAddFlat()`** — a flat 1D grid-stride loop over `rows * cols`.
4. **Verify all three agree**, then time them.

Before you run it: **predict how much slower the swapped version will be.** Write
the number down, then measure. The gap between your guess and the result is the
lesson.

## Build and run

```bash
cmake --build build --target p2_06_matrix_add_2d -j
./build/bin/p2/p2_06_matrix_add_2d
```

## Expected output

RTX 3060, 4096×4096:

```
--- Correctness ---
  [PASS] row-major mapping (x = column)   (max abs err 0.000e+00)
  [PASS] column-major mapping (x = row)   (max abs err 0.000e+00)
  [PASS] flat 1D grid-stride              (max abs err 0.000e+00)

--- Performance ---
Variant                           Time (ms)      GB/s    Speedup
----------------------------------------------------------------
CPU (single thread)                  12.288     16.38      1.00x
GPU 2D, x = column                    0.605    332.81     20.31x
GPU 2D, x = row (swapped)             1.001    201.06     12.27x
GPU 1D flat grid-stride               0.629    320.21     19.54x

  x = column :   332.8 GB/s  (92% of peak)
  x = row    :   201.1 GB/s  (56% of peak)
  Swapping two lines cost 1.7x.
```

### Why 1.7× and not 4×?

**L2 cache.** This GPU has 2304 KB of it, and neighbouring blocks re-read sectors
that a previous block already pulled in, so the cache absorbs much of the waste.
Change the block shape to 256×1 and the cache stops helping — Phase 3 exercise 01
does exactly that and measures the full effect.

The honest lesson is therefore two-sided: the penalty is real and large, *and* the
hardware partially rescues you. Never assume either.

## Key takeaways

- **`x` must map to the contiguous axis.** Getting it backwards is correct and
  slow, and nothing warns you.
- **A 2D launch is a convenience, not a requirement.** The flat 1D version is
  within 4% here.
- **L2 hides part of a bad access pattern**, which is exactly why bad patterns
  survive in real code — they are slow, but not catastrophically so, until the
  working set grows.
- Predict, then measure. A wrong prediction is the most useful result you can get.

## Going further

- Rerun with `dim3 block(256, 1)`. The gap should widen sharply — why?
- Use `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` on both
  kernels and compare the sector counts directly.
- Try a non-square matrix (4096×1024). Does the penalty change?
- Read the [Best Practices Guide on coalesced access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory).
