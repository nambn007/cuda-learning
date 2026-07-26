<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 2 · CUDA Fundamentals

> **4–6 weeks · ⭐⭐ · 12 exercises** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## What this phase is for

Writing kernels that are **correct**, and measuring what they actually cost.
Optimisation is Phase 3. Here the goals are: the execution model, device memory
management, index arithmetic, and honest reporting of effective bandwidth.

One result from this phase surprises nearly everyone and is worth stating up
front: **for simple element-wise work the GPU is slower than the CPU end to end**,
because the PCIe round trip costs more than the arithmetic saves. Knowing exactly
when that flips is a large part of being useful with a GPU.

## Exercises

| # | Exercise | Difficulty | Core idea |
|---|---|---|---|
| 01 | [hello-cuda](exercises/01-hello-cuda/) | ⭐ | `__global__`, launch syntax, global id, bounds checks, async errors |
| 02 | [thread-indexing](exercises/02-thread-indexing/) | ⭐ | 1D/2D/3D grids and the grid-stride loop |
| 03 | vector-add | ⭐ | The full memory cycle — and why PCIe often wins |
| 04 | saxpy | ⭐⭐ | Effective bandwidth as *the* metric for memory-bound kernels |
| 05 | error-handling | ⭐⭐ | Sticky errors, sync vs async failures, `compute-sanitizer` |
| 06 | matrix-add-2d | ⭐ | 2D launches on real 2D data |
| 07 | matmul-naive | ⭐⭐ | The GPU baseline for every later matmul |
| 08 | transpose-naive | ⭐⭐ | A kernel that is correct and slow — the setup for Phase 3 |
| 09 | rgb-to-grayscale | ⭐⭐ | Image data, PPM I/O, per-pixel parallelism |
| 10 | box-blur | ⭐⭐ | Stencils, halos, boundary handling |
| 11 | unified-memory | ⭐⭐ | `cudaMallocManaged`, page migration, prefetching |
| 12 | device-buffer-class | ⭐⭐ | RAII around device memory — Phase 1/04 applied |

## Build and run

```bash
./scripts/build.sh
ctest --test-dir build -L p2 --output-on-failure
```

## Checklist

- [ ] I can write `blockIdx.x * blockDim.x + threadIdx.x` without thinking
- [ ] I bounds-check every kernel, and I know what happens when I do not
- [ ] I check launch errors *and* execution errors, and know the difference
- [ ] I can allocate, copy and free device memory without leaking
- [ ] I use grid-stride loops by default
- [ ] I report effective bandwidth, not milliseconds
- [ ] I can explain when a GPU is *not* worth it for a given kernel
- [ ] I have used `compute-sanitizer` on a kernel that was actually broken

## Checkpoint

Given a new element-wise or 2D problem, you can write a correct kernel, verify it
against a CPU reference, report its effective bandwidth as a percentage of your
GPU's peak, and say whether the transfers make the whole thing worthwhile.
