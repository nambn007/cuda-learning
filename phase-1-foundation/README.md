<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 1 · Foundation

> **3–4 weeks · ⭐ · 8 exercises** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## Why this phase exists

You cannot optimise a GPU kernel if you cannot explain why a CPU loop is slow.
Every idea here reappears on the GPU under a different name:

| Learned here | Reappears as |
|---|---|
| Row-major layout, stride-1 access | Memory coalescing (Phase 3/01) |
| Cache blocking | Shared-memory tiling (Phase 3/04) |
| Cache line | Memory sector / transaction (Phase 3/01) |
| AVX2 lanes | Warp lanes (Phase 2/01) |
| Multiple accumulators | Instruction-level parallelism, occupancy (Phase 3/15) |
| Register spilling | `__launch_bounds__`, local memory (Phase 4/11) |
| Arithmetic intensity, roofline | Nsight Compute's roofline chart (Phase 3/16) |
| RAII, move semantics | The device-buffer class (Phase 2/12) |
| Allocation is expensive | Memory pools, `cudaMallocAsync` |

Skip this phase only if you can already state your machine's cache line size,
sustained DRAM bandwidth and ridge point from memory.

## Exercises

| # | Exercise | Difficulty | What you build |
|---|---|---|---|
| 01 | [matmul-naive-cpu](exercises/01-matmul-naive-cpu/) | ⭐ | The baseline every later matmul is compared against, plus an honest benchmark |
| 02 | [matmul-cache-blocking](exercises/02-matmul-cache-blocking/) | ⭐⭐ | `ikj` reordering and tiling — 10× from memory layout alone |
| 03 | [memory-pool-allocator](exercises/03-memory-pool-allocator/) | ⭐⭐ | An arena and a pool allocator with an intrusive free list |
| 04 | [modern-cpp-toolkit](exercises/04-modern-cpp-toolkit/) | ⭐⭐ | `Matrix<T>`: RAII, moves, templates, lambdas — proven zero-cost |
| 05 | [cache-and-bandwidth](exercises/05-cache-and-bandwidth/) | ⭐⭐ | Measure your cache line, cache levels and DRAM bandwidth |
| 06 | [simd-intrinsics](exercises/06-simd-intrinsics/) | ⭐⭐⭐ | AVX2 by hand; why SIMD does nothing for memory-bound code |
| 07 | [roofline-model](exercises/07-roofline-model/) | ⭐⭐⭐ | Your machine's roofline, from measurements |
| 08 | [gpu-device-query](exercises/08-gpu-device-query/) | ⭐ | Your GPU's peaks, ridge point and occupancy budget |

Exercises 01–07 are pure C++ and need no GPU. Exercise 08 does.

## Build and run

```bash
./scripts/build.sh
ctest --test-dir build -L p1 --output-on-failure
```

## Checklist

- [ ] I can explain why `ijk` matmul is ~10× slower than `ikj`
- [ ] I know my cache line size **because I measured it**
- [ ] I know my sustained DRAM bandwidth and my single-core ridge point
- [ ] I can write an allocator with no individual `free`, and say when that is fine
- [ ] I can write a movable RAII wrapper and prove the move does not allocate
- [ ] I can explain why SIMD helps a polynomial but not a SAXPY
- [ ] I can classify a kernel as memory or compute bound before touching it
- [ ] I know my GPU's peak bandwidth, peak FP32, ridge point and registers/thread

## Checkpoint

Write down, for your own machine: cache line size, sustained bandwidth,
single-core ridge point, and the four GPU numbers from exercise 08. Phase 3
assumes you have them.
