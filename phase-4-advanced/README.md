<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 4 · Advanced

> **8–10 weeks · ⭐⭐⭐⭐ · 13 exercises** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## What this phase is for

Phase 3 made a single kernel fast. Phase 4 is about everything *around* the
kernel: overlapping work so the GPU is never idle, scaling beyond one device,
synchronising correctly between threads, and reading the machine code the
compiler actually produced.

A theme worth naming: **at this level, correctness gets harder.** Streams
introduce ordering bugs, atomics introduce memory-ordering bugs, and multi-GPU
introduces both. `compute-sanitizer --tool racecheck` becomes a routine tool
rather than a last resort.

## Exercises

| # | Exercise | Difficulty | Core idea |
|---|---|---|---|
| 01 | pinned-memory | ⭐⭐ | Pageable vs pinned transfer bandwidth |
| 02 | streams-basics | ⭐⭐⭐ | Concurrency, and the default-stream trap |
| 03 | streams-pipeline | ⭐⭐⭐ | Overlap H2D → kernel → D2H in chunks |
| 04 | events-and-sync | ⭐⭐⭐ | `cudaEvent`, `cudaStreamWaitEvent`, dependency graphs by hand |
| 05 | cuda-graphs | ⭐⭐⭐⭐ | Capture a pipeline, eliminate per-launch overhead |
| 06 | cooperative-groups | ⭐⭐⭐⭐ | Tiled partitions, grid-wide synchronisation |
| 07 | dynamic-parallelism | ⭐⭐⭐⭐ | Kernels launching kernels (CDP2 semantics, CUDA 12+) |
| 08 | multi-gpu-basics | ⭐⭐⭐ | Device enumeration, peer access, P2P copies |
| 09 | multi-gpu-matmul | ⭐⭐⭐⭐ | Splitting work across devices |
| 10 | lock-free-queue | ⭐⭐⭐⭐⭐ | `atomicCAS`, `__threadfence`, memory ordering on a GPU |
| 11 | register-pressure | ⭐⭐⭐ | `__launch_bounds__`, spilling, the occupancy trade-off |
| 12 | ptx-and-sass | ⭐⭐⭐⭐ | `cuobjdump`, `nvdisasm`, inline PTX |
| 13 | persistent-kernel | ⭐⭐⭐⭐ | Megakernels and on-device producer/consumer |

Exercises 08 and 09 detect a single-GPU machine and skip cleanly — they are not
failures.

## Build and run

```bash
./scripts/build.sh
ctest --test-dir build -L p4 --output-on-failure
```

## Checklist

- [ ] I can overlap a transfer with a kernel and prove it on a timeline
- [ ] I know why the default stream serialises everything, and how to avoid it
- [ ] I can express a dependency between two streams without a full sync
- [ ] I can convert a repeated launch sequence into a CUDA graph
- [ ] I can explain when `__threadfence()` is required and when it is not
- [ ] I can read a SASS listing and find the instruction that stalls
- [ ] I have used `__launch_bounds__` to trade registers for occupancy, and measured both
- [ ] I have found a real race with `compute-sanitizer --tool racecheck`

## Checkpoint

You can take a multi-stage pipeline, overlap its stages, keep every available GPU
busy, and justify each synchronisation point in it.
