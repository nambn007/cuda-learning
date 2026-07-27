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
| 01 ✅ | [pinned-and-streams](exercises/01-pinned-and-streams/) | ⭐⭐⭐ | Pinned memory, and turning `H2D + kernel + D2H` into `max(...)` |
| 02 ⚠️ | [cuda-graphs](exercises/02-cuda-graphs/) | ⭐⭐⭐⭐ | Replay a recorded DAG with one launch; remove the CPU from the inner loop |
| 03 ⚠️ | [atomics-and-ordering](exercises/03-atomics-and-ordering/) | ⭐⭐⭐⭐⭐ | Privatisation, `atomicCAS`, and why `__threadfence` is not optional |

✅ verified on an RTX 3060 · ⚠️ code complete, not yet run on the reference machine
(so its README carries no measured numbers)

### Deferred

Specified in [docs/ROADMAP.md](../docs/ROADMAP.md) but not written yet:
**events-and-sync**, **cooperative-groups**, **dynamic-parallelism**,
**multi-gpu-basics**, **multi-gpu-matmul**, **lock-free-queue**,
**register-pressure**, **ptx-and-sass**, **persistent-kernel**.

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
