<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 3 · Intermediate

> **6–8 weeks · ⭐⭐⭐ · 16 exercises** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## What this phase is for

This is the heart of the curriculum: where "it works" becomes "it is fast", and
where everything you measured in Phase 1 pays off.

The order is deliberate. Memory comes first (01–05) because on a GPU almost
everything is memory bound — your GPU's ridge point from
[Phase 1/08](../phase-1-foundation/exercises/08-gpu-device-query/) is around 37
FLOP/byte, and a vector add manages 0.08. Then the parallel patterns (06–14) that
every real kernel is built from, and finally the tuning and profiling skills
(15–16) that tell you which of them to reach for.

## Exercises

**Status: 7 of 16 implemented.** Exercises 01–07 are complete and verified on an
RTX 3060. Exercises 08–16 are specified below but **not yet written** — the folders
are absent from the tree, and `scripts/new-exercise.sh` will scaffold them when you
or a contributor get to them.

| # | Exercise | Difficulty | Core idea |
|---|---|---|---|
| 01 ✅ | memory-coalescing | ⭐⭐⭐ | The single biggest GPU performance factor, measured |
| 02 ✅ | shared-memory-basics | ⭐⭐ | `__shared__`, `__syncthreads()`, cooperation within a block |
| 03 ✅ | bank-conflicts | ⭐⭐⭐ | 32 banks; why padding by one element fixes a 32× slowdown |
| 04 ✅ | tiled-matmul | ⭐⭐⭐ | Phase 1/02's blocking in shared memory — and why the win comes from register tiling, not the tiling itself |
| 05 ✅ | transpose-optimized | ⭐⭐⭐ | Coalesced read *and* write through a shared tile |
| 06 ✅ | reduction-variants | ⭐⭐⭐ | Six kernels, each faster than the last — the classic study |
| 07 ✅ | warp-shuffle | ⭐⭐⭐ | `__shfl_down_sync`, ballot, vote — registers instead of shared memory |
| 08 🚧 | atomics | ⭐⭐ | `atomicAdd`, `atomicCAS`, contention, custom float atomics |
| 09 🚧 | histogram | ⭐⭐⭐ | Privatisation: global contention becomes shared-memory contention |
| 10 🚧 | scan-hillis-steele | ⭐⭐⭐ | Inclusive scan within a block |
| 11 🚧 | scan-blelloch | ⭐⭐⭐⭐ | Work-efficient scan, arbitrary length, multi-block |
| 12 🚧 | stream-compaction | ⭐⭐⭐ | Scan + scatter — the backbone of filtering on a GPU |
| 13 🚧 | conv-1d-constant | ⭐⭐ | `__constant__` memory and its broadcast cache |
| 14 🚧 | conv-2d-shared | ⭐⭐⭐ | 2D stencil with a haloed shared tile |
| 15 🚧 | occupancy-tuning | ⭐⭐⭐ | Registers vs shared memory vs block size; when high occupancy hurts |
| 16 🚧 | profiling-nsight | ⭐⭐⭐ | Nsight Systems and Nsight Compute on your own kernels |

## Build and run

```bash
./scripts/build.sh
ctest --test-dir build -L p3 --output-on-failure
```

Exercise 16 needs `ncu` (ships with the CUDA toolkit) and ideally `nsys`.

## Checklist

- [ ] I can explain coalescing to someone else, with numbers I measured
- [ ] I can spot a bank conflict by reading a kernel's indexing
- [ ] My tiled matmul beats my naive matmul by at least 5×
- [ ] I can write a reduction that reaches >80% of peak bandwidth
- [ ] I use warp shuffles instead of shared memory where it applies
- [ ] I can implement a work-efficient scan for an arbitrary length
- [ ] I can read an Nsight Compute report and name the limiter
- [ ] I know a case where *lowering* occupancy made a kernel faster

## Checkpoint

Given a slow kernel you can profile it, name the limiter from the metrics, apply
the right fix, and prove the improvement with a before/after measurement.
