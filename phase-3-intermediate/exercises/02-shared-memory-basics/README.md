<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 02 - Shared memory and `__syncthreads`

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐ | Time: ~2 h | Prerequisites: [01](../01-memory-coalescing/) | **Requires a GPU**

## Goal

Learn the tool that the rest of Phase 3 is built on — a small, fast, block-private
scratchpad you manage by hand — including the barrier it requires and the
occupancy it costs.

## Background

Shared memory is on-chip storage private to a thread block. Roughly **100× lower
latency than global memory**, and there is very little of it: 48 KB per block by
default, 100 KB per SM on an RTX 3060, divided among every block resident there.

It solves exactly two problems, and every later exercise in Phase 3 is one of them:

| Problem | Fix | Where |
|---|---|---|
| **Reuse** — several threads need the same value | read it from global once, serve it from shared | tiled matmul (04), convolution (14), reduction (06) |
| **Reordering** — an access cannot be coalesced | stage it in shared, where coalescing rules do not apply, and rearrange there | transpose (05), histogram (09) |

**`__syncthreads()`** is a barrier: no thread passes until every thread arrives,
and all shared-memory writes issued before it become visible to all threads after
it.

### Two rules that catch people

1. **Every thread in the block must reach every `__syncthreads()`.** One inside
   `if (tid < 100)` in a 256-thread block is undefined behaviour — typically a hang.
2. **Since Volta, threads in a warp are *not* implicitly synchronised.** Code that
   omitted the barrier and "worked" on Kepler because a warp moved in lockstep is
   broken on every modern GPU. If you mean warp-level synchronisation, write
   `__syncwarp()`.

**Static vs dynamic.** `__shared__ float tile[256]` fixes the size at compile time.
`extern __shared__ float tile[]` plus `kernel<<<grid, block, bytes>>>` makes it a
launch parameter — use that when the tile size is a runtime tuning knob.

## Your task

Open `main.cu` and write six kernels:

1. **`reverseGlobal`** — no shared memory. This baseline exists to make a point:
   shared memory is not always the answer.
2. **`reverseShared`** — staged through a tile.
3. **`reverseSharedBroken`** — the same, with the barrier deleted. **Predict what
   fraction of elements will be wrong before you run it.**
4. **`reverseSharedDynamic`** — dynamic shared memory.
5. **`broadcastGlobal` / `broadcastShared`** — the reuse pattern. Predict the
   speedup from the read counts, then measure it.

Then tabulate blocks-per-SM against shared-memory-per-block to see the occupancy
cost.

## Build and run

```bash
cmake --build build --target p3_02_shared_memory_basics -j
./build/bin/p3/p3_02_shared_memory_basics
compute-sanitizer --tool racecheck ./build/bin/p3/p3_02_shared_memory_basics
```

## Expected output

RTX 3060, 4M floats, 256-thread blocks:

```
--- Correctness ---
  [PASS] reverse via global memory
  [PASS] reverse via shared memory
  [PASS] reverse via dynamic shared memory

--- What happens without __syncthreads() ---
  2001632 of 4194304 elements wrong (47.7%)

--- Reuse ---
Variant                           Time (ms)   Speedup
------------------------------------------------------
global memory                         1.336     1.00x
shared memory                         0.583     2.29x

  Global reads per output element
    naive  : 257      shared : 1      ratio : 257x fewer reads
    measured : 2.29x faster
```

### Two things to sit with

**47.7% wrong.** Not a subtle corruption — nearly half the array. Thread `tid`
reads `tile[count-1-tid]`, written by a *different* thread, very possibly in a warp
that has not run yet. Without the barrier there is no ordering between them at all.

**257× fewer reads, 2.29× faster.** The gap is L1: the block fits in cache, so the
naive version's repetition is mostly absorbed. Shared memory still wins because it
is **explicit** — guaranteed on-chip, guaranteed not to be evicted by another
block's traffic. Cache is a hope; shared memory is a promise.

## Key takeaways

- **Shared memory is for reuse and for reordering.** If your kernel does neither,
  it will not help.
- **The barrier is not optional**, and a missing one can produce *right* answers on
  your machine and wrong ones elsewhere. Use `compute-sanitizer --tool racecheck`.
- **Warps are not implicitly synchronised on modern hardware.**
- **Shared memory costs occupancy.** Ask for 48 KB per block and only 2 blocks fit
  per SM, leaving far fewer warps to hide latency with. That trade-off is
  [exercise 15](../15-occupancy-tuning/).

## Going further

- Run the broken kernel with a 32-thread block. Does it still fail? Why is
  "it works" the most dangerous outcome here?
- Replace `__syncthreads()` with `__syncwarp()` in a 32-thread block. When is that
  legitimate?
- Sweep the block size 32→1024 for the broadcast kernel. Where is the optimum, and
  which constraint sets it?
- **Next:** [03 bank-conflicts](../03-bank-conflicts/) — shared memory has its own
  access rules, and breaking them costs up to 32×.
