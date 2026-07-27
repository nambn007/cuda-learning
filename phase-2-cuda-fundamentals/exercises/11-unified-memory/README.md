<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 11 - Unified memory: one pointer, and what it costs

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [03](../03-vector-add/) | **Requires a GPU**

## Goal

Trade explicit `cudaMemcpy` for a single pointer that works everywhere, measure
exactly what that convenience costs, and learn the one thing managed memory can do
that explicit allocation cannot do at all.

## Background

`cudaMallocManaged` returns **one pointer valid on both the host and the device**.
No `cudaMemcpy`, no parallel `h_`/`d_` variables, no confusion about which one you
are holding.

The price: the driver must move the pages for you, and it only learns where they
are needed by trapping a **page fault**. On Pascal and later:

```
host writes   →  pages migrate to host memory
kernel reads  →  every first touch faults; the driver migrates 4 KB;
                 the warp stalls
host reads    →  they all migrate back
```

**Fault-driven migration at 4 KB granularity is far slower than one bulk DMA.**
The fix is to tell the driver what you already know:

| Call | Effect |
|---|---|
| `cudaMemPrefetchAsync(p, bytes, device)` | move it now, in bulk |
| `cudaMemAdvise(..., SetReadMostly)` | replicate read-only data on both sides |
| `cudaMemAdvise(..., SetPreferredLocation)` | pin the home node |

**Oversubscription** is the feature with no explicit-API equivalent. You can
allocate *more managed memory than the GPU physically has*, and the driver pages it
in and out. That is what makes it possible to run a model larger than VRAM. With
`cudaMalloc` you would have to tile the problem by hand.

## Your task

Open `main.cu` and run five experiments.

> **The rule that makes this exercise meaningful:** every variant must do exactly
> the same work — *host writes both arrays → GPU computes → host reads back*.
> Leaving the host-side initialisation out of one variant and not the others is the
> easiest way to get a meaningless answer, because managed memory's entire cost is
> the movement that the initialisation triggers.
>
> Also check correctness with a **separate single call**, not inside the timing
> loop — the loop runs the body several times and this kernel accumulates into `y`.

1. **Explicit** `cudaMalloc` + `cudaMemcpy` — the baseline.
2. **Managed, no hints.**
3. **Managed + `cudaMemPrefetchAsync`** before each side touches the data.
4. **Kernel alone**, with the data already resident.
5. **Oversubscription** — allocate 1.5× `totalGlobalMem` both ways and read the
   two error codes.

## Build and run

```bash
cmake --build build --target p2_11_unified_memory -j
cd build/testrun && ../bin/p2/p2_11_unified_memory
```

## Expected output

RTX 3060, 256 MB per array:

```
--- Comparison ---
Variant                           Time (ms)   Speedup
------------------------------------------------------
explicit malloc + memcpy            153.851     1.00x
managed, no hints                   260.437     0.59x
managed + prefetch                  173.759     0.89x

  Managed memory without hints is 1.69x the explicit version.
  With prefetching it is 1.13x.

--- Kernel time with data already on the device ---
  kernel alone                 3.109 ms
  259.0 GB/s  (72% of peak)

--- Oversubscription ---
  This GPU has 11.6 GB. Trying to allocate 17.4 GB...
  cudaMalloc                   cudaErrorMemoryAllocation
  cudaMallocManaged            cudaSuccess
  [PASS] managed memory can exceed device memory
```

Three conclusions, all measured:

1. **Naive managed memory costs 1.69×** — real, but not catastrophic.
2. **Prefetching recovers almost all of it** (1.13×). The data movement is
   *identical*; only the granularity changes.
3. **Once resident, managed memory is exactly as fast** as explicit memory — it is
   the same physical memory.

## Key takeaways

- **Managed memory is not slower memory.** It is the same DRAM. What costs is
  *fault-driven migration*, and a prefetch removes it.
- **Always pair `cudaMallocManaged` with `cudaMemPrefetchAsync`** once you know the
  access pattern. The convenience is free; the ignorance is not.
- **Oversubscription has no alternative.** If the working set exceeds VRAM,
  managed memory is the only thing that runs at all.
- **Fair benchmarks require identical work.** Getting this wrong is easy and the
  resulting number looks plausible.

### When to use which

**Managed:** prototyping; irregular or pointer-chasing structures (trees, graphs)
where you cannot say in advance which pages are needed; working sets larger than
VRAM.

**Explicit:** production kernels with a known regular access pattern; anything that
overlaps transfer with compute using streams (Phase 4/03).

## Going further

- Add `cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly, device)` for the
  read-only array. Does it help?
- Profile with `nsys profile --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true`
  and count the faults in the naive version.
- Touch the *whole* oversized allocation. How does throughput change when the
  working set exceeds VRAM?
- Read the [Programming Guide on Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming)
  and the NVIDIA blog post *Maximizing Unified Memory Performance*.
