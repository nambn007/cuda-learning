<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 01 - Pinned memory and stream overlap

> **Phase 4 · Advanced** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [Phase 2/03](../../../phase-2-cuda-fundamentals/exercises/03-vector-add/) | **Requires a GPU** | ✅ verified on an RTX 3060

## Goal

Phase 2/03 ended badly: for simple element-wise work the GPU *lost*, because the
PCIe round trip cost more than the arithmetic saved. This is the first real answer
to that.

## Background

**Pinned (page-locked) host memory.** Ordinary allocations can be swapped out, so
the driver cannot hand their addresses to the DMA engine — it stages the copy
through an internal pinned buffer first. `cudaMallocHost` gives memory the OS
cannot move, which the DMA engine reads directly.

It is also a **hard requirement for asynchrony**: `cudaMemcpyAsync` on *pageable*
memory silently behaves synchronously. A "pipeline" built on pageable memory
overlaps nothing at all — and looks correct while doing so.

**Streams.** A stream is an ordered queue of GPU work; different streams may run
concurrently. A GPU has separate copy engines per direction plus the SMs, so with
enough streams H2D, compute and D2H can all be in flight at once:

```
sequential  =  H2D + kernel + D2H
overlapped  →  max(H2D, kernel, D2H) + one chunk of latency
```

**The trap.** The default stream synchronises with every other stream. One launch
or copy that forgets its stream argument collapses the timeline — and the answer
stays correct, so no test catches it.

## Your task

Open `main.cu`:

1. Allocate pinned memory, compare its H2D bandwidth with a `std::vector`.
2. Build the sequential pipeline.
3. Build the chunked pipeline across `kStreams` streams. **Every** operation names
   its stream. Zero the output before verifying, so a pipeline that silently does
   nothing cannot pass by reusing the previous result.
4. Reproduce the default-stream trap by deleting the stream argument from the
   *kernel launch only*, and explain the size of the penalty you measure.
5. Confirm the overlap with `nsys profile --stats=true`. A correct result proves
   nothing about concurrency.

## Build and run

```bash
cmake --build build --target p4_01_pinned_and_streams -j
./build/bin/p4/p4_01_pinned_and_streams
nsys profile --stats=true ./build/bin/p4/p4_01_pinned_and_streams
```

## Expected output

RTX 3060, 256 MB per buffer, 2 copy engines:

```
--- Transfer bandwidth ---
pageable host -> device              33.248      8.07 GB/s     1.00x
pinned host -> device                21.711     12.36 GB/s     1.53x

--- Summary ---
sequential                                    44.228 ms     1.00x
overlapped, 4 streams                         26.939 ms     1.64x
overlapped, but one default-stream call       29.158 ms     1.52x

  kernel alone     :  1.79 ms
  sequential total : 44.23 ms
  overlapped total : 26.94 ms  (1.64x)
```

### Read the default-stream row carefully

The bug costs only **8%** here — which is *not* evidence that it is a mild bug. The
kernel is 1.79 ms out of 44 ms, so the transfers dominate and H2D still overlaps
with D2H even when the kernel does not. Make the kernel the expensive part and the
same bug costs you everything.

> **A small measured penalty does not mean a small bug.** It means this particular
> workload was not sensitive to it.

## Key takeaways

- **Pinned memory is ~1.5× faster and is a *requirement* for async copies**, not an
  optimisation. `cudaMemcpyAsync` on pageable memory is silently synchronous.
- **Overlap converts a sum into a maximum.** 1.64× here, and it grows with how well
  balanced the three stages are.
- **The default stream is a barrier.** One missing stream argument serialises
  everything, silently.
- **Verify concurrency on a timeline.** The answer being right tells you nothing
  about whether anything overlapped.

## Going further

- Sweep `kChunks` = 2, 4, 8, 16, 64. Too few and there is nothing to overlap; too
  many and per-launch overhead dominates.
- Make the kernel 10× heavier and rerun. How do both the speedup and the
  default-stream penalty change?
- Try `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` — what does it change
  about the trap?
- Pin too much memory (several GB) and watch the host slow down. Pinned pages are a
  machine-wide resource.
