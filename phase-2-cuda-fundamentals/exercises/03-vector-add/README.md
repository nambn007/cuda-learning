<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 03 - Vector addition, and the cost of getting data there

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐ | Time: ~1 h | Prerequisites: [02](../02-thread-indexing/) | **Requires a GPU**

## Goal

Write the "hello world" of GPU computing — and then discover that **the GPU loses**.
This is the most useful negative result in the whole curriculum: it teaches you
what a GPU is actually for, before you spend weeks optimising something that
should never have been on a GPU.

## Background

The kernel is trivial: `c[i] = a[i] + b[i]`. Per element it reads 8 bytes, writes
4, and performs one addition. Arithmetic intensity is **1/12 = 0.083 FLOP/byte**.
Your GPU's ridge point (from
[Phase 1/08](../../../phase-1-foundation/exercises/08-gpu-device-query/)) is around
**37**. This kernel is off the left edge of the roofline — pure memory traffic
with a rounding error of arithmetic attached.

That alone would be fine; the GPU has ~360 GB/s against the CPU's ~17 GB/s. The
problem is getting there. **PCIe runs at roughly 8–12 GB/s**, about 30–40× slower
than the GPU's own memory. So the sequence

```
copy a  (64 MB over PCIe)  →  copy b  (64 MB)  →  kernel  →  copy c back (64 MB)
```

spends almost all of its time on the arrows, not on the kernel.

The measurement that matters is therefore not "how fast is the kernel" but "how
fast is the whole thing, starting from where the data actually lives". Blog posts
quote the first number. Production systems live or die by the second.

## Your task

Open `main.cu`:

1. **`vectorAdd()`** — a grid-stride loop.
2. **The four steps** — `cudaMalloc` ×3, `cudaMemcpy` H2D ×2, launch,
   `cudaMemcpy` D2H, `cudaFree` ×3. Use `size_t` for byte counts; `int n *
   sizeof(float)` overflows silently at 512M elements.
3. **Three measurements** — the CPU version, the kernel alone, and the full round
   trip. Then decide which one you would put in a report.

## Build and run

```bash
cmake --build build --target p2_03_vector_add -j
./build/bin/p2/p2_03_vector_add
```

## Expected output

Measured on an RTX 3060 with `n = 16M`:

```
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
CPU (single thread)                  13.071     15.40       1.28     1.00x
GPU kernel only                       0.627    321.25      26.77    20.86x
GPU + PCIe round trip                24.365      8.26       0.69     0.54x

  Kernel efficiency : 321.3 of 360.0 GB/s = 89% of theoretical peak

  host -> device (64 MB)               8.06 ms      8.3 GB/s
  device -> host (64 MB)               7.81 ms      8.6 GB/s
  kernel                               0.63 ms

    The GPU is 1.86x SLOWER than the CPU once transfers are counted,
    even though the kernel itself is 20.9x faster.
```

Read those three rows carefully. The kernel is at **89% of the hardware's
theoretical peak** — there is nothing left to optimise, it is already as fast as
this GPU can move memory. And the complete operation is still **1.86× slower than
a single CPU thread**.

## Key takeaways

- **The kernel is not the program.** Quoting kernel-only time is the most common
  way to lie with GPU benchmarks — usually to yourself.
- **PCIe is the bottleneck** for anything that does not stay resident: ~8 GB/s
  versus the GPU's ~360 GB/s.
- **89% of peak means done.** When a kernel is at the roofline, stop optimising it
  and change the algorithm or the data flow instead.
- A GPU is worth it when: data **stays resident** across many kernels; arithmetic
  intensity is **high** (matmul, Phase 3/04); transfers **overlap** with compute
  (streams, Phase 4/03); or several operations are **fused** into one pass
  (Phase 5/14).

## Going further

- Rerun with `n = 1 << 20` and `n = 1 << 26`. Does the ratio change? Why not?
- Replace `cudaMalloc` + `cudaMemcpy` with `cudaMallocManaged`. Faster or slower?
  (Phase 2/11 investigates properly.)
- Use pinned host memory (`cudaMallocHost`). PCIe bandwidth roughly doubles —
  that is Phase 4/01.
- Chain ten vector adds on the device before copying back. At what point does the
  GPU win? That number is the real answer to "should this run on a GPU".
