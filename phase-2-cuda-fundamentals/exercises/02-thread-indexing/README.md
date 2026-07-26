<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 02 - Thread indexing in 1D, 2D and 3D

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐ | Time: ~1 h | Prerequisites: [01](../01-hello-cuda/) | **Requires a GPU**

## Goal

Map data of any shape onto threads, and learn the **grid-stride loop** — the
launch pattern used by essentially every production CUDA kernel.

## Background

A launch produces a 1D/2D/3D grid of blocks, each holding a 1D/2D/3D block of
threads. Picking that shape is a kernel's first design decision, and getting the
index arithmetic wrong is the most common beginner bug.

**The rule:** match the grid shape to the *data* shape, and make sure adjacent
**threads** touch adjacent **addresses**.

```cuda
// 1D
int gid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int i = y * width + x;                    // x fastest-varying

// 3D
int z = blockIdx.z * blockDim.z + threadIdx.z;
int i = (z * height + y) * width + x;
```

**Why `x` must be the contiguous axis.** Threads in a warp differ in `threadIdx.x`
first. If `x` indexes the column, a warp reads 32 adjacent addresses and the
hardware merges them into a few transactions. Swap `x` and `y` and the kernel
still produces the right answer — roughly **10× slower**. Phase 3 exercise 01
measures exactly this.

**The grid-stride loop** decouples grid size from data size:

```cuda
int gid    = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = gid; i < n; i += stride) out[i] = i;
```

Four benefits: one launch config works for **any** `n`; the grid can be sized to
the *GPU* (a few blocks per SM) instead of the data; accesses stay coalesced
because consecutive threads still take consecutive addresses in each iteration;
and the bounds check comes free from the loop condition.

It is also sometimes the *only* option — `gridDim.y`/`z` are capped at 65535 and
`blockDim.z` at 64.

## Your task

Open `main.cu` and implement `index1D`, `index2D`, `index3D` and `gridStride`,
then write the launches. Suggested block shapes: `256` (1D), `dim3(16,16)` (2D),
`dim3(8,8,4)` (3D) — all 256 threads, all multiples of 32.

## Build and run

```bash
cmake --build build --target p2_02_thread_indexing -j
./build/bin/p2/p2_02_thread_indexing
```

## Expected output

```
--- 2D - one thread per pixel ---
  1920x1080 image -> grid(120, 68) x block(16, 16) = 2088960 threads
  covers 1920x1088 pixels, 15360 of them idle
  [PASS] 2D indexing (n=2073600, exact match)

--- Grid-stride loop ---
  n = 100000, but launching only <<<112, 256>>> = 28672 threads
  each thread handles about 3.5 elements
  [PASS] grid-stride loop (n=100000, exact match)
  [PASS] same launch config, n = 7 (n=7, exact match)
```

That last line is the point: the identical launch handles `n = 100000` and
`n = 7`.

## Key takeaways

- Index arithmetic scales the same way in every dimension; only the flattening
  changes.
- **Check every bound.** A 2D grid over a 1920×1080 image covers 1920×1088.
- **`x` is the contiguous axis.** Getting this backwards costs ~10× and produces
  correct results, so nothing warns you.
- **Prefer the grid-stride loop.** It is size-independent, GPU-sized, still
  coalesced, and self-bounds-checking.

## Going further

- Swap `x` and `y` in `index2D` and time both. (Phase 3/01 does this properly.)
- Use `cudaOccupancyMaxPotentialBlockSize()` to pick the grid instead of
  `SMs × 4`. Does it agree?
- Try `dim3 block(32, 32)` = 1024 threads. Does it still launch? Is it faster?
- Read the [Programming Guide, "Thread Hierarchy"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)
  and the NVIDIA blog post [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
