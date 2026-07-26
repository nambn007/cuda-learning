<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 01 - Hello CUDA

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐ | Time: ~1 h | Prerequisites: [Phase 1/08](../../../phase-1-foundation/exercises/08-gpu-device-query/) | **Requires a GPU**

## Goal

Write, launch and verify your first kernel. Three things happen here that never
happen in ordinary C++: a function runs on a different processor, it runs
thousands of times at once, and **the call returns before the work is done**.

## Background

**`__global__`** marks a function that is compiled for the GPU, called from the
CPU, and must return `void`. Every thread created by the launch runs the same
body — that is the whole SIMT model.

**The launch syntax** `kernel<<<grid, block>>>(args)` is the only non-C++ syntax
in CUDA. It creates `grid` blocks of `block` threads each.

**Four built-in variables** are visible inside any kernel:

| | meaning | range |
|---|---|---|
| `threadIdx.x` | thread index within its block | `[0, blockDim.x)` |
| `blockIdx.x` | block index within the grid | `[0, gridDim.x)` |
| `blockDim.x` | threads per block | |
| `gridDim.x` | blocks in the grid | |

From which comes **the most important line in CUDA**:

```cuda
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

**The bounds check is mandatory.** With `n = 1000` and 256 threads per block you
launch 1024 threads; 24 of them have `gid >= n`. Writing past the allocation
usually does not crash a GPU — it silently corrupts whatever is next in memory.
Always `if (gid < n)`.

**Launches are asynchronous**, so failures come in two flavours and need two
different checks:

- `cudaGetLastError()` — problems with the *launch*: too many threads per block,
  too much shared memory. Reported immediately.
- `cudaDeviceSynchronize()` — problems during *execution*: illegal address,
  assertion failure. Only visible after waiting.

**Block ordering is not defined.** Blocks are scheduled onto SMs as resources free
up. The `printf` lines will come out shuffled, and any code that depends on block
ordering is broken.

## Your task

Open `main.cu`:

1. **`helloCUDA()`** — `printf` the block and thread index.
2. **`writeGlobalId(out, n)`** — compute the global id, bounds-check it, store it.
3. **The memory plumbing** — `cudaMalloc` → launch → `cudaMemcpy` back →
   `cudaFree`, then verify `host[i] == i`.

## Build and run

```bash
cmake --build build --target p2_01_hello_cuda -j
./build/bin/p2/p2_01_hello_cuda
```

Or without CMake:

```bash
nvcc -arch=native -I common -o hello \
     phase-2-cuda-fundamentals/exercises/01-hello-cuda/solution.cu && ./hello
```

## Expected output

```
--- Kernel printf ---
  Launching <<<2, 4>>> = 2 blocks x 4 threads = 8 threads

  Hello from block 1, thread 0 (global id 4)
  Hello from block 1, thread 1 (global id 5)
  ...
  Hello from block 0, thread 0 (global id 0)

--- Verifiable kernel ---
  n = 1000, block = 256, grid = 4 (1024 threads, 24 idle)
  [PASS] every thread wrote its global id (n=1000, exact match)
```

## Key takeaways

- `gid = blockIdx.x * blockDim.x + threadIdx.x` — commit it to memory.
- **Always bounds-check.** Out-of-range GPU writes corrupt silently.
- A launch is **asynchronous**; check the launch *and* the execution.
- **Block execution order is undefined.** Never rely on it.
- Make `blockDim` a **multiple of 32** — the hardware schedules in warps of 32,
  and a block of 100 threads wastes 28 slots in every warp it occupies.
- `printf` proves a kernel *ran*; only a verified result proves it was *correct*.

## Going further

- Launch `<<<1, 2048>>>`. It fails — `cudaGetLastError()` tells you why. (Max
  threads per block is 1024.)
- Remove the bounds check and run under `compute-sanitizer ./hello`. It reports
  the exact out-of-range access.
- Add `printf("%d\n", warpSize)` inside the kernel. Where does that value come from?
- Read the [CUDA C++ Programming Guide, "Programming Model"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model).
