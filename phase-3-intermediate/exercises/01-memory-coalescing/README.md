<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 01 - Memory coalescing, measured

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [Phase 1/05](../../../phase-1-foundation/exercises/05-cache-and-bandwidth/), [Phase 2/06](../../../phase-2-cuda-fundamentals/exercises/06-matrix-add-2d/) | **Requires a GPU**

## Goal

Measure the single most important performance fact about GPUs. Everything else in
Phase 3 is a technique for working around cases where you cannot satisfy it
directly.

## Background

> **Memory is not moved in bytes, it is moved in blocks.**

You measured this on the CPU in [Phase 1/05](../../../phase-1-foundation/exercises/05-cache-and-bandwidth/),
where the block was a 64-byte cache line. On a GPU it is a **32-byte sector**.
When a warp issues a load, the hardware works out how many distinct sectors the 32
addresses fall into and fetches **all of them**:

| Access | Sectors per warp | Bytes used |
|---|---|---|
| 32 consecutive floats | 4 | 128 / 128 |
| 32 floats, stride 8 | **32** | 128 / 1024 |

The second case moves 8× the data for the same result. Nothing in the source looks
different and the answer is identical.

Three effects, measured separately:

**1. Stride.** As the gap between neighbouring threads' addresses grows, sectors
per warp grow with it — until every thread has its own sector and the curve
flattens. You cannot waste more than one whole sector per access.

**2. Alignment.** Even a perfectly contiguous access pays one extra sector per warp
if it does not start on a sector boundary.

**3. The mapping.** "Give each thread a contiguous chunk" is correct on a CPU and
catastrophic on a GPU. This is the single most common porting mistake.

## Your task

Open `main.cu` and write four kernels: `stridedRead`, `offsetRead`,
`blockPerThread` and `interleaved`. Then run three sweeps.

**Write your prediction down before each experiment.** In particular, before
experiment 3: how much slower do you expect chunk-per-thread to be? Most people
guess 2×.

For the stride sweep, print the model's `sectorsPerWarp(stride)` and a *predicted*
bandwidth alongside the measured one. The relationship between those two columns is
the whole lesson.

## Build and run

```bash
cmake --build build --target p3_01_memory_coalescing -j
./build/bin/p3/p3_01_memory_coalescing
```

## Expected output

RTX 3060, 256 MB array:

```
--- Experiment 1 - stride ---
  stride     time (ms)   useful GB/s  sectors/warp   predicted   of peak
  ----------------------------------------------------------------------------
  1              1.752         306.5           4.0       306.5       85%
  2              1.240         216.5           8.0       324.7       60%
  4              1.002         133.9          16.0       334.7       37%
  8              0.908          73.9          32.0       332.5       21%
  16             0.455          73.8          32.0       332.1       20%
  32             0.229          73.4          32.0       330.3       20%
  128            0.072          58.5          32.0       263.3       16%

--- Experiment 2 - alignment ---
  offset     time (ms)   useful GB/s vs offset 0
  ---------------------------------------------------
  0              1.750         306.7       1.00x
  1              1.929         278.2       0.91x
  4              1.924         279.0       0.91x
  8              1.804         297.6       0.97x
  16             1.800         298.2       0.97x
  31             1.911         280.9       0.92x
  32             1.745         307.7       1.00x

--- Experiment 3 ---
32-element chunk per thread          19.765     27.16 GB/s     1.00x
interleaved (grid-stride)             1.743    308.04 GB/s    11.34x
```

### Reading this

**The `predicted` column stays near 300 GB/s in every row.** That is the point: the
memory system is working *just as hard* at stride 128 as at stride 1. It is moving
data you throw away. Useful bandwidth collapses 4× while the bus stays saturated.

**Alignment is periodic, not monotonic.** Offsets 8, 16 and 32 are back at full
speed; 1, 2, 4 and 31 are not. 8 floats = 32 bytes = exactly one sector, so what
matters is `offset % 8`, not how large the offset is. The penalty is ~9% rather
than the 25% the sector count suggests, because one warp's extra sector is usually
the next warp's first sector, and L2 serves it.

**11.34×.** Chunk-per-thread is the loop structure that was *right* on a CPU. Watch
for it when porting.

## Key takeaways

- **Adjacent threads must touch adjacent addresses.** That is the whole rule.
- **Bad access patterns saturate the bus without delivering bandwidth.** Never
  judge a kernel by GB/s alone — judge it by *useful* GB/s.
- **The stride penalty saturates** at one sector per thread. Beyond that, more
  stride costs nothing extra.
- **Alignment costs one sector per warp**, and it is periodic in the sector size.
- **The CPU-correct mapping is GPU-catastrophic.**

## Going further

- `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` on each stride —
  do the measured sector counts match `sectorsPerWarp()`?
- Repeat with `double` (8 bytes). At what stride does the collapse start now?
- Shrink the array to 1 MB so it fits in L2 and rerun. Where does the penalty go?
- **Next:** [02 shared-memory-basics](../02-shared-memory-basics/) introduces the
  staging area that lets you satisfy this rule even when the algorithm fights you;
  [04](../04-tiled-matmul/) and [05](../05-transpose-optimized/) apply it.
