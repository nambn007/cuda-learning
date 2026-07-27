<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 07 - Naive matrix multiplication on the GPU

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [06](../06-matrix-add-2d/), [Phase 1/01](../../../phase-1-foundation/exercises/01-matmul-naive-cpu/) | **Requires a GPU**

## Goal

Write the GPU baseline that Phase 3 and Phase 5 are measured against — and then
work out why a **perfectly coalesced** kernel still reaches only ~5% of the GPU's
peak FLOP/s. The answer is the reason shared memory exists.

## Background

One thread per output element. Thread `(row, col)` walks the whole of row `row` of
`A` and the whole of column `col` of `B`:

```cuda
float acc = 0.0f;
for (int k = 0; k < n; ++k)
    acc += A[row * n + k] * B[k * n + col];
C[row * n + col] = acc;
```

**Check the access pattern first.** Within a warp, threads differ in `col`:

| Access | Across the 32 threads | Verdict |
|---|---|---|
| `A[row * n + k]` | the **same** address for all 32 | broadcast — the hardware handles this well |
| `B[k * n + col]` | 32 **consecutive** addresses | perfectly coalesced |

So this kernel is *already* doing memory access correctly. And it is still slow.

**Why.** Each output element costs `2N` FLOPs and issues `2N` global loads, so

```
arithmetic intensity = 2N / (2N × 4 bytes) = 0.25 FLOP per byte loaded
```

and it stays 0.25 no matter how large `N` gets. Against a ridge point near 37,
that is hopeless.

Note the distinction carefully: **the algorithm is not memory bound** — it does
`N³` work on `N²` data, so its intrinsic intensity grows with `N`. **This kernel
is**, because every thread re-reads its row of `A` and column of `B` from global
memory instead of sharing them with the 255 other threads in its block that need
exactly the same values.

At `n = 1024` that is **683× more loads than the algorithm requires**.

## Your task

Open `main.cu`:

1. **`matmulNaive()`** — accumulate into a local `float acc`, exactly as on the
   CPU in Phase 1/01.
2. **`matmulNaiveSwapped()`** — swap `row` and `col`, time both.
3. **Verify** with `rtol = 1e-4` — summing 1024 terms with FMA drifts measurably
   from the CPU's separate multiply and add.
4. **Analyse**: achieved GFLOP/s as a percentage of peak, bytes requested vs bytes
   needed, and the ratio.

## Build and run

```bash
cmake --build build --target p2_07_matmul_naive -j
./build/bin/p2/p2_07_matmul_naive
```

## Expected output

RTX 3060, `n = 1024`:

```
--- Performance ---
Variant                           Time (ms)    GFLOP/s   Speedup
----------------------------------------------------------------
CPU ikj (single thread)             154.834      13.87     1.00x
GPU naive (coalesced B)               2.680     801.36    57.78x
GPU naive (strided B)                12.130     177.03    12.76x

--- Where the performance went ---
  Achieved              : 801.4 GFLOP/s  (6% of this GPU's FP32 peak)
  Bytes REQUESTED       : 8.6 GB per call
  Bytes actually needed : 0.013 GB
  Ratio                 : 683x more loads than the algorithm requires
```

**58× faster than the CPU, and still only 6% of what the GPU can do.** Both facts
matter. Swapping the axes costs another 4.5×.

> The load *rate* the program prints works out to several times the DRAM peak,
> which is impossible — and informative. It means the L1 and L2 caches served most
> of those requests. The kernel is not limited by DRAM bandwidth; it is limited by
> how fast the memory pipeline can issue and service 8.6 GB of load **requests**.

## Key takeaways

- **Coalescing is necessary, not sufficient.** It decides how efficiently a
  request is served; it cannot reduce how many requests you make.
- **"Memory bound" is a property of the kernel, not the algorithm.** The same
  maths can be either, depending on how you stage the data.
- **A broadcast access is fine.** All 32 threads reading one address costs one
  transaction, not 32.
- 58× over one CPU thread sounds impressive until you compare against the *GPU's*
  own ceiling. Always state both.

## Going further

- Try `dim3 block(32, 32)` (1024 threads). Faster or slower? Check occupancy with
  `ncu --set full`.
- Compute the ratio for `n = 2048`. Does the 683× grow?
- Profile with `ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` and
  confirm the request count matches the model.
- **Next:** [Phase 3/04 tiled-matmul](../../../phase-3-intermediate/exercises/04-tiled-matmul/)
  stages tiles in `__shared__` memory so each fetched value serves 16 threads.
  [Phase 5/03](../../../phase-5-expert/exercises/03-cublas-gemm/) then compares
  both against cuBLAS.
