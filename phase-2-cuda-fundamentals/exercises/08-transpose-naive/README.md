<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 08 - Naive transpose: the kernel you cannot coalesce

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [06](../06-matrix-add-2d/) | **Requires a GPU**

## Goal

Meet the first kernel where **being careful is not enough**. Transpose does no
arithmetic — one read and one write per element, exactly like a copy — and yet it
runs at a third of copy speed. No rearrangement of the indices fixes it. That is
what sets up shared memory in Phase 3.

## Background

```cuda
out[col][row] = in[row][col];
```

A warp varies one index. Whichever you pick, one side is contiguous and the other
is scattered:

| Warp varies | Read `in[row*cols + col]` | Write `out[col*rows + row]` |
|---|---|---|
| `col` | 32 consecutive floats ✅ | 32 addresses `rows` apart ❌ |
| `row` | 32 addresses `cols` apart ❌ | 32 consecutive floats ✅ |

**The scatter is not a bug in your indexing. The scatter *is* the transpose.**

**Measure against a copy, not against peak.** A copy moves exactly the same bytes,
so it is the speed of light for this problem. "93 GB/s" means nothing on its own;
"29% of copy" is the number that tells you how much room is left.

**When you must scatter one side, scatter the reads.** The asymmetry is real:

- A scattered **read** can hit in L1/L2 — a neighbouring warp may already have
  pulled that sector in, and misses are hidden by other resident warps.
- A scattered **write** has no such escape. It must eventually reach DRAM, and a
  partial-sector write wastes the rest of that sector with no reuse to recover it.

## Your task

Open `main.cu`:

1. **`copyKernel()`** — the baseline.
2. **`transposeReadCoalesced()`** — `x` → `col`.
3. **`transposeWriteCoalesced()`** — `x` → `row`. Same body, different grid shape.
4. **Report each transpose as a percentage of copy bandwidth.**

Suggested block: `dim3(32, 8)` = 256 threads, 32 wide so a warp spans exactly one
row of the block.

Then, before reading the solution: **try to rearrange the indices so both sides are
coalesced.** Convince yourself of the answer.

## Build and run

```bash
cmake --build build --target p2_08_transpose_naive -j
./build/bin/p2/p2_08_transpose_naive
```

## Expected output

RTX 3060, 4096×4096:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
copy (speed of light)                 0.415    323.63     1.00x
transpose, coalesced read             1.442     93.11     0.29x
transpose, coalesced write            0.990    135.54     0.42x
CPU transpose                       160.717      0.84     0.00x

  copy                               323.6 GB/s  ( 90% of DRAM peak)
  transpose, coalesced read           93.1 GB/s    29% of copy
  transpose, coalesced write         135.5 GB/s    42% of copy
```

Three things worth noticing:

1. **Copy reaches 90% of DRAM peak** — the hardware is fine, the problem is the
   access pattern.
2. **The best transpose reaches 42% of copy.** Over half the bandwidth is thrown
   away on partial sectors.
3. **The CPU manages 0.84 GB/s** — a 190× gap, because the strided write thrashes
   its cache in exactly the same way, only worse.

## Key takeaways

- **Some access patterns cannot be coalesced by rearranging indices.** The
  scatter is intrinsic to the operation.
- **Measure against the right baseline.** GB/s against DRAM peak flatters a
  transpose; GB/s against a copy tells the truth.
- **Coalesce the writes if you must choose.** Reads have L2 and latency hiding to
  fall back on; writes do not.
- The same effect ruins the CPU version even harder — this is Phase 1/05's stride
  experiment, on both processors at once.

## Going further

- Try block shapes `(32,8)`, `(16,16)`, `(32,32)`. Does the ranking change?
- `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` on both
  transposes: count the write sectors and compare with the model.
- Transpose a non-square matrix (4096×1024). Does anything change structurally?
- **Next:** [Phase 3/05 transpose-optimized](../../../phase-3-intermediate/exercises/05-transpose-optimized/)
  stages a tile in `__shared__` memory so both sides can be contiguous — and then
  hits bank conflicts, which is [Phase 3/03](../../../phase-3-intermediate/exercises/03-bank-conflicts/).
