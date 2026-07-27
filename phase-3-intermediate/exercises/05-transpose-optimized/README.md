<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 05 - Transpose with a shared-memory tile

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [02](../02-shared-memory-basics/), [03](../03-bank-conflicts/), [Phase 2/08](../../../phase-2-cuda-fundamentals/exercises/08-transpose-naive/) | **Requires a GPU**

## Goal

Fix the kernel that Phase 2 proved could not be fixed by rearranging indices —
and, in the process, learn when a bank conflict actually costs you anything.

## Background

Phase 2/08 established that `out[col][row] = in[row][col]` cannot be coalesced on
both sides: whichever index a warp varies, one side scatters. **The scatter *is* the
transpose.**

The fix is shared memory's second use ([exercise 02](../02-shared-memory-basics/)):
**reordering**. Shared memory has no coalescing requirement, so:

1. read a `TILE×TILE` block from global memory — **coalesced**
2. write it into a shared tile
3. `__syncthreads()`
4. read the shared tile **transposed** — free, no coalescing rules apply
5. write it out to global memory — **coalesced**

Both global accesses are now contiguous. The transpose happens entirely on chip.

**Except** that step 4 walks a *column* of the shared tile:

```cuda
tile[threadIdx.x][threadIdx.y + j]
  address = tx * 32 + (ty + j)
  bank    = (ty + j) % 32     ← identical for all 32 threads → 32-way conflict
```

Padding to `tile[32][33]` makes the bank `(tx + ty + j) % 32` — all 32 banks, for
128 bytes per tile. That is exercise 03's fix, applied where it belongs.

**One thing to get right:** in step 3 the output tile lives at the *transposed
position in the grid*, not just transposed inside the tile. Recompute `x` and `y`
from `blockIdx.y` and `blockIdx.x` swapped.

## Your task

Open `main.cu` and write four kernels: `copyKernel`, `transposeNaive`,
`transposeSharedNoPad`, `transposeSharedPadded`.

**Predict before measuring:** in exercise 03 the identical 32-way conflict cost
**15×**. Do you expect the same here? If not, what is different about this kernel?

## Build and run

```bash
cmake --build build --target p3_05_transpose_optimized -j
./build/bin/p3/p3_05_transpose_optimized
```

## Expected output

RTX 3060, 4096×4096:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
copy (speed of light)                 0.407    329.82     1.00x
transpose naive                       1.440     93.22     0.28x
transpose shared, no padding          0.440    304.82     0.92x
transpose shared, padded              0.417    322.17     0.98x

--- As a fraction of copy speed ---
  copy                               329.8 GB/s    100%
  naive                               93.2 GB/s     28%
  shared, no padding                 304.8 GB/s     92%
  shared, padded                     322.2 GB/s     98%

  naive -> shared          3.27x
  shared -> shared padded  1.06x
  naive -> shared padded   3.46x
```

### The bank conflict cost 6%, not 15×

Staging in shared memory is the big win: **3.27×**, from 28% to 92% of copy speed.

The unpadded version *does* have a genuine 32-way bank conflict — the arithmetic
above is not wrong. But removing it is worth only **1.06×** here, versus **15×** in
exercise 03.

The difference is **what the kernel is bound by**. Exercise 03 was pure
shared-memory traffic in a tight loop, so every serialised cycle showed up in the
total. This kernel pushes 128 MB through DRAM, and the extra shared-memory cycles
hide behind that latency.

> **A bank conflict only costs you when shared memory is the bottleneck.**

Pad anyway — 128 bytes for 6% is an easy trade, and the moment you optimise the
global side further the conflict stops being hidden. But **measure before assuming
it is your problem.**

### Why 98% and not 100%

Three structural reasons: two `__syncthreads()` per tile that a copy does not need;
the write goes to a different 4096-float row for each tile column, losing the DRAM
page locality a straight copy enjoys; and the shared memory limits blocks per SM.

Closing the last 2% means diagonal block reordering or vectorised loads — or not
transposing at all, since many algorithms can consume a matrix in either
orientation if you let them.

## Key takeaways

- **Shared memory can make an uncoalesceable access coalesced**, by moving the
  scatter on chip. This is its second use, distinct from reuse.
- **The output tile's grid position transposes too**, not just its contents.
- **Bank conflicts matter in proportion to how shared-memory-bound you are.** The
  same 32-way conflict cost 15× in one kernel and 6% in another.
- **Measure which resource is binding before optimising it.** This is the single
  most repeated lesson in Phase 3.

## Going further

- Profile both shared versions with
  `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` — the
  conflict count should differ hugely even though the runtime barely does.
- Try `kBlockRows` = 4, 8, 16, 32. What limits it?
- Implement diagonal block reordering (`blockIdx` remapped) to spread DRAM page
  access. How much of the last 2% does it recover?
- Add `float4` loads. Does the tile still fit conveniently?
