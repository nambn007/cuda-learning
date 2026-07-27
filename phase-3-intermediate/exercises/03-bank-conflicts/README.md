<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 03 - Shared memory bank conflicts

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~1.5 h | Prerequisites: [02](../02-shared-memory-basics/) | **Requires a GPU**

## Goal

Shared memory has its own access rules, entirely separate from global-memory
coalescing. Break them and a single misplaced index costs **15×**. Fix them with
one extra column.

## Background

Shared memory is not a flat block of storage. It is split into **32 banks**, each
4 bytes wide, with consecutive words in consecutive banks:

```
bank = (address / 4) % 32

word:  0   1   2  ...  31  32  33 ...
bank:  0   1   2  ...  31   0   1 ...
```

Each bank serves **one address per cycle**:

| A warp's access | Cost |
|---|---|
| 32 threads → 32 different banks | 1 cycle ✅ |
| 32 threads → the **same address** | 1 cycle ✅ (broadcast) |
| *k* threads → different addresses in one bank | *k* cycles ❌ |

With stride `s`, a warp touches `32/gcd(s,32)` distinct banks, so it serialises
into `gcd(s,32)` cycles. **What matters is the stride modulo 32, not its size** —
stride 33 is conflict-free while stride 32 is the worst case.

### The classic trap, and the classic fix

```cuda
__shared__ float tile[32][32];
tile[threadIdx.x][i]     // address tid*32 + i  →  bank i % 32
                         // EVERY thread hits the same bank: 32-way
```

```cuda
__shared__ float tile[32][33];      // one extra column
tile[threadIdx.x][i]     // address tid*33 + i  →  bank (tid + i) % 32
                         // all 32 banks: no conflict
```

128 wasted bytes per tile. You will use this again in
[04 tiled-matmul](../04-tiled-matmul/) and
[05 transpose-optimized](../05-transpose-optimized/).

## Your task

Open `main.cu` and write five kernels: `sharedStride`, `tileRowAccess`,
`tileColumnAccess`, `tileColumnPadded`, `sharedBroadcast`.

**Work out the bank number on paper before writing each kernel**, and predict:
what happens at stride 33? Is a broadcast a 32-way conflict?

## Build and run

```bash
cmake --build build --target p3_03_bank_conflicts -j
./build/bin/p3/p3_03_bank_conflicts
```

## Expected output

RTX 3060, one warp per block:

```
--- Experiment 1 - stride ---
  stride      time (ms) predicted ways  vs stride 1  banks hit
  --------------------------------------------------------------------
  1               0.181              1        1.00x         32
  2               0.182              2        1.00x         16
  4               0.348              4        1.92x          8
  8               0.700              8        3.86x          4
  16              1.370             16        7.56x          2
  32              2.739             32       15.11x          1
  33              0.181              1        1.00x         32

--- Experiment 2 - a 2D tile ---
tile[32][32], by row                  0.179     1.00x
tile[32][32], by column               2.746     0.07x
tile[32][33], by column (padded)      0.166     1.08x

  Column access costs 15.34x. One extra column - 128 bytes per tile -
  recovers 101% of it.

--- Experiment 3 - broadcast ---
  Every thread reads the SAME address: 0.166 ms (0.92x vs stride 1)
```

### Reading this honestly

**The measured slowdown is about half the predicted way count** (32-way costs
~15×, not 32×), and stride 2 shows no penalty at all. That is not the model being
wrong — each loop iteration also does an add, a mask and a branch, and a couple of
extra shared-memory cycles hide behind them. Only once the conflict degree is large
does shared memory become the binding constraint.

**Predict the ratio from the model; measure the absolute cost.**

## Key takeaways

- **Bank conflicts are a shared-memory phenomenon, unrelated to coalescing.** A
  kernel can be perfectly coalesced and still lose 15× here.
- **`gcd(stride, 32)` is the whole model.** Stride 33 beats stride 32.
- **Broadcast is free.** 32 threads wanting *one* address is the best case, not the
  worst. A conflict needs 32 *different* addresses in one bank.
- **Pad by one element** whenever a tile is written by row and read by column. It
  is the cheapest fix in CUDA.

## Going further

- `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` — do the
  measured conflict counts match `gcd(s, 32)`?
- Try `double` tiles (8 bytes). How does the bank mapping change?
- Pad by 2, 4 and 8 instead of 1. Which paddings work, and why?
- Increase the block to 256 threads. Does the effect survive, or does having more
  warps hide it?
