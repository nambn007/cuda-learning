<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 07 - Warp-level primitives

> **Phase 3 · Intermediate** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [06](../06-reduction-variants/) | **Requires a GPU**

## Goal

Exchange values directly between threads' registers — no shared memory, no
barriers, no memory traffic. And then discover that the most celebrated
application of these primitives **buys nothing on a modern compiler**, which is a
more useful lesson than the technique itself.

## Background

| Intrinsic | Effect |
|---|---|
| `__shfl_sync(mask, v, lane)` | read lane `lane`'s `v` |
| `__shfl_up_sync(mask, v, d)` | read lane `me − d` |
| `__shfl_down_sync(mask, v, d)` | read lane `me + d` |
| `__shfl_xor_sync(mask, v, m)` | read lane `me XOR m` |
| `__ballot_sync(mask, p)` | 32-bit word, one bit per lane |
| `__all_sync` / `__any_sync` | is `p` true for all / any lane |
| `__activemask()` | which lanes are currently active |

**The `_sync` suffix and the mask are not decoration.** Before Volta a warp always
moved in lockstep and the old intrinsics needed no mask. Since Volta threads
diverge independently, so you must declare which lanes are expected to
participate. Use `0xffffffff` when the whole warp is provably active, and
`__activemask()` when it may not be — inside a grid-stride loop whose last
iteration leaves lanes behind, for example. **A mask that does not match reality is
undefined behaviour, and the symptom is a wrong answer that appears only under
load.**

**`__shfl_xor_sync` versus `__shfl_down_sync`.** Both reduce in 5 steps. `down`
leaves the answer only in lane 0; `xor` (a butterfly) leaves it in *every* lane.
Same cost, and no `if (lane == 0)` afterwards.

## Your task

Open `main.cu` and write: `warpAllReduceSum`, `warpInclusiveScan`,
`blockReduceKernel`, `voteDemo`, `filterNaive`, `filterWarpAggregated`.

For the last one, **predict the speedup, measure it, and then — whatever the
number — disassemble the *naive* kernel**:

```bash
cuobjdump -sass build/bin/p3/p3_07_warp_shuffle | grep -B3 RED
```

The answer to "why did I get that number" is in there.

## Build and run

```bash
cmake --build build --target p3_07_warp_shuffle -j
./build/bin/p3/p3_07_warp_shuffle
```

## Expected output

RTX 3060, 16M floats, 25% pass rate:

```
  [PASS] inclusive scan within each warp
  [PASS] block reduction is correct
  [PASS] __all_sync / __any_sync / __ballot_sync + __popc

--- Warp-aggregated atomics ---
Counting elements that pass a filter
one atomic per passing thread         0.377    178.09 GB/s     1.00x
one atomic per warp (ballot)          0.390    172.01 GB/s     0.97x
```

### 0.97×. No gain. Here is why

Disassemble `filterNaive` and you find:

```
REDUX.SUM UR10, R4                ← warp-wide sum, one instruction
ISETP.EQ.U32.AND P1, PT, ...      ← pick a single lane
@P1 RED.E.ADD.STRONG.GPU [...]    ← ONE atomic for the whole warp
```

**nvcc already performed the aggregation.** It has done so since CUDA 9, and on
Ampere it uses the hardware `REDUX.SUM` instruction — which is *faster* than the
`__ballot_sync` + `__popc` sequence you would write by hand. The manual version is
not wrong; it is redundant.

> **The lesson is not "aggregate your atomics". It is: read the SASS before
> hand-optimising.** A celebrated technique from a 2017 blog post can be dead code
> by 2020 because the compiler absorbed it. Checking costs one command.

**Manual aggregation still earns its keep when:**

- the compiler cannot see the pattern — a data-dependent increment, or an atomic
  behind a function call
- you need each lane's **offset**, not just the total. A warp scan over the ballot
  gives every passing thread its own output slot from a single atomic — exactly how
  [stream compaction](../12-stream-compaction/) works
- you are aggregating something other than a counter

## Key takeaways

- **Shuffles move data between registers.** No shared memory, no barriers, no
  memory traffic — 5 instructions for a warp reduction or scan.
- **`xor` gives every lane the result; `down` gives it to lane 0.** Pick by what
  you need next.
- **Always pass a correct mask.** `__activemask()` when the warp may be partial.
- **Verify that an optimisation is still needed before applying it.** The compiler
  is a moving target, and `cuobjdump -sass` is how you check.

## Going further

- Implement a warp-level `scan` over the ballot mask (`__popc(mask & lanemask_lt())`)
  to give each passing thread a unique output index. That is exercise 12's core.
- Rerun the filter with a 100% pass rate and with 1%. Does the ranking change?
- Use `__reduce_add_sync` (sm_80+) directly and compare with your hand-written
  butterfly.
- Read [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
  — and note the date on it while you do.
