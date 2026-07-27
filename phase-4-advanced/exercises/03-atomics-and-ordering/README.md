<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 03 - Atomics, contention and memory ordering

> **Phase 4 · Advanced** | Difficulty: ⭐⭐⭐⭐⭐ | Time: ~3 h | Prerequisites: [Phase 3/07](../../../phase-3-intermediate/exercises/07-warp-shuffle/) | **Requires a GPU**
>
> ⚠️ **Not yet verified on the reference machine** — no measured numbers below. Run
> it and fill in your own.

## Goal

Three topics, in increasing difficulty: **contention**, **building atomics that do
not exist**, and **memory ordering** — where a kernel can be correct on your GPU
and wrong on someone else's.

## Background

**1. Contention.** `atomicAdd` on one global address from every thread serialises.
The fix is **privatisation**: each block keeps its own counters in shared memory,
then combines once. Global atomics drop from one per *element* to a few per
*block*. This is the most useful atomic technique there is.

**2. `atomicCAS`.** Only a handful of atomics exist in hardware. A float maximum, a
custom combine — anything else is a compare-and-swap retry loop:

```cuda
do { assumed = old;
     old = atomicCAS(addr, assumed, f(assumed)); }
while (assumed != old);
```

Correct, but it retries under contention. Reduce within the block first.

**3. Memory ordering.** **Atomicity is not visibility.** `atomicAdd` guarantees the
counter is right; it says *nothing* about whether the data you wrote just before it
is visible to the thread that reads the counter afterwards. `__threadfence()` is
what orders them.

| | Scope |
|---|---|
| `__threadfence_block()` | within the block |
| `__threadfence()` | device-wide |
| `__threadfence_system()` | host and peer GPUs too |

## Your task

Open `main.cu`: `histogramGlobal`, `histogramPrivatised`, `atomicMaxFloat` +
`maxKernel`, and `singlePassReduce`.

For the last one: write it **with** the fence, then delete the fence and run it a
few hundred times. It will almost certainly still be right — **that is precisely
why the bug is expensive.** It surfaces on someone else's hardware, months later,
under load.

## Build and run

```bash
cmake --build build --target p4_03_atomics_and_ordering -j
./build/bin/p4/p4_03_atomics_and_ordering
compute-sanitizer --tool racecheck ./build/bin/p4/p4_03_atomics_and_ordering
```

## Key takeaways

- **Privatise before you optimise anything else about an atomic.**
- **`atomicCAS` builds any atomic you need**, at the cost of a retry loop.
- **Atomicity ≠ visibility.** The fence is not optional, and omitting it usually
  produces correct results — which is the trap.
- A memory-ordering bug cannot be caught by testing on one machine.
