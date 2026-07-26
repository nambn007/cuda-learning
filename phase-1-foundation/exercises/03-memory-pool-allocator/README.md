<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 03 - A memory pool allocator

> **Phase 1 · Foundation** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: pointers, `new`/`delete`

## Goal

Write the two allocators that every high-performance system ends up needing — a
**bump (arena) allocator** and a **pool allocator** — and measure how much faster
they are than `new`/`delete`. This is the CPU-side rehearsal for a problem that is
far worse on the GPU.

## Background

**Why does a CUDA course care about allocators?** Because `cudaMalloc` costs
roughly **100–500 microseconds**. It enters the driver, may synchronise the whole
device, and can reorganise the GPU's virtual address space. A kernel that runs in
50 µs preceded by a 200 µs allocation spends 80% of its time not computing.

Every serious CUDA codebase solves this the same way: allocate one big slab up
front and sub-allocate from it yourself. CUDA 11.2+ ships this as
`cudaMallocAsync` backed by a `cudaMemPool_t`; PyTorch has its caching allocator;
TensorRT has its workspace. All of them are the two data structures below.

**Bump allocator (arena).** One buffer plus an offset. `allocate()` aligns the
offset, returns the pointer, advances the offset — three instructions, no locking,
no search, no fragmentation. The catch: you cannot free one object, only
everything at once via `reset()`. That is fine whenever objects share a lifetime,
which covers per-frame, per-request and per-batch work.

**Pool allocator.** Fixed-size blocks with a free list, so individual frees work.
The elegant part: a *free* block contains no user data, so its first bytes are
spare — store the "next free block" pointer right there. The free list costs zero
extra memory. `allocate()` pops the head, `deallocate()` pushes onto the head;
both are O(1).

**Alignment.** `alignUp(n, a) = (n + a - 1) & ~(a - 1)` works because alignment is
always a power of two. Misaligned loads are slower on x86 and a hard fault on some
architectures — and on a GPU, misaligned access silently destroys coalescing.

## Your task

Open `main.cpp`:

1. **`BumpAllocator`** — constructor, `allocate(size, alignment)`, `reset()`.
   Return `nullptr` when the arena is exhausted; never over-run the buffer.
2. **`PoolAllocator`** — thread all blocks onto the free list in the constructor,
   then implement `allocate()` and `deallocate()`. Use `std::memcpy` to read and
   write the embedded pointer (it is well-defined even when the storage is raw).
3. **Benchmark all three** with `timeCpuMs()` and fill in a `ResultTable`.

Use **placement new** (`new (raw) Node{...}`) to construct objects in storage you
allocated yourself.

## Build and run

```bash
cmake --build build --target p1_03_memory_pool_allocator -j
./build/bin/p1/p1_03_memory_pool_allocator
```

## Expected output

```
--- Correctness ---
  [PASS] new/delete checksum
  [PASS] bump allocator checksum
  [PASS] pool allocator checksum
  [PASS] pool reuses freed blocks
  [PASS] arena reports exhaustion

--- Performance ---
Variant                           Time (ms)  ...   Speedup
new / delete                          ~14.0            1.00x
bump allocator                         ~1.6            ~9x
pool allocator                         ~2.6            ~5x
```

## Key takeaways

- **Allocation is not free.** `new` walks free lists, may take a lock and may
  call into the kernel; on the GPU the equivalent is ~1000× worse.
- **Constrain the problem to make it fast.** The arena is fast *because* it gives
  up individual free.
- **Intrusive data structures cost nothing.** Storing the free-list link inside
  the free block is the whole trick.
- Raw storage + **placement new** is how you separate allocation from
  construction — the same split as `cudaMalloc` + a constructor kernel.

## Going further

- Add a `reset()`-safe `Arena::create<T>(args...)` that also records destructors.
- Make `PoolAllocator` thread-safe with a lock-free `compare_exchange` on the head
  — then compare that with the GPU lock-free queue in Phase 4 exercise 10.
- Read the CUDA docs on
  [stream-ordered memory allocation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator).
