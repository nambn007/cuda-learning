<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 12 - A RAII device buffer

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~2 h | Prerequisites: [Phase 1/04](../../../phase-1-foundation/exercises/04-modern-cpp-toolkit/), [03](../03-vector-add/) | **Requires a GPU**

## Goal

Apply Phase 1's `Matrix<T>` to device memory, and **prove** — by measuring free
VRAM — that the RAII version cannot leak while the C-style version does. This is
the last exercise of Phase 2, and the class you build here is used for the rest of
the curriculum.

## Background

A leaked host allocation is reclaimed when the process exits. **A forgotten
`cudaFree` leaks device memory that nothing reclaims until the CUDA context is
destroyed** — so a service leaking 1 MB per request eventually dies with
`cudaErrorMemoryAllocation` and no clue where it went.

Worse, the C-style shape *invites* the bug:

```cuda
float* d = nullptr;
cudaMalloc(&d, bytes);
if (somethingWrong) return -1;      // leaked
...
cudaFree(d);
```

Every early return, every thrown exception, every `goto fail` is a leak. RAII
removes the possibility instead of reminding you about it.

### Three design decisions worth arguing about

**Copy is deleted, not implemented.** An implicit 16 MB device-to-device copy is
exactly the kind of expensive operation that should never happen by accident.
Provide an explicit `clone()` so the cost is visible at the call site.

**Move is `noexcept`.** Not decoration: without it `std::vector` copies instead of
moving when it reallocates — and since the copy is deleted, code that puts these in
a container **will not compile at all**. `noexcept` is what makes the class usable.

**The destructor does not use `CUDA_CHECK`.** `CUDA_CHECK` calls `exit()`, and
during process teardown the context may already be gone, in which case `cudaFree`
legitimately returns an error. Exiting from a destructor is worse than the leak it
would be reporting.

## Your task

Open `main.cu`:

1. **`DeviceBuffer<T>`** — constructor, destructor, deleted copy, `noexcept` move,
   `clone()`, `copyFromHost()`, `copyToHost()`, `zero()`, accessors. Throw
   `std::out_of_range` on an oversized copy; silent truncation on a GPU is how you
   get corruption three kernels later.
2. **`PinnedBuffer<T>`** — the same pattern around `cudaMallocHost` /
   `cudaFreeHost`.
3. **Prove it** with `cudaMemGetInfo()`: measure free VRAM before and after 20
   leaky calls, then after 20 RAII scopes unwound by a thrown exception.

## Build and run

```bash
cmake --build build --target p2_12_device_buffer_class -j
./build/bin/p2/p2_12_device_buffer_class
```

## Expected output

RTX 3060, 16 MB buffers:

```
--- Basic use ---
  [PASS] scale by 3 (n=4194304, max abs err 0.000e+00)

--- Leaks ---
  20 early returns from the raw C-style version leaked 320.0 MB
  [PASS] the raw version really does leak
  20 thrown exceptions through the RAII version leaked 0.0 MB
  [PASS] RAII leaks nothing, even when unwinding

--- Move semantics ---
  [PASS] move transfers the pointer
  [PASS] moved-from buffer is empty
  [PASS] clone allocates separate storage

--- Use in containers ---
  [PASS] vector holds four live buffers

--- Pinned host memory ---
pageable host memory                  1.970      8.52 GB/s     1.00x
pinned host memory                    1.327     12.65 GB/s     1.48x
```

**320 MB versus 0 MB.** That is not a style preference, it is the difference
between a service that runs for a week and one that does not.

## Key takeaways

- **Device leaks are worse than host leaks.** Nothing reclaims them until the
  context dies.
- **RAII makes the leak impossible**, including on paths you have not thought of —
  early returns, exceptions, code somebody adds six months from now.
- **Delete copy for expensive resources.** Make the cost explicit with `clone()`.
- **`noexcept` on moves is what makes a type container-usable.**
- **Never `CUDA_CHECK` in a destructor.**
- **Pinned memory is ~1.5× faster** and is a machine-wide scarce resource — the
  second reason RAII matters here.

## Going further

- Add a `DeviceBuffer<T>::resize()` that preserves contents. When is that worth it
  versus allocating a new one?
- Add a stream-aware constructor using `cudaMallocAsync` (CUDA 11.2+). Compare the
  allocation cost — this is Phase 1/03's memory pool, built into the driver.
- Add a debug mode that records every live allocation with its call site, and
  prints anything still outstanding at exit.
- Compare with `thrust::device_vector` (Phase 5/01). What does it do differently,
  and why?

---

**That completes Phase 2.** You can now write correct CUDA kernels, manage device
memory safely, and report honest performance numbers.
[Phase 3](../../../phase-3-intermediate/) is where they get fast.
