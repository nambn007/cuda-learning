<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 04 - Modern C++ toolkit: RAII, moves, templates, lambdas

> **Phase 1 · Foundation** | Difficulty: ⭐⭐ | Time: ~2 h | Prerequisites: [03](../03-memory-pool-allocator/)

## Goal

Build one small class, `Matrix<T>`, that exercises the four C++ features you will
lean on for the rest of the curriculum. In Phase 2 exercise 12 you rewrite this
same class around `cudaMalloc`/`cudaFree`, so getting it right here pays off twice.

## Background

**RAII** — a resource is owned by an object, and the destructor releases it. There
is no code path, not even a thrown exception or an early `return`, that can leak.
This matters far more on a GPU than on a CPU: a leaked host allocation is
reclaimed when the process exits, but a forgotten `cudaFree` inside a long-running
service leaks device memory until the whole context is destroyed.

**Move semantics** — a move steals the pointer instead of duplicating the data.
For a `Matrix` that is three assignments versus copying megabytes. Two rules:

- A moved-from object must remain **valid and destructible** — set its size to 0
  and null its pointer.
- Mark move operations **`noexcept`**. `std::vector` only moves its elements while
  reallocating if the move constructor is `noexcept`; otherwise it silently copies
  them to preserve its strong exception guarantee.

**Templates** — one implementation, many types. On the GPU you will write one
kernel wrapper and instantiate it for `float`, `double` and `__half`.

**Lambdas** — `apply()` is templated on the callable rather than taking a
`std::function`. That means the lambda body is *inlined into the loop*: no virtual
call, no allocation, identical machine code to a hand-written loop. Thrust and CUB
rely on exactly this to make device functors free.

## Your task

Open `main.cpp` and implement `Matrix<T>` (TODO 1–7):

1. Constructor — `std::make_unique<T[]>(rows * cols)`, notify `AllocationTracker`.
2. Destructor — notify the tracker (`unique_ptr` frees the memory).
3. Copy constructor — **deep** copy.
4. Move constructor — steal, then empty the source. `noexcept`.
5. Copy and move assignment. Handle self-assignment; copy-and-swap is the tidy way.
6. `operator()(r, c)` — row-major indexing.
7. `apply(fn)` — templated on the callable.

Then (TODO 8) write checks proving: element access round-trips, a move performs
**no** allocation, a moved-from matrix is empty, and nothing leaks.

## Build and run

```bash
cmake --build build --target p1_04_modern_cpp_toolkit -j
./build/bin/p1/p1_04_modern_cpp_toolkit
```

## Expected output

```
--- RAII and element access ---
  [PASS] constructor allocated once
  [PASS] element access round-trips
  [PASS] nothing leaked after scope exit

--- Copy versus move ---
  [PASS] copy allocates new storage
  [PASS] move allocates nothing
  [PASS] moved-from matrix is empty

--- Zero-cost abstraction ---
Variant                           Time (ms)      GB/s ...
raw std::vector loop                   ~6.5      ~5.1
Matrix<T>::apply(lambda)               ~6.5      ~5.1
```

The last two rows being equal *is* the result: the abstraction is free.

## Key takeaways

- **RAII means no leak is possible**, not "no leak if you remember".
- A move is **O(1)**; a copy is **O(n)**. Returning by value is cheap once the
  move constructor exists.
- **`noexcept` on moves is load-bearing**, not documentation.
- Templates + lambdas are resolved at compile time — **zero-cost abstraction** is
  literal, and you can prove it with a benchmark.

## Going further

- Add `Matrix<T>::create(rows, cols)` returning `std::optional<Matrix<T>>` for a
  failure path that does not throw.
- Delete the copy constructor entirely and see which call sites stop compiling —
  that is exactly what a device buffer should do.
- Compile with `-fno-elide-constructors` and rerun. How many allocations now?
- Read *Effective Modern C++* (Scott Meyers), items 23–30 on move semantics and
  perfect forwarding.
