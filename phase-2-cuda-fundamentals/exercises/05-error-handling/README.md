<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 05 - Error handling: recoverable, sticky, and asynchronous

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1 h | Prerequisites: [03](../03-vector-add/) | **Requires a GPU**

## Goal

Unusually for this repository, the goal is to **make CUDA fail on purpose** and
watch how each kind of failure behaves. GPU bugs feel worse than CPU bugs mainly
because the error is reported far away from the code that caused it — this
exercise shows exactly why.

## Background

### Two kinds of error

**Recoverable** — detected *before* anything ran: an invalid launch configuration,
a failed allocation, a bad argument. `cudaGetLastError()` returns it, clears the
flag, and the context carries on working.

**Sticky** — raised *during* execution: an illegal address, a device-side assert.
These **destroy the CUDA context**. Every subsequent call — `cudaMalloc`,
`cudaMemcpy`, any launch — returns the same error forever. The only recovery is to
exit the process.

### The trap that costs people whole afternoons

A failing API call **both returns the error and sets the last-error flag**.
Reading the return value does *not* clear the flag:

```cuda
cudaError_t e = cudaMalloc(&p, huge);   // returns cudaErrorMemoryAllocation
cudaGetLastError();                     // ALSO returns it - the flag was set
cudaGetLastError();                     // only now is it clear
```

So if you ignore one failed call, the next `CUDA_CHECK_LAST()` after a completely
unrelated kernel launch reports *that* error — and you go hunting through a kernel
that was never broken. **Check the return value of every call, at the call.**

### `peek` versus `get`

| Call | Returns the error | Clears the flag |
|---|---|---|
| `cudaPeekAtLastError()` | yes | **no** |
| `cudaGetLastError()` | yes | **yes** |

### Asynchrony

A launch returns before the kernel runs, so an error inside a kernel surfaces at
the next synchronisation point — often an innocent `cudaMemcpy` hundreds of lines
away. When a failure makes no sense:

```bash
CUDA_LAUNCH_BLOCKING=1 ./your_program
```

The runtime then waits after every launch and reports the error at the line that
caused it. It is slow; use it only to diagnose.

### Small overruns are the dangerous ones

A write a few kilobytes past the end usually is **not** trapped — the runtime
allocates in large pages, so it lands in memory the process already owns and
silently corrupts it. Only `compute-sanitizer` finds those:

```bash
compute-sanitizer ./your_program                    # invalid access
compute-sanitizer --tool racecheck  ./your_program  # data races
compute-sanitizer --tool synccheck  ./your_program  # bad __syncthreads
compute-sanitizer --tool initcheck  ./your_program  # uninitialised reads
```

## Your task

Open `main.cu` and run three experiments:

1. **Recoverable** — launch with >1024 threads per block, read the error yourself
   (not `CUDA_CHECK_LAST()`, which calls `exit`), confirm it is
   `cudaErrorInvalidConfiguration`, then prove a valid launch still works.
2. **Peek vs get** — trigger a failure, then call each twice.
3. **Sticky** — write far outside an allocation. Check the error immediately after
   the launch (is it an error yet? why not?), then after
   `cudaDeviceSynchronize()`, then try any unrelated call. **Put this last** — the
   context is dead afterwards.

Then run the binary under `compute-sanitizer`.

## Build and run

```bash
cmake --build build --target p2_05_error_handling -j
./build/bin/p2/p2_05_error_handling
compute-sanitizer ./build/bin/p2/p2_05_error_handling
```

## Expected output

```
--- Recoverable errors ---
  <<<4, 256>>> (valid)                       ok
  <<<1, 2048>>> (too many threads per block) cudaErrorInvalidConfiguration
  <<<4, 256>>> (valid again)                 ok
  cudaMalloc(1 PB)                           cudaErrorMemoryAllocation
  cudaGetLastError() after that              cudaErrorMemoryAllocation
  cudaGetLastError() again                   cudaSuccess

--- Sticky errors (this destroys the context) ---
  launch error       : cudaSuccess          <- the launch was fine
  after synchronise  : cudaErrorIllegalAddress
  cudaMalloc(16)     : cudaErrorIllegalAddress    <- context is dead
```

Note that the launch itself reports **success**. The damage happens later.

## Key takeaways

- **Recoverable errors leave the context usable; sticky ones do not.** Only
  process exit recovers from a sticky error.
- **A failed call sets the flag as well as returning the error.** An unchecked
  failure gets blamed on the next kernel you launch.
- **`peek` inspects, `get` consumes.**
- **Launches are asynchronous**, so the reported location is not the real
  location. `CUDA_LAUNCH_BLOCKING=1` fixes that while you debug.
- **Small overruns corrupt silently.** Run `compute-sanitizer` before believing a
  kernel is correct.

## The five rules

1. Wrap every runtime call in `CUDA_CHECK(...)`.
2. After every launch, `CUDA_CHECK_LAST()` for configuration errors.
3. While debugging, `CUDA_CHECK_KERNEL()` to catch execution errors at the launch
   site. Remove it from timing loops — it costs a full device sync.
4. When an error appears somewhere impossible, suspect an earlier unchecked one
   and rerun with `CUDA_LAUNCH_BLOCKING=1`.
5. Run `compute-sanitizer` before believing any kernel.

## Going further

- Add `assert(idx < n)` inside a kernel and trigger it. What error do you get?
- Use `cudaGetErrorName()` versus `cudaGetErrorString()`. When is each better?
- Read the [CUDA Runtime API error handling docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html).
