<!-- Language: **English** | [Tiếng Việt](ROADMAP.vi.md) -->

# Roadmap

> From zero to GPU expert in 6–18 months. 63 exercises, 5 projects, one
> continuous argument: **memory is the bottleneck, and you must measure to know.**

[← back to the repository](../README.md)

---

## How the curriculum is put together

Every exercise follows the same shape — **read the theory, fill in the `TODO`s,
verify against a CPU reference, measure, compare.** Nothing is asserted that the
program does not demonstrate.

Three threads run the whole way through and are worth naming, because noticing
them is most of the learning:

1. **Blocks, not bytes.** Hardware moves data in fixed-size units. A CPU cache
   line (Phase 1/05) is a GPU memory sector (Phase 3/01). Using part of a block
   wastes the rest.
2. **Reuse beats bandwidth.** Cache blocking (Phase 1/02) *is* shared-memory
   tiling (Phase 3/04) *is* kernel fusion (Phase 5/14).
3. **Latency is hidden by parallelism.** Multiple accumulators on a CPU
   (Phase 1/06–07) become many resident warps on a GPU (Phase 3/15).

Times assume ~20 h/week. Halve them if you already write systems C++.

---

## Phase 1 · Foundation

**3–4 weeks · ⭐ · [directory](../phase-1-foundation/)**

You cannot optimise a GPU kernel if you cannot explain why a CPU loop is slow.
Every concept here reappears on the GPU under a different name.

| # | Exercise | Core idea |
|---|---|---|
| 01 | [matmul-naive-cpu](../phase-1-foundation/exercises/01-matmul-naive-cpu/) | Row-major layout, FLOP counting, honest benchmarking (warmup + median) |
| 02 | [matmul-cache-blocking](../phase-1-foundation/exercises/02-matmul-cache-blocking/) | Loop order is worth 10×; blocking pays only once the cache gives up |
| 03 | [memory-pool-allocator](../phase-1-foundation/exercises/03-memory-pool-allocator/) | Arena and pool allocators — why `cudaMalloc` must be avoided in hot paths |
| 04 | [modern-cpp-toolkit](../phase-1-foundation/exercises/04-modern-cpp-toolkit/) | RAII, move semantics, templates, lambdas — the tools for a device-buffer class |
| 05 | [cache-and-bandwidth](../phase-1-foundation/exercises/05-cache-and-bandwidth/) | Measure your cache line, your cache levels, your DRAM bandwidth |
| 06 | [simd-intrinsics](../phase-1-foundation/exercises/06-simd-intrinsics/) | AVX2 lanes ≈ warp lanes; SIMD only helps compute-bound code |
| 07 | [roofline-model](../phase-1-foundation/exercises/07-roofline-model/) | Build your own roofline; find the register cliff |
| 08 | [gpu-device-query](../phase-1-foundation/exercises/08-gpu-device-query/) | Your GPU's peaks, ridge point and occupancy budget |

**Checkpoint.** You can state, for your own machine: the cache line size, the
sustained DRAM bandwidth, the single-core ridge point, and — for your GPU — the
peak bandwidth, peak FP32, ridge point and registers-per-thread budget.

---

## Phase 2 · CUDA Fundamentals

**4–6 weeks · ⭐⭐ · [directory](../phase-2-cuda-fundamentals/)**

Write correct kernels and understand what a launch actually costs. Optimisation
comes later; correctness and honest measurement come first.

| # | Exercise | Core idea |
|---|---|---|
| 01 | [hello-cuda](../phase-2-cuda-fundamentals/exercises/01-hello-cuda/) | `__global__`, launch syntax, global id, bounds checks, async errors |
| 02 | [thread-indexing](../phase-2-cuda-fundamentals/exercises/02-thread-indexing/) | 1D/2D/3D grids, and the grid-stride loop |
| 03 | vector-add | The full `cudaMalloc`/`Memcpy`/`Free` cycle — and why PCIe often wins |
| 04 | saxpy | Effective bandwidth as the metric for memory-bound kernels |
| 05 | error-handling | Sticky errors, sync vs async failures, `compute-sanitizer` |
| 06 | matrix-add-2d | 2D launches on real 2D data |
| 07 | matmul-naive | The GPU baseline every later matmul is measured against |
| 08 | transpose-naive | A kernel that is *correct* and *slow* — sets up Phase 3 |
| 09 | rgb-to-grayscale | Image data, PPM I/O, per-pixel parallelism |
| 10 | box-blur | Stencils, halos, boundary handling |
| 11 | unified-memory | `cudaMallocManaged`, page migration, `cudaMemPrefetchAsync` |
| 12 | device-buffer-class | RAII around device memory — Phase 1/04 applied |

**Checkpoint.** You can write a correct kernel for a new problem, manage device
memory without leaking, report effective bandwidth, and explain why a vector add
is *slower* on the GPU once transfers are counted.

---

## Phase 3 · Intermediate

**6–8 weeks · ⭐⭐⭐ · [directory](../phase-3-intermediate/)**

The heart of the curriculum. This is where "it works" becomes "it is fast", and
where the payoff for Phase 1 arrives.

| # | Exercise | Core idea |
|---|---|---|
| 01 | memory-coalescing | The single biggest GPU performance factor, measured |
| 02 | shared-memory-basics | `__shared__`, `__syncthreads__`, block-local cooperation |
| 03 | bank-conflicts | 32 banks, why padding by one element fixes a 32× slowdown |
| 04 | tiled-matmul | Phase 1/02's blocking, in shared memory — here it is worth 5–10× |
| 05 | transpose-optimized | Coalesced read *and* write via a shared tile |
| 06 | reduction-variants | Six kernels, each faster than the last — the classic study |
| 07 | warp-shuffle | `__shfl_down_sync`, ballot, vote — registers instead of shared memory |
| 08 | atomics | `atomicAdd`, `atomicCAS`, contention, custom float atomics |
| 09 | histogram | Privatisation: turning global contention into shared-memory contention |
| 10 | scan-hillis-steele | Inclusive scan within a block |
| 11 | scan-blelloch | Work-efficient scan, arbitrary length, multi-block |
| 12 | stream-compaction | Scan + scatter, the backbone of filtering on a GPU |
| 13 | conv-1d-constant | `__constant__` memory and its broadcast cache |
| 14 | conv-2d-shared | 2D stencil with a haloed shared tile |
| 15 | occupancy-tuning | Registers vs shared memory vs block size; when high occupancy hurts |
| 16 | profiling-nsight | Nsight Systems and Nsight Compute on your own kernels |

**Checkpoint.** Given a slow kernel you can profile it, name the limiter from the
metrics, apply the right fix, and prove the improvement.

---

## Phase 4 · Advanced

**8–10 weeks · ⭐⭐⭐⭐ · [directory](../phase-4-advanced/)**

Beyond one kernel: overlapping, scaling out, and reading what the compiler
actually produced.

| # | Exercise | Core idea |
|---|---|---|
| 01 | pinned-memory | Pageable vs pinned transfer bandwidth |
| 02 | streams-basics | Concurrency, the default-stream trap |
| 03 | streams-pipeline | Overlap H2D → kernel → D2H, chunked |
| 04 | events-and-sync | `cudaEvent`, `cudaStreamWaitEvent`, dependency graphs by hand |
| 05 | cuda-graphs | Capture a pipeline; eliminate per-launch overhead |
| 06 | cooperative-groups | Tiled partitions, grid-wide synchronisation |
| 07 | dynamic-parallelism | Kernels launching kernels (CDP2 semantics, CUDA 12+) |
| 08 | multi-gpu-basics | Device enumeration, peer access, P2P copies |
| 09 | multi-gpu-matmul | Splitting work across devices |
| 10 | lock-free-queue | `atomicCAS`, `__threadfence`, memory ordering on a GPU |
| 11 | register-pressure | `__launch_bounds__`, spilling, the occupancy trade-off |
| 12 | ptx-and-sass | `cuobjdump`, `nvdisasm`, inline PTX |
| 13 | persistent-kernel | Megakernels and producer/consumer on the device |

Exercises 08 and 09 detect a single-GPU machine and skip cleanly.

**Checkpoint.** You can overlap transfer with compute, keep multiple GPUs busy,
read SASS to explain a stall, and reason about memory ordering between threads.

---

## Phase 5 · Expert

**6–8 weeks · ⭐⭐⭐⭐⭐ · [directory](../phase-5-expert/)**

Stop writing everything yourself: know the ecosystem, and know when a library
will beat you (it usually will).

| # | Exercise | Core idea |
|---|---|---|
| 01 | thrust-basics | STL-style algorithms on the device |
| 02 | cub-primitives | Block- and device-level building blocks vs your hand-written versions |
| 03 | cublas-gemm | The column-major trap; how far your tiled matmul really is |
| 04 | cublas-batched | Batched and strided-batched GEMM |
| 05 | curand-monte-carlo | Device-side RNG, and what "parallel random" means |
| 06 | cufft-convolution | Convolution via FFT |
| 07 | cusparse-spmv | CSR SpMV — irregular, memory bound, unavoidable |
| 08 | mixed-precision | FP16/BF16/TF32, `half2` vectorisation, accuracy vs speed |
| 09 | tensor-core-wmma | The WMMA API (needs sm_70+) |
| 10 | cutlass-gemm | Templated GEMM, epilogue fusion *(optional dependency)* |
| 11 | pytorch-extension | A custom CUDA op callable from Python *(optional dependency)* |
| 12 | nvrtc-jit | Runtime compilation and the driver API |
| 13 | cuda-opengl-interop | Zero-copy visualisation *(optional dependency)* |
| 14 | kernel-fusion | Fewer passes over memory — the highest-leverage optimisation left |

Optional-dependency exercises build only with `-DCL_ENABLE_OPTIONAL=ON`.

**Checkpoint.** You reach for the right library first, can call a Tensor Core
kernel, and can explain when writing your own is justified.

---

## Phase 6 · Mastery

**Ongoing · 🏆 · [directory](../phase-6-mastery/)**

Each project ships an architecture document, a milestone breakdown, evaluation
criteria, and one core module implemented as a worked example. The rest is yours.

| Project | What you will build |
|---|---|
| gpu-ray-tracer | Path tracing with BVH traversal, multiple bounces, denoising |
| mini-dl-framework | Autograd, forward/backward kernels, SGD/Adam, operator fusion |
| fluid-simulation | Navier–Stokes or SPH, real-time, with visualisation |
| gpu-database | Columnar storage, GPU filter/join/aggregate, query compilation |
| llm-inference-engine | KV-cache management, paged attention, batched decoding |

**Checkpoint.** Two finished projects, a profiler report for each, and a written
account of what you optimised and why.

---

## Progress tracking

| Level | Criterion | Evidence |
|---|---|---|
| Beginner | Correct kernels, device memory managed safely | Phases 1–2 complete |
| Intermediate | Profile, diagnose, optimise | Phase 3 complete |
| Advanced | Streams, multi-GPU, async execution, SASS | Phase 4 complete |
| Expert | Libraries, Tensor Cores, custom framework ops | Phase 5 complete |
| Master | Shipped projects, community contributions | Phase 6 ongoing |

Benchmarks to hold yourself to:

| Kernel | Beginner | Expert |
|---|---|---|
| SGEMM 4096³ | < 5% of cuBLAS | > 80% of cuBLAS |
| Reduction | < 10% of peak bandwidth | > 90% of peak bandwidth |
| 2D convolution | naive | > 70% of cuDNN |
| Memory copy | < 30% of theoretical | > 85% of theoretical |

---

## Suggested weekly rhythm

| Day | Focus | Hours |
|---|---|---|
| Mon | Theory — read the exercise README and its references | 2–3 |
| Tue | Code — fill in the `TODO`s | 3–4 |
| Wed | Theory + code | 2–3 |
| Thu | Code — finish, verify, measure | 3–4 |
| Fri | Profile and optimise; compare against the reference solution | 2–3 |
| Sat | Project work | 4–6 |
| Sun | Review, papers, blog posts | 2–3 |

---

## Further reading

**Books, in reading order**

1. *CUDA by Example* — Sanders & Kandrot (beginner)
2. *Programming Massively Parallel Processors*, 4th ed. — Kirk & Hwu (the standard text)
3. *Professional CUDA C Programming* — Cheng, Grossman & McKercher
4. *The CUDA Handbook* — Nicholas Wilt (reference)

**Documentation**

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

**Papers**

- Volkov, *Understanding Latency Hiding on GPUs* (2016)
- Micikevicius et al., *Mixed Precision Training* (2018)
- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023)
- Williams, Waterman & Patterson, *Roofline* (CACM 2009)

**Courses**

- [NVIDIA DLI — Fundamentals of Accelerated Computing with CUDA C/C++](https://www.nvidia.com/en-us/training/)
- [Coursera — GPU Programming Specialization (Johns Hopkins)](https://www.coursera.org/specializations/gpu-programming)
- [Udacity CS344 — Intro to Parallel Programming](https://github.com/udacity/cs344) (archived, still excellent)

**Blogs**

- [NVIDIA Technical Blog](https://developer.nvidia.com/blog)
- [Simon Boehm — How to optimise a CUDA matmul kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Lei Mao's blog](https://leimao.github.io/)

**Tools**

- [Nsight Systems](https://developer.nvidia.com/nsight-systems) · [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [Compiler Explorer](https://godbolt.org/) — has CUDA support, useful for reading PTX
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Additional links are collected in [`resources/README.md`](../resources/README.md).

---

> **The one piece of advice that matters.** CUDA cannot be learned from theory.
> Code every day, profile everything, and always compare against the best
> available implementation (cuBLAS, cuDNN, CUB). The gap between a kernel that
> *runs* and a kernel that is *fast* is where the learning actually happens.
