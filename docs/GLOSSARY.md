<!-- Language: **English** | [Tiếng Việt](GLOSSARY.vi.md) -->

# Glossary

[← back to the repository](../README.md)

The Vietnamese column gives the term used in the `.vi.md` documents. The English
term is the one you will meet in NVIDIA documentation, so **learn the concept in
whichever language you prefer, but keep the English name** — it is what you will
search for.

---

## Execution model

| English | Tiếng Việt | Meaning |
|---|---|---|
| Thread | Thread / luồng | The smallest unit of execution. Has its own registers and program counter. |
| Warp | Warp | 32 threads issuing one instruction together. The real scheduling unit. |
| Thread block | Block / khối thread | A group of threads that runs on one SM and can share `__shared__` memory and synchronise. |
| Grid | Grid / lưới | All the blocks created by one launch. |
| Kernel | Kernel | A `__global__` function that runs on the GPU. |
| Launch | Launch / khởi chạy | `kernel<<<grid, block>>>(args)`. Asynchronous. |
| SM (Streaming Multiprocessor) | SM / bộ đa xử lý dòng | The hardware core a block runs on. A GPU has tens of them. |
| SIMT | SIMT | Single Instruction, Multiple Threads — NVIDIA's variant of SIMD. |
| Warp divergence | Phân kỳ warp | Threads in a warp taking different branches; both sides execute serially. |
| Occupancy | Occupancy / độ chiếm dụng | Resident warps per SM ÷ the maximum. More warps hide more latency. |
| Grid-stride loop | Vòng lặp grid-stride | Each thread walks the array in steps of the total thread count. |

## Memory

| English | Tiếng Việt | Meaning |
|---|---|---|
| Global memory | Bộ nhớ toàn cục | Device DRAM. Large, slow, visible to everyone. |
| Shared memory | Shared memory / bộ nhớ chia sẻ | On-chip scratchpad, per block. A manually managed cache. |
| Register | Thanh ghi | Fastest storage, private to a thread. Scarce. |
| Local memory | Bộ nhớ cục bộ | Per-thread spill space that actually lives in DRAM. Slow. |
| Constant memory | Bộ nhớ hằng | 64 KB read-only region with a broadcast cache. |
| Texture memory | Bộ nhớ texture | Read-only path with hardware interpolation and 2D locality. |
| Unified / managed memory | Bộ nhớ hợp nhất | One pointer valid on host and device; pages migrate on demand. |
| Pinned (page-locked) memory | Bộ nhớ ghim | Host memory the OS cannot swap, so DMA can read it directly. |
| Coalescing | Coalescing / gộp truy cập | Merging the addresses of a warp into as few transactions as possible. |
| Bank conflict | Xung đột bank | Two threads in a warp hitting different rows of the same shared-memory bank; serialised. |
| Cache line / sector | Cache line / sector | The fixed-size block memory actually moves in. |
| Bandwidth | Băng thông | Bytes per second. |
| Latency | Độ trễ | Time until one request returns. Hidden by parallelism, not reduced by it. |

## Performance

| English | Tiếng Việt | Meaning |
|---|---|---|
| Throughput | Thông lượng | Work completed per unit time. |
| Arithmetic intensity | Cường độ số học | FLOPs per byte moved. The x-axis of the roofline. |
| Roofline model | Mô hình roofline | `min(peak compute, intensity × peak bandwidth)`. |
| Ridge point | Điểm gãy | Where the two roofs meet: the intensity at which a kernel stops being memory bound. |
| Memory bound | Nghẽn bộ nhớ | Limited by data movement. Fix by moving fewer bytes. |
| Compute bound | Nghẽn tính toán | Limited by arithmetic. Fix with better instructions. |
| FLOP / GFLOP/s | FLOP / GFLOP/s | Floating point operation; billions of them per second. |
| Effective bandwidth | Băng thông hữu hiệu | Useful bytes ÷ time — as opposed to bytes the bus moved. |
| Speedup | Mức tăng tốc | Baseline time ÷ optimised time. |
| Register spilling | Tràn thanh ghi | Too many live values, so the compiler stores them in local memory. |
| ILP | Song song mức lệnh | Independent instructions in flight at once; the cure for latency. |

## Parallel patterns

| English | Tiếng Việt | Meaning |
|---|---|---|
| Reduction | Reduction / rút gọn | Many values to one (sum, max, ...). |
| Scan / prefix sum | Scan / tổng tiền tố | Running totals across an array. |
| Stream compaction | Nén dòng | Keeping only the elements that pass a predicate. |
| Stencil | Stencil | Each output reads a fixed neighbourhood of the input. |
| Tiling / blocking | Tiling / chia khối | Working on a cache- or shared-memory-sized chunk to create reuse. |
| Halo | Halo / vành biên | The extra border a tile must load for a stencil. |
| Privatisation | Riêng tư hoá | Per-block copies to turn global contention into local contention. |
| Kernel fusion | Gộp kernel | Merging kernels to avoid a round trip through DRAM. |
| Atomic operation | Phép toán nguyên tử | Read-modify-write that no other thread can interleave with. |

## Tooling and API

| English | Tiếng Việt | Meaning |
|---|---|---|
| Compute capability | Compute capability | The GPU's feature/ISA level, e.g. `sm_86`. Not a speed. |
| PTX | PTX | Portable virtual assembly, JIT-compiled by the driver. |
| SASS | SASS | The real machine code for one architecture. |
| Stream | Stream | An ordered queue of GPU work. Different streams may overlap. |
| Event | Event | A marker in a stream, used for timing and cross-stream dependencies. |
| CUDA Graph | CUDA Graph | A recorded DAG of operations, replayed with almost no launch overhead. |
| Cooperative groups | Cooperative groups | An API for synchronising subsets of threads, up to the whole grid. |
| Dynamic parallelism | Song song động | A kernel launching another kernel. |
| Peer access (P2P) | Truy cập ngang hàng | One GPU reading another GPU's memory directly. |
| Nsight Systems | Nsight Systems | Whole-application timeline profiler (`nsys`). |
| Nsight Compute | Nsight Compute | Per-kernel deep profiler (`ncu`). |
| compute-sanitizer | compute-sanitizer | Detects invalid memory access, races and sync errors. |

## Libraries

| Name | Purpose |
|---|---|
| cuBLAS | Dense linear algebra (GEMM and friends) |
| cuDNN | Deep learning primitives (convolution, pooling, normalisation) |
| cuFFT | Fast Fourier transforms |
| cuRAND | Random number generation |
| cuSPARSE | Sparse matrices |
| cuSOLVER | Linear solvers and eigenvalue problems |
| Thrust | STL-style algorithms on the device |
| CUB | Block- and device-level primitives, the layer under Thrust |
| CUTLASS | Templated, tunable GEMM building blocks |
| NCCL | Multi-GPU collective communication |
| TensorRT | Inference optimisation and deployment |

---

## Naming conventions in this repository

| Symbol | Meaning |
|---|---|
| `d_x` | a pointer to **device** memory |
| `h_x` | a pointer to **host** memory |
| `p2_03_vector_add` | phase 2, exercise 03 — the **starter** |
| `p2_03_vector_add_sol` | the same exercise — the **reference solution** |
| `TODO` | something for you to write |
| `[PASS]` / `[FAIL]` / `[SKIP]` | a verification result; `[SKIP]` is not a failure |
