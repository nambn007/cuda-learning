# 🗺️ CUDA Mastery Roadmap – Lộ trình chi tiết

> **Mục tiêu**: Từ zero đến chuyên gia CUDA trong 6-18 tháng

---

## ⚡ Phase 1: Foundation – Nền tảng (3-4 tuần)

### 🎯 Mục tiêu
Nắm vững C/C++ và hiểu kiến trúc phần cứng GPU trước khi bắt đầu CUDA.

### 📚 Kiến thức cần nắm

#### 1.1 C/C++ nâng cao (Tuần 1-2)
- [ ] **Pointers & Memory Management**
  - Con trỏ, con trỏ hàm, con trỏ void
  - `malloc`, `calloc`, `realloc`, `free`
  - Stack vs Heap memory
  - Memory leaks và cách phát hiện (Valgrind)
- [ ] **C++ Modern Features**
  - Templates và template metaprogramming cơ bản
  - Smart pointers (`unique_ptr`, `shared_ptr`)
  - Move semantics và rvalue references
  - Lambda expressions
  - `constexpr` và compile-time computation
- [ ] **Build Systems**
  - Makefile nâng cao
  - CMake cơ bản đến trung cấp
  - Compiler flags và optimization levels (`-O0`, `-O2`, `-O3`, `-Ofast`)
- [ ] **Performance Profiling cơ bản**
  - `gprof`, `perf`, `Valgrind/Cachegrind`
  - Cache miss, branch prediction
  - SIMD intrinsics cơ bản (SSE, AVX)

#### 1.2 Kiến trúc máy tính & GPU (Tuần 3-4)
- [ ] **CPU Architecture Review**
  - Pipeline, superscalar, out-of-order execution
  - Cache hierarchy (L1, L2, L3)
  - Memory bandwidth và latency
  - NUMA architecture
- [ ] **GPU Architecture Fundamentals**
  - Lịch sử phát triển GPU: từ graphics pipeline đến GPGPU
  - So sánh CPU vs GPU: throughput vs latency
  - Streaming Multiprocessor (SM) architecture
  - CUDA Cores, Tensor Cores, RT Cores
  - Warp, thread block, grid concepts (lý thuyết)
- [ ] **NVIDIA GPU Generations**
  - Kepler → Maxwell → Pascal → Volta → Turing → Ampere → Hopper → Blackwell
  - Compute Capability và ý nghĩa
  - Kiến trúc SM qua các thế hệ

### 📖 Tài liệu tham khảo
| Tài liệu | Loại | Link |
|-----------|------|------|
| Computer Organization and Design (Patterson & Hennessy) | Sách | ISBN: 978-0128203316 |
| NVIDIA GPU Architecture Whitepapers | Whitepaper | developer.nvidia.com |
| C++ Primer (5th Edition) | Sách | ISBN: 978-0321714114 |

### 🏋️ Bài tập
1. Viết matrix multiplication bằng C++ thuần (naive + cache-optimized)
2. Implement memory pool allocator đơn giản
3. Profile và so sánh performance giữa các phiên bản
4. Vẽ sơ đồ kiến trúc GPU mà bạn đang sử dụng

---

## ⚡ Phase 2: CUDA Fundamentals – Lập trình CUDA cơ bản (4-6 tuần)

### 🎯 Mục tiêu
Viết được CUDA kernel đầu tiên, hiểu execution model và memory model cơ bản.

### 📚 Kiến thức cần nắm

#### 2.1 CUDA Programming Model (Tuần 1-2)
- [ ] **Hello CUDA**
  - Cài đặt CUDA Toolkit
  - File `.cu` và `nvcc` compiler
  - `__global__`, `__device__`, `__host__` qualifiers
  - Kernel launch syntax: `kernel<<<gridDim, blockDim>>>(args)`
  - `cudaDeviceSynchronize()`
- [ ] **Thread Hierarchy**
  - Thread → Warp → Block → Grid
  - `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
  - 1D, 2D, 3D thread organization
  - Tính toán global thread ID
  - Boundary checking trong kernel
- [ ] **Error Handling**
  - `cudaError_t` và error checking macro
  - `cudaGetLastError()` vs `cudaPeekAtLastError()`
  - Best practices cho error handling

#### 2.2 Memory Management cơ bản (Tuần 3-4)
- [ ] **Device Memory**
  - `cudaMalloc`, `cudaFree`
  - `cudaMemcpy` (H2D, D2H, D2D)
  - `cudaMemset`
  - Unified Memory (`cudaMallocManaged`) – giới thiệu
- [ ] **Memory Types Overview**
  - Global Memory
  - Shared Memory (giới thiệu)
  - Local Memory / Registers
  - Constant Memory (giới thiệu)
  - Texture Memory (giới thiệu)
- [ ] **Basic Optimization Concepts**
  - Coalesced memory access là gì
  - Occupancy là gì
  - Ý nghĩa của warp size (32)

#### 2.3 Các pattern cơ bản (Tuần 5-6)
- [ ] **Vector Operations**
  - Vector addition
  - Vector scaling (SAXPY/DAXPY)
  - Dot product (giới thiệu reduction)
- [ ] **Matrix Operations**
  - Matrix addition
  - Naive matrix multiplication
  - Matrix transpose
- [ ] **Image Processing cơ bản**
  - RGB to Grayscale conversion
  - Simple blur filter (box blur)

### 📖 Tài liệu tham khảo
| Tài liệu | Loại | Link |
|-----------|------|------|
| CUDA C++ Programming Guide | Official Doc | docs.nvidia.com/cuda |
| CUDA by Example (Sanders & Kandrot) | Sách | ISBN: 978-0131387683 |
| Professional CUDA C Programming (Cheng et al.) | Sách | ISBN: 978-1118739327 |
| NVIDIA CUDA Samples | Code | github.com/NVIDIA/cuda-samples |

### 🏋️ Bài tập
1. **Vector Add**: Viết kernel cộng hai vector, so sánh với CPU
2. **SAXPY**: `y = a*x + y` trên GPU, đo speedup
3. **Matrix Multiply**: Naive implementation, đo bandwidth
4. **Image Blur**: Đọc ảnh PNG, áp dụng box blur trên GPU
5. **Error Handling**: Tạo wrapper class cho CUDA memory management

### ✅ Checkpoint
> Bạn nên có thể: viết kernel đơn giản, quản lý bộ nhớ GPU, hiểu thread hierarchy, đo thời gian thực thi

---

## ⚡ Phase 3: Intermediate CUDA – Tối ưu hóa & Memory (6-8 tuần)

### 🎯 Mục tiêu
Master memory hierarchy, tối ưu hóa kernel, và sử dụng profiling tools.

### 📚 Kiến thức cần nắm

#### 3.1 Memory Optimization (Tuần 1-3)
- [ ] **Shared Memory Deep Dive**
  - `__shared__` declaration
  - Bank conflicts và cách tránh
  - Tiled matrix multiplication
  - Dynamic shared memory allocation
  - Shared memory as software-managed cache
- [ ] **Global Memory Optimization**
  - Coalesced access patterns chi tiết
  - Structure of Arrays (SoA) vs Array of Structures (AoS)
  - Memory alignment và padding
  - Vectorized loads (`float2`, `float4`, `int4`)
- [ ] **Constant Memory**
  - `__constant__` declaration
  - `cudaMemcpyToSymbol`
  - Constant cache behavior
  - Use cases: convolution kernels, lookup tables
- [ ] **Texture Memory & Surface Memory**
  - Texture objects (bindless textures)
  - Hardware interpolation
  - Boundary handling modes
  - Read-only cache (`__ldg()`)
- [ ] **Unified Memory (Advanced)**
  - Page migration mechanics
  - `cudaMemPrefetchAsync`
  - `cudaMemAdvise`
  - Performance implications

#### 3.2 Execution Optimization (Tuần 4-5)
- [ ] **Warp-Level Programming**
  - Warp divergence và control flow
  - Warp shuffle operations (`__shfl_sync`, `__shfl_down_sync`, etc.)
  - Warp vote functions (`__ballot_sync`, `__all_sync`, `__any_sync`)
  - Cooperative groups cơ bản
- [ ] **Occupancy Optimization**
  - Factors affecting occupancy: registers, shared memory, block size
  - `cudaOccupancyMaxPotentialBlockSize`
  - CUDA Occupancy Calculator
  - Khi nào occupancy cao KHÔNG tốt
- [ ] **Instruction-Level Optimization**
  - Arithmetic intensity
  - Instruction throughput vs memory throughput
  - Fast math (`--use_fast_math`, `__fdividef`, `__expf`)
  - Loop unrolling (`#pragma unroll`)

#### 3.3 Parallel Patterns (Tuần 6-7)
- [ ] **Reduction**
  - Sequential addressing reduction
  - Warp shuffle reduction
  - Multi-block reduction
  - Atomic operations (`atomicAdd`, `atomicCAS`, etc.)
- [ ] **Scan (Prefix Sum)**
  - Inclusive vs Exclusive scan
  - Hillis-Steele scan
  - Blelloch scan
  - Work-efficient scan
- [ ] **Histogram**
  - Naive histogram with atomics
  - Shared memory privatization
  - Sorting-based histogram
- [ ] **Compact / Stream Compaction**
  - Scatter và gather operations
  - Predicated compaction

#### 3.4 Profiling & Debugging (Tuần 8)
- [ ] **NVIDIA Nsight Systems**
  - Timeline analysis
  - API trace
  - Kernel launch analysis
  - Memory transfer analysis
- [ ] **NVIDIA Nsight Compute**
  - Kernel profiling chi tiết
  - Memory workload analysis
  - Compute workload analysis
  - Roofline model analysis
  - Speed of light (SOL) metrics
- [ ] **cuda-gdb & compute-sanitizer**
  - Race condition detection
  - Memory error detection
  - Synchronization errors
- [ ] **Đọc hiểu Performance Metrics**
  - DRAM throughput, L2 hit rate
  - Warp execution efficiency
  - Achieved occupancy
  - Instruction per clock (IPC)

### 📖 Tài liệu tham khảo
| Tài liệu | Loại | Link |
|-----------|------|------|
| CUDA C++ Best Practices Guide | Official Doc | docs.nvidia.com/cuda |
| Programming Massively Parallel Processors (Kirk & Hwu) | Sách | ISBN: 978-0323912310 |
| Nsight Compute Documentation | Tool Doc | docs.nvidia.com/nsight-compute |
| GPU Gems 3 – Chapter 39 (Parallel Prefix Sum) | Article | developer.nvidia.com |

### 🏋️ Bài tập
1. **Tiled MatMul**: Implement với shared memory, so sánh với naive
2. **Parallel Reduction**: Implement tất cả các biến thể, benchmark
3. **Prefix Sum**: Implement Blelloch scan cho array lớn
4. **Convolution 2D**: Sử dụng constant memory cho kernel, shared memory cho input tile
5. **Profile Session**: Profile tất cả bài tập trước đó bằng Nsight, viết report

### ✅ Checkpoint
> Bạn nên có thể: tối ưu hóa memory access, tránh bank conflicts, sử dụng profiler, implement parallel patterns cơ bản

---

## ⚡ Phase 4: Advanced CUDA – Kỹ thuật nâng cao (8-10 tuần)

### 🎯 Mục tiêu
Master CUDA streams, multi-GPU, dynamic parallelism, và advanced optimization.

### 📚 Kiến thức cần nắm

#### 4.1 Asynchronous Execution (Tuần 1-3)
- [ ] **CUDA Streams**
  - Default stream behavior
  - Creating và managing streams
  - Overlapping computation và data transfer
  - Stream priorities
  - Per-thread default stream
- [ ] **CUDA Events**
  - Timing với events
  - Stream synchronization với events
  - `cudaEventRecord`, `cudaEventSynchronize`
  - `cudaStreamWaitEvent`
- [ ] **CUDA Graphs**
  - Graph capture mode
  - Explicit graph construction
  - Graph instantiation và launch
  - Graph update
  - Performance benefits
- [ ] **Pinned Memory & Async Transfers**
  - `cudaMallocHost` / `cudaHostAlloc`
  - `cudaMemcpyAsync`
  - Write-Combined memory
  - Mapped pinned memory (zero-copy)

#### 4.2 Multi-GPU Programming (Tuần 4-5)
- [ ] **Multi-GPU Basics**
  - `cudaSetDevice`
  - Peer-to-peer access (`cudaDeviceEnablePeerAccess`)
  - P2P memory copy
  - Multi-GPU topology (NVLink, PCIe)
- [ ] **Multi-GPU Patterns**
  - Data parallelism across GPUs
  - Model parallelism concepts
  - Halo exchange pattern
  - Load balancing
- [ ] **NCCL (NVIDIA Collective Communication Library)**
  - AllReduce, Broadcast, AllGather
  - Ring-based vs tree-based algorithms
  - Integration với multi-GPU code

#### 4.3 Dynamic Parallelism & Cooperative Groups (Tuần 6-7)
- [ ] **Dynamic Parallelism**
  - Launching kernels from kernels
  - Memory visibility rules
  - Synchronization in nested kernels
  - Use cases: adaptive algorithms, recursive algorithms
- [ ] **Cooperative Groups (Advanced)**
  - Thread block groups
  - Grid-level synchronization
  - Multi-grid groups
  - Tiled partitions
  - Custom group types

#### 4.4 Advanced Optimization (Tuần 8-10)
- [ ] **Register Optimization**
  - Register pressure
  - `__launch_bounds__`
  - Register spilling
  - Trade-off: registers vs occupancy
- [ ] **Memory Access Patterns**
  - Global memory transaction sizes
  - L2 cache residency control (Ampere+)
  - `cudaAccessPolicyWindow`
  - Persistent kernels
- [ ] **Atomic Operations (Advanced)**
  - System-wide atomics
  - Custom atomic operations với `atomicCAS`
  - Lock-free data structures
  - Memory ordering và `__threadfence`
- [ ] **PTX & SASS**
  - Đọc PTX assembly cơ bản
  - `cuobjdump` để xem SASS
  - Inline PTX assembly
  - Hiểu scheduling và latency hiding

### 📖 Tài liệu tham khảo
| Tài liệu | Loại | Link |
|-----------|------|------|
| CUDA C++ Programming Guide (Advanced chapters) | Official Doc | docs.nvidia.com/cuda |
| GTC Presentations | Conference | nvidia.com/gtc |
| CUDA Graphs documentation | Official Doc | docs.nvidia.com/cuda |
| NCCL Developer Guide | Official Doc | docs.nvidia.com/nccl |

### 🏋️ Bài tập
1. **Pipeline**: Overlap H2D transfer → Kernel → D2H transfer với streams
2. **Multi-GPU MatMul**: Chia matrix ra nhiều GPU, tính song song
3. **CUDA Graph**: Convert một pipeline phức tạp sang CUDA Graph
4. **Merge Sort**: Implement với dynamic parallelism
5. **Lock-free Queue**: Implement trên GPU với atomics

### ✅ Checkpoint
> Bạn nên có thể: sử dụng streams và events, lập trình multi-GPU, đọc PTX, tối ưu hóa nâng cao

---

## ⚡ Phase 5: Expert Topics – Chuyên sâu & Thư viện (6-8 tuần)

### 🎯 Mục tiêu
Làm chủ CUDA ecosystem, thư viện chuyên dụng, và ứng dụng thực tế.

### 📚 Kiến thức cần nắm

#### 5.1 CUDA Libraries (Tuần 1-2)
- [ ] **cuBLAS**
  - GEMM operations
  - Batched operations
  - cuBLAS-Lt cho mixed precision
  - Tensor Core acceleration
- [ ] **cuDNN**
  - Convolution algorithms
  - Forward/backward pass
  - Workspace management
  - Fusion opportunities
- [ ] **cuFFT**
  - 1D, 2D, 3D FFT
  - Batched FFT
  - Multi-GPU FFT
- [ ] **Thrust & CUB**
  - Thrust algorithms (sort, reduce, scan, transform)
  - CUB block-level và device-level primitives
  - Custom operators và iterators
- [ ] **cuSPARSE & cuSOLVER**
  - Sparse matrix formats (CSR, CSC, COO, BSR)
  - Sparse-dense operations
  - Linear system solvers

#### 5.2 Mixed Precision & Tensor Cores (Tuần 3-4)
- [ ] **FP16 / BF16 / TF32 / FP8**
  - Floating point format so sánh
  - `half` và `__nv_bfloat16` types
  - Precision vs Performance trade-offs
  - Automatic mixed precision concepts
- [ ] **Tensor Core Programming**
  - WMMA (Warp Matrix Multiply Accumulate) API
  - `nvcuda::wmma` namespace
  - Fragment types và operations
  - MMA PTX instructions
  - Tensor Core trong deep learning
- [ ] **CUTLASS**
  - GEMM templates
  - Epilogue fusion
  - Custom tile sizes
  - Profiling CUTLASS kernels

#### 5.3 CUDA cho AI/ML (Tuần 5-6)
- [ ] **Custom CUDA Kernels cho Deep Learning**
  - Custom PyTorch extensions (C++/CUDA)
  - TorchScript custom ops
  - ONNX Runtime custom ops
  - Triton kernel language (so sánh)
- [ ] **Inference Optimization**
  - TensorRT overview
  - Quantization (INT8, FP8)
  - Kernel fusion strategies
  - Memory optimization cho inference
- [ ] **Training Optimization**
  - Gradient accumulation
  - Data loading pipeline
  - Communication/computation overlap
  - Flash Attention concept

#### 5.4 Interoperability (Tuần 7-8)
- [ ] **CUDA + OpenGL/Vulkan**
  - Graphics interop
  - Shared buffers
  - Real-time visualization
- [ ] **CUDA + Python**
  - PyCUDA
  - Numba CUDA
  - CuPy
  - `ctypes` / `cffi` binding
- [ ] **CUDA Driver API**
  - Driver API vs Runtime API
  - Context management
  - Module loading
  - JIT compilation với NVRTC

### 📖 Tài liệu tham khảo
| Tài liệu | Loại | Link |
|-----------|------|------|
| cuBLAS / cuDNN / cuFFT Documentation | Official Doc | docs.nvidia.com |
| CUTLASS GitHub Repository | Code | github.com/NVIDIA/cutlass |
| Mixed Precision Training (Micikevicius et al., 2018) | Paper | arxiv.org |
| Flash Attention (Dao et al., 2022) | Paper | arxiv.org |
| Triton Language Documentation | Doc | triton-lang.org |

### 🏋️ Bài tập
1. **cuBLAS GEMM**: Benchmark so với custom kernel
2. **Tensor Core GEMM**: Implement với WMMA API
3. **PyTorch Extension**: Viết custom CUDA op cho PyTorch
4. **Inference Engine**: Build simple inference engine với kernel fusion
5. **Real-time Visualization**: CUDA compute + OpenGL render

### ✅ Checkpoint
> Bạn nên có thể: sử dụng thành thạo CUDA libraries, lập trình Tensor Cores, viết custom ops cho frameworks

---

## ⚡ Phase 6: Mastery – Dự án thực tế & Chuyên gia (Ongoing)

### 🎯 Mục tiêu
Áp dụng tất cả kiến thức vào dự án thực tế, đóng góp cho cộng đồng, và liên tục cập nhật.

### 📚 Dự án thực tế đề xuất

#### 6.1 Dự án cấp độ Portfolio
- [ ] **GPU-accelerated Ray Tracer**
  - Path tracing với BVH acceleration
  - Multiple bounces, soft shadows
  - Denoising
- [ ] **Custom Deep Learning Framework**
  - Forward/backward pass engine
  - Autograd system
  - Optimizers (SGD, Adam)
  - Operator fusion
- [ ] **Real-time Fluid Simulation**
  - Navier-Stokes solver
  - SPH (Smoothed Particle Hydrodynamics)
  - Real-time visualization
- [ ] **GPU Database Engine**
  - Columnar storage
  - GPU-accelerated queries (filter, join, aggregate)
  - Query compilation
- [ ] **LLM Inference Engine**
  - KV-cache management
  - PagedAttention implementation
  - Batched inference
  - Speculative decoding

#### 6.2 Đóng góp cộng đồng
- [ ] Contribute to NVIDIA open-source projects (CUTLASS, cuDF, etc.)
- [ ] Viết blog posts / tutorials về CUDA
- [ ] Answer questions trên Stack Overflow / NVIDIA Forums
- [ ] Present tại GTC hoặc local meetups
- [ ] Publish benchmark results và optimizations

#### 6.3 Cập nhật liên tục
- [ ] Theo dõi GTC keynotes và sessions
- [ ] Đọc NVIDIA Technical Blog
- [ ] Theo dõi CUDA Toolkit releases
- [ ] Nghiên cứu kiến trúc GPU mới
- [ ] Học thêm: SYCL, HIP (AMD), oneAPI (Intel)

---

## 📊 Metrics đánh giá tiến độ

### Mức độ thành thạo

| Level | Tiêu chí | Bằng chứng |
|-------|----------|------------|
| **Beginner** | Viết được kernel cơ bản, quản lý bộ nhớ | Hoàn thành Phase 1-2 |
| **Intermediate** | Tối ưu hóa kernel, sử dụng profiler | Hoàn thành Phase 3 |
| **Advanced** | Multi-GPU, streams, async execution | Hoàn thành Phase 4 |
| **Expert** | Library integration, Tensor Cores, custom ops | Hoàn thành Phase 5 |
| **Master** | Dự án thực tế, đóng góp cộng đồng, mentoring | Phase 6 ongoing |

### Performance Benchmarks (tự đánh giá)

| Benchmark | Beginner | Expert |
|-----------|----------|--------|
| MatMul (4096x4096) | < 5% cuBLAS | > 80% cuBLAS |
| Reduction | < 10% peak bandwidth | > 90% peak bandwidth |
| Convolution | Naive | > 70% cuDNN |
| Memory Bandwidth | < 30% theoretical | > 85% theoretical |

---

## 🔗 Tài nguyên bổ sung

### Sách (theo thứ tự đọc)
1. 📘 **CUDA by Example** – Sanders & Kandrot (beginner)
2. 📗 **Programming Massively Parallel Processors** – Kirk & Hwu (intermediate)
3. 📕 **Professional CUDA C Programming** – Cheng, Grossman, McKercher (intermediate-advanced)
4. 📙 **CUDA Handbook** – Nicholas Wilt (reference)

### Online Courses
1. 🎓 NVIDIA DLI – Fundamentals of Accelerated Computing with CUDA C/C++
2. 🎓 Coursera – GPU Programming Specialization (Johns Hopkins)
3. 🎓 Udacity – Intro to Parallel Programming (CS344)

### Communities
- 💬 NVIDIA Developer Forums
- 💬 r/CUDA (Reddit)
- 💬 GPU Computing Discord servers
- 💬 Stack Overflow [cuda] tag

### Blogs & Channels
- 📝 NVIDIA Technical Blog (developer.nvidia.com/blog)
- 📝 Lei Mao's Blog (leimao.github.io)
- 📝 Simon Boehm's Blog (siboehm.com)
- 🎥 NVIDIA GTC Sessions (YouTube)

---

## 📅 Weekly Schedule Template

```
Monday:    Lý thuyết (đọc sách / documentation)      2-3h
Tuesday:   Coding (implement concepts)                 3-4h
Wednesday: Lý thuyết + Coding                         2-3h
Thursday:  Coding (bài tập thực hành)                  3-4h
Friday:    Profiling & Optimization (review code)      2-3h
Saturday:  Dự án lớn (project work)                    4-6h
Sunday:    Review + đọc papers / blog posts            2-3h
                                          Total: ~20-25h/week
```

---

> **Lời khuyên quan trọng nhất**: CUDA không thể học chỉ bằng lý thuyết. Hãy **code mỗi ngày**, **profile mọi thứ**, và **so sánh với implementation tốt nhất** (cuBLAS, cuDNN). Sự khác biệt giữa kernel "chạy được" và kernel "chạy nhanh" là nơi bạn thực sự học được CUDA.

---

*Lộ trình được thiết kế dựa trên kinh nghiệm thực tế trong GPU Computing. Điều chỉnh tốc độ phù hợp với trình độ hiện tại của bạn.*
