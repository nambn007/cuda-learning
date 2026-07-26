<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 08 - Know your GPU

> **Phase 1 · Foundation** | Difficulty: ⭐ | Time: ~1 h | Prerequisites: [07](../07-roofline-model/) | **Requires a GPU**

## Goal

Your first `.cu` file — and it launches no kernel. Instead you interrogate the
hardware and work out on paper what it *could* do, so that every performance
number from Phase 2 onwards has something to be measured against.

Finish this exercise and you will be able to answer, for your own card: how many
threads can run at once, how many bytes per second it can read, how many FLOPs a
kernel must do per byte to stop being memory bound, and how many registers a
thread may use before occupancy collapses.

## Background

`cudaGetDeviceProperties()` fills a `cudaDeviceProp` struct with about 80 fields.
Four groups matter for the rest of the curriculum:

**Compute capability** (`major.minor`, written `sm_86`) — the ISA version. It
determines which instructions exist: Tensor Cores need `sm_70+`, asynchronous
copies `sm_80+`, L2 residency control `sm_80+`. It is *not* a performance number;
a big `sm_75` card beats a small `sm_86` one.

**Theoretical peaks** — computed, not reported:

```
peak GB/s    = 2 × memoryClockRate(kHz) × 1e3 × (memoryBusWidth / 8) / 1e9
peak GFLOP/s = 2 × coresPerSM × multiProcessorCount × clockRate(kHz) × 1e3 / 1e9
```

The first `2` is because GDDR is *double data rate*; the second is because one FMA
counts as two operations. `coresPerSM` is not exposed by the runtime — it is a
hardware table, provided in `common/cuda_helper.h`.

**The ridge point** — `peak GFLOP/s / peak GB/s`, in FLOP per byte. On an RTX 3060
that is about **37**: a kernel must do 37 floating point operations for every byte
it touches before the GPU becomes compute bound. Vector add does 0.08. This one
number explains why Phase 3 is mostly about memory.

**The occupancy budget** — an SM has a fixed number of registers and a fixed
amount of shared memory, split among all resident threads. On an RTX 3060, 65536
registers ÷ 1536 threads = **42 registers per thread** at full occupancy. A kernel
holding an 8×8 float tile in registers has already spent 64. Going over does not
fail — it silently halves your occupancy. That is what Phase 3 exercise 15 is about.

> **Note for CUDA 13 users:** `clockRate` and `memoryClockRate` were removed from
> `cudaDeviceProp`. The code guards on `CUDART_VERSION`; read the clocks from
> `nvidia-smi -q -d CLOCK` instead.

## Your task

Open `main.cu`:

1. **Peak bandwidth** from `memoryClockRate` and `memoryBusWidth`.
2. **Peak FP32** from `coresPerSM()`, `multiProcessorCount` and `clockRate`.
3. **Ridge point**, and compare it with the CPU ridge point you measured in
   exercise 07.
4. **Occupancy budget**: max warps/SM, max threads on the whole GPU, registers per
   thread and shared memory per thread at 100% occupancy.

## Build and run

```bash
cmake --build build --target p1_08_gpu_device_query -j
./build/bin/p1/p1_08_gpu_device_query
```

## Expected output

Measured on an RTX 3060:

```
  Device 0: NVIDIA GeForce RTX 3060
  Compute capability : 8.6 (Ampere, sm_86)
  SMs                : 28
  FP32 lanes         : 3584 (128 per SM)
  Peak bandwidth     : 360.0 GB/s (theoretical)
  Peak FP32          : 13167.6 GFLOP/s (theoretical)

--- Roofline ---
  ridge point                  36.58 FLOP/byte

--- Occupancy budget (per SM) ---
  max resident threads / SM          1536
  max resident warps / SM            48
  registers / SM                     65536
  At 100% occupancy each thread may use at most:
  registers / thread                 42
  shared memory / thread             66.7 bytes
  whole GPU, fully occupied          43008 threads
```

## Key takeaways

- **Compute capability is a feature level, not a speed.**
- **Theoretical peaks are computed from clocks and widths**, and you will never
  reach them — 80% of peak bandwidth is an excellent result.
- **The GPU's ridge point is far higher than the CPU's**, so an even larger
  fraction of GPU kernels are memory bound.
- **Occupancy is a resource budget**, and registers are the scarcest resource.
  Roughly 42 per thread is not much.

## Going further

- Run `nvidia-smi -q` and find the same numbers there. Which are missing?
- Compute how long it takes to read the entire framebuffer once at peak bandwidth.
  That is the floor for any kernel touching all of memory.
- Write down the four answers listed at the end of the program — Phase 3 assumes
  you know them for your own card.
- Read the [CUDA C++ Programming Guide, "Compute Capabilities"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)
  appendix and find your architecture's row.
