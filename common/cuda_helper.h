#pragma once
// ============================================================
// common/cuda_helper.h - CUDA error checking, timing, device info
// ============================================================
// Include this in every CUDA exercise. It pulls in the CUDA-free
// helpers too, so a single include is usually enough:
//
//     #include "cuda_helper.h"
//
// The build system puts <repo>/common on the include path, so
// plain quoted includes work from any exercise directory.
// ============================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "report.h"
#include "timer.h"
#include "verify.h"

// ============================================================
// Error checking
// ============================================================
// Nearly every CUDA runtime call returns a cudaError_t, and
// ignoring it is the single most common source of "my kernel
// silently does nothing" bugs. Wrap every call:
//
//     CUDA_CHECK(cudaMalloc(&d_a, bytes));
//
// The do/while(0) wrapper makes the macro behave like a single
// statement, so it is safe inside an unbraced if/else.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d\n  in: %s\n  -> %s (%s)\n",      \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_),          \
                    cudaGetErrorName(err_));                                      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Kernel launches do NOT return an error code, so they need a
// separate check. cudaGetLastError() clears the sticky error flag
// and reports launch-configuration problems (too many threads,
// too much shared memory, ...) immediately.
#define CUDA_CHECK_LAST()                                                         \
    do {                                                                          \
        cudaError_t err_ = cudaGetLastError();                                    \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA kernel launch error at %s:%d\n  -> %s (%s)\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err_),                 \
                    cudaGetErrorName(err_));                                      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Launch errors are asynchronous: an illegal memory access inside
// the kernel only surfaces at the next synchronisation point. Use
// this right after a launch while debugging - it costs a full
// device synchronise, so do not leave it inside a timing loop.
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        CUDA_CHECK_LAST();                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                      \
    } while (0)

// ============================================================
// Small utilities
// ============================================================

// Number of blocks needed to cover `n` items with `blockSize`
// threads each. Integer ceiling division: never use (n / b + 1),
// which over-allocates a block when n is an exact multiple of b.
__host__ __device__ inline int ceilDiv(int n, int b) { return (n + b - 1) / b; }
__host__ __device__ inline size_t ceilDiv(size_t n, size_t b) { return (n + b - 1) / b; }

// ============================================================
// GpuTimer - wall-clock timing on the device timeline
// ============================================================
// CUDA events are recorded *in the stream*, so they measure GPU
// work without being fooled by asynchronous launches. Timing a
// kernel with a host-side clock requires a manual synchronise and
// includes launch overhead, which is usually not what you want.
struct GpuTimer {
    cudaEvent_t startEvent{}, stopEvent{};

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }

    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(startEvent, stream)); }
    void stop(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(stopEvent, stream)); }

    // Blocks until the stop event has been reached on the device.
    float elapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        return ms;
    }

    void print(const char* label) { printf("[%s] %.3f ms\n", label, elapsedMs()); }
};

// ------------------------------------------------------------
// timeGpuMs - benchmark a kernel launch, return the median ms
// ------------------------------------------------------------
// Usage:
//   double ms = timeGpuMs(50, [&]{ myKernel<<<g, b>>>(args); });
//
// The callable should only *launch* work; the helper handles the
// synchronisation. Warmup iterations absorb one-time costs such as
// PTX JIT compilation, context setup and clock ramp-up.
template <typename Fn>
double timeGpuMs(int iters, Fn&& launch, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    GpuTimer t;
    for (int i = 0; i < iters; ++i) {
        t.start();
        launch();
        t.stop();
        samples.push_back(t.elapsedMs());
    }
    CUDA_CHECK(cudaGetLastError());
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// ============================================================
// Device information
// ============================================================

// Marketing name of the architecture behind a compute capability.
inline const char* archName(int major, int minor) {
    switch (major) {
        case 5: return "Maxwell";
        case 6: return "Pascal";
        case 7: return (minor >= 5) ? "Turing" : "Volta";
        case 8: return (minor >= 9) ? "Ada Lovelace" : "Ampere";
        case 9: return "Hopper";
        case 10:
        case 12: return "Blackwell";
        default: return "Unknown";
    }
}

// FP32 lanes ("CUDA cores") per SM. This is a hardware property
// that the runtime does not expose, so it has to be a table.
inline int coresPerSM(int major, int minor) {
    switch (major) {
        case 5: return 128;                       // Maxwell
        case 6: return (minor == 0) ? 64 : 128;   // GP100 vs GP10x
        case 7: return 64;                        // Volta, Turing
        case 8: return (minor == 0) ? 64 : 128;   // A100 vs GA10x / Ada
        case 9: return 128;                       // Hopper
        case 10:
        case 12: return 128;                      // Blackwell
        default: return 0;
    }
}

inline void printDeviceInfo(int deviceId = 0) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("------------------------------------------------------------\n");
    printf("  Device %d: %s\n", deviceId, prop.name);
    printf("  Compute capability : %d.%d (%s, sm_%d%d)\n", prop.major, prop.minor,
           archName(prop.major, prop.minor), prop.major, prop.minor);
    printf("  SMs                : %d\n", prop.multiProcessorCount);
    int cores = coresPerSM(prop.major, prop.minor);
    if (cores > 0)
        printf("  FP32 lanes         : %d (%d per SM)\n",
               cores * prop.multiProcessorCount, cores);
    printf("  Global memory      : %.2f GB\n",
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared mem / block : %zu KB (opt-in max %zu KB)\n",
           prop.sharedMemPerBlock / 1024, prop.sharedMemPerBlockOptin / 1024);
    printf("  Shared mem / SM    : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Registers / block  : %d\n", prop.regsPerBlock);
    printf("  Warp size          : %d\n", prop.warpSize);
    printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads/SM     : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  L2 cache           : %d KB\n", prop.l2CacheSize / 1024);
    printf("  Memory bus width   : %d-bit\n", prop.memoryBusWidth);

    // clockRate / memoryClockRate were removed from cudaDeviceProp in
    // CUDA 13. Guard so the same source builds on 11.x, 12.x and 13.x.
#if CUDART_VERSION < 13000
    // GDDR is double data rate, hence the factor of 2.
    double peakBwGBs = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("  Peak bandwidth     : %.1f GB/s (theoretical)\n", peakBwGBs);
    if (cores > 0) {
        // One FMA counts as two floating point operations.
        double peakFp32 = 2.0 * cores * prop.multiProcessorCount * prop.clockRate * 1e3 / 1e9;
        printf("  Peak FP32          : %.1f GFLOP/s (theoretical)\n", peakFp32);
    }
#endif
    printf("  Async engines      : %d (concurrent copy+kernel: %s)\n", prop.asyncEngineCount,
           prop.deviceOverlap ? "yes" : "no");
    printf("  Unified addressing : %s\n", prop.unifiedAddressing ? "yes" : "no");
    printf("  Managed memory     : %s\n", prop.managedMemory ? "yes" : "no");
    printf("  Cooperative launch : %s\n", prop.cooperativeLaunch ? "yes" : "no");
    printf("------------------------------------------------------------\n");
}

// Theoretical peak DRAM bandwidth in GB/s, or 0 when the runtime
// no longer exposes the clock. Use it to judge memory-bound kernels:
// reaching ~80% of this number is close to the practical maximum.
inline double theoreticalBandwidthGBs(int deviceId = 0) {
#if CUDART_VERSION < 13000
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
#else
    (void)deviceId;
    return 0.0;
#endif
}

// ============================================================
// Capability guards
// ============================================================
// Exercises that need a specific feature call these first. They
// print a clear message and let main() return 0 (a skip, not a
// failure), so the whole suite can be run on any GPU.

inline int skipExercise(const char* reason) {
    printf("\n  [SKIP] %s\n", reason);
    printf("  Nothing was verified. This is not a failure.\n\n");
    return 0;
}

// True when the current device is at least the requested compute
// capability. Example: hasComputeCapability(7, 0) for Tensor Cores.
inline bool hasComputeCapability(int major, int minor, int deviceId = 0) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return (prop.major > major) || (prop.major == major && prop.minor >= minor);
}

inline int deviceCount() {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return 0;
    return n;
}

// Call at the start of main() in every CUDA exercise.
inline void requireCudaDevice() {
    int n = deviceCount();
    if (n == 0) {
        fprintf(stderr,
                "No CUDA device found.\n"
                "  - Is the NVIDIA driver installed?  (run: nvidia-smi)\n"
                "  - Inside Docker, did you pass the GPU through?\n"
                "    (docker run --gpus all ... or the deploy block in "
                "docker-compose.yml)\n");
        exit(EXIT_FAILURE);
    }
}
