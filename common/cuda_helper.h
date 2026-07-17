#pragma once
// ============================================================
// common/cuda_helper.h – CUDA utility macros & helpers
// ============================================================
// Include this in every exercise for error checking and timing.
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================
// Error Checking Macros
// ============================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================
// GPU Timer (using CUDA Events)
// ============================================================

struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    void Stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
    }

    float ElapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

    void PrintElapsed(const char* label) {
        float ms = ElapsedMs();
        printf("[%s] Elapsed: %.3f ms\n", label, ms);
    }
};

// ============================================================
// Device Info
// ============================================================

inline void printDeviceInfo(int deviceId = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ============================================================
// CPU Timer (for host-side timing)
// ============================================================

#include <chrono>

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t_start, t_stop;

    void Start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void Stop() {
        t_stop = std::chrono::high_resolution_clock::now();
    }

    float ElapsedMs() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start);
        return duration.count() / 1000.0f;
    }

    void PrintElapsed(const char* label) {
        float ms = ElapsedMs();
        printf("[%s] Elapsed: %.3f ms\n", label, ms);
    }
};
