// ============================================================
// 11 - Unified memory: one pointer, and what it costs  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void axpy(float a, const float* x, float* y, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) y[i] = a * x[i] + y[i];
}

int main() {
    printBanner("Phase 2 / 11 - Unified memory");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (!prop.managedMemory) {
        return skipExercise("this device does not support managed memory");
    }

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;
    const float alpha = 2.0f;

    printf("  n = %zu floats (%.0f MB per array)\n", n, bytes / (1024.0 * 1024.0));
    printf("  Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "yes" : "no");
    printf("  Pageable memory access   : %s\n",
           prop.pageableMemoryAccess ? "yes (HMM/ATS)" : "no");

    // ========================================================
    // 1. Explicit memory - the baseline from exercise 03
    // ========================================================
    // Every variant below does exactly the same work, so that the
    // comparison is fair:
    //     host writes both arrays -> GPU computes -> host reads the
    //     result back.
    // Leaving the host-side initialisation out of one variant and
    // not the others is the easiest way to get a meaningless answer
    // here, because managed memory's cost is precisely the movement
    // that the initialisation triggers.
    printSection("Explicit cudaMalloc + cudaMemcpy");
    double msExplicit = 0.0;
    {
        std::vector<float> hx(n), hy(n);
        float *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&dy, bytes));

        auto round = [&] {
            for (size_t i = 0; i < n; ++i) {
                hx[i] = 1.0f;
                hy[i] = 2.0f;
            }
            CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
            axpy<<<grid, block>>>(alpha, dx, dy, n);
            CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        // Check once, outside the timing loop. Timing loops run the
        // body several times, and this one accumulates into y.
        round();
        reportCheck("explicit result is correct", hy[0] == 2.0f + alpha * 1.0f);

        msExplicit = timeCpuMs(3, round, /*warmup=*/1);
        printKV("host init + H2D + kernel + D2H", msExplicit, "ms");
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    // ========================================================
    // 2. Managed memory, no hints
    // ========================================================
    // One pointer, no cudaMemcpy anywhere. Every page the kernel
    // touches for the first time raises a fault, the driver
    // migrates 4 KB, and the warp waits.
    printSection("Managed memory, no hints");
    double msNaive = 0.0;
    {
        float *x = nullptr, *y = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));

        auto round = [&] {
            // Host writes: pages migrate to the host.
            for (size_t i = 0; i < n; ++i) {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
            // Kernel reads: every page faults back to the device.
            axpy<<<grid, block>>>(alpha, x, y, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Host reads: they all migrate again.
            volatile float sink = y[0];
            (void)sink;
        };

        round();
        reportCheck("managed result is correct", y[0] == 2.0f + alpha * 1.0f);

        msNaive = timeCpuMs(3, round, /*warmup=*/1);
        printKV("host init + kernel + host read", msNaive, "ms");
        printf("  No cudaMemcpy anywhere - and it shows.\n");

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
    }

    // ========================================================
    // 3. Managed memory with prefetching
    // ========================================================
    // Same code, plus two lines telling the driver what you already
    // know: move these pages now, in bulk, instead of discovering
    // the need one fault at a time.
    printSection("Managed memory + cudaMemPrefetchAsync");
    double msPrefetch = 0.0;
    {
        float *x = nullptr, *y = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        auto round = [&] {
            // Tell the driver the host is about to write, so the
            // pages arrive in one transfer instead of one fault at
            // a time.
            CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId));
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId));
            CUDA_CHECK(cudaDeviceSynchronize());
            for (size_t i = 0; i < n; ++i) {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
            // ...and now the device is about to read.
            CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, device));
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, device));
            axpy<<<grid, block>>>(alpha, x, y, n);
            CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId));
            CUDA_CHECK(cudaDeviceSynchronize());
            volatile float sink = y[0];
            (void)sink;
        };

        round();
        reportCheck("prefetched result is correct", y[0] == 2.0f + alpha * 1.0f);

        msPrefetch = timeCpuMs(3, round, /*warmup=*/1);
        printKV("with explicit prefetch", msPrefetch, "ms");

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
    }

    // ========================================================
    // 4. Kernel-only time, once the data is resident
    // ========================================================
    printSection("Kernel time with data already on the device");
    double msKernelOnly = 0.0;
    {
        float *x = nullptr, *y = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        for (size_t i = 0; i < n; ++i) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
        CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, device));
        CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, device));
        CUDA_CHECK(cudaDeviceSynchronize());

        msKernelOnly = timeGpuMs(20, [&] { axpy<<<grid, block>>>(alpha, x, y, n); });
        printKV("kernel alone", msKernelOnly, "ms");
        printf("  %.1f GB/s", gbPerSec(axpyBytes(n), msKernelOnly));
        const double peak = theoreticalBandwidthGBs();
        if (peak > 0.0) printf("  (%.0f%% of peak)", 100.0 * gbPerSec(axpyBytes(n), msKernelOnly) / peak);
        printf("\n  Once the pages are resident, managed memory is EXACTLY as fast\n");
        printf("  as explicitly allocated memory - it is the same physical memory.\n");

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
    }

    // ========================================================
    // Summary
    // ========================================================
    printSection("Comparison");
    ResultTable table;
    table.add("explicit malloc + memcpy", msExplicit);
    table.add("managed, no hints", msNaive);
    table.add("managed + prefetch", msPrefetch);
    table.print("Full round trip (host init, kernel, host read)");

    printf("\n  Managed memory without hints is %.2fx the explicit version.\n",
           msNaive / msExplicit);
    printf("  With prefetching it is %.2fx.\n", msPrefetch / msExplicit);
    printf("\n  The difference is entirely about GRANULARITY. Fault-driven\n");
    printf("  migration moves 4 KB at a time and stalls a warp for each fault;\n");
    printf("  a prefetch is one bulk DMA. The data movement is identical - only\n");
    printf("  the driver's knowledge of when to do it changes.\n");

    // ========================================================
    // The feature with no explicit-API equivalent
    // ========================================================
    printSection("Oversubscription");
    {
        const size_t oversized = static_cast<size_t>(prop.totalGlobalMem * 1.5);
        printf("  This GPU has %.1f GB. Trying to allocate %.1f GB...\n\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               oversized / (1024.0 * 1024.0 * 1024.0));

        void* p = nullptr;
        cudaError_t explicitErr = cudaMalloc(&p, oversized);
        printf("  %-28s %s\n", "cudaMalloc", cudaGetErrorName(explicitErr));
        if (explicitErr == cudaSuccess) CUDA_CHECK(cudaFree(p));
        cudaGetLastError();  // clear the flag

        float* managed = nullptr;
        cudaError_t managedErr = cudaMallocManaged(&managed, oversized);
        printf("  %-28s %s\n", "cudaMallocManaged", cudaGetErrorName(managedErr));

        if (managedErr == cudaSuccess) {
            // Touch a small slice to prove it is really usable.
            const size_t slice = 64u * 1024u * 1024u / sizeof(float);
            axpy<<<grid, block>>>(1.0f, managed, managed, slice);
            cudaError_t useErr = cudaDeviceSynchronize();
            printf("  %-28s %s\n", "kernel on an oversized buffer",
                   cudaGetErrorName(useErr));
            reportCheck("managed memory can exceed device memory", useErr == cudaSuccess);
            CUDA_CHECK(cudaFree(managed));
        } else {
            printf("\n  Oversubscription was refused. It needs Pascal or newer plus a\n");
            printf("  64-bit Linux host; on Windows under WDDM it is not available.\n");
            cudaGetLastError();
        }

        printf("\n  This is what managed memory is FOR. Explicit allocation cannot\n");
        printf("  do it at all: if the working set does not fit in VRAM you have to\n");
        printf("  tile the problem by hand. With managed memory the driver pages it\n");
        printf("  in and out for you - slowly, but it runs. That is how a model\n");
        printf("  larger than the card can be trained at all.\n");
    }

    printSection("When to use which");
    printf("  Managed memory:\n");
    printf("    - prototyping, and any code where clarity beats the last 10%%\n");
    printf("    - irregular or pointer-chasing structures (trees, graphs) where\n");
    printf("      you cannot say in advance which pages are needed\n");
    printf("    - working sets larger than VRAM\n");
    printf("    - ALWAYS with cudaMemPrefetchAsync once you know the access pattern\n");
    printf("\n  Explicit memory:\n");
    printf("    - production kernels with a known, regular access pattern\n");
    printf("    - anything where you want to overlap transfer with compute using\n");
    printf("      streams (Phase 4 exercise 03)\n");
    printf("\n  Also worth knowing: cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly,\n");
    printf("  device) replicates read-only data on both sides instead of migrating\n");
    printf("  it back and forth, and SetPreferredLocation pins its home node.\n");

    return verifySummary();
}
