// ============================================================
// 11 - Unified memory: one pointer, and what it costs  [STARTER]
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
    printBanner("Phase 2 / 11 - Unified memory (starter)");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (!prop.managedMemory) {
        return skipExercise("this device does not support managed memory");
    }

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    printf("  n = %zu floats (%.0f MB per array)\n", n, bytes / (1024.0 * 1024.0));

    // ========================================================
    // The one rule that makes this exercise meaningful
    // ========================================================
    // Every variant must do EXACTLY the same work:
    //     host writes both arrays -> GPU computes -> host reads back
    // Leaving the host-side initialisation out of one variant and
    // not the others is the easiest way to get a meaningless answer,
    // because managed memory's whole cost is the movement that the
    // initialisation triggers.
    //
    // Also: check correctness with a SEPARATE single call, not
    // inside the timing loop. The loop runs the body several times
    // and this kernel accumulates into y.

    // --------------------------------------------------------
    // TODO 1: explicit cudaMalloc + cudaMemcpy (the baseline)
    // --------------------------------------------------------
    printSection("Explicit cudaMalloc + cudaMemcpy");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 2: cudaMallocManaged, with no hints at all
    // --------------------------------------------------------
    // One pointer, valid on both sides. No cudaMemcpy anywhere:
    //     cudaMallocManaged(&x, bytes);
    //     for (...) x[i] = ...;        // host touches it
    //     axpy<<<...>>>(a, x, y, n);   // device touches it
    //     cudaDeviceSynchronize();
    //     float v = y[0];              // host touches it again
    //
    // Every page the kernel touches for the first time raises a
    // fault and the driver migrates 4 KB while the warp waits.
    printSection("Managed memory, no hints");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: the same thing with cudaMemPrefetchAsync
    // --------------------------------------------------------
    //     cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId);  // before host writes
    //     cudaMemPrefetchAsync(x, bytes, device);           // before the kernel
    //     cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId);  // before host reads
    //
    // The data movement is identical. Only the driver's knowledge
    // of WHEN to do it changes - one bulk DMA instead of thousands
    // of 4 KB faults.
    printSection("Managed memory + cudaMemPrefetchAsync");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: time the kernel alone, with the data already resident
    // --------------------------------------------------------
    // Prefetch to the device, synchronise, THEN time the kernel.
    // Compare its GB/s with what you measured in exercise 04.
    // Managed memory that is already in the right place is not
    // slower - it is the same physical memory.
    printSection("Kernel time with data already on the device");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: oversubscription - the feature with no alternative
    // --------------------------------------------------------
    // Try to allocate 1.5x prop.totalGlobalMem with cudaMalloc and
    // then with cudaMallocManaged. Read the two error codes.
    //
    // Then launch a kernel on a slice of the managed allocation to
    // prove it is genuinely usable.
    //
    // This is what managed memory is FOR: a working set larger than
    // VRAM. Explicit allocation cannot do it at all.
    printSection("Oversubscription");
    printf("  TODO\n");

    printTodoNotice("implement the five experiments in main.cu");
    return verifySummary();
}
