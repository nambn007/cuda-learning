// ============================================================
// 03 - Atomics, contention and memory ordering  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: histogram with global atomics
// ------------------------------------------------------------
//     atomicAdd(&bins[v[i]], 1u)   for every element
// With skewed data, thousands of threads hit the same address.
__global__ void histogramGlobal(const unsigned char* v, size_t n, unsigned int* bins) {
    (void)v;
    (void)n;
    (void)bins;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the same, privatised into shared memory
// ------------------------------------------------------------
//   __shared__ unsigned int local[kBins];
//   zero it cooperatively, __syncthreads()
//   accumulate into local[] instead of bins[]
//   __syncthreads()
//   one atomicAdd per non-zero bin into bins[]
//
// Global atomics drop from one per ELEMENT to at most kBins per
// BLOCK. Predict the speedup before measuring.
__global__ void histogramPrivatised(const unsigned char* v, size_t n, unsigned int* bins) {
    (void)v;
    (void)n;
    (void)bins;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: a float atomicMax, built from atomicCAS
// ------------------------------------------------------------
// There is no atomicMax for float. Build one:
//
//   unsigned int* p = reinterpret_cast<unsigned int*>(addr);
//   unsigned int old = *p, assumed;
//   do { assumed = old;
//        if (__uint_as_float(assumed) >= value) break;
//        old = atomicCAS(p, assumed, __float_as_uint(value)); }
//   while (assumed != old);
//
// Then reduce within the block first, so only one thread per block
// performs the global CAS - otherwise the retry loop thrashes.
__device__ float atomicMaxFloat(float* addr, float value) {
    (void)addr;
    (void)value;
    return 0.0f;  // TODO
}

__global__ void maxKernel(const float* v, size_t n, float* result) {
    (void)v;
    (void)n;
    (void)result;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: a single-pass reduction, and the fence it needs
// ------------------------------------------------------------
// Each block reduces its slice, writes partial[blockIdx.x], then
// atomically increments a counter. The block that sees the counter
// reach gridDim.x - 1 knows it is last and reduces all the partials
// itself - so the whole reduction takes ONE launch.
//
// The subtle part: between writing partial[] and incrementing the
// counter you need
//
//     __threadfence();
//
// ATOMICITY IS NOT VISIBILITY. atomicAdd guarantees the counter is
// correct; it says nothing about whether your partial sum is
// visible to the block that reads it.
//
// Write it WITH the fence. Then delete the fence and run it a few
// hundred times. It will almost certainly still be right - and that
// is precisely why the bug is expensive.
__device__ unsigned int retirementCount = 0;

__global__ void singlePassReduce(const float* v, size_t n, float* partial, float* result) {
    (void)v;
    (void)n;
    (void)partial;
    (void)result;
    // TODO
}

int main() {
    printBanner("Phase 4 / 03 - Atomics and memory ordering (starter)");
    requireCudaDevice();

    // --------------------------------------------------------
    // TODO 5: verify and benchmark all three
    // --------------------------------------------------------
    // reference.h has histogramCPU(), maxCPU() and sumCPU().
    // For the histogram, report global atomics per launch alongside
    // the times: one per element versus kBins per block.
    printSection("Experiments");
    printf("  TODO\n");

    printTodoNotice("implement the four kernels and the comparisons");
    return verifySummary();
}
