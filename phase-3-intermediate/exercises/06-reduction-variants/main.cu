// ============================================================
// 06 - Parallel reduction, seven ways  [STARTER]
// ============================================================
// Every version below is CORRECT. Each is faster than the last for
// a different reason, and naming the reason is the exercise.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: v1 - interleaved addressing, divergent branch
// ------------------------------------------------------------
//   s[tid] = in[i];  __syncthreads();
//   for (stride = 1; stride < blockDim.x; stride *= 2) {
//       if (tid % (2*stride) == 0) s[tid] += s[tid + stride];
//       __syncthreads();
//   }
//   if (tid == 0) out[blockIdx.x] = s[0];
//
// Use dynamic shared memory (`extern __shared__ float s[];` plus a
// third launch argument), so all variants share one declaration.
//
// Before writing it: which threads of a warp take that branch?
__global__ void reduce1(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: v2 - same tree, contiguous active threads
// ------------------------------------------------------------
//   unsigned index = 2 * stride * tid;
//   if (index < blockDim.x) s[index] += s[index + stride];
//
// Divergence is gone. What has it been replaced by? (Compute the
// shared-memory stride and apply exercise 03's model.)
__global__ void reduce2(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: v3 - sequential addressing
// ------------------------------------------------------------
//   for (stride = blockDim.x / 2; stride > 0; stride >>= 1)
//       if (tid < stride) s[tid] += s[tid + stride];
//
// No divergence, no bank conflicts. What is still wasteful?
__global__ void reduce3(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: v4 - first add during load
// ------------------------------------------------------------
// Each thread loads TWO elements and adds them before the tree
// starts, so the grid halves and nobody is idle on the first step.
// Launch with half the blocks.
__global__ void reduce4(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 5: v5 - unroll the last warp with __shfl_down_sync
// ------------------------------------------------------------
// Once 32 threads remain they are one warp, so values can move
// directly between lanes' REGISTERS:
//
//   for (int offset = warpSize/2; offset > 0; offset >>= 1)
//       v += __shfl_down_sync(0xffffffff, v, offset);
//
// The old trick was to drop __syncthreads() and mark the array
// `volatile`, relying on warp lockstep. That has been UNSAFE since
// Volta - use the shuffle.
__device__ __forceinline__ float warpReduceSum(float v) {
    (void)v;
    return 0.0f;  // TODO
}

__global__ void reduce5(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 6: v6 - grid-stride sweep + warp shuffle
// ------------------------------------------------------------
// Size the grid to the GPU, have each thread sum many elements in a
// coalesced sweep, then reduce once per block.
//
// Predict: this looks strictly better than v5. Is it?
__global__ void reduce6(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 7: v7 - the same sweep with FOUR accumulators
// ------------------------------------------------------------
// Only attempt this after you have measured v6 and been surprised.
// The fix is one you already applied on a CPU, in Phase 1 exercise
// 07. Work out what v6's sweep is actually bound by first.
__global__ void reduce7(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 3 / 06 - Parallel reduction (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    std::vector<float> host(n);
    fillRandom(host.data(), n, 0.5f, 1.5f, 101);
    const double golden = sumCPU(host.data(), n);

    printf("  n = %zu floats (%.0f MB), reference sum %.4f\n", n,
           n * sizeof(float) / (1024.0 * 1024.0), golden);

    const double peak = theoreticalBandwidthGBs();
    if (peak > 0.0)
        printf("  Bandwidth floor: %.3f ms at %.0f GB/s - this kernel is memory bound\n",
               reductionBytes(n) / (peak * 1e9) * 1e3, peak);

    // --------------------------------------------------------
    // TODO 8: run each variant, verify, then time them all
    // --------------------------------------------------------
    // Each kernel writes one partial sum per block; sum those on the
    // host in double. Compare against `golden` with a RELATIVE
    // tolerance of 1e-4 - every variant sums in a different order,
    // and float addition is not associative, so they land on
    // slightly different values. All of them are equally correct.
    //
    // Report GB/s and percentage of peak for each. The goal is not
    // "faster than the last one" but "how close to the floor".
    printSection("Results");
    printf("  TODO\n");

    printTodoNotice("implement the seven kernels and the comparison");
    return verifySummary();
}
