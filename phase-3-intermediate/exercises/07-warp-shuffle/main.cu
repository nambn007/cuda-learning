// ============================================================
// 07 - Warp-level primitives  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

static constexpr unsigned kFullMask = 0xffffffffu;

// ------------------------------------------------------------
// TODO 1: butterfly reduction - every lane gets the total
// ------------------------------------------------------------
// Exercise 06 used __shfl_down_sync, which leaves the answer only
// in lane 0. __shfl_xor_sync exchanges symmetrically:
//
//   for (int offset = 16; offset > 0; offset >>= 1)
//       v += __shfl_xor_sync(0xffffffff, v, offset);
//
// Same five instructions, but afterwards EVERY lane holds the sum.
__device__ __forceinline__ float warpAllReduceSum(float v) {
    (void)v;
    return 0.0f;  // TODO
}

// ------------------------------------------------------------
// TODO 2: warp-wide inclusive scan in five shuffles
// ------------------------------------------------------------
//   for (int offset = 1; offset < 32; offset <<= 1) {
//       float n = __shfl_up_sync(0xffffffff, v, offset);
//       if (lane >= offset) v += n;
//   }
//
// The `if` matters: lanes below `offset` have no source lane, and
// __shfl_up_sync returns their own value in that case.
__device__ __forceinline__ float warpInclusiveScan(float v) {
    (void)v;
    return 0.0f;  // TODO
}

__global__ void scanKernel(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: a block reduction built from warp reductions
// ------------------------------------------------------------
// Grid-stride sweep, warp reduce, one value per warp into shared
// memory, then one warp combines them. Compare with exercise 06's v6.
__global__ void blockReduceKernel(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: vote functions
// ------------------------------------------------------------
//   __all_sync(mask, p)   is p true for every active lane?
//   __any_sync(mask, p)   is p true for any?
//   __ballot_sync(mask, p) one bit per lane; __popc counts them
//
// Have lane 0 of each warp record all three for its warp.
__global__ void voteDemo(const float* in, int* flags, size_t n) {
    (void)in;
    (void)flags;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 5: filtering, the naive way
// ------------------------------------------------------------
// Every passing thread does its own atomicAdd(counter, 1).
__global__ void filterNaive(const float* in, int* counter, size_t n) {
    (void)in;
    (void)counter;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 6: filtering with warp-aggregated atomics
// ------------------------------------------------------------
//   unsigned active = __activemask();
//   unsigned mask   = __ballot_sync(active, keep);
//   int count       = __popc(mask);
//   unsigned leader = __ffs(active) - 1;
//   if (lane == leader && count > 0) atomicAdd(counter, count);
//
// Use __activemask() rather than a hard-coded full mask: a
// grid-stride loop can leave lanes behind on the last iteration.
//
// Predict the speedup before you measure. Then measure it. Then -
// whatever the number is - disassemble the NAIVE kernel:
//
//   cuobjdump -sass build/bin/p3/p3_07_warp_shuffle | grep -B3 RED
//
// and see what the compiler emitted for it. The answer to "why did
// I get that number" is in there, and it is the real content of
// this exercise.
__global__ void filterWarpAggregated(const float* in, int* counter, size_t n) {
    (void)in;
    (void)counter;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 3 / 07 - Warp-level primitives (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    std::vector<float> host(n);
    fillRandom(host.data(), n, 0.0f, 1.0f, 111);
    const size_t kept = countKeptCPU(host.data(), n);
    printf("  n = %zu, %zu pass the filter (%.1f%%)\n", n, kept,
           100.0 * static_cast<double>(kept) / static_cast<double>(n));

    // --------------------------------------------------------
    // TODO 7: verify each kernel, then benchmark the two filters
    // --------------------------------------------------------
    // warpScanCPU() in reference.h gives the golden scan.
    printSection("Experiments");
    printf("  TODO\n");

    printTodoNotice("implement the warp primitives and the two filter kernels");
    return verifySummary();
}
