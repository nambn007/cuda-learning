// ============================================================
// 01 - Thrust and CUB  [STARTER]
// ============================================================

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: bring across your best reduction from Phase 3/06
// ------------------------------------------------------------
// Use v7 - grid-stride sweep with four independent accumulators
// plus a warp-shuffle combine. The comparison is only meaningful
// against a genuine effort.
__global__ void handWrittenReduce(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 5 / 01 - Thrust and CUB (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    std::vector<float> host(n);
    fillRandom(host.data(), n, 0.5f, 1.5f, 151);
    printf("  n = %zu floats (%.0f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

    // --------------------------------------------------------
    // TODO 2: reduction three ways
    // --------------------------------------------------------
    //   a) your hand-written kernel
    //   b) cub::DeviceReduce::Sum
    //   c) thrust::reduce
    //
    // CUB never allocates for you. Its two-call convention:
    //     cub::DeviceReduce::Sum(nullptr, tempBytes, ...);  // ask
    //     cudaMalloc(&d_temp, tempBytes);                   // allocate
    //     cub::DeviceReduce::Sum(d_temp, tempBytes, ...);   // run
    // That is deliberate: one scratch buffer can be reused across
    // many calls instead of allocating inside a hot loop.
    //
    // Report each as a PERCENTAGE OF PEAK BANDWIDTH, since that is
    // the ceiling for a reduction. Predict where your kernel lands
    // before you look.
    printSection("Reduction");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: inclusive scan with CUB and with Thrust
    // --------------------------------------------------------
    // Verify against inclusiveScanCPU() with a RELATIVE tolerance -
    // accumulating 16M floats drifts from the double reference.
    //
    // A scan looks inherently sequential and is not: CUB uses a
    // single-pass decoupled look-back algorithm that gets close to
    // copy bandwidth.
    printSection("Inclusive scan");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: thrust::sort
    // --------------------------------------------------------
    // Check the output is ordered, and report keys per second.
    // Then consider how long a competitive GPU radix sort would
    // take you to write.
    printSection("Sort");
    printf("  TODO\n");

    printTodoNotice("implement the reduction and call the library algorithms");
    return verifySummary();
}
