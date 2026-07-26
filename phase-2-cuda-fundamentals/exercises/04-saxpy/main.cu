// ============================================================
// 04 - SAXPY and vectorised loads  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: the straightforward kernel
// ------------------------------------------------------------
//     y[i] = a * x[i] + y[i]
// Grid-stride loop, as always.
//
// Note that `a * x[i] + y[i]` compiles to a single FMA instruction
// (one multiply-add with one rounding step), which is why the
// result may differ in the last bit from a separate multiply
// followed by an add. That is a feature, not a bug - but it is why
// this exercise compares with a tolerance.
__global__ void saxpy(float a, const float* x, float* y, size_t n) {
    (void)a;
    (void)x;
    (void)y;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the same kernel with vectorised loads
// ------------------------------------------------------------
// float4 is a built-in struct of four floats with 16-byte
// alignment. Loading one issues a SINGLE 16-byte memory
// instruction instead of four 4-byte ones.
//
//     const float4* x4 = reinterpret_cast<const float4*>(x);
//     float4 xv = x4[i];        // one instruction, four floats
//     xv.x, xv.y, xv.z, xv.w    // the components
//
// The pointer must be 16-byte aligned for this to be legal.
// cudaMalloc always returns memory aligned to at least 256 bytes,
// so a base pointer is fine; an arbitrary offset into an array is
// not.
//
// n4 = n / 4 is the number of float4 elements. Handle any
// remainder (n % 4) with a scalar tail.
__global__ void saxpyVectorised(float a, const float4* x, float4* y, size_t n4) {
    (void)a;
    (void)x;
    (void)y;
    (void)n4;
    // TODO
}

int main() {
    printBanner("Phase 2 / 04 - SAXPY (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    SaxpyProblem problem(n);
    std::vector<float> result(n);

    printSection("Correctness");
    printf("  TODO: allocate, copy, launch both kernels, verify against the golden\n");
    printf("        result with checkArray(..., rtol=1e-5, atol=1e-6)\n");

    // --------------------------------------------------------
    // TODO 3: report EFFECTIVE BANDWIDTH, not time
    // --------------------------------------------------------
    // saxpyBytes(n) is the traffic; gbPerSec() turns it into GB/s.
    // Then divide by theoreticalBandwidthGBs() to get the number
    // that actually tells you whether the kernel is good:
    // the percentage of peak.
    //
    // Anything above ~80% means the kernel is done. Below ~50%
    // means something is wrong - usually the access pattern.
    printSection("Performance");

    printTodoNotice("implement saxpy(), saxpyVectorised() and the measurements");
    return verifySummary();
}
