// ============================================================
// 03 - Vector addition, and the cost of getting data there [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: the kernel
// ------------------------------------------------------------
// c[i] = a[i] + b[i], using a grid-stride loop so the same launch
// configuration works for any n (see exercise 02).
__global__ void vectorAdd(const float* a, const float* b, float* c, size_t n) {
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 2 / 03 - Vector addition (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    printf("  n = %zu floats (%.0f MB per array)\n", n,
           n * sizeof(float) / (1024.0 * 1024.0));

    VectorProblem problem(n);
    std::vector<float> result(n, 0.0f);

    // --------------------------------------------------------
    // TODO 2: the standard four steps
    // --------------------------------------------------------
    //   1. cudaMalloc three device buffers
    //   2. cudaMemcpy a and b host -> device
    //   3. launch the kernel
    //   4. cudaMemcpy c device -> host, then cudaFree everything
    //
    // Prefer `size_t` for the byte counts: n * sizeof(float) with an
    // int n overflows silently at 512M elements.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    (void)d_a;
    (void)d_b;
    (void)d_c;

    printSection("Correctness");
    bool ok = checkArray("c == a + b", result, problem.golden, 1e-6, 1e-6);
    if (!ok) {
        printTodoNotice("implement vectorAdd() and the memory plumbing in main.cu");
        return verifySummary();
    }

    // --------------------------------------------------------
    // TODO 3: measure three different things
    // --------------------------------------------------------
    // a) the CPU version                       -> timeCpuMs
    // b) the kernel alone                      -> timeGpuMs
    // c) the whole round trip: H2D + kernel + D2H
    //
    // Then answer honestly: which one would you ship?
    printSection("Performance");
    printf("  TODO: measure CPU, kernel-only, and the full round trip\n");

    return verifySummary();
}
