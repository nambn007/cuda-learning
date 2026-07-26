// ============================================================
// 01 - Naive matrix multiplication on the CPU  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// The textbook triple loop
// ------------------------------------------------------------
// Two details matter even in this "obvious" version:
//
// 1. `acc` is a local variable, so the compiler keeps the running
//    sum in a register. Accumulating straight into C[i * n + j]
//    forces a load and a store on every one of the N^3 iterations,
//    because the compiler cannot prove that C does not alias A or
//    B and therefore may not cache the value.
//
// 2. `restrict`-style aliasing rules are the reason the signature
//    takes const pointers for the inputs: it tells both the reader
//    and the optimiser that A and B are not written.
void matmulNaive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) {
                // A[i][k] walks along a row  -> stride 1, cache friendly.
                // B[k][j] walks down a column -> stride n, one cache
                // line per element once n gets large. This is the
                // bottleneck that exercise 02 attacks.
                acc += A[idx(i, k, n)] * B[idx(k, j, n)];
            }
            C[idx(i, j, n)] = acc;
        }
    }
}

int main() {
    printBanner("Phase 1 / 01 - Naive matrix multiplication");

    const int n = 512;
    printf("  Matrix size: %d x %d (%.1f MB per matrix)\n", n, n,
           n * n * sizeof(float) / (1024.0 * 1024.0));

    MatmulProblem problem(n);
    std::vector<float> C(problem.elements(), 0.0f);

    printSection("Correctness");
    matmulNaive(problem.A.data(), problem.B.data(), C.data(), n);
    checkArray("C == A * B", C, problem.golden, /*rtol=*/1e-4, /*atol=*/1e-4);

    printSection("Performance");
    // Warmup passes are not optional: the first pass pays for page
    // faults on the freshly allocated output and for the CPU
    // ramping up from its idle clock. timeCpuMs() handles that and
    // returns the median, which ignores a single unlucky sample.
    double ms = timeCpuMs(5, [&] {
        matmulNaive(problem.A.data(), problem.B.data(), C.data(), n);
    });

    ResultTable table;
    table.add("naive ijk", ms, matmulMinBytes(n), matmulFlops(n));
    table.print("CPU matrix multiplication");

    printf("\n  Reading the numbers\n");
    printf("    GFLOP/s is the useful metric here: %.2f\n",
           gflops(matmulFlops(n), ms));
    printf("    The GB/s column is misleading for this kernel - it assumes\n");
    printf("    each matrix is touched once, but the naive loop re-reads all\n");
    printf("    of B for every row of A. Exercise 02 measures that properly.\n");

    return verifySummary();
}
