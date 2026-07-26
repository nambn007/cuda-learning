// ============================================================
// 01 - Naive matrix multiplication on the CPU  [STARTER]
// ============================================================
// Fill in every TODO, then rebuild:
//   cmake --build build --target p1_01_matmul_naive_cpu -j
//   ./build/bin/p1/p1_01_matmul_naive_cpu
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// TODO 1: implement C = A * B
// ------------------------------------------------------------
// All three matrices are n x n and row-major (see reference.h).
// The straightforward definition is three nested loops:
//
//     C[i][j] = sum over k of A[i][k] * B[k][j]
//
// Accumulate into a local float variable inside the j loop rather
// than into C[i * n + j] directly. Writing to memory on every
// iteration prevents the compiler from keeping the running sum in
// a register, and costs a surprising amount of performance.
void matmulNaive(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO: replace this with the triple loop.
}

int main() {
    printBanner("Phase 1 / 01 - Naive matrix multiplication (starter)");

    const int n = 512;
    printf("  Matrix size: %d x %d (%.1f MB per matrix)\n", n, n,
           n * n * sizeof(float) / (1024.0 * 1024.0));

    MatmulProblem problem(n);
    std::vector<float> C(problem.elements(), 0.0f);

    // --------------------------------------------------------
    // Correctness first. A fast wrong answer is worth nothing.
    // --------------------------------------------------------
    printSection("Correctness");
    matmulNaive(problem.A.data(), problem.B.data(), C.data(), n);
    bool ok = checkArray("C == A * B", C, problem.golden, /*rtol=*/1e-4, /*atol=*/1e-4);

    if (!ok) {
        printTodoNotice("implement matmulNaive() in main.cpp");
        return verifySummary();
    }

    // --------------------------------------------------------
    // TODO 2: measure it
    // --------------------------------------------------------
    // Use timeCpuMs(iterations, callable) from timer.h. It runs a
    // couple of untimed warmup passes and then returns the MEDIAN
    // of `iterations` timed passes.
    //
    // Replace the 0.0 below with a real measurement, for example:
    //     double ms = timeCpuMs(5, [&]{
    //         matmulNaive(problem.A.data(), problem.B.data(), C.data(), n);
    //     });
    printSection("Performance");
    double ms = 0.0;  // TODO

    if (ms <= 0.0) {
        printTodoNotice("measure matmulNaive() with timeCpuMs()");
        return verifySummary();
    }

    // --------------------------------------------------------
    // TODO 3: turn the time into a rate
    // --------------------------------------------------------
    // matmulFlops(n) returns the operation count. gflops(flops, ms)
    // in report.h converts it into GFLOP/s.
    ResultTable table;
    table.add("naive ijk", ms, matmulMinBytes(n), matmulFlops(n));
    table.print("CPU matrix multiplication");

    printf("\n  Think about it: how many GFLOP/s did you get, and how does that\n");
    printf("  compare with your CPU's peak? A modern core running AVX2 FMA can\n");
    printf("  do roughly 30-100 GFLOP/s on a single thread.\n");

    return verifySummary();
}
