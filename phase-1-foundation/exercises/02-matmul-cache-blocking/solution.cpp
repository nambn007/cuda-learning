// ============================================================
// 02 - Cache blocking and loop order  [SOLUTION]
// ============================================================

#include <algorithm>
#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// ikj: the cheapest optimisation in this whole repository
// ------------------------------------------------------------
// Nothing changes mathematically, only the order of the loops.
// What changes physically:
//
//   ijk inner loop: A[i][k] stride 1, B[k][j] stride n  -> B misses
//   ikj inner loop: B[k][j] stride 1, C[i][j] stride 1  -> both hit
//
// The stride-1 inner loop is also a shape the compiler recognises,
// so -O3 will emit AVX FMA instructions for it. That is where most
// of the speedup actually comes from.
void matmulIKJ(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + static_cast<size_t>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            // Hoisted out of the j loop: it does not depend on j.
            const float a = A[idx(i, k, n)];
            const float* brow = B + static_cast<size_t>(k) * n;
            float* crow = C + static_cast<size_t>(i) * n;
            for (int j = 0; j < n; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

// ------------------------------------------------------------
// Blocked (tiled) multiplication
// ------------------------------------------------------------
// ikj is cache friendly *within* the inner loop but still sweeps
// the entire B matrix once per row of A. For n = 768, B is 2.25 MB
// - larger than a typical 1 MB L2 slice - so every sweep starts
// from DRAM again.
//
// Blocking restricts the working set: while we compute one T x T
// tile of C we only touch a T x T tile of A and a T x T tile of B.
// Three tiles of 64x64 floats are 48 KB, which fits comfortably in
// L2 and mostly in L1. Each element of B is now read n/T times
// instead of n times.
//
// This is *exactly* the idea behind shared-memory tiling in
// Phase 3: shared memory is a manually managed L1, and the tile
// loop structure is identical.
void matmulBlocked(const float* A, const float* B, float* C, int n, int tile) {
    std::fill(C, C + static_cast<size_t>(n) * n, 0.0f);

    for (int ii = 0; ii < n; ii += tile) {
        const int iMax = std::min(ii + tile, n);
        for (int kk = 0; kk < n; kk += tile) {
            const int kMax = std::min(kk + tile, n);
            for (int jj = 0; jj < n; jj += tile) {
                const int jMax = std::min(jj + tile, n);

                // Inner triple loop over one tile, in ikj order so
                // the innermost access stays stride 1.
                for (int i = ii; i < iMax; ++i) {
                    float* crow = C + static_cast<size_t>(i) * n;
                    for (int k = kk; k < kMax; ++k) {
                        const float a = A[idx(i, k, n)];
                        const float* brow = B + static_cast<size_t>(k) * n;
                        for (int j = jj; j < jMax; ++j) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

#ifdef _OPENMP
#include <omp.h>
// One more axis of parallelism before we reach the GPU: give each
// core its own stripe of output rows. The tiles are independent,
// so no synchronisation is needed - the same property that makes
// this problem a good fit for thousands of GPU threads.
void matmulBlockedOMP(const float* A, const float* B, float* C, int n, int tile) {
    std::fill(C, C + static_cast<size_t>(n) * n, 0.0f);
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += tile) {
        const int iMax = std::min(ii + tile, n);
        for (int kk = 0; kk < n; kk += tile) {
            const int kMax = std::min(kk + tile, n);
            for (int jj = 0; jj < n; jj += tile) {
                const int jMax = std::min(jj + tile, n);
                for (int i = ii; i < iMax; ++i) {
                    float* crow = C + static_cast<size_t>(i) * n;
                    for (int k = kk; k < kMax; ++k) {
                        const float a = A[idx(i, k, n)];
                        const float* brow = B + static_cast<size_t>(k) * n;
                        for (int j = jj; j < jMax; ++j) crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}
#endif

int main() {
    printBanner("Phase 1 / 02 - Cache blocking");

    const int n = 1024;
    const int tile = 64;
    printf("  Matrix size: %d x %d (%.2f MB per matrix), tile: %d\n", n, n,
           n * n * sizeof(float) / (1024.0 * 1024.0), tile);

    MatmulProblem problem(n);
    std::vector<float> C(problem.elements(), 0.0f);

    printSection("Correctness");
    matmulIKJ(problem.A.data(), problem.B.data(), C.data(), n);
    checkArray("ikj order", C, problem.golden, 1e-3, 1e-3);

    matmulBlocked(problem.A.data(), problem.B.data(), C.data(), n, tile);
    checkArray("blocked", C, problem.golden, 1e-3, 1e-3);

#ifdef _OPENMP
    matmulBlockedOMP(problem.A.data(), problem.B.data(), C.data(), n, tile);
    checkArray("blocked + OpenMP", C, problem.golden, 1e-3, 1e-3);
#endif

    // The GB/s column is left blank on purpose: for this kernel the
    // bytes actually moved depend on cache behaviour, which is what
    // we are trying to measure. GFLOP/s is the honest metric here.
    printSection("Performance");
    ResultTable table;

    double msNaive = timeCpuMs(2, [&] {
        matmulGolden(problem.A.data(), problem.B.data(), C.data(), n);
    }, /*warmup=*/1);
    table.add("naive ijk (baseline)", msNaive, 0.0, matmulFlops(n));

    double msIkj = timeCpuMs(3, [&] {
        matmulIKJ(problem.A.data(), problem.B.data(), C.data(), n);
    });
    table.add("ikj order", msIkj, 0.0, matmulFlops(n));

    double msBlocked = timeCpuMs(3, [&] {
        matmulBlocked(problem.A.data(), problem.B.data(), C.data(), n, tile);
    });
    table.add("blocked ikj", msBlocked, 0.0, matmulFlops(n));

#ifdef _OPENMP
    double msOmp = timeCpuMs(3, [&] {
        matmulBlockedOMP(problem.A.data(), problem.B.data(), C.data(), n, tile);
    });
    char label[64];
    snprintf(label, sizeof(label), "blocked + OpenMP (%d threads)", omp_get_max_threads());
    table.add(label, msOmp, 0.0, matmulFlops(n));
#endif

    table.print("Loop order and blocking");

    // ========================================================
    // When does blocking actually pay?
    // ========================================================
    // Only once the working set stops fitting in cache. On a
    // machine with a large L3, ikj and blocked are neck and neck at
    // n = 512 - the hardware is already doing the caching for you.
    // Watch the ratio move as n grows.
    printSection("When blocking starts to matter");
    printf("  Blocking helps only when the matrices no longer fit in cache.\n");
    printf("  B alone occupies 4*n*n bytes; compare that with your L2/L3 size\n");
    printf("  (exercise 05 measures it).\n\n");
    printf("  %-8s %12s %12s %12s %10s\n", "n", "B size", "ikj (ms)", "blocked (ms)",
           "speedup");
    printf("  --------------------------------------------------------------\n");

    for (int size : {256, 512, 1024, 2048}) {
        MatmulProblem p(size);
        std::vector<float> out(p.elements(), 0.0f);

        double a = timeCpuMs(1, [&] {
            matmulIKJ(p.A.data(), p.B.data(), out.data(), size);
        }, /*warmup=*/1);
        double b = timeCpuMs(1, [&] {
            matmulBlocked(p.A.data(), p.B.data(), out.data(), size, tile);
        }, /*warmup=*/1);

        printf("  %-8d %9.1f MB %12.1f %12.1f %9.2fx\n", size,
               size * size * sizeof(float) / (1024.0 * 1024.0), a, b, a / b);
    }
    printf("\n  If the last row is not faster when blocked, your cache is simply\n");
    printf("  large enough to hold the whole matrix. Increase n until it is not.\n");

    printSection("Tile size sweep");
    printf("  The best tile is the largest one whose three tiles still fit in\n");
    printf("  cache. Too small and loop overhead dominates; too large and you\n");
    printf("  are back to streaming from DRAM.\n\n");
    printf("  %-10s %12s %12s\n", "tile", "time (ms)", "GFLOP/s");
    for (int t : {16, 32, 64, 128, 256}) {
        double ms = timeCpuMs(2, [&] {
            matmulBlocked(problem.A.data(), problem.B.data(), C.data(), n, t);
        });
        printf("  %-10d %12.2f %12.2f\n", t, ms, gflops(matmulFlops(n), ms));
    }

    return verifySummary();
}
