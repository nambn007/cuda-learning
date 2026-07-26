// ============================================================
// 02 - Cache blocking and loop order  [STARTER]
// ============================================================

#include <algorithm>
#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// TODO 1: the ikj loop order
// ------------------------------------------------------------
// Same maths, different traversal. Swap the j and k loops so that
// k is in the middle and j is innermost:
//
//     for i:
//       for k:
//         a = A[i][k]                 // loop invariant, hoist it
//         for j:
//           C[i][j] += a * B[k][j]    // both walk a ROW, stride 1
//
// Now the inner loop reads B along a row and writes C along a row,
// both stride 1, so the hardware prefetcher and the vector units
// can do their job. C must be zeroed first because we accumulate
// into it across iterations of k.
void matmulIKJ(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO: implement the ikj order.
}

// ------------------------------------------------------------
// TODO 2: cache blocking (tiling)
// ------------------------------------------------------------
// ikj still streams the whole of B once per row of A. Blocking
// fixes that: work on a T x T tile of C at a time, so the tiles of
// A and B it needs stay resident in cache while they are reused.
//
//     for ii in steps of T:
//       for kk in steps of T:
//         for jj in steps of T:
//           for i in [ii, ii+T):
//             for k in [kk, kk+T):
//               a = A[i][k]
//               for j in [jj, jj+T):
//                 C[i][j] += a * B[k][j]
//
// Use std::min(ii + tile, n) as the loop bound so a size that is
// not a multiple of the tile still works.
void matmulBlocked(const float* A, const float* B, float* C, int n, int tile) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    (void)tile;
    // TODO: implement the blocked version.
}

int main() {
    printBanner("Phase 1 / 02 - Cache blocking (starter)");

    const int n = 1024;
    const int tile = 64;
    printf("  Matrix size: %d x %d, tile: %d\n", n, n, tile);

    MatmulProblem problem(n);
    std::vector<float> C(problem.elements(), 0.0f);

    printSection("Correctness");

    std::fill(C.begin(), C.end(), 0.0f);
    matmulIKJ(problem.A.data(), problem.B.data(), C.data(), n);
    bool okIkj = checkArray("ikj order", C, problem.golden, 1e-3, 1e-3);

    std::fill(C.begin(), C.end(), 0.0f);
    matmulBlocked(problem.A.data(), problem.B.data(), C.data(), n, tile);
    bool okBlocked = checkArray("blocked", C, problem.golden, 1e-3, 1e-3);

    if (!okIkj || !okBlocked) {
        printTodoNotice("implement matmulIKJ() and matmulBlocked() in main.cpp");
        return verifySummary();
    }

    printSection("Performance");
    ResultTable table;

    double msNaive = timeCpuMs(2, [&] {
        matmulGolden(problem.A.data(), problem.B.data(), C.data(), n);
    }, /*warmup=*/1);
    table.add("naive ijk (baseline)", msNaive, 0.0, matmulFlops(n));

    // TODO 3: time matmulIKJ and matmulBlocked the same way and add
    // them to the table. Remember to zero C first - both accumulate.
    table.print("Loop order and blocking");

    printf("\n  Expect ikj to be several times faster than ijk, and blocking to\n");
    printf("  add another gain once the matrix stops fitting in L2.\n");

    return verifySummary();
}
