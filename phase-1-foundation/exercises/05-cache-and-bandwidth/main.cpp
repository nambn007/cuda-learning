// ============================================================
// 05 - Cache lines and memory bandwidth  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// TODO 1: strided sum
// ------------------------------------------------------------
// Sum data[0], data[stride], data[2*stride], ... up to n.
// Return the sum - do not just discard it, or the optimiser will
// delete the whole loop and you will measure nothing.
float strideSum(const float* data, size_t n, int stride) {
    (void)data;
    (void)n;
    (void)stride;
    return 0.0f;  // TODO
}

// ------------------------------------------------------------
// TODO 2: STREAM triad
// ------------------------------------------------------------
//     a[i] = b[i] + scalar * c[i]
// Three arrays, one pass, no reuse: the classic way to measure
// sustained memory bandwidth. Per element the machine must read 8
// bytes and write 4, so 12 bytes of traffic per element.
void triad(float* a, const float* b, const float* c, float scalar, size_t n) {
    (void)a;
    (void)b;
    (void)c;
    (void)scalar;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 1 / 05 - Cache lines and memory bandwidth (starter)");

    std::vector<float> data(kStreamElements, 1.0f);

    printSection("Correctness");
    // With stride 1 the sum of an all-ones array is just n.
    float s1 = strideSum(data.data(), 1024, 1);
    float s4 = strideSum(data.data(), 1024, 4);
    bool ok = reportCheck("strideSum(stride=1) == 1024", s1 == 1024.0f) &&
              reportCheck("strideSum(stride=4) == 256", s4 == 256.0f);

    std::vector<float> a(1024, 0.0f), b(1024, 2.0f), c(1024, 3.0f);
    triad(a.data(), b.data(), c.data(), 2.0f, 1024);
    ok = reportCheck("triad computes b + 2*c", a[0] == 8.0f) && ok;

    if (!ok) {
        printTodoNotice("implement strideSum() and triad() in main.cpp");
        return verifySummary();
    }

    // --------------------------------------------------------
    // TODO 3: sweep the stride and print two bandwidth columns
    // --------------------------------------------------------
    // For each stride in kStrides:
    //   - time strideSum over the whole 64 MB array
    //   - touched = kStreamElements / stride
    //   - useful GB/s  = gbPerSec(usefulBytes(touched), ms)
    //   - DRAM GB/s    = gbPerSec(dramBytes(touched, stride), ms)
    //
    // Watch what happens to each column as the stride grows past
    // 16 floats (= one 64-byte cache line).
    printSection("Stride sweep");
    printf("  TODO: implement the sweep\n");

    // --------------------------------------------------------
    // TODO 4: sustained bandwidth with triad
    // --------------------------------------------------------
    printSection("Sustained bandwidth (STREAM triad)");
    printf("  TODO: implement the triad benchmark\n");

    return verifySummary();
}
