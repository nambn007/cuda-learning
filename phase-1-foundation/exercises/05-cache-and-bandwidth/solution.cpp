// ============================================================
// 05 - Cache lines and memory bandwidth  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// The returned value is consumed by main(), which is what stops
// the optimiser from deleting these loops entirely. A benchmark
// whose result is unused measures nothing.
float strideSum(const float* data, size_t n, int stride) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += static_cast<size_t>(stride)) sum += data[i];
    return sum;
}

void triad(float* a, const float* b, const float* c, float scalar, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + scalar * c[i];
}

int main() {
    printBanner("Phase 1 / 05 - Cache lines and memory bandwidth");

    std::vector<float> data(kStreamElements, 1.0f);
    printf("  Array: %zu floats (%.0f MB), cache line assumed %zu bytes\n",
           kStreamElements, kStreamElements * sizeof(float) / (1024.0 * 1024.0),
           kCacheLineBytes);

    printSection("Correctness");
    reportCheck("strideSum(stride=1) == 1024", strideSum(data.data(), 1024, 1) == 1024.0f);
    reportCheck("strideSum(stride=4) == 256", strideSum(data.data(), 1024, 4) == 256.0f);
    {
        std::vector<float> a(1024, 0.0f), b(1024, 2.0f), c(1024, 3.0f);
        triad(a.data(), b.data(), c.data(), 2.0f, 1024);
        reportCheck("triad computes b + 2*c", a[0] == 8.0f);
    }

    // ========================================================
    // Experiment 1: what a stride costs
    // ========================================================
    printSection("Stride sweep");
    printf("  'useful' counts only the bytes the program asked for.\n");
    printf("  'DRAM' counts whole 64-byte lines, which is what the bus moved.\n\n");
    printf("  %-8s %12s %14s %14s %10s\n", "stride", "time (ms)", "useful GB/s",
           "DRAM GB/s", "efficiency");
    printf("  ------------------------------------------------------------------\n");

    volatile float sink = 0.0f;  // keeps the sums alive
    for (int stride : kStrides) {
        double ms = timeCpuMs(3, [&] {
            sink = strideSum(data.data(), kStreamElements, stride);
        });
        size_t touched = kStreamElements / static_cast<size_t>(stride);
        double useful = gbPerSec(usefulBytes(touched), ms);
        double dram = gbPerSec(dramBytes(touched, stride), ms);
        printf("  %-8d %12.2f %14.1f %14.1f %9.0f%%\n", stride, ms, useful, dram,
               100.0 * useful / (dram > 0 ? dram : 1.0));
    }

    printf("\n  How to read this table\n");
    printf("    stride 1  : every byte of every line is used -> useful == DRAM.\n");
    printf("    stride 16 : 16 floats = 64 bytes = exactly one line, so each\n");
    printf("                line now yields ONE useful float. Useful bandwidth\n");
    printf("                has collapsed to about 1/16, while the DRAM column\n");
    printf("                barely moved: the bus is just as busy, moving data\n");
    printf("                you throw away.\n");
    printf("    stride >16: no further loss - you were already wasting a whole\n");
    printf("                line per access.\n");
    printf("\n    This is memory coalescing, seen from the CPU. In Phase 3 the\n");
    printf("    same graph reappears for a GPU warp, where the block size is a\n");
    printf("    32-byte sector instead of a 64-byte line.\n");

    // ========================================================
    // Experiment 2: where the cache levels are
    // ========================================================
    printSection("Working-set sweep (finding L1, L2, L3)");
    printf("  A small array is re-read from cache; a large one from DRAM. The\n");
    printf("  steps in this column are the boundaries of your cache levels.\n\n");
    printf("  %-14s %12s %14s\n", "working set", "time (ms)", "GB/s");
    printf("  ----------------------------------------------\n");

    for (size_t kb : kWorkingSetsKB) {
        size_t elements = kb * 1024 / sizeof(float);
        if (elements > kStreamElements) continue;

        // Touch the same working set enough times to make the
        // measurement stable and to give the cache a chance to
        // actually hold it.
        size_t passes = (64u * 1024u * 1024u) / (kb * 1024u);
        if (passes < 1) passes = 1;

        double ms = timeCpuMs(3, [&] {
            float s = 0.0f;
            for (size_t p = 0; p < passes; ++p) s += strideSum(data.data(), elements, 1);
            sink = s;
        });
        double bytes = static_cast<double>(elements) * sizeof(float) * passes;
        printf("  %10zu KB %12.2f %14.1f\n", kb, ms, gbPerSec(bytes, ms));
    }

    // ========================================================
    // Experiment 3: sustained bandwidth
    // ========================================================
    printSection("Sustained bandwidth (STREAM triad)");
    {
        const size_t n = kStreamElements;
        std::vector<float> a(n, 0.0f), b(n, 1.0f), c(n, 2.0f);

        double ms = timeCpuMs(5, [&] { triad(a.data(), b.data(), c.data(), 3.0f, n); });

        // Two reads plus one write per element. The write also
        // costs a read on most CPUs (read-for-ownership) unless
        // the compiler emits non-temporal stores, so 16 B/element
        // is often closer to the truth than 12.
        double bytes = static_cast<double>(n) * 3 * sizeof(float);
        ResultTable table;
        table.add("triad a = b + s*c", ms, bytes, static_cast<double>(n) * 2);
        table.print("Sustained memory bandwidth");

        printf("\n  A desktop CPU with dual-channel DDR4-3200 has a theoretical\n");
        printf("  peak near 51 GB/s and typically sustains 60-80%% of it on one\n");
        printf("  thread. Compare that with the GPU number printed by any Phase 2\n");
        printf("  exercise: an RTX 3060 has 360 GB/s of theoretical bandwidth.\n");
        printf("  That ratio, roughly 10x, is the real reason to move work to a GPU.\n");

        reportCheck("triad produced the expected values", a[0] == 1.0f + 3.0f * 2.0f);
    }

    (void)sink;
    return verifySummary();
}
