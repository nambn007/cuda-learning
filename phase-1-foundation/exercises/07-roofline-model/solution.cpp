// ============================================================
// 07 - The roofline model  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// A kernel whose arithmetic intensity is a parameter
// ------------------------------------------------------------
// TILE elements are held in registers at once, each with its own
// accumulator. Because the accumulators are independent, the FMA
// pipeline never has to wait on its own previous result, and the
// inner `t` loop is trivially vectorisable.
//
// TILE is a template parameter so the compiler sees a constant
// trip count and can fully unroll - which is what lets it keep the
// tile in registers instead of on the stack.
template <int TILE>
void intensityKernel(const float* in, float* out, size_t n, int fmaCount) {
    size_t i = 0;
    for (; i + TILE <= n; i += TILE) {
        float x[TILE], acc[TILE];
        for (int t = 0; t < TILE; ++t) {
            x[t] = in[i + t];
            acc[t] = 1.0f;
        }
        for (int k = 0; k < fmaCount; ++k) {
            // No dependency across t: TILE independent FMA chains.
            for (int t = 0; t < TILE; ++t) acc[t] = acc[t] * x[t] + 1.0f;
        }
        for (int t = 0; t < TILE; ++t) out[i + t] = acc[t];
    }
    for (; i < n; ++i) {
        float a = 1.0f;
        const float v = in[i];
        for (int k = 0; k < fmaCount; ++k) a = a * v + 1.0f;
        out[i] = a;
    }
}

void triad(float* a, const float* b, const float* c, float scalar, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + scalar * c[i];
}

static void printBar(double value, double maxValue, int width = 28) {
    int filled = maxValue > 0.0 ? static_cast<int>(width * value / maxValue) : 0;
    if (filled > width) filled = width;
    printf("|");
    for (int i = 0; i < width; ++i) putchar(i < filled ? '#' : ' ');
    printf("|");
}

// TILE has to be a compile-time constant, so a runtime tile width
// needs an explicit dispatch. Templates are instantiated for each
// width we want to try.
static double timeTile(int tile, const float* in, float* out, size_t n, int fma, int iters) {
    switch (tile) {
        case 4: return timeCpuMs(iters, [&] { intensityKernel<4>(in, out, n, fma); });
        case 8: return timeCpuMs(iters, [&] { intensityKernel<8>(in, out, n, fma); });
        case 16: return timeCpuMs(iters, [&] { intensityKernel<16>(in, out, n, fma); });
        case 32: return timeCpuMs(iters, [&] { intensityKernel<32>(in, out, n, fma); });
        case 64: return timeCpuMs(iters, [&] { intensityKernel<64>(in, out, n, fma); });
        case 128: return timeCpuMs(iters, [&] { intensityKernel<128>(in, out, n, fma); });
        default: return 0.0;
    }
}

int main() {
    printBanner("Phase 1 / 07 - The roofline model");

    printSection("Correctness");
    {
        std::vector<float> in(64, 2.0f), out(64, 0.0f);
        intensityKernel<16>(in.data(), out.data(), 64, 1);
        reportCheck("fma=1 on x=2 gives 3", out[0] == 3.0f);
        intensityKernel<16>(in.data(), out.data(), 64, 2);
        reportCheck("fma=2 on x=2 gives 7", out[0] == 7.0f);

        std::vector<float> a(64, 0.0f), b(64, 1.0f), c(64, 2.0f);
        triad(a.data(), b.data(), c.data(), 3.0f, 64);
        reportCheck("triad computes b + 3*c", a[0] == 7.0f);
    }

    // ========================================================
    // Roof 1: sustained memory bandwidth
    // ========================================================
    printSection("Roof 1 - memory bandwidth");
    double peakGBs = 0.0;
    {
        const size_t n = kMemoryElements;
        std::vector<float> a(n, 0.0f), b(n, 1.0f), c(n, 2.0f);
        double ms = timeCpuMs(5, [&] { triad(a.data(), b.data(), c.data(), 3.0f, n); });
        peakGBs = gbPerSec(static_cast<double>(n) * 3 * sizeof(float), ms);
        printKV("sustained bandwidth", peakGBs, "GB/s  (STREAM triad, 64 MB, 1 thread)");
    }

    // ========================================================
    // Roof 2: FMA throughput - and the register cliff
    // ========================================================
    // How many independent chains you keep in flight decides how
    // close to peak you get. Too few and the FMA latency dominates;
    // too many and the tile no longer fits in registers, so the
    // compiler spills it to the stack and throughput collapses.
    //
    // That collapse is REGISTER SPILLING, and it is the exact same
    // failure mode you will hit on the GPU in Phase 4 exercise 11 -
    // except there it costs you occupancy as well.
    printSection("Roof 2 - FMA throughput (and the register cliff)");
    double peakGflops = 0.0;
    int bestTile = 0;
    {
        const size_t n = kComputeElements;  // fits in cache: no DRAM traffic
        std::vector<float> in(n, 1.0001f), out(n, 0.0f);
        const int fma = 256;

        printf("  %-14s %14s   %s\n", "tile width", "GFLOP/s", "independent FMA chains");
        printf("  --------------------------------------------------------------\n");

        double previous = 0.0;
        bool sawCliff = false;
        for (int tile : {4, 8, 16, 32, 64, 128}) {
            double ms = timeTile(tile, in.data(), out.data(), n, fma, 10);
            double g = gflops(intensityFlops(n, fma), ms);
            const char* note = "";
            if (previous > 0.0 && g < previous * 0.7) {
                note = "  <- register spill";
                sawCliff = true;
            }
            printf("  %-14d %14.1f   %d%s\n", tile, g, tile, note);
            if (g > peakGflops) {
                peakGflops = g;
                bestTile = tile;
            }
            previous = g;
        }

        printf("\n  Best: tile %d at %.1f GFLOP/s.\n", bestTile, peakGflops);
        printf("  Throughput climbs as long as adding chains keeps the FMA pipeline\n");
        printf("  busier - an FMA has ~4 cycles of latency but issues every cycle, so\n");
        printf("  one chain alone reaches a quarter of peak.\n");
        if (sawCliff) {
            printf("\n  Then it COLLAPSES. That is REGISTER SPILLING: the tile no longer\n");
            printf("  fits in the CPU's 16 vector registers, so the compiler keeps it on\n");
            printf("  the stack and every accumulator update becomes memory traffic.\n");
        } else {
            printf("\n  No collapse showed up here - your compiler kept every tile in\n");
            printf("  registers. Add 256 to the list and it will eventually spill.\n");
        }
        printf("  Remember this shape. On a GPU the same cliff costs you occupancy as\n");
        printf("  well as speed, because registers are shared by all resident threads\n");
        printf("  (Phase 4 exercise 11).\n");
    }

    // ========================================================
    // The roofline
    // ========================================================
    const double ridge = ridgePoint(peakGflops, peakGBs);
    printSection("The roofline");
    printKV("peak bandwidth", peakGBs, "GB/s");
    printKV("peak throughput", peakGflops, "GFLOP/s");
    printKV("ridge point", ridge, "FLOP/byte");
    printf("\n  Below %.1f FLOP/byte this machine is memory bound; above it,\n", ridge);
    printf("  compute bound.\n");
    printf("\n  Both roofs here are SINGLE THREADED. A whole chip has many cores\n");
    printf("  sharing one memory controller, so its compute roof scales with core\n");
    printf("  count while its bandwidth roof does not - the full-chip ridge point\n");
    printf("  is therefore several times higher than this one. An RTX 3060 sits\n");
    printf("  near 37 FLOP/byte (13100 GFLOP/s over 360 GB/s).\n");

    // ========================================================
    // The sweep: cross the ridge
    // ========================================================
    printSection("Intensity sweep");
    printf("  %-6s %8s %12s %12s %8s  %s\n", "FMAs", "AI", "achieved", "roofline", "of roof",
           "attainable performance");
    printf("  ---------------------------------------------------------------"
           "-------------------------\n");

    const size_t n = kMemoryElements;
    std::vector<float> in(n, 1.0001f), out(n, 0.0f);

    // Use the tile width that won the sweep above, otherwise the
    // achieved column would be measured against a roof it cannot
    // possibly reach.
    for (int fma : kIntensitySweep) {
        double ms = timeTile(bestTile, in.data(), out.data(), n, fma, 3);
        double ai = intensityOf(fma);
        double achieved = gflops(intensityFlops(n, fma), ms);
        double roof = rooflineBound(ai, peakGflops, peakGBs);
        const char* regime = (ai < ridge) ? "memory bound" : "compute bound";

        printf("  %-6d %8.1f %12.1f %12.1f %7.0f%%  ", fma, ai, achieved, roof,
               100.0 * achieved / (roof > 0 ? roof : 1.0));
        printBar(achieved, peakGflops);
        printf(" %s\n", regime);
    }

    printf("\n  How to read this\n");
    printf("    The 'roofline' column is the fastest this machine could possibly\n");
    printf("    run a kernel of that intensity. Low-AI rows are capped by the\n");
    printf("    sloped bandwidth roof no matter how good the code is; the only way\n");
    printf("    up is to move fewer bytes. High-AI rows are capped by the flat\n");
    printf("    compute roof, where better instructions are what help.\n");

    printSection("Where real kernels sit");
    printf("  %-34s %10s  %s\n", "kernel", "AI", "regime");
    printf("  -------------------------------------------------------------\n");
    struct Known {
        const char* name;
        double ai;
    };
    const Known known[] = {
        {"vector copy (a[i] = b[i])", 0.0},
        {"SAXPY (y = a*x + y)", 2.0 / 12.0},
        {"dot product", 2.0 / 8.0},
        {"3-point stencil", 5.0 / 12.0},
        {"naive matmul (no reuse)", 2.0 / 12.0},
        {"tiled matmul, tile 32", 8.0},
        {"dense GEMM, register blocked", 60.0},
    };
    for (const Known& k : known)
        printf("  %-34s %10.2f  %s\n", k.name, k.ai,
               k.ai < ridge ? "memory bound" : "compute bound");

    printf("\n  Almost everything real sits on the LEFT of the ridge. That is why\n");
    printf("  Phase 3 of this curriculum is mostly about memory, not arithmetic,\n");
    printf("  and why the ridge point rising every hardware generation makes the\n");
    printf("  problem worse, not better.\n");

    reportCheck("bandwidth roof is plausible (> 1 GB/s)", peakGBs > 1.0);
    reportCheck("compute roof is plausible (> 1 GFLOP/s)", peakGflops > 1.0);
    reportCheck("more chains beat fewer (ILP matters)", bestTile >= 8);

    return verifySummary();
}
