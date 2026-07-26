// ============================================================
// 07 - The roofline model  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ------------------------------------------------------------
// TODO 1: a kernel whose arithmetic intensity is a parameter
// ------------------------------------------------------------
// For each element: read one float, perform `fmaCount` fused
// multiply-adds on it, write one float.
//
//   out[i] = ((...((1*x + 1)*x + 1)...)*x + 1)      // fmaCount times
//
// Traffic is fixed at 8 bytes per element while arithmetic scales
// with fmaCount, so sweeping fmaCount sweeps the arithmetic
// intensity - which is what draws the roofline.
//
// Structure it as a REGISTER TILE of TILE elements:
//
//   for i in steps of TILE:
//       load x[0..TILE) from in, set acc[0..TILE) = 1
//       repeat fmaCount times:
//           for t in 0..TILE:  acc[t] = acc[t] * x[t] + 1
//       store acc[0..TILE) to out
//   handle the tail (n % TILE) with a scalar loop
//
// Why a tile and not one element at a time? An FMA has ~4 cycles of
// latency but issues every cycle. A single accumulator waits on its
// own previous result and reaches a quarter of peak. TILE
// independent accumulators keep the pipeline full - and the inner
// `t` loop has no cross-iteration dependency, so it vectorises.
//
// TILE is a template parameter so the trip count is a compile-time
// constant and the compiler can fully unroll the inner loop.
template <int TILE>
void intensityKernel(const float* in, float* out, size_t n, int fmaCount) {
    (void)in;
    (void)out;
    (void)n;
    (void)fmaCount;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: a pure streaming kernel for the bandwidth roof
// ------------------------------------------------------------
//     a[i] = b[i] + scalar * c[i]     (STREAM triad)
void triad(float* a, const float* b, const float* c, float scalar, size_t n) {
    (void)a;
    (void)b;
    (void)c;
    (void)scalar;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 1 / 07 - The roofline model (starter)");

    printSection("Correctness");
    {
        std::vector<float> in(64, 2.0f), out(64, 0.0f);
        intensityKernel<16>(in.data(), out.data(), 64, 1);
        bool ok = reportCheck("fma=1 on x=2 gives 3", out[0] == 3.0f);
        intensityKernel<16>(in.data(), out.data(), 64, 2);
        ok = reportCheck("fma=2 on x=2 gives 7", out[0] == 7.0f) && ok;

        std::vector<float> a(64, 0.0f), b(64, 1.0f), c(64, 2.0f);
        triad(a.data(), b.data(), c.data(), 3.0f, 64);
        ok = reportCheck("triad computes b + 3*c", a[0] == 7.0f) && ok;

        if (!ok) {
            printTodoNotice("implement intensityKernel<TILE>() and triad()");
            return verifySummary();
        }
    }

    // --------------------------------------------------------
    // TODO 3: measure the bandwidth roof
    // --------------------------------------------------------
    // Time triad() over kMemoryElements (64 MB, far bigger than any
    // cache) and convert with gbPerSec(3 * n * sizeof(float), ms).
    printSection("Roof 1 - memory bandwidth");
    double peakGBs = 0.0;  // TODO

    // --------------------------------------------------------
    // TODO 4: measure the compute roof, and find the register cliff
    // --------------------------------------------------------
    // Run intensityKernel with fmaCount = 256 over kComputeElements
    // (small enough to stay in cache, so no DRAM traffic pollutes
    // the number) for TILE = 4, 8, 16 and 32. Print all four.
    //
    // You should see throughput RISE from 4 to 16 chains as the FMA
    // pipeline fills up, then usually COLLAPSE at 32 - the tile no
    // longer fits in the vector registers and the compiler spills it
    // to the stack. Take the best of the four as your compute roof.
    printSection("Roof 2 - FMA throughput (and the register cliff)");
    double peakGflops = 0.0;  // TODO

    if (peakGBs <= 0.0 || peakGflops <= 0.0) {
        printTodoNotice("measure the two roofs");
        return verifySummary();
    }

    // --------------------------------------------------------
    // TODO 5: sweep the intensity and compare against the roofline
    // --------------------------------------------------------
    // For each fmaCount in kIntensitySweep, run over kMemoryElements
    // and print:
    //   AI         = intensityOf(fmaCount)
    //   achieved   = gflops(intensityFlops(n, fmaCount), ms)
    //   roof       = rooflineBound(AI, peakGflops, peakGBs)
    //   efficiency = achieved / roof
    // and label each row memory bound or compute bound by comparing
    // AI against ridgePoint(peakGflops, peakGBs).
    printSection("Intensity sweep");
    printf("  TODO: implement the sweep\n");

    return verifySummary();
}
