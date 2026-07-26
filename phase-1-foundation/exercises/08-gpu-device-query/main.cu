// ============================================================
// 08 - Know your GPU  [STARTER]
// ============================================================
// The first .cu file of the curriculum. It launches no kernel: the
// goal is to interrogate the hardware and work out, on paper, what
// it is capable of - so that every performance number from Phase 2
// onwards has something to be compared against.
// ============================================================

#include <cstdio>

#include "cuda_helper.h"

int main() {
    printBanner("Phase 1 / 08 - Know your GPU (starter)");
    requireCudaDevice();

    int devices = deviceCount();
    printf("  CUDA devices found: %d\n", devices);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printDeviceInfo(0);

    // --------------------------------------------------------
    // TODO 1: theoretical peak memory bandwidth
    // --------------------------------------------------------
    //   bandwidth (GB/s) = 2 * memoryClockRate(kHz) * 1e3
    //                        * (memoryBusWidth / 8) / 1e9
    // The factor 2 is because GDDR is double data rate: it moves
    // data on both the rising and the falling clock edge.
    printSection("Theoretical peaks");
    double peakBandwidthGBs = 0.0;  // TODO

    // --------------------------------------------------------
    // TODO 2: theoretical peak FP32 throughput
    // --------------------------------------------------------
    //   GFLOP/s = 2 * coresPerSM * multiProcessorCount
    //               * clockRate(kHz) * 1e3 / 1e9
    // The factor 2 is one FMA counted as two operations.
    // coresPerSM(major, minor) is provided by cuda_helper.h.
    double peakFp32Gflops = 0.0;  // TODO

    if (peakBandwidthGBs <= 0.0 || peakFp32Gflops <= 0.0) {
        printTodoNotice("compute the theoretical peaks in main.cu");
        return verifySummary();
    }

    printKV("peak bandwidth", peakBandwidthGBs, "GB/s");
    printKV("peak FP32", peakFp32Gflops, "GFLOP/s");

    // --------------------------------------------------------
    // TODO 3: the ridge point of this GPU
    // --------------------------------------------------------
    // ridge = peak GFLOP/s / peak GB/s, in FLOP per byte. Compare
    // it with the CPU ridge point you measured in exercise 07 -
    // the GPU's is several times higher, which means even MORE of
    // your kernels will be memory bound.
    printSection("Roofline");
    // TODO

    // --------------------------------------------------------
    // TODO 4: occupancy limits
    // --------------------------------------------------------
    // Work out and print:
    //   - maximum resident warps per SM
    //     (maxThreadsPerMultiProcessor / warpSize)
    //   - maximum resident threads on the whole GPU
    //   - registers available per thread at full occupancy
    //     (regsPerMultiprocessor / maxThreadsPerMultiProcessor)
    //   - shared memory available per thread at full occupancy
    //
    // These four numbers are the budget every kernel you write in
    // Phase 3 has to fit inside.
    printSection("Occupancy budget");
    // TODO

    return verifySummary();
}
