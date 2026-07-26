// ============================================================
// 08 - Know your GPU  [SOLUTION]
// ============================================================

#include <cstdio>

#include "cuda_helper.h"

int main() {
    printBanner("Phase 1 / 08 - Know your GPU");
    requireCudaDevice();

    const int devices = deviceCount();
    printf("  CUDA devices found: %d\n", devices);
    printf("  CUDA runtime version: %d.%d\n", CUDART_VERSION / 1000,
           (CUDART_VERSION % 1000) / 10);

    int driverVersion = 0, runtimeVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    printf("  Driver supports up to CUDA %d.%d\n", driverVersion / 1000,
           (driverVersion % 1000) / 10);
    printf("  (The driver version must be >= the toolkit version, or nothing runs.)\n");

    for (int dev = 0; dev < devices; ++dev) {
        printDeviceInfo(dev);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // ====================================================
        // Theoretical peaks
        // ====================================================
        printSection("Theoretical peaks");

        double peakBandwidthGBs = 0.0;
        double peakFp32Gflops = 0.0;
        const int cores = coresPerSM(prop.major, prop.minor);

#if CUDART_VERSION < 13000
        // GDDR is double data rate: data moves on both clock edges.
        peakBandwidthGBs = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
        // One FMA = two floating point operations.
        peakFp32Gflops = 2.0 * cores * prop.multiProcessorCount * prop.clockRate * 1e3 / 1e9;

        printKV("memory clock", prop.memoryClockRate / 1000.0, "MHz");
        printKV("core clock", prop.clockRate / 1000.0, "MHz");
        printKV("peak bandwidth", peakBandwidthGBs, "GB/s");
        printKV("peak FP32", peakFp32Gflops, "GFLOP/s");
        printKV("peak FP32 (TFLOP/s)", peakFp32Gflops / 1000.0, "TFLOP/s");
#else
        printf("  clockRate / memoryClockRate were removed from cudaDeviceProp in\n");
        printf("  CUDA 13. Read the clocks from `nvidia-smi -q -d CLOCK` instead.\n");
#endif

        // ====================================================
        // Roofline
        // ====================================================
        if (peakBandwidthGBs > 0.0 && peakFp32Gflops > 0.0) {
            printSection("Roofline");
            const double ridge = peakFp32Gflops / peakBandwidthGBs;
            printKV("ridge point", ridge, "FLOP/byte");
            printf("\n  A kernel needs to perform at least %.0f floating point operations\n",
                   ridge);
            printf("  for every byte it reads or writes before this GPU becomes compute\n");
            printf("  bound. Below that, no amount of instruction tuning helps - only\n");
            printf("  moving fewer bytes does.\n\n");
            printf("  For comparison:\n");
            printf("    vector add     0.08 FLOP/byte  -> hopelessly memory bound\n");
            printf("    naive matmul   0.17 FLOP/byte  -> memory bound\n");
            printf("    tiled matmul  ~%4.0f FLOP/byte  -> depends on the tile size\n", 8.0);
            printf("    dense GEMM     %.0f+ FLOP/byte  -> compute bound, use Tensor Cores\n",
                   ridge);
            printf("\n  Compare with the CPU ridge point from exercise 07: this GPU's is\n");
            printf("  several times higher, so even MORE of your kernels will be limited\n");
            printf("  by memory rather than by arithmetic.\n");
        }

        // ====================================================
        // Occupancy budget
        // ====================================================
        // These are the resource limits every kernel competes for.
        // Exceed any one of them and fewer thread blocks fit on an
        // SM, which means fewer warps to hide memory latency with.
        printSection("Occupancy budget (per SM)");

        const int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        const int maxThreadsGpu = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        const double regsPerThread =
            static_cast<double>(prop.regsPerMultiprocessor) / prop.maxThreadsPerMultiProcessor;
        const double smemPerThread =
            static_cast<double>(prop.sharedMemPerMultiprocessor) / prop.maxThreadsPerMultiProcessor;

        printf("  %-34s %d\n", "max resident threads / SM", prop.maxThreadsPerMultiProcessor);
        printf("  %-34s %d\n", "max resident warps / SM", maxWarpsPerSM);
        printf("  %-34s %d\n", "max resident blocks / SM", prop.maxBlocksPerMultiProcessor);
        printf("  %-34s %d\n", "registers / SM", prop.regsPerMultiprocessor);
        printf("  %-34s %zu KB\n", "shared memory / SM", prop.sharedMemPerMultiprocessor / 1024);
        printf("\n  At 100%% occupancy each thread may use at most:\n");
        printf("  %-34s %.0f\n", "registers / thread", regsPerThread);
        printf("  %-34s %.1f bytes\n", "shared memory / thread", smemPerThread);
        printf("\n  %-34s %d threads\n", "whole GPU, fully occupied", maxThreadsGpu);
        printf("\n  Read those two limits carefully. %.0f registers per thread is not\n",
               regsPerThread);
        printf("  many - a kernel that keeps a 8x8 tile of floats in registers has\n");
        printf("  already spent 64 of them. Exceeding the budget does not fail; it\n");
        printf("  silently halves your occupancy, which is why Phase 3 exercise 15\n");
        printf("  exists.\n");

        // ====================================================
        // Feature availability
        // ====================================================
        printSection("Features relevant to later phases");
        printf("  %-40s %s\n", "Tensor Cores (Phase 5/09)",
               (prop.major >= 7) ? "yes" : "NO - sm_70+ required");
        printf("  %-40s %s\n", "Cooperative launch (Phase 4/06)",
               prop.cooperativeLaunch ? "yes" : "no");
        printf("  %-40s %s\n", "Managed memory (Phase 2/11)",
               prop.managedMemory ? "yes" : "no");
        printf("  %-40s %s\n", "Concurrent copy + kernel (Phase 4/03)",
               prop.asyncEngineCount > 0 ? "yes" : "no");
        printf("  %-40s %d\n", "Copy engines (Phase 4/03)", prop.asyncEngineCount);
        printf("  %-40s %s\n", "Multi-GPU peer access (Phase 4/08)",
               devices > 1 ? "possible" : "only one GPU present");
        printf("  %-40s %s\n", "L2 persistence control (Phase 4)",
               prop.major >= 8 ? "yes (Ampere+)" : "no");

        // ====================================================
        // Sanity checks
        // ====================================================
        reportCheck("compute capability is at least 5.0",
                    prop.major > 5 || (prop.major == 5 && prop.minor >= 0));
        reportCheck("warp size is 32", prop.warpSize == 32);
        reportCheck("device reports at least one SM", prop.multiProcessorCount > 0);
        reportCheck("cores-per-SM table knows this architecture", cores > 0);
    }

    printSection("Fill this in for your own notes");
    printf("  Write down the answers - you will need them in every later phase:\n");
    printf("    1. How many threads can be resident at once on this whole GPU?\n");
    printf("    2. How many bytes per second can it read from DRAM?\n");
    printf("    3. How many FLOPs must a kernel do per byte to be compute bound?\n");
    printf("    4. How many registers may a thread use before occupancy drops?\n");
    printf("    5. How does each of those compare with the CPU numbers you\n");
    printf("       measured in exercises 05 and 07?\n");

    return verifySummary();
}
