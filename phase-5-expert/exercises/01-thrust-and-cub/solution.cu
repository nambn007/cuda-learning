// ============================================================
// 01 - Thrust and CUB  [SOLUTION]
// ============================================================

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// The best hand-written reduction from Phase 3 exercise 06 (v7),
// so the comparison is against a genuine effort rather than a straw
// man.
__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void handWrittenReduce(const float* in, float* out, size_t n) {
    __shared__ float warpSums[32];
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned warp = threadIdx.x / warpSize;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        a0 += in[i];
        a1 += in[i + stride];
        a2 += in[i + 2 * stride];
        a3 += in[i + 3 * stride];
    }
    for (; i < n; i += stride) a0 += in[i];

    float v = warpReduceSum((a0 + a1) + (a2 + a3));
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const unsigned warpCount = (blockDim.x + warpSize - 1) / warpSize;
        v = (lane < warpCount) ? warpSums[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

int main() {
    printBanner("Phase 5 / 01 - Thrust and CUB");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;

    printf("  n = %zu floats (%.0f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

    std::vector<float> host(n);
    fillRandom(host.data(), n, 0.5f, 1.5f, 151);
    const double goldenSum = sumCPU(host.data(), n);

    float* d_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_raw, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================
    // Reduction: hand-written vs CUB vs Thrust
    // ========================================================
    printSection("Reduction");
    {
        // --- hand-written (Phase 3/06 v7) ---
        float* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));
        handWrittenReduce<<<grid, block>>>(d_raw, d_partial, n);
        CUDA_CHECK_LAST();
        std::vector<float> partial(grid);
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double gotHand = sumCPU(partial.data(), grid);
        reportCheck("hand-written reduction", std::fabs(gotHand - goldenSum) / goldenSum < 1e-4);

        // --- CUB device reduce ---
        // CUB never allocates for you. Call it once with a null
        // temp pointer to learn the size, allocate, call again.
        float* d_cubOut = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cubOut, sizeof(float)));
        void* d_temp = nullptr;
        size_t tempBytes = 0;
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, tempBytes, d_raw, d_cubOut,
                                          static_cast<int>(n)));
        CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, tempBytes, d_raw, d_cubOut,
                                          static_cast<int>(n)));
        float gotCub = 0.0f;
        CUDA_CHECK(cudaMemcpy(&gotCub, d_cubOut, sizeof(float), cudaMemcpyDeviceToHost));
        reportCheck("cub::DeviceReduce::Sum",
                    std::fabs(gotCub - goldenSum) / goldenSum < 1e-3);
        printf("  CUB temporary storage: %zu bytes\n", tempBytes);

        // --- Thrust ---
        thrust::device_vector<float> tv(host.begin(), host.end());
        float gotThrust = thrust::reduce(tv.begin(), tv.end(), 0.0f, thrust::plus<float>());
        reportCheck("thrust::reduce",
                    std::fabs(gotThrust - goldenSum) / goldenSum < 1e-3);

        double msHand =
            timeGpuMs(10, [&] { handWrittenReduce<<<grid, block>>>(d_raw, d_partial, n); });
        double msCub = timeGpuMs(10, [&] {
            cub::DeviceReduce::Sum(d_temp, tempBytes, d_raw, d_cubOut, static_cast<int>(n));
        });
        double msThrust = timeGpuMs(
            10, [&] { thrust::reduce(thrust::device, d_raw, d_raw + n, 0.0f); });

        const double bytes = static_cast<double>(n) * sizeof(float);
        ResultTable t;
        t.add("hand-written (Phase 3/06 v7)", msHand, bytes);
        t.add("cub::DeviceReduce::Sum", msCub, bytes);
        t.add("thrust::reduce", msThrust, bytes);
        t.print("Sum of 16M floats");

        const double peak = theoreticalBandwidthGBs();
        if (peak > 0.0) {
            printf("\n  As a fraction of peak bandwidth (the ceiling for a reduction):\n");
            printf("    hand-written : %.0f%%\n", 100.0 * gbPerSec(bytes, msHand) / peak);
            printf("    CUB          : %.0f%%\n", 100.0 * gbPerSec(bytes, msCub) / peak);
            printf("    Thrust       : %.0f%%\n", 100.0 * gbPerSec(bytes, msThrust) / peak);
        }

        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_cubOut));
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ========================================================
    // Scan
    // ========================================================
    printSection("Inclusive scan");
    {
        std::vector<float> golden(n);
        inclusiveScanCPU(host.data(), golden.data(), n);

        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

        void* d_temp = nullptr;
        size_t tempBytes = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp, tempBytes, d_raw, d_out,
                                                 static_cast<int>(n)));
        CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp, tempBytes, d_raw, d_out,
                                                 static_cast<int>(n)));

        std::vector<float> result(n);
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        // A scan accumulates 16M floats, so the tail drifts from the
        // double-precision reference. Relative tolerance only.
        checkArray("cub::DeviceScan::InclusiveSum", result, golden, 1e-2, 1e-1);

        double msCub = timeGpuMs(10, [&] {
            cub::DeviceScan::InclusiveSum(d_temp, tempBytes, d_raw, d_out, static_cast<int>(n));
        });
        double msThrust = timeGpuMs(10, [&] {
            thrust::inclusive_scan(thrust::device, d_raw, d_raw + n, d_out);
        });

        // A scan reads once and writes once.
        const double bytes = static_cast<double>(n) * 2.0 * sizeof(float);
        ResultTable t;
        t.add("cub::DeviceScan", msCub, bytes);
        t.add("thrust::inclusive_scan", msThrust, bytes);
        t.print("Inclusive scan over 16M floats");

        printf("\n  A scan looks inherently sequential and is not: CUB uses a\n");
        printf("  single-pass 'decoupled look-back' algorithm that reaches close to\n");
        printf("  copy bandwidth. Writing that yourself is a multi-week project.\n");

        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ========================================================
    // Sort - where hand-writing is clearly a bad idea
    // ========================================================
    printSection("Sort");
    {
        thrust::device_vector<float> tv(host.begin(), host.end());
        double msSort = timeGpuMs(3, [&] {
            thrust::device_vector<float> copy = tv;
            thrust::sort(copy.begin(), copy.end());
        });

        thrust::device_vector<float> sorted = tv;
        thrust::sort(sorted.begin(), sorted.end());
        std::vector<float> result(n);
        thrust::copy(sorted.begin(), sorted.end(), result.begin());

        bool ordered = true;
        for (size_t i = 1; i < n; ++i)
            if (result[i] < result[i - 1]) {
                ordered = false;
                break;
            }
        reportCheck("thrust::sort produces a sorted sequence", ordered);
        printKV("thrust::sort (includes a device copy)", msSort, "ms");
        printf("  %.0f million keys per second\n", n / (msSort * 1e-3) / 1e6);
        printf("\n  This is radix sort. Hand-writing a competitive GPU radix sort is\n");
        printf("  a serious project; calling it is one line.\n");
    }

    printSection("Which to reach for");
    printf("  Thrust  whole-array operations, prototyping, when the code reading\n");
    printf("          well matters more than the last 10%%\n");
    printf("  CUB     inside your own kernels (cub::BlockReduce, BlockScan), or\n");
    printf("          when you want control over temporary storage\n");
    printf("  Neither only after you have measured both and beaten them\n");
    printf("\n  Note CUB's two-call convention: it never allocates for you. Call it\n");
    printf("  with a null temp pointer to learn the size, allocate, call again.\n");
    printf("  That is deliberate - it lets you reuse one scratch buffer across\n");
    printf("  many calls instead of allocating inside a hot loop.\n");

    CUDA_CHECK(cudaFree(d_raw));
    return verifySummary();
}
