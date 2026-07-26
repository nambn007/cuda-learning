// ============================================================
// 04 - SAXPY and vectorised loads  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void saxpy(float a, const float* x, float* y, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) {
        // Compiles to a single FMA: one multiply-add with one
        // rounding step, not two. Slightly more accurate than a
        // separate multiply and add, and slightly different from
        // the CPU result - hence the tolerance below.
        y[i] = a * x[i] + y[i];
    }
}

__global__ void saxpyVectorised(float a, const float4* x, float4* y, size_t n4) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n4; i += stride) {
        // One 16-byte load instead of four 4-byte loads. The bytes
        // moved are identical; the number of memory INSTRUCTIONS
        // drops by 4, which matters when the kernel has nothing
        // else to do while it waits.
        float4 xv = x[i];
        float4 yv = y[i];
        yv.x = a * xv.x + yv.x;
        yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z;
        yv.w = a * xv.w + yv.w;
        y[i] = yv;
    }
}

// Handles the elements that do not fill a whole float4.
__global__ void saxpyTail(float a, const float* x, float* y, size_t start, size_t n) {
    size_t i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    printBanner("Phase 2 / 04 - SAXPY");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    printf("  n = %zu floats, alpha = %.1f, traffic = %.0f MB\n", n, kAlpha,
           saxpyBytes(n) / (1024.0 * 1024.0));

    SaxpyProblem problem(n);
    std::vector<float> result(n);

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, problem.x.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;

    // Resets y before each run - SAXPY accumulates into y, so a
    // repeated benchmark would otherwise compute something else
    // every iteration.
    auto resetY = [&] {
        CUDA_CHECK(cudaMemcpy(d_y, problem.y0.data(), bytes, cudaMemcpyHostToDevice));
    };

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");

    resetY();
    saxpy<<<grid, block>>>(kAlpha, d_x, d_y, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    // rtol 1e-5: the GPU fuses the multiply-add, the CPU reference
    // does not, so the two differ in the last bit or two.
    checkArray("scalar saxpy", result, problem.golden, 1e-5, 1e-6);

    const size_t n4 = n / 4;
    const size_t tailStart = n4 * 4;

    resetY();
    saxpyVectorised<<<grid, block>>>(kAlpha, reinterpret_cast<const float4*>(d_x),
                                     reinterpret_cast<float4*>(d_y), n4);
    CUDA_CHECK_LAST();
    if (tailStart < n) {
        saxpyTail<<<ceilDiv(n - tailStart, size_t(256)), 256>>>(kAlpha, d_x, d_y, tailStart, n);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    checkArray("vectorised saxpy (float4)", result, problem.golden, 1e-5, 1e-6);

    // ========================================================
    // Performance
    // ========================================================
    printSection("Performance");

    double msCpu = timeCpuMs(5, [&] {
        std::vector<float> y = problem.y0;
        saxpyCPU(kAlpha, problem.x.data(), y.data(), n);
    });

    double msScalar = timeGpuMs(20, [&] { saxpy<<<grid, block>>>(kAlpha, d_x, d_y, n); });

    double msVector = timeGpuMs(20, [&] {
        saxpyVectorised<<<grid, block>>>(kAlpha, reinterpret_cast<const float4*>(d_x),
                                         reinterpret_cast<float4*>(d_y), n4);
    });

    ResultTable table;
    table.add("CPU (single thread)", msCpu, saxpyBytes(n), saxpyFlops(n));
    table.add("GPU scalar", msScalar, saxpyBytes(n), saxpyFlops(n));
    table.add("GPU float4", msVector, saxpyBytes(n), saxpyFlops(n));
    table.print("SAXPY: y = a*x + y");

    // ========================================================
    // The only metric that matters for a memory-bound kernel
    // ========================================================
    printSection("Percentage of peak bandwidth");

    const double peak = theoreticalBandwidthGBs();
    const double bwScalar = gbPerSec(saxpyBytes(n), msScalar);
    const double bwVector = gbPerSec(saxpyBytes(n), msVector);

    if (peak > 0.0) {
        printf("  Theoretical peak      : %.1f GB/s\n", peak);
        printf("  Scalar kernel         : %.1f GB/s  (%.0f%% of peak)\n", bwScalar,
               100.0 * bwScalar / peak);
        printf("  float4 kernel         : %.1f GB/s  (%.0f%% of peak)\n", bwVector,
               100.0 * bwVector / peak);
        printf("\n  How to read this\n");
        printf("    > 80%%  the kernel is finished; look elsewhere for speed\n");
        printf("    50-80%% usually fine for a kernel doing real work\n");
        printf("    < 50%%  something is wrong, and it is almost always the\n");
        printf("           access pattern (Phase 3 exercise 01)\n");
    }

    printf("\n  Why float4 helps at all\n");
    printf("    The bytes moved are identical. What changes is the number of\n");
    printf("    memory INSTRUCTIONS: one 16-byte request instead of four 4-byte\n");
    printf("    ones. A memory-bound kernel has nothing to do while it waits, so\n");
    printf("    issuing fewer, larger requests keeps more data in flight per warp.\n");
    printf("    The gain is usually a few percent - real, but not transformative.\n");
    printf("    Do not reach for it before the access pattern is already coalesced.\n");

    printf("\n  Alignment matters\n");
    printf("    reinterpret_cast to float4 is only legal on a 16-byte aligned\n");
    printf("    pointer. cudaMalloc returns at least 256-byte alignment, so a base\n");
    printf("    pointer is safe - but `d_x + 1` is not, and the failure mode is a\n");
    printf("    misaligned-address error at run time.\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    reportCheck("scalar kernel exceeds 50% of peak bandwidth",
                peak <= 0.0 || bwScalar > 0.5 * peak);
    reportCheck("float4 kernel is not slower than scalar", msVector <= msScalar * 1.05);

    return verifySummary();
}
