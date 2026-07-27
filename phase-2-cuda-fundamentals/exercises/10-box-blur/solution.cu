// ============================================================
// 10 - Box blur: stencils, halos and separability  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__device__ __forceinline__ int clampDev(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ------------------------------------------------------------
// Naive 2D stencil
// ------------------------------------------------------------
// Every thread reads the full (2R+1)^2 window. Neighbouring threads
// read overlapping windows, so each input pixel is fetched up to
// (2R+1)^2 times - 81 at radius 4. The L1 and L2 caches recover a
// lot of that, which is why this is not 81x slower than a copy, but
// it is still the dominant cost.
__global__ void blur2D(const unsigned char* in, unsigned char* out, int width, int height,
                       int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int window = (2 * radius + 1) * (2 * radius + 1);
    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int sy = clampDev(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            const int sx = clampDev(x + dx, 0, width - 1);
            sum += in[static_cast<size_t>(sy) * width + sx];
        }
    }
    out[static_cast<size_t>(y) * width + x] =
        static_cast<unsigned char>((sum + window / 2) / window);
}

// ------------------------------------------------------------
// Separable: horizontal pass then vertical pass
// ------------------------------------------------------------
// A box filter is a product of a horizontal and a vertical box, so
// two 1D passes give exactly the same answer as one 2D pass. Cost
// drops from (2R+1)^2 reads per pixel to 2*(2R+1): 81 -> 18 at
// radius 4.
//
// This is a saving from ALGEBRA, not from any GPU technique. Look
// for it before reaching for shared memory - it is usually the
// larger win, and it composes with every other optimisation.
//
// The intermediate buffer holds a horizontal sum, which can reach
// 255 * 9 = 2295, so it must be wider than a byte. Using
// `unsigned short` keeps the arithmetic exact and the traffic low.
__global__ void blurHorizontal(const unsigned char* in, unsigned short* tmp, int width,
                               int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int sum = 0;
    const unsigned char* row = in + static_cast<size_t>(y) * width;
    for (int dx = -radius; dx <= radius; ++dx) sum += row[clampDev(x + dx, 0, width - 1)];
    tmp[static_cast<size_t>(y) * width + x] = static_cast<unsigned short>(sum);
}

__global__ void blurVertical(const unsigned short* tmp, unsigned char* out, int width,
                             int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int window = (2 * radius + 1) * (2 * radius + 1);
    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        sum += tmp[static_cast<size_t>(clampDev(y + dy, 0, height - 1)) * width + x];
    }
    out[static_cast<size_t>(y) * width + x] =
        static_cast<unsigned char>((sum + window / 2) / window);
}

// ------------------------------------------------------------
// The boundary cost, isolated
// ------------------------------------------------------------
// Interior-only: no clamping at all, so no divergence and no
// address arithmetic for the borders. It computes the wrong answer
// near the edge on purpose - it exists only to show what the
// boundary handling was costing.
__global__ void blur2DNoBoundary(const unsigned char* in, unsigned char* out, int width,
                                 int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < radius || y < radius || x >= width - radius || y >= height - radius) return;

    const int window = (2 * radius + 1) * (2 * radius + 1);
    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            sum += in[static_cast<size_t>(y + dy) * width + (x + dx)];
    out[static_cast<size_t>(y) * width + x] =
        static_cast<unsigned char>((sum + window / 2) / window);
}

int main() {
    printBanner("Phase 2 / 10 - Box blur");
    requireCudaDevice();

    Image src = loadOrCreateTestImage("input.ppm", kImageWidth, kImageHeight);
    std::vector<unsigned char> gray = toGray(src);
    const int width = src.width, height = src.height;
    const size_t pixels = gray.size();
    const int radius = kRadius;

    printf("  %d x %d grayscale, radius %d (%dx%d window = %d taps)\n", width, height,
           radius, 2 * radius + 1, 2 * radius + 1,
           (2 * radius + 1) * (2 * radius + 1));

    std::vector<unsigned char> golden(pixels), result(pixels);
    boxBlurCPU(gray.data(), golden.data(), width, height, radius);

    unsigned char *d_in = nullptr, *d_out = nullptr;
    unsigned short* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, pixels));
    CUDA_CHECK(cudaMalloc(&d_out, pixels));
    CUDA_CHECK(cudaMalloc(&d_tmp, pixels * sizeof(unsigned short)));
    CUDA_CHECK(cudaMemcpy(d_in, gray.data(), pixels, cudaMemcpyHostToDevice));

    const dim3 block(32, 8);
    const dim3 grid(ceilDiv(width, static_cast<int>(block.x)),
                    ceilDiv(height, static_cast<int>(block.y)));

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");

    CUDA_CHECK(cudaMemset(d_out, 0, pixels));
    blur2D<<<grid, block>>>(d_in, d_out, width, height, radius);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, pixels, cudaMemcpyDeviceToHost));
    // All integer arithmetic, so this must match exactly - unlike
    // the float luma in exercise 09.
    checkArrayExact("naive 2D stencil", result.data(), golden.data(), pixels);

    CUDA_CHECK(cudaMemset(d_out, 0, pixels));
    blurHorizontal<<<grid, block>>>(d_in, d_tmp, width, height, radius);
    CUDA_CHECK_LAST();
    blurVertical<<<grid, block>>>(d_tmp, d_out, width, height, radius);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, pixels, cudaMemcpyDeviceToHost));
    checkArrayExact("separable (two 1D passes)", result.data(), golden.data(), pixels);

    printf("\n  The separable version is not an approximation - a box filter is a\n");
    printf("  product of a horizontal and a vertical box, so the two agree exactly.\n");

    // ========================================================
    // Performance
    // ========================================================
    printSection("Performance");

    double msCpu = timeCpuMs(3, [&] {
        boxBlurCPU(gray.data(), result.data(), width, height, radius);
    });
    double ms2D =
        timeGpuMs(10, [&] { blur2D<<<grid, block>>>(d_in, d_out, width, height, radius); });
    double msSep = timeGpuMs(10, [&] {
        blurHorizontal<<<grid, block>>>(d_in, d_tmp, width, height, radius);
        blurVertical<<<grid, block>>>(d_tmp, d_out, width, height, radius);
    });
    double msNoBoundary = timeGpuMs(10, [&] {
        blur2DNoBoundary<<<grid, block>>>(d_in, d_out, width, height, radius);
    });

    ResultTable table;
    table.add("CPU naive 2D", msCpu, naiveReadBytes(pixels, radius));
    table.add("GPU naive 2D", ms2D, naiveReadBytes(pixels, radius));
    table.add("GPU 2D, no boundary handling", msNoBoundary, naiveReadBytes(pixels, radius));
    table.add("GPU separable (2 passes)", msSep, separableReadBytes(pixels, radius));
    table.print("Box blur, radius 4");

    // ========================================================
    // Interpretation
    // ========================================================
    printSection("Where the time goes");

    const double w = 2.0 * radius + 1.0;
    printf("  Reads per output pixel\n");
    printf("    naive 2D    : %.0f  ((2R+1)^2)\n", w * w);
    printf("    separable   : %.0f  (2 * (2R+1))\n", 2.0 * w);
    printf("    ratio       : %.1fx fewer reads\n", (w * w) / (2.0 * w));
    printf("    measured    : %.2fx faster\n", ms2D / msSep);

    printf("\n  Separability is ALGEBRA, not a GPU trick. It applies on a CPU too,\n");
    printf("  it composes with every other optimisation, and at radius 8 it would\n");
    printf("  be worth %.1fx instead of %.1fx. Always check for it before reaching\n",
           (17.0 * 17.0) / (2.0 * 17.0), (w * w) / (2.0 * w));
    printf("  for shared memory.\n");

    printf("\n  Boundary handling cost %.2fx\n", ms2D / msNoBoundary);
    printf("    That is larger than most people expect, and the reason is not warp\n");
    printf("    divergence - only the handful of warps on the actual border\n");
    printf("    diverge. It is the sheer count of clamp operations: %.0f integer\n",
           2.0 * w * w);
    printf("    min/max pairs per output pixel, all of them in the innermost loop,\n");
    printf("    on a kernel that is otherwise almost pure loads.\n");
    printf("    The interior-only kernel skips them and computes the wrong answer\n");
    printf("    near the border, which is why it exists only as a measurement.\n");
    printf("\n    The production fix is to pad the input by `radius` pixels on every\n");
    printf("    side so no clamping is needed at all - you trade a little memory\n");
    printf("    for the entire cost. The alternative is to split the launch into an\n");
    printf("    interior kernel with no clamping and a thin border kernel with it.\n");

    printf("\n  What is still on the table\n");
    printf("    Even the separable version re-reads each pixel %.0f times. Phase 3\n", 2.0 * w);
    printf("    exercise 14 stages a tile plus its halo in __shared__ memory so\n");
    printf("    each input pixel is fetched from global memory exactly once, and\n");
    printf("    Phase 3 exercise 13 puts the filter weights in __constant__ memory.\n");
    printf("    A box blur also has a running-sum formulation that is O(1) per\n");
    printf("    pixel regardless of radius - worth deriving on paper.\n");

    // Save the result to look at.
    Image out;
    out.resize(width, height, 1);
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, pixels, cudaMemcpyDeviceToHost));
    blurHorizontal<<<grid, block>>>(d_in, d_tmp, width, height, radius);
    blurVertical<<<grid, block>>>(d_tmp, d_out, width, height, radius);
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, pixels, cudaMemcpyDeviceToHost));
    if (savePPM("output_blur.pgm", out)) {
        printf("\n  Wrote output_blur.pgm - the checkerboard should be soft and the\n");
        printf("  circle should have a fuzzy edge. If the borders look wrong, the\n");
        printf("  clamping is wrong.\n");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_tmp));

    reportCheck("separable is faster than naive 2D", msSep < ms2D);
    reportCheck("GPU beats the CPU", ms2D < msCpu);
    return verifySummary();
}
