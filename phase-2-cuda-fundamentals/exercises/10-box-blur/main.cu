// ============================================================
// 10 - Box blur: stencils, halos and separability  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__device__ __forceinline__ int clampDev(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ------------------------------------------------------------
// TODO 1: the naive 2D stencil
// ------------------------------------------------------------
// For each output pixel, average the (2R+1)x(2R+1) window centred
// on it. Clamp out-of-range coordinates to the edge with
// clampDev(), so border pixels repeat rather than wrap or crash.
//
// Use integer arithmetic and round with (sum + window/2) / window,
// exactly as boxBlurCPU does - then the two agree bit for bit and
// you can use checkArrayExact().
__global__ void blur2D(const unsigned char* in, unsigned char* out, int width, int height,
                       int radius) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    (void)radius;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the separable version, in two passes
// ------------------------------------------------------------
// A box filter is a product of a horizontal box and a vertical
// box, so blurring rows and then columns gives EXACTLY the same
// answer as one 2D pass - it is not an approximation.
//
//   pass 1: tmp[y][x] = sum of in[y][x-R .. x+R]
//   pass 2: out[y][x] = (sum of tmp[y-R .. y+R][x] + window/2) / window
//
// Cost drops from (2R+1)^2 reads per pixel to 2*(2R+1): at radius
// 4 that is 18 instead of 81.
//
// Why is tmp `unsigned short` and not `unsigned char`? Work out
// the largest value a horizontal sum can reach.
__global__ void blurHorizontal(const unsigned char* in, unsigned short* tmp, int width,
                               int height, int radius) {
    (void)in;
    (void)tmp;
    (void)width;
    (void)height;
    (void)radius;
    // TODO
}

__global__ void blurVertical(const unsigned short* tmp, unsigned char* out, int width,
                             int height, int radius) {
    (void)tmp;
    (void)out;
    (void)width;
    (void)height;
    (void)radius;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: measure what the boundary handling costs
// ------------------------------------------------------------
// The same 2D kernel, but processing only the interior - no
// clamping at all. It computes the WRONG answer near the edges on
// purpose; it exists only so you can measure what the clamping and
// the resulting warp divergence were costing.
__global__ void blur2DNoBoundary(const unsigned char* in, unsigned char* out, int width,
                                 int height, int radius) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    (void)radius;
    // TODO
}

int main() {
    printBanner("Phase 2 / 10 - Box blur (starter)");
    requireCudaDevice();

    Image src = loadOrCreateTestImage("input.ppm", kImageWidth, kImageHeight);
    std::vector<unsigned char> gray = toGray(src);
    const int width = src.width, height = src.height;

    printf("  %d x %d grayscale, radius %d (%d taps per pixel)\n", width, height, kRadius,
           (2 * kRadius + 1) * (2 * kRadius + 1));

    std::vector<unsigned char> golden(gray.size());
    boxBlurCPU(gray.data(), golden.data(), width, height, kRadius);

    // --------------------------------------------------------
    // TODO 4: allocate, copy, run all variants, verify
    // --------------------------------------------------------
    // The intermediate buffer needs pixels * sizeof(unsigned short).
    // Suggested block: dim3(32, 8).
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: measure, and separate the two effects
    // --------------------------------------------------------
    // Report reads per output pixel for each variant alongside the
    // measured time, so you can see how much of the predicted 4.5x
    // saving actually materialises - and where the rest went.
    //
    // Then write the blurred image out with savePPM() and look at
    // it. The checkerboard should be soft and the circle should
    // have a fuzzy edge; if the borders look wrong, the clamping is
    // wrong.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the four kernels and the analysis in main.cu");
    return verifySummary();
}
