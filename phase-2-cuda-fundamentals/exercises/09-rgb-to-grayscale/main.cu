// ============================================================
// 09 - RGB to grayscale: AoS versus SoA  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// A __device__ helper both kernels can call. __forceinline__ keeps
// it from becoming a real function call.
__device__ __forceinline__ unsigned char luma(unsigned char r, unsigned char g,
                                              unsigned char b) {
    float y = kWeightR * r + kWeightG * g + kWeightB * b;
    int v = static_cast<int>(y + 0.5f);
    return static_cast<unsigned char>(v > 255 ? 255 : (v < 0 ? 0 : v));
}

// ------------------------------------------------------------
// TODO 1: Array of Structures - the layout the file gives you
// ------------------------------------------------------------
// rgb holds RGBRGBRGB..., so pixel i is at bytes 3i, 3i+1, 3i+2.
// Grid-stride loop over `pixels`.
__global__ void grayInterleaved(const unsigned char* rgb, unsigned char* gray,
                                size_t pixels) {
    (void)rgb;
    (void)gray;
    (void)pixels;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: Structure of Arrays - the layout the GPU wants
// ------------------------------------------------------------
// Three separate planes, so each read is a plain contiguous byte
// load across the warp.
__global__ void grayPlanar(const unsigned char* r, const unsigned char* g,
                           const unsigned char* b, unsigned char* gray, size_t pixels) {
    (void)r;
    (void)g;
    (void)b;
    (void)gray;
    (void)pixels;
    // TODO
}

// ------------------------------------------------------------
// TODO 3 (harder): keep AoS but load four pixels at a time
// ------------------------------------------------------------
// 4 pixels = 12 bytes = exactly three uchar4. Read three uchar4,
// unpack the twelve components into four (r, g, b) triples, and
// write one uchar4 of output.
//
//     uchar4 a = rgb12[i*3 + 0];   // R0 G0 B0 R1
//     uchar4 b = rgb12[i*3 + 1];   // G1 B1 R2 G2
//     uchar4 c = rgb12[i*3 + 2];   // B2 R3 G3 B3
//
// Work out which component goes where - getting it wrong produces a
// visibly wrong image, which is the nice thing about image kernels.
__global__ void grayInterleavedVec4(const uchar4* rgb12, uchar4* gray4, size_t quads) {
    (void)rgb12;
    (void)gray4;
    (void)quads;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: the experiment that decides the AoS/SoA question
// ------------------------------------------------------------
// Extract ONLY the red channel, from each layout.
//   from AoS: out[i] = rgb[i * 3]      -> stride 3, a warp touches
//                                         96 bytes and uses 32
//   from SoA: out[i] = r[i]            -> plain contiguous copy
//
// Time both. The gap between them - not the gap between the two
// grayscale kernels - is the real cost of an interleaved layout.
__global__ void redFromInterleaved(const unsigned char* rgb, unsigned char* out,
                                   size_t pixels) {
    (void)rgb;
    (void)out;
    (void)pixels;
    // TODO
}

__global__ void redFromPlanar(const unsigned char* r, unsigned char* out, size_t pixels) {
    (void)r;
    (void)out;
    (void)pixels;
    // TODO
}

int main() {
    printBanner("Phase 2 / 09 - RGB to grayscale (starter)");
    requireCudaDevice();

    // Loads input.ppm if present, otherwise generates one.
    Image img = loadOrCreateTestImage("input.ppm", kImageWidth, kImageHeight);
    const size_t pixels = img.pixelCount();
    printf("  %d x %d = %zu pixels\n", img.width, img.height, pixels);

    std::vector<unsigned char> golden(pixels);
    rgbToGrayCPU(img.data.data(), golden.data(), pixels);

    // --------------------------------------------------------
    // TODO 5: build the planar copies, allocate, copy, run, verify
    // --------------------------------------------------------
    // deinterleave() in reference.h splits RGBRGB into three planes.
    //
    // Verify with checkImage(), NOT checkArrayExact(). nvcc contracts
    // `0.299f*r + 0.587f*g + 0.114f*b` into FMA instructions, which
    // round once instead of twice, so a small fraction of pixels
    // come out one grey level away from the host result. That is
    // expected; see the note in reference.h.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 6: measure everything, then predict before you look
    // --------------------------------------------------------
    // Before running, write down your prediction for each:
    //   a) AoS versus SoA for grayscale - which wins, and by how much?
    //   b) uchar4 versus scalar AoS?
    //   c) AoS versus SoA for extracting only the red channel?
    //
    // At least one of those answers is probably not what the usual
    // "prefer SoA on a GPU" advice would lead you to expect. Working
    // out WHY is the exercise.
    //
    // Finally write the result out with savePPM("output_gray.pgm", ...)
    // and look at it. An image kernel that is subtly wrong usually
    // LOOKS wrong, which no assertion gives you for free.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the five kernels and the comparisons in main.cu");
    return verifySummary();
}
