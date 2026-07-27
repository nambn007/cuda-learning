#pragma once
// ============================================================
// reference.h - stencils, halos and boundary handling
// ============================================================
// A box blur replaces every pixel by the average of the (2R+1)^2
// pixels around it. It is the simplest STENCIL: each output depends
// on a fixed neighbourhood of the input.
//
// Stencils are everywhere - convolution, image filters, PDE
// solvers, cellular automata - and they all share three problems:
//
// 1. REUSE. Neighbouring output pixels read overlapping input. With
//    radius 4, each input pixel is read (2*4+1)^2 = 81 times by the
//    naive kernel. Phase 3 exercise 14 stages a tile in shared
//    memory so each is read once.
//
// 2. BOUNDARIES. Pixels near the edge have no neighbours on one
//    side. Clamping (repeat the edge pixel) is the usual answer,
//    and the `if` it needs is the reason stencil kernels have
//    warp divergence at the borders.
//
// 3. SEPARABILITY. A box blur is separable: blurring horizontally
//    then vertically gives the same answer as one 2D pass, but
//    costs 2*(2R+1) reads per pixel instead of (2R+1)^2. At radius
//    4 that is 18 instead of 81 - a 4.5x saving from ALGEBRA, not
//    from any GPU trick. Always look for this before optimising a
//    stencil.
// ============================================================

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ppm.h"
#include "verify.h"

inline constexpr int kImageWidth = 2048;
inline constexpr int kImageHeight = 2048;
inline constexpr int kRadius = 4;  // 9x9 window

// Clamp to the edge: out-of-range coordinates repeat the border
// pixel. Cheap, artefact-free, and what every image library does.
inline int clampIndex(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Grayscale, single channel, so the exercise is about the stencil
// rather than about channel bookkeeping.
inline void boxBlurCPU(const unsigned char* in, unsigned char* out, int width, int height,
                       int radius) {
    const int window = (2 * radius + 1) * (2 * radius + 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = clampIndex(y + dy, 0, height - 1);
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = clampIndex(x + dx, 0, width - 1);
                    sum += in[static_cast<size_t>(sy) * width + sx];
                }
            }
            // Integer arithmetic throughout, so host and device
            // agree bit for bit - unlike exercise 09.
            out[static_cast<size_t>(y) * width + x] =
                static_cast<unsigned char>((sum + window / 2) / window);
        }
    }
}

// Reads performed by the naive 2D kernel: (2R+1)^2 per output pixel.
inline double naiveReadBytes(size_t pixels, int radius) {
    const double w = 2.0 * radius + 1.0;
    return static_cast<double>(pixels) * (w * w + 1.0);
}

// Reads performed by the separable version: two passes of (2R+1).
inline double separableReadBytes(size_t pixels, int radius) {
    const double w = 2.0 * radius + 1.0;
    return static_cast<double>(pixels) * 2.0 * (w + 1.0);
}

// Convert an RGB test image to a single grayscale plane.
inline std::vector<unsigned char> toGray(const Image& img) {
    std::vector<unsigned char> gray(img.pixelCount());
    for (size_t i = 0; i < img.pixelCount(); ++i) {
        if (img.channels == 1) {
            gray[i] = img.data[i];
        } else {
            // Integer weights (77, 151, 28)/256 so this stays exact.
            const int r = img.data[i * 3 + 0];
            const int g = img.data[i * 3 + 1];
            const int b = img.data[i * 3 + 2];
            gray[i] = static_cast<unsigned char>((77 * r + 151 * g + 28 * b) >> 8);
        }
    }
    return gray;
}
