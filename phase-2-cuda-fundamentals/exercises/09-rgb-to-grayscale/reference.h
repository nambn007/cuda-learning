#pragma once
// ============================================================
// reference.h - RGB to grayscale, and AoS versus SoA
// ============================================================
// The arithmetic is one line:
//
//   gray = 0.299*R + 0.587*G + 0.114*B
//
// (ITU-R BT.601 luma weights: the eye is far more sensitive to
// green than to blue, so a plain average looks wrong.)
//
// The interesting part is the LAYOUT, and the result is not the one
// the usual advice would lead you to expect.
//
// An image is stored interleaved, RGBRGBRGB..., which is an Array
// of Structures. The standard GPU advice is "prefer Structure of
// Arrays" - three separate planes RRRR...GGGG...BBBB. Measure it
// here and the two come out the SAME, because this kernel needs all
// three channels of every pixel: a warp reading AoS touches 96
// consecutive bytes, and a warp reading three SoA planes touches
// three runs of 32. Same sectors, same traffic.
//
// AoS only hurts when you need PART of each structure. Then the
// stride between the fields you want wastes most of every sector.
// This exercise measures both cases, so the rule you take away is
// the correct one:
//
//   AoS vs SoA matters in proportion to how much of each structure
//   you actually read.
//
// What does help here is vectorising the loads: have each thread
// handle four pixels so twelve byte-loads become three 4-byte
// loads. That is an instruction-count win, not a layout win.
// ============================================================

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ppm.h"
#include "verify.h"

inline constexpr int kImageWidth = 4096;
inline constexpr int kImageHeight = 2048;

// BT.601 luma weights.
inline constexpr float kWeightR = 0.299f;
inline constexpr float kWeightG = 0.587f;
inline constexpr float kWeightB = 0.114f;

inline unsigned char rgbToGrayScalar(unsigned char r, unsigned char g, unsigned char b) {
    float y = kWeightR * r + kWeightG * g + kWeightB * b;
    // Round rather than truncate, and clamp: the weights sum to
    // 1.0 so overflow is impossible, but rounding differences would
    // otherwise show up as off-by-one against the GPU.
    int v = static_cast<int>(y + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return static_cast<unsigned char>(v);
}

inline void rgbToGrayCPU(const unsigned char* rgb, unsigned char* gray, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        gray[i] = rgbToGrayScalar(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }
}

// Split an interleaved RGB buffer into three contiguous planes.
inline void deinterleave(const unsigned char* rgb, unsigned char* r, unsigned char* g,
                         unsigned char* b, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
}

// 3 bytes read + 1 byte written per pixel, either layout.
inline double grayscaleBytes(size_t pixels) { return static_cast<double>(pixels) * 4.0; }

// 1 byte read + 1 byte written per pixel.
inline double singleChannelBytes(size_t pixels) {
    return static_cast<double>(pixels) * 2.0;
}

// ------------------------------------------------------------
// Comparing images: exact equality is the wrong test
// ------------------------------------------------------------
// The host computes 0.299f*r + 0.587f*g + 0.114f*b with separate
// multiplies and adds. nvcc contracts the same expression into FMA
// instructions by default (-fmad=true), which rounds once instead
// of twice. When the true value sits near x.5, the two round to
// different integers, and roughly one pixel in a few hundred comes
// out one grey level apart.
//
// That is not a bug, and chasing bit-exactness here would mean
// giving up FMA. The right test for an image kernel is "no pixel is
// off by more than one level", plus a bound on how many differ at
// all. Compile with -fmad=false if you ever do need bit-exact
// agreement with a host reference.
inline bool checkImage(const char* label, const unsigned char* got,
                       const unsigned char* expected, size_t n, int maxDiff = 1,
                       double maxFraction = 0.02) {
    size_t differing = 0;
    int worst = 0;
    size_t worstIndex = 0;
    for (size_t i = 0; i < n; ++i) {
        int d = static_cast<int>(got[i]) - static_cast<int>(expected[i]);
        if (d < 0) d = -d;
        if (d > 0) ++differing;
        if (d > worst) {
            worst = d;
            worstIndex = i;
        }
    }
    const double fraction = static_cast<double>(differing) / static_cast<double>(n);
    const bool ok = (worst <= maxDiff) && (fraction <= maxFraction);
    if (ok) {
        printf("  [PASS] %s (max diff %d level, %.3f%% of pixels differ)\n", label, worst,
               100.0 * fraction);
    } else {
        printf("  [FAIL] %s (max diff %d at index %zu, %.3f%% of pixels differ)\n", label,
               worst, worstIndex, 100.0 * fraction);
        ++verifyFailureCount();
    }
    return ok;
}
