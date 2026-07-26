#pragma once
// ============================================================
// reference.h - SIMD on the CPU, SIMT on the GPU
// ============================================================
// An AVX2 register holds 8 floats and one instruction operates on
// all 8 at once. That is SIMD: Single Instruction, Multiple Data.
//
// A CUDA warp holds 32 threads that issue one instruction together.
// NVIDIA calls it SIMT (Single Instruction, Multiple Threads), and
// the difference is mostly about how divergence is handled:
//
//   SIMD (AVX2)          SIMT (CUDA warp)
//   ---------------      -----------------------------------
//   8 lanes              32 lanes
//   you write the        you write scalar code; the hardware
//   shuffles by hand     groups 32 of them into one instruction
//   masks via _mm256_    divergence handled by the scheduler,
//   blendv / cmp         but both branches still execute
//
// The lesson that transfers: work is done in fixed-width groups,
// and a group that is not full - a remainder loop, or a warp where
// half the threads took the other branch - wastes hardware.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kVectorLength = 1u << 22;  // 4M floats = 16 MB

// SAXPY: y = a * x + y. One multiply-add per element, 8 bytes read
// and 4 written -> arithmetic intensity 2/12 FLOP per byte. Deeply
// memory bound; SIMD cannot fix that.
inline void saxpyGolden(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// Dot product: also memory bound, but it needs a horizontal
// reduction at the end - the same problem you will solve on the
// GPU with __shfl_down_sync in Phase 3.
inline float dotGolden(const float* x, const float* y, size_t n) {
    // Accumulate in double so the reference is not itself limited
    // by float rounding over 4M terms.
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * y[i];
    return static_cast<float>(sum);
}

// A deliberately compute-bound kernel: a degree-15 polynomial
// evaluated with Horner's rule. 30 FLOPs for 8 bytes of traffic,
// so arithmetic intensity is ~3.75 - far to the right of the
// roofline ridge point. This is where SIMD actually shines.
inline constexpr int kPolyDegree = 15;

inline float polyGolden(float x) {
    float acc = 1.0f;
    for (int i = 0; i < kPolyDegree; ++i) acc = acc * x + static_cast<float>(i + 1);
    return acc;
}

inline void polyArrayGolden(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = polyGolden(in[i]);
}

inline double polyFlops(size_t n) {
    // Two FLOPs (one FMA) per Horner step.
    return static_cast<double>(n) * kPolyDegree * 2.0;
}
