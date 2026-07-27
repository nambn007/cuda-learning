#pragma once
// ============================================================
// reference.h - Thrust and CUB
// ============================================================
// Phase 3 had you write a reduction seven times and a scan twice.
// That was worth doing once. Doing it again in production is not.
//
// Thrust is an STL-style layer: thrust::sort, reduce, transform,
// inclusive_scan on device vectors, with iterators and functors.
// It is header-only and ships with the CUDA toolkit.
//
// CUB is the layer underneath. It exposes the same algorithms at
// three levels - warp, block and device - so you can drop
// cub::BlockReduce into the middle of your own kernel rather than
// calling out to a whole separate launch. Thrust is implemented on
// top of it.
//
// Which to reach for:
//   Thrust  whole-array operations, prototyping, when the code
//           reads better than it needs to run
//   CUB     inside your own kernels, or when you need the extra
//           10-20% and control over temporary storage
//   neither only when you have measured both and lost
//
// The point of this exercise is calibration: find out how far your
// hand-written Phase 3 kernels actually are from the library, so
// that in future you know when writing your own is justified.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 16u * 1024u * 1024u;

inline double sumCPU(const float* v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    return s;
}

inline void inclusiveScanCPU(const float* in, float* out, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += in[i];
        out[i] = static_cast<float>(acc);
    }
}
