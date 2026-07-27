#pragma once
// ============================================================
// reference.h - RAII for device memory
// ============================================================
// Phase 1 exercise 04 built Matrix<T> around new[]/delete[]. This
// exercise builds the same thing around cudaMalloc/cudaFree, and
// the stakes are higher.
//
// A leaked host allocation is reclaimed when the process exits. A
// forgotten cudaFree leaks DEVICE memory, which nothing reclaims
// until the CUDA context is destroyed - so a long-running service
// that leaks 1 MB per request eventually fails with
// cudaErrorMemoryAllocation and no clue where it went.
//
// Worse, the classic C-style shape invites the bug:
//
//     float* d = nullptr;
//     cudaMalloc(&d, bytes);
//     if (somethingWrong) return -1;      // <- leaked
//     ...
//     cudaFree(d);
//
// Every early return, every thrown exception, every `goto fail` is
// a leak. RAII removes the possibility rather than reminding you
// about it.
//
// Two design decisions worth arguing about, both of which this
// exercise takes a position on:
//
//   COPY is deleted. An implicit device-to-device copy of a large
//   buffer is exactly the kind of expensive operation that should
//   never happen by accident. Make it explicit: clone().
//
//   MOVE is noexcept. Otherwise std::vector<DeviceBuffer<T>> will
//   copy instead of moving when it grows - and the copy is deleted,
//   so it will not even compile. noexcept is what makes the class
//   usable inside containers.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 4u * 1024u * 1024u;  // 16 MB

inline void scaleCPU(float a, const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a * in[i];
}
