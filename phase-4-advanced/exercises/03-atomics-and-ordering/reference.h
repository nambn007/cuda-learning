#pragma once
// ============================================================
// reference.h - atomics, contention and memory ordering
// ============================================================
// An atomic is a read-modify-write that no other thread can
// interleave with. On a GPU with tens of thousands of threads, the
// interesting question is never "is it correct" but "how many
// threads want the same address at the same time".
//
// Three separate topics, in increasing order of difficulty:
//
// 1. CONTENTION. atomicAdd on ONE global address from every thread
//    serialises. The fix is PRIVATISATION: give each block its own
//    counter in shared memory, then combine per block. One global
//    atomic per block instead of one per thread. This is the same
//    idea as the histogram in Phase 3, and it is the single most
//    useful atomic technique there is.
//
// 2. atomicCAS. Only a handful of atomic operations exist in
//    hardware. Anything else - a float maximum, a custom combine -
//    is built from compare-and-swap in a retry loop:
//
//        old = *addr;
//        do { assumed = old;
//             old = atomicCAS(addr, assumed, f(assumed)); }
//        while (assumed != old);
//
//    The loop retries until nobody else changed the value in
//    between. Under heavy contention it can retry many times, so it
//    is correct but not automatically fast.
//
// 3. MEMORY ORDERING. Atomicity is not visibility. atomicAdd
//    guarantees the counter is right; it says NOTHING about whether
//    the data you wrote before it is visible to the thread that
//    reads the counter afterwards. __threadfence() is what orders
//    those writes.
//
//    Getting this wrong produces a kernel that works on your GPU,
//    at your block count, on a Tuesday.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 16u * 1024u * 1024u;
inline constexpr int kBlockSize = 256;
inline constexpr int kBins = 256;

inline void histogramCPU(const unsigned char* v, size_t n, unsigned int* bins) {
    for (int i = 0; i < kBins; ++i) bins[i] = 0;
    for (size_t i = 0; i < n; ++i) ++bins[v[i]];
}

inline float maxCPU(const float* v, size_t n) {
    float m = v[0];
    for (size_t i = 1; i < n; ++i)
        if (v[i] > m) m = v[i];
    return m;
}

inline double sumCPU(const float* v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    return s;
}
