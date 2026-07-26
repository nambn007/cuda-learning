#pragma once
// ============================================================
// reference.h - what the Matrix<T> class has to satisfy
// ============================================================
// The C++ features drilled here are the ones you will use in
// every later phase:
//
//   RAII            -> Phase 2/12 wraps cudaMalloc + cudaFree
//   move semantics  -> returning a device buffer without copying
//   templates       -> one kernel wrapper for float, double, half
//   lambdas         -> Thrust and CUB take functors everywhere
//
// A GPU buffer is the perfect RAII candidate: forgetting a
// cudaFree leaks device memory that nothing else will reclaim
// until the process exits.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// Every allocation performed by Matrix<T> bumps this counter and
// every deallocation decrements it. The tests use it to prove that
// a move really did move instead of secretly copying, and that
// nothing leaks.
struct AllocationTracker {
    static inline long long live = 0;   // outstanding allocations
    static inline long long total = 0;  // allocations ever made

    static void onAllocate() {
        ++live;
        ++total;
    }
    static void onFree() { --live; }
    static void reset() {
        live = 0;
        total = 0;
    }
};

// A workload used to compare the class against a raw std::vector,
// so the abstraction can be shown to cost nothing.
inline double expectedSum(size_t rows, size_t cols) {
    // Sum of (r * cols + c) * 0.5 over all elements.
    double n = static_cast<double>(rows * cols);
    return 0.5 * (n - 1.0) * n / 2.0;
}
