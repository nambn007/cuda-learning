#pragma once
// ============================================================
// common/timer.h - Host-side timing utilities
// ============================================================
// Pure C++ (no CUDA headers). Used by every exercise, including
// the CPU-only ones in Phase 1.
//
// Why a median instead of a mean?
//   A single run is noise: the OS may preempt you, the CPU may
//   still be ramping its clock, caches may be cold. We therefore
//   run a few untimed warmup iterations, then time N iterations
//   and report the MEDIAN, which is immune to a single outlier.
// ============================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

// ------------------------------------------------------------
// CpuTimer - manual start/stop stopwatch
// ------------------------------------------------------------
struct CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0, t1;

    void start() { t0 = Clock::now(); }
    void stop() { t1 = Clock::now(); }

    // Elapsed time in milliseconds between the last start() and stop().
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void print(const char* label) const {
        printf("[%s] %.3f ms\n", label, elapsedMs());
    }
};

// ------------------------------------------------------------
// timeCpuMs - benchmark a callable, return the median time in ms
// ------------------------------------------------------------
// Usage:
//   double ms = timeCpuMs(20, [&]{ matmulNaive(A, B, C, N); });
//
// `warmup` iterations run untimed so that caches are warm and the
// CPU frequency has settled before measurement starts.
template <typename Fn>
double timeCpuMs(int iters, Fn&& fn, int warmup = 2) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        CpuTimer t;
        t.start();
        fn();
        t.stop();
        samples.push_back(t.elapsedMs());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}
