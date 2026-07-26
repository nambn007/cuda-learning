#pragma once
// ============================================================
// common/verify.h - Correctness checking and test data
// ============================================================
// Pure C++ (no CUDA headers).
//
// Every exercise in this repository must answer two questions:
//   1. Is the result CORRECT?  -> this header
//   2. Is the result FAST?     -> timer.h / cuda_helper.h + report.h
//
// A fast kernel that computes the wrong answer is worth nothing,
// so correctness is always checked first and the program exits
// with a non-zero status when a check fails. That makes the whole
// exercise set usable as a regression test suite.
// ============================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ------------------------------------------------------------
// Global pass/fail bookkeeping
// ------------------------------------------------------------
inline int& verifyFailureCount() {
    static int failures = 0;
    return failures;
}

// Report a boolean check. Returns the value so it can be chained.
inline bool reportCheck(const char* label, bool ok, const char* detail = nullptr) {
    if (ok) {
        printf("  [PASS] %s\n", label);
    } else {
        printf("  [FAIL] %s%s%s\n", label,
               detail ? " - " : "", detail ? detail : "");
        ++verifyFailureCount();
    }
    return ok;
}

// Call this as the last statement of main():
//     return verifySummary();
// Exit code 0 means every check passed, 1 means at least one failed.
inline int verifySummary() {
    int failures = verifyFailureCount();
    printf("------------------------------------------------------------\n");
    if (failures == 0) {
        printf("  RESULT: all checks passed\n");
    } else {
        printf("  RESULT: %d check(s) FAILED\n", failures);
    }
    printf("------------------------------------------------------------\n");
    return failures == 0 ? 0 : 1;
}

// ------------------------------------------------------------
// Floating point comparison
// ------------------------------------------------------------
// Floating point addition is not associative, so a GPU kernel that
// sums numbers in a different order than the CPU will produce a
// slightly different result. Never compare floats with '=='.
//
// We use a mixed absolute/relative tolerance:
//     |a - b| <= atol + rtol * |b|
// The absolute term handles values near zero, where a relative
// tolerance would be meaninglessly strict.
template <typename T>
struct CompareStats {
    bool ok = true;
    size_t mismatches = 0;
    size_t firstIndex = 0;
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    T firstGot{};
    T firstExpected{};
};

template <typename T>
CompareStats<T> compareArrays(const T* got, const T* expected, size_t n,
                              double rtol = 1e-4, double atol = 1e-5) {
    CompareStats<T> s;
    for (size_t i = 0; i < n; ++i) {
        double g = static_cast<double>(got[i]);
        double e = static_cast<double>(expected[i]);

        // NaN never compares equal to anything, so flag it explicitly.
        if (std::isnan(g) != std::isnan(e) || std::isinf(g) != std::isinf(e)) {
            if (s.mismatches == 0) {
                s.firstIndex = i;
                s.firstGot = got[i];
                s.firstExpected = expected[i];
            }
            ++s.mismatches;
            s.ok = false;
            continue;
        }

        double absErr = std::fabs(g - e);
        double relErr = absErr / (std::fabs(e) + 1e-30);
        if (absErr > s.maxAbsErr) s.maxAbsErr = absErr;
        if (std::fabs(e) > 1e-6 && relErr > s.maxRelErr) s.maxRelErr = relErr;

        if (absErr > atol + rtol * std::fabs(e)) {
            if (s.mismatches == 0) {
                s.firstIndex = i;
                s.firstGot = got[i];
                s.firstExpected = expected[i];
            }
            ++s.mismatches;
            s.ok = false;
        }
    }
    return s;
}

// Compare two float arrays and print a PASS/FAIL line.
inline bool checkArray(const char* label, const float* got, const float* expected,
                       size_t n, double rtol = 1e-4, double atol = 1e-5) {
    CompareStats<float> s = compareArrays(got, expected, n, rtol, atol);
    if (s.ok) {
        printf("  [PASS] %s (n=%zu, max abs err %.3e, max rel err %.3e)\n",
               label, n, s.maxAbsErr, s.maxRelErr);
        return true;
    }
    printf("  [FAIL] %s (n=%zu, %zu mismatch(es))\n", label, n, s.mismatches);
    printf("         first at index %zu: got %.8g, expected %.8g\n",
           s.firstIndex, static_cast<double>(s.firstGot),
           static_cast<double>(s.firstExpected));
    printf("         max abs err %.3e, max rel err %.3e\n", s.maxAbsErr, s.maxRelErr);
    ++verifyFailureCount();
    return false;
}

inline bool checkArray(const char* label, const std::vector<float>& got,
                       const std::vector<float>& expected,
                       double rtol = 1e-4, double atol = 1e-5) {
    if (got.size() != expected.size()) {
        printf("  [FAIL] %s (size mismatch: %zu vs %zu)\n", label, got.size(),
               expected.size());
        ++verifyFailureCount();
        return false;
    }
    return checkArray(label, got.data(), expected.data(), got.size(), rtol, atol);
}

// Integer arrays are compared exactly - integer arithmetic is exact.
template <typename T>
bool checkArrayExact(const char* label, const T* got, const T* expected, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            printf("  [FAIL] %s - first mismatch at index %zu: got %lld, expected %lld\n",
                   label, i, static_cast<long long>(got[i]),
                   static_cast<long long>(expected[i]));
            ++verifyFailureCount();
            return false;
        }
    }
    printf("  [PASS] %s (n=%zu, exact match)\n", label, n);
    return true;
}

// ------------------------------------------------------------
// Deterministic test data
// ------------------------------------------------------------
// A fixed seed means every run - and every machine - produces the
// same input, so a failure is always reproducible.
inline std::mt19937& rng(unsigned seed = 1234u) {
    static std::mt19937 gen(seed);
    return gen;
}

inline void fillRandom(float* p, size_t n, float lo = -1.0f, float hi = 1.0f,
                       unsigned seed = 1234u) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = dist(gen);
}

inline void fillRandomInt(int* p, size_t n, int lo = 0, int hi = 255,
                          unsigned seed = 1234u) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = dist(gen);
}

template <typename T>
void fillConstant(T* p, size_t n, T value) {
    for (size_t i = 0; i < n; ++i) p[i] = value;
}

template <typename T>
void fillSequence(T* p, size_t n, T start = T(0), T step = T(1)) {
    T v = start;
    for (size_t i = 0; i < n; ++i, v += step) p[i] = v;
}

// Print the first `count` elements - handy when debugging a kernel.
template <typename T>
void printArray(const char* label, const T* p, size_t n, size_t count = 8) {
    printf("  %s: [", label);
    size_t m = n < count ? n : count;
    for (size_t i = 0; i < m; ++i) printf("%s%.4g", i ? ", " : "", static_cast<double>(p[i]));
    if (n > m) printf(", ... (%zu total)", n);
    printf("]\n");
}
