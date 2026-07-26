#pragma once
// ============================================================
// common/report.h - Console output and performance metrics
// ============================================================
// Pure C++ (no CUDA headers).
//
// Every exercise prints the same shape of output so that results
// are comparable across exercises and across machines:
//
//   ============================================================
//     Exercise title
//   ============================================================
//     ...checks...
//   Variant                        Time (ms)     GB/s   GFLOP/s  Speedup
//   ---------------------------------------------------------------
//   CPU baseline                      42.100     1.42      0.51    1.00x
//   GPU naive                          1.310    45.60     16.30   32.14x
// ============================================================

#include <cstdio>
#include <string>
#include <vector>

// ------------------------------------------------------------
// Derived performance metrics
// ------------------------------------------------------------
// Effective bandwidth: how many bytes the kernel had to move
// through DRAM, divided by the time it took. Compare this against
// the theoretical peak of your card (printed by printDeviceInfo)
// to know whether a memory-bound kernel is doing well.
inline double gbPerSec(double bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return bytes / (ms * 1.0e-3) / 1.0e9;
}

// Throughput in billions of floating point operations per second.
// Compare against the theoretical peak for compute-bound kernels.
inline double gflops(double flops, double ms) {
    if (ms <= 0.0) return 0.0;
    return flops / (ms * 1.0e-3) / 1.0e9;
}

// Arithmetic intensity = FLOPs performed per byte moved.
// This is the x-axis of the roofline model: kernels to the left of
// the ridge point are memory bound, kernels to the right are
// compute bound. See phase-1-foundation/exercises/07-roofline-model.
inline double arithmeticIntensity(double flops, double bytes) {
    if (bytes <= 0.0) return 0.0;
    return flops / bytes;
}

// ------------------------------------------------------------
// Section headers
// ------------------------------------------------------------
inline void printBanner(const char* title) {
    printf("============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

inline void printSection(const char* title) {
    printf("\n--- %s ---\n", title);
}

inline void printKV(const char* key, const char* value) {
    printf("  %-28s %s\n", key, value);
}

inline void printKV(const char* key, double value, const char* unit = "") {
    printf("  %-28s %.3f %s\n", key, value, unit);
}

// ------------------------------------------------------------
// ResultTable - one row per implementation variant
// ------------------------------------------------------------
// The first row added becomes the speedup baseline, so add the
// slow reference implementation first.
struct ResultTable {
    struct Row {
        std::string name;
        double ms;
        double bytes;  // bytes moved; 0 to hide the GB/s column value
        double flops;  // FLOPs performed; 0 to hide the GFLOP/s column value
    };

    std::vector<Row> rows;
    double baselineMs = 0.0;

    void add(const std::string& name, double ms, double bytes = 0.0, double flops = 0.0) {
        if (rows.empty()) baselineMs = ms;
        rows.push_back({name, ms, bytes, flops});
    }

    void print(const char* title = "Performance") const {
        printf("\n%s\n", title);
        printf("%-32s %10s %9s %10s %9s\n", "Variant", "Time (ms)", "GB/s", "GFLOP/s",
               "Speedup");
        printf("--------------------------------------------------------------------------\n");
        for (const Row& r : rows) {
            char bw[32] = "-";
            char fl[32] = "-";
            if (r.bytes > 0.0) snprintf(bw, sizeof(bw), "%.2f", gbPerSec(r.bytes, r.ms));
            if (r.flops > 0.0) snprintf(fl, sizeof(fl), "%.2f", gflops(r.flops, r.ms));
            double speedup = (r.ms > 0.0 && baselineMs > 0.0) ? baselineMs / r.ms : 0.0;
            printf("%-32s %10.3f %9s %10s %8.2fx\n", r.name.c_str(), r.ms, bw, fl, speedup);
        }
        printf("--------------------------------------------------------------------------\n");
    }
};

// ------------------------------------------------------------
// Hints shown when an exercise is still unsolved
// ------------------------------------------------------------
// The starter files (main.cu) call this so that running an
// unfinished exercise prints an explanation instead of a crash.
inline void printTodoNotice(const char* what) {
    printf("\n");
    printf("  ------------------------------------------------------------\n");
    printf("  This is the STARTER file and it is not finished yet.\n");
    printf("  TODO: %s\n", what);
    printf("\n");
    printf("  Open this file, implement the parts marked with TODO, then\n");
    printf("  rebuild. Compare with the *_sol target if you get stuck.\n");
    printf("  ------------------------------------------------------------\n");
}
