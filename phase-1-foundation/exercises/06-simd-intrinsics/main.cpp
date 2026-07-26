// ============================================================
// 06 - SIMD intrinsics: the CPU's warp  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

#if defined(__x86_64__) || defined(__i386__)
#define HAS_X86_SIMD 1
#include <immintrin.h>
#else
#define HAS_X86_SIMD 0
#endif

#if HAS_X86_SIMD

// The target attribute lets these functions use AVX2 and FMA even
// though the rest of the file is compiled for the baseline ISA.
// No special CMake flags, and the binary still starts on a CPU
// without AVX2 as long as you check before calling.
#define AVX2_FUNC __attribute__((target("avx2,fma")))

// ------------------------------------------------------------
// TODO 1: SAXPY with AVX2
// ------------------------------------------------------------
//   __m256            holds 8 floats
//   _mm256_set1_ps(a) broadcasts one float into all 8 lanes
//   _mm256_loadu_ps   loads 8 floats (u = unaligned, always safe)
//   _mm256_fmadd_ps(a, b, c) computes a*b + c in one instruction
//   _mm256_storeu_ps  stores 8 floats
//
// Process the array 8 elements at a time, then finish the leftover
// tail (n % 8) with a scalar loop. That tail is the CPU version of
// a partially filled warp: the hardware is there, you are just not
// using all of it.
AVX2_FUNC void saxpyAVX2(float a, const float* x, float* y, size_t n) {
    (void)a;
    (void)x;
    (void)y;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: dot product with AVX2
// ------------------------------------------------------------
// Keep a __m256 accumulator, then reduce it to one float at the
// end. The horizontal reduction is the interesting part:
//
//   _mm256_extractf128_ps(v, 1)  upper 128 bits
//   _mm256_castps256_ps128(v)    lower 128 bits
//   _mm_add_ps                   add the two halves -> 4 floats
//   _mm_hadd_ps                  pairwise add -> 2, then -> 1
//   _mm_cvtss_f32                extract lane 0
//
// This log2(lanes) tree is exactly the shape of the warp shuffle
// reduction you will write in Phase 3 exercise 07.
AVX2_FUNC float dotAVX2(const float* x, const float* y, size_t n) {
    (void)x;
    (void)y;
    (void)n;
    return 0.0f;  // TODO
}

// ------------------------------------------------------------
// TODO 3: the polynomial, vectorised
// ------------------------------------------------------------
// Same Horner loop as polyGolden, but on 8 elements at a time.
// This kernel is compute bound, so this is where you should see a
// real speedup close to 8x.
AVX2_FUNC void polyAVX2(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

#endif  // HAS_X86_SIMD

int main() {
    printBanner("Phase 1 / 06 - SIMD intrinsics (starter)");

#if !HAS_X86_SIMD
    printf("\n  [SKIP] This exercise uses x86 AVX2 intrinsics; this is not an x86 CPU.\n\n");
    return 0;
#else
    if (!__builtin_cpu_supports("avx2")) {
        printf("\n  [SKIP] This CPU does not support AVX2.\n\n");
        return 0;
    }

    const size_t n = kVectorLength;
    std::vector<float> x(n), y(n), yRef(n), out(n), outRef(n);
    fillRandom(x.data(), n, -1.0f, 1.0f, 7);
    fillRandom(y.data(), n, -1.0f, 1.0f, 8);
    yRef = y;

    printSection("Correctness");
    saxpyGolden(2.5f, x.data(), yRef.data(), n);
    saxpyAVX2(2.5f, x.data(), y.data(), n);
    bool ok = checkArray("saxpy AVX2", y, yRef, 1e-5, 1e-6);

    polyArrayGolden(x.data(), outRef.data(), n);
    polyAVX2(x.data(), out.data(), n);
    ok = checkArray("polynomial AVX2", out, outRef, 1e-4, 1e-4) && ok;

    float dRef = dotGolden(x.data(), x.data(), n);
    float dVec = dotAVX2(x.data(), x.data(), n);
    ok = reportCheck("dot AVX2", std::fabs(dRef - dVec) < 1e-2f * std::fabs(dRef)) && ok;

    if (!ok) {
        printTodoNotice("implement saxpyAVX2(), dotAVX2() and polyAVX2()");
        return verifySummary();
    }

    // TODO 4: benchmark scalar versus AVX2 for all three kernels and
    // explain why saxpy speeds up far less than the polynomial.
    printSection("Performance");
    printf("  TODO: add the benchmarks\n");

    return verifySummary();
#endif
}
