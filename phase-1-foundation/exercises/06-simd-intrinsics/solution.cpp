// ============================================================
// 06 - SIMD intrinsics: the CPU's warp  [SOLUTION]
// ============================================================

#include <cmath>
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
#define AVX2_FUNC __attribute__((target("avx2,fma")))

// ------------------------------------------------------------
// SAXPY: y = a*x + y, eight elements per instruction
// ------------------------------------------------------------
AVX2_FUNC void saxpyAVX2(float a, const float* x, float* y, size_t n) {
    const __m256 va = _mm256_set1_ps(a);  // a broadcast to all 8 lanes

    size_t i = 0;
    // Main loop: 8 floats per iteration.
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        // One instruction: multiply and add, with a single rounding.
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(y + i, vy);
    }
    // Tail: the elements that do not fill a whole vector. On a GPU
    // this is the `if (i < n)` bounds check every kernel needs, and
    // for the same reason.
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

// ------------------------------------------------------------
// Horizontal reduction: 8 lanes down to 1 value
// ------------------------------------------------------------
AVX2_FUNC static inline float horizontalSum(__m256 v) {
    // 8 -> 4: add the upper 128-bit half to the lower half.
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    // 4 -> 2 -> 1 with pairwise adds.
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

AVX2_FUNC float dotAVX2(const float* x, const float* y, size_t n) {
    // Four independent accumulators, not one. An FMA has ~4 cycles
    // of latency but issues every cycle, so a single accumulator
    // would stall on its own dependency chain and reach a quarter
    // of peak. The GPU equivalent is instruction-level parallelism
    // inside a thread - see Phase 3 exercise 06.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = horizontalSum(acc);
    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

// ------------------------------------------------------------
// Compute-bound kernel: this is where SIMD really pays
// ------------------------------------------------------------
AVX2_FUNC void polyAVX2(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 acc = _mm256_set1_ps(1.0f);
        for (int k = 0; k < kPolyDegree; ++k) {
            acc = _mm256_fmadd_ps(acc, x, _mm256_set1_ps(static_cast<float>(k + 1)));
        }
        _mm256_storeu_ps(out + i, acc);
    }
    for (; i < n; ++i) out[i] = polyGolden(in[i]);
}
#endif  // HAS_X86_SIMD

int main() {
    printBanner("Phase 1 / 06 - SIMD intrinsics");

#if !HAS_X86_SIMD
    printf("\n  [SKIP] This exercise uses x86 AVX2 intrinsics; this is not an x86 CPU.\n\n");
    return 0;
#else
    if (!__builtin_cpu_supports("avx2")) {
        printf("\n  [SKIP] This CPU does not support AVX2.\n\n");
        return 0;
    }

    const size_t n = kVectorLength;
    printf("  Vector length: %zu floats (%.0f MB)\n", n,
           n * sizeof(float) / (1024.0 * 1024.0));
    printf("  AVX2 register: 256 bit = 8 floats per instruction\n");

    std::vector<float> x(n), y(n), yRef(n), out(n), outRef(n);
    fillRandom(x.data(), n, -1.0f, 1.0f, 7);
    fillRandom(y.data(), n, -1.0f, 1.0f, 8);

    printSection("Correctness");
    yRef = y;
    saxpyGolden(2.5f, x.data(), yRef.data(), n);
    {
        std::vector<float> yv = y;
        saxpyAVX2(2.5f, x.data(), yv.data(), n);
        checkArray("saxpy AVX2", yv, yRef, 1e-5, 1e-6);
    }

    polyArrayGolden(x.data(), outRef.data(), n);
    polyAVX2(x.data(), out.data(), n);
    checkArray("polynomial AVX2", out, outRef, 1e-3, 1e-3);

    float dRef = dotGolden(x.data(), x.data(), n);
    float dVec = dotAVX2(x.data(), x.data(), n);
    reportCheck("dot AVX2 matches within float tolerance",
                std::fabs(dRef - dVec) < 1e-2f * std::fabs(dRef));

    printSection("Memory-bound kernel: SAXPY");
    {
        std::vector<float> ys = y, yv = y;
        double msScalar = timeCpuMs(10, [&] { saxpyGolden(2.5f, x.data(), ys.data(), n); });
        double msVec = timeCpuMs(10, [&] { saxpyAVX2(2.5f, x.data(), yv.data(), n); });

        double bytes = static_cast<double>(n) * 3 * sizeof(float);  // read x, read y, write y
        ResultTable t;
        t.add("scalar", msScalar, bytes, static_cast<double>(n) * 2);
        t.add("AVX2 (8 lanes)", msVec, bytes, static_cast<double>(n) * 2);
        t.print("SAXPY - 2 FLOP per 12 bytes");
        printf("\n  Expect well under 8x. This kernel is limited by DRAM, and\n");
        printf("  wider registers do not make memory faster. Compare the GB/s\n");
        printf("  column with the sustained bandwidth you measured in exercise 05.\n");
    }

    printSection("Compute-bound kernel: degree-15 polynomial");
    {
        double msScalar = timeCpuMs(5, [&] { polyArrayGolden(x.data(), outRef.data(), n); });
        double msVec = timeCpuMs(5, [&] { polyAVX2(x.data(), out.data(), n); });

        double bytes = static_cast<double>(n) * 2 * sizeof(float);
        ResultTable t;
        t.add("scalar", msScalar, bytes, polyFlops(n));
        t.add("AVX2 (8 lanes)", msVec, bytes, polyFlops(n));
        t.print("Horner polynomial - 30 FLOP per 8 bytes");
        printf("\n  Arithmetic intensity is ~3.75 FLOP/byte here, so the kernel is\n");
        printf("  compute bound and SIMD delivers close to its theoretical 8x.\n");
        printf("  (The scalar version may already be auto-vectorised by -O3, in\n");
        printf("  which case the gap is small - check with -fopt-info-vec.)\n");
    }

    printSection("Dot product: reduction across lanes");
    {
        volatile float sink = 0.0f;
        double msScalar = timeCpuMs(10, [&] { sink = dotGolden(x.data(), x.data(), n); });
        double msVec = timeCpuMs(10, [&] { sink = dotAVX2(x.data(), x.data(), n); });
        (void)sink;

        double bytes = static_cast<double>(n) * 2 * sizeof(float);
        ResultTable t;
        t.add("scalar (double accum)", msScalar, bytes, static_cast<double>(n) * 2);
        t.add("AVX2, 4 accumulators", msVec, bytes, static_cast<double>(n) * 2);
        t.print("Dot product");
        printf("\n  Four independent accumulators, not one: an FMA has ~4 cycles of\n");
        printf("  latency but issues every cycle, so a single accumulator stalls on\n");
        printf("  its own dependency chain and reaches a quarter of peak.\n");
    }

    printf("\n  Carry this to the GPU\n");
    printf("    AVX2 = 8 lanes, one instruction. A CUDA warp = 32 lanes, one\n");
    printf("    instruction. The tail loop here (n %% 8) is the same waste as a\n");
    printf("    partially filled warp, and the horizontal reduction above is the\n");
    printf("    same tree you will write with __shfl_down_sync in Phase 3.\n");
    printf("    The difference: on the GPU you write scalar code and the hardware\n");
    printf("    forms the vectors for you.\n");

    return verifySummary();
#endif
}
