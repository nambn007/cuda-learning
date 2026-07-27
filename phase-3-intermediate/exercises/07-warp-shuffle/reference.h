#pragma once
// ============================================================
// reference.h - warp-level primitives
// ============================================================
// A warp's 32 threads share an instruction stream, and the hardware
// lets them exchange register values directly - no shared memory,
// no __syncthreads(), no memory traffic at all.
//
//   __shfl_sync(mask, v, srcLane)     read lane srcLane's v
//   __shfl_up_sync(mask, v, delta)    read lane (me - delta)
//   __shfl_down_sync(mask, v, delta)  read lane (me + delta)
//   __shfl_xor_sync(mask, v, mask2)   read lane (me XOR mask2)
//
//   __ballot_sync(mask, pred)   32-bit word, one bit per lane
//   __all_sync / __any_sync     is the predicate true for all/any
//   __activemask()              which lanes are currently active
//
// The `_sync` suffix and the mask are not decoration. Before Volta,
// a warp always moved in lockstep and the older intrinsics had no
// mask. Since Volta threads diverge independently, so you must
// state which lanes are expected to participate - normally
// 0xffffffff, meaning all of them. Passing a mask that does not
// match reality is undefined behaviour.
//
// Three things these buy you:
//
//   REDUCTION   log2(32) = 5 shuffles, versus a shared-memory tree
//               with barriers (exercise 06).
//   SCAN        __shfl_up_sync gives an inclusive scan in 5 steps.
//   AGGREGATION __ballot_sync + __popc lets ONE lane perform an
//               atomic on behalf of the whole warp, which turns 32
//               contended atomics into 1.
//
// That last one comes with a twist that is the real content of this
// exercise: measure it and you will find it buys nothing, because
// nvcc has performed the same transformation automatically since
// CUDA 9 - on Ampere using the hardware REDUX.SUM instruction. The
// exercise asks you to prove that from the disassembly rather than
// take it on faith, and the transferable habit is: READ THE SASS
// BEFORE HAND-OPTIMISING.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 16u * 1024u * 1024u;
inline constexpr int kBlockSize = 256;

// Predicate used by the filtering benchmark: roughly 1 element in 8
// passes, which is a realistic sparsity for a filter kernel.
inline bool keepPredicate(float v) { return v > 0.75f; }

inline size_t countKeptCPU(const float* v, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i)
        if (keepPredicate(v[i])) ++c;
    return c;
}

// Inclusive prefix sum within each group of 32, for checking the
// warp-scan kernel.
inline void warpScanCPU(const float* in, float* out, size_t n) {
    for (size_t base = 0; base < n; base += 32) {
        float acc = 0.0f;
        const size_t count = (base + 32 <= n) ? 32 : (n - base);
        for (size_t i = 0; i < count; ++i) {
            acc += in[base + i];
            out[base + i] = acc;
        }
    }
}
