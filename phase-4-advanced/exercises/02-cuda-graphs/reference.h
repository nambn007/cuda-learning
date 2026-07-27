#pragma once
// ============================================================
// reference.h - CUDA graphs
// ============================================================
// Every kernel launch costs the CPU a few microseconds of driver
// work: validating arguments, writing a command packet, ringing the
// doorbell. That is invisible next to a 2 ms kernel and dominant
// next to a 5 us one.
//
// Deep learning inference, iterative solvers and physics steps all
// have the same shape - dozens of small kernels, launched in the
// same order, thousands of times. The launch overhead becomes the
// bottleneck and the GPU sits idle between kernels waiting for the
// CPU to catch up.
//
// A CUDA graph records that whole sequence once, as a DAG of nodes
// with their dependencies, and replays it with a SINGLE launch. The
// driver validates the work once instead of every time.
//
// Two ways to build one:
//
//   STREAM CAPTURE  put a stream into capture mode, issue the work
//                   exactly as you normally would, and end capture.
//                   Almost no code changes; the usual choice.
//
//   EXPLICIT API    cudaGraphAddKernelNode etc. More work, but you
//                   control the dependency structure directly.
//
// The catch: a graph captures POINTERS AND VALUES, not intent. If
// the next iteration uses a different buffer, you must either
// rebuild the graph or update the node parameters with
// cudaGraphExecKernelNodeSetParams. Capturing once and replaying
// with stale pointers is the classic graph bug.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// Small enough that per-launch overhead is a large fraction of the
// total - which is exactly the regime graphs are for.
inline constexpr size_t kElements = 1u << 16;  // 64K floats
inline constexpr int kKernelsPerIteration = 20;
inline constexpr int kIterations = 200;

inline void chainCPU(float* v, size_t n, int steps) {
    for (int s = 0; s < steps; ++s)
        for (size_t i = 0; i < n; ++i) v[i] = v[i] * 1.0001f + 0.5f;
}
