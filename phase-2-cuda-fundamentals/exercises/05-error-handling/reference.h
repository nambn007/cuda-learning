#pragma once
// ============================================================
// reference.h - two kinds of CUDA error
// ============================================================
// CUDA errors fall into two categories, and confusing them is the
// reason GPU bugs feel so much worse than CPU bugs.
//
// RECOVERABLE errors happen before anything ran: an invalid launch
// configuration, an out-of-memory allocation, a bad argument.
// cudaGetLastError() returns them, clears the flag, and the context
// carries on working normally.
//
// STICKY errors happen while a kernel is executing: an illegal
// address, a device-side assert. They destroy the CUDA context.
// EVERY subsequent call - including cudaMalloc, cudaMemcpy and even
// cudaDeviceReset in older versions - returns the same error
// forever. The only cure is to exit the process.
//
// That is why the guidance "check every call" is not pedantry. An
// unchecked sticky error turns into a cascade of nonsensical
// failures thousands of lines later, and the stack trace points
// nowhere near the kernel that actually broke.
//
// The second trap is ASYNCHRONY. A launch returns immediately, so
// the error it eventually produces surfaces at the next
// synchronisation point - which may be a completely unrelated
// cudaMemcpy. Under CUDA_LAUNCH_BLOCKING=1 the runtime waits after
// every launch, so errors are reported where they happen. It is
// slow, and it is the first thing to try when a failure makes no
// sense.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kSafeElements = 1024;
