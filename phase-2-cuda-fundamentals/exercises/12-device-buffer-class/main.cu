// ============================================================
// 12 - A RAII device buffer  [STARTER]
// ============================================================
// Phase 1 exercise 04 built Matrix<T> around new[]/delete[]. Build
// the same thing around cudaMalloc/cudaFree - where the stakes are
// higher, because a leaked device allocation is not reclaimed until
// the CUDA context is destroyed.
// ============================================================

#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ============================================================
// TODO 1: DeviceBuffer<T>
// ============================================================
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // TODO: cudaMalloc(&ptr_, count * sizeof(T)) when count > 0.
    explicit DeviceBuffer(size_t count) : count_(count) {
        (void)count;
        // TODO
    }

    // TODO: cudaFree(ptr_) if it is non-null.
    //
    // Do NOT use CUDA_CHECK here. CUDA_CHECK calls exit(), and
    // during process teardown the context may already be gone, in
    // which case cudaFree legitimately returns an error. Exiting
    // from a destructor is worse than the leak it would report.
    ~DeviceBuffer() {
        // TODO
    }

    // TODO 2: delete the copy operations.
    //
    // Why delete rather than implement? An implicit device-to-device
    // copy of a 16 MB buffer is exactly the kind of expensive
    // operation that should never happen by accident. Provide an
    // explicit clone() instead, so the cost is visible at the call
    // site.

    // TODO 3: move constructor and move assignment, both noexcept.
    //
    // noexcept is load-bearing, not decoration: without it,
    // std::vector copies instead of moving when it reallocates -
    // and since the copy is deleted, code that puts these in a
    // vector will not even compile.

    // TODO 4: clone(), copyFromHost(), copyToHost(), zero()
    //
    // For the copy helpers, throw std::out_of_range if the caller
    // asks for more elements than the buffer holds. Silent
    // truncation on a GPU is how you get corruption three kernels
    // later.

    // TODO 5: accessors.
    //
    // An implicit `operator T*()` lets a buffer be passed straight
    // to a kernel:  myKernel<<<g, b>>>(buf, n);
    // Some codebases consider that too implicit and require .get().
    // Pick one and be consistent.

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    bool empty() const { return ptr_ == nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// ============================================================
// TODO 6: PinnedBuffer<T> around cudaMallocHost / cudaFreeHost
// ============================================================
// Page-locked host memory cannot be swapped out, so the DMA engine
// reads it directly instead of going via a staging buffer. Expect
// transfers roughly 1.5-2x faster - and it is what makes
// cudaMemcpyAsync genuinely asynchronous (Phase 4 exercise 01).
//
// It is a scarce, machine-wide resource: pinning too much starves
// the operating system. RAII matters doubly here.

__global__ void scaleKernel(float a, const float* in, float* out, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) out[i] = a * in[i];
}

// Free device memory in bytes - use this to PROVE nothing leaks
// rather than assuming it.
static size_t freeDeviceMemory() {
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    return freeBytes;
}

// The bug the class exists to prevent. Every early return in this
// shape leaks, and there is no way to make it safe except by not
// writing it.
static int rawStyleWithEarlyReturn(size_t n, bool failEarly) {
    float* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return -1;
    if (failEarly) return -2;  // LEAK
    cudaFree(d);
    return 0;
}

int main() {
    printBanner("Phase 2 / 12 - A RAII device buffer (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    std::vector<float> host(n), golden(n);
    fillRandom(host.data(), n, -1.0f, 1.0f, 71);
    scaleCPU(3.0f, host.data(), golden.data(), n);

    // --------------------------------------------------------
    // TODO 7: prove the class works and that nothing leaks
    // --------------------------------------------------------
    // Suggested experiments, in order:
    //   a) allocate, copy in, run scaleKernel, copy out, verify
    //   b) call rawStyleWithEarlyReturn(n, true) twenty times and
    //      measure freeDeviceMemory() before and after. How much
    //      leaked?
    //   c) do the same with a DeviceBuffer that goes out of scope
    //      because of a THROWN EXCEPTION. How much leaks now?
    //   d) move a buffer and check the source is empty afterwards
    //   e) put four buffers in a std::vector - this is the test
    //      that fails to compile if your move is not noexcept
    //   f) compare pageable and pinned host-to-device bandwidth
    printSection("Basic use");
    printf("  TODO\n");

    printTodoNotice("implement DeviceBuffer<T>, PinnedBuffer<T> and the experiments");
    return verifySummary();
}
