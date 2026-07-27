// ============================================================
// 12 - A RAII device buffer  [SOLUTION]
// ============================================================

#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ============================================================
// DeviceBuffer<T> - owns a cudaMalloc allocation
// ============================================================
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
        }
    }

    ~DeviceBuffer() {
        // Never CUDA_CHECK in a destructor: it calls exit(), and
        // during process teardown the context may already be gone,
        // in which case cudaFree legitimately returns an error.
        if (ptr_) cudaFree(ptr_);
    }

    // Copying is DELETED on purpose. An implicit device-to-device
    // copy of a large buffer is exactly the kind of expensive
    // operation that should never happen by accident. If you want
    // one, say so: clone().
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // noexcept is load-bearing: without it std::vector would try to
    // copy on reallocation, and the copy is deleted, so the code
    // would not compile at all.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // An explicit deep copy, for when you really do want one.
    DeviceBuffer clone() const {
        DeviceBuffer copy(count_);
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(copy.ptr_, ptr_, bytes(), cudaMemcpyDeviceToDevice));
        }
        return copy;
    }

    void copyFromHost(const T* src, size_t count) {
        if (count > count_) throw std::out_of_range("DeviceBuffer::copyFromHost overflow");
        CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copyFromHost(const std::vector<T>& src) { copyFromHost(src.data(), src.size()); }

    void copyToHost(T* dst, size_t count) const {
        if (count > count_) throw std::out_of_range("DeviceBuffer::copyToHost overflow");
        CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copyToHost(std::vector<T>& dst) const { copyToHost(dst.data(), dst.size()); }

    void zero() {
        if (ptr_) CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
    }

    // Implicit conversion to T* so the buffer can be passed
    // straight to a kernel: myKernel<<<g,b>>>(buf, n).
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }

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
// PinnedBuffer<T> - page-locked host memory
// ============================================================
// The same pattern for cudaMallocHost. Pinned memory cannot be
// swapped out, so the DMA engine can read it directly and transfers
// run about twice as fast - and, crucially, can be asynchronous
// (Phase 4 exercise 01).
//
// It is a scarce resource: pinning too much starves the operating
// system. RAII matters here for a second reason - not just leaking
// but leaking something the whole machine shares.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMallocHost(&ptr_, count_ * sizeof(T)));
    }
    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }

    T* get() { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

__global__ void scaleKernel(float a, const float* in, float* out, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) out[i] = a * in[i];
}

// Free device memory, in bytes. Used to prove nothing leaks.
static size_t freeDeviceMemory() {
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    return freeBytes;
}

// ------------------------------------------------------------
// The bug the class exists to prevent
// ------------------------------------------------------------
// Every early return in this shape is a leak. There is no way to
// make it safe except by not writing it.
static int rawStyleWithEarlyReturn(size_t n, bool failEarly) {
    float* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return -1;

    if (failEarly) {
        // Somebody adds a validation check six months later and
        // does not notice the cudaFree below.
        return -2;  // LEAK
    }

    cudaFree(d);
    return 0;
}

int main() {
    printBanner("Phase 2 / 12 - A RAII device buffer");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const size_t n = kElements;
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;

    std::vector<float> host(n), golden(n), result(n);
    fillRandom(host.data(), n, -1.0f, 1.0f, 71);
    scaleCPU(3.0f, host.data(), golden.data(), n);

    // ========================================================
    // 1. It works, and it is shorter
    // ========================================================
    printSection("Basic use");
    {
        DeviceBuffer<float> in(n), out(n);
        in.copyFromHost(host);
        // The implicit conversion to float* lets a buffer be passed
        // straight to a kernel.
        scaleKernel<<<grid, block>>>(3.0f, in, out, n);
        CUDA_CHECK_LAST();
        out.copyToHost(result);
        checkArray("scale by 3", result, golden, 1e-6, 1e-6);
    }
    printf("  Four lines, no cudaMalloc, no cudaFree, no way to forget either.\n");

    // ========================================================
    // 2. Nothing leaks, even on an early return
    // ========================================================
    printSection("Leaks");

    const size_t before = freeDeviceMemory();
    for (int i = 0; i < 20; ++i) {
        rawStyleWithEarlyReturn(n, /*failEarly=*/true);
    }
    const size_t afterRaw = freeDeviceMemory();
    const long long leaked = static_cast<long long>(before) - static_cast<long long>(afterRaw);
    printf("  20 early returns from the raw C-style version leaked %.1f MB\n",
           leaked / (1024.0 * 1024.0));
    reportCheck("the raw version really does leak", leaked > 0);

    // The same thing with RAII. Even a thrown exception unwinds
    // through the destructor.
    const size_t beforeRaii = freeDeviceMemory();
    for (int i = 0; i < 20; ++i) {
        try {
            DeviceBuffer<float> buf(n);
            throw std::runtime_error("simulated failure");
        } catch (const std::exception&) {
            // swallowed on purpose
        }
    }
    const size_t afterRaii = freeDeviceMemory();
    const long long leakedRaii =
        static_cast<long long>(beforeRaii) - static_cast<long long>(afterRaii);
    printf("  20 thrown exceptions through the RAII version leaked %.1f MB\n",
           leakedRaii / (1024.0 * 1024.0));
    reportCheck("RAII leaks nothing, even when unwinding",
                leakedRaii < static_cast<long long>(4 * 1024 * 1024));

    // ========================================================
    // 3. Move, and why copy is deleted
    // ========================================================
    printSection("Move semantics");
    {
        DeviceBuffer<float> a(n);
        a.copyFromHost(host);
        const float* original = a.get();

        DeviceBuffer<float> b = std::move(a);
        reportCheck("move transfers the pointer", b.get() == original);
        reportCheck("moved-from buffer is empty", a.empty() && a.size() == 0);

        // b == a would not compile: the copy constructor is deleted.
        // An accidental 16 MB device-to-device copy is exactly the
        // kind of thing that should require you to ask for it.
        DeviceBuffer<float> c = b.clone();
        reportCheck("clone allocates separate storage", c.get() != b.get());
        c.copyToHost(result);
        checkArray("clone copied the contents", result, host, 1e-6, 1e-6);
    }

    // ========================================================
    // 4. It works inside containers
    // ========================================================
    // This is what the noexcept on the move constructor buys. Take
    // it away and std::vector tries to copy on reallocation, the
    // copy is deleted, and this block stops compiling.
    printSection("Use in containers");
    {
        std::vector<DeviceBuffer<float>> buffers;
        for (int i = 0; i < 4; ++i) {
            DeviceBuffer<float> buf(n / 4);
            buf.copyFromHost(host.data(), n / 4);
            buffers.push_back(std::move(buf));
        }
        reportCheck("vector holds four live buffers", buffers.size() == 4);
        reportCheck("each buffer is distinct",
                    buffers[0].get() != buffers[1].get() &&
                        buffers[1].get() != buffers[2].get());
    }
    reportCheck("all container buffers released",
                freeDeviceMemory() >= afterRaii - 4 * 1024 * 1024);

    // ========================================================
    // 5. Pinned host memory - the same pattern, a scarcer resource
    // ========================================================
    printSection("Pinned host memory");
    {
        const size_t bytes = n * sizeof(float);
        PinnedBuffer<float> pinned(n);
        for (size_t i = 0; i < n; ++i) pinned[i] = host[i];

        DeviceBuffer<float> dev(n);

        double msPageable = timeCpuMs(10, [&] {
            CUDA_CHECK(cudaMemcpy(dev.get(), host.data(), bytes, cudaMemcpyHostToDevice));
        });
        double msPinned = timeCpuMs(10, [&] {
            CUDA_CHECK(cudaMemcpy(dev.get(), pinned.get(), bytes, cudaMemcpyHostToDevice));
        });

        ResultTable t;
        t.add("pageable host memory", msPageable, static_cast<double>(bytes));
        t.add("pinned host memory", msPinned, static_cast<double>(bytes));
        t.print("Host to device transfer");

        printf("\n  Pinned memory cannot be swapped out, so the DMA engine reads it\n");
        printf("  directly instead of going through a staging buffer. That is worth\n");
        printf("  about %.2fx here, and it is also what makes cudaMemcpyAsync\n",
               msPageable / msPinned);
        printf("  actually asynchronous - Phase 4 exercise 01.\n");
        printf("\n  It is a scarce resource: pinning too much starves the operating\n");
        printf("  system. RAII matters doubly here, because what you would leak is\n");
        printf("  something the whole machine shares.\n");

        reportCheck("pinned transfer is at least as fast", msPinned <= msPageable * 1.05);
    }

    printSection("Design notes");
    printf("  Copy is DELETED, not implemented.\n");
    printf("    An implicit 16 MB device-to-device copy is exactly the kind of\n");
    printf("    expensive operation that should never happen by accident. clone()\n");
    printf("    makes the cost visible at the call site.\n");
    printf("\n  Move is NOEXCEPT.\n");
    printf("    Without it, std::vector copies instead of moving when it grows -\n");
    printf("    and since the copy is deleted, the container block above would not\n");
    printf("    even compile. noexcept is what makes the class usable.\n");
    printf("\n  The destructor does NOT use CUDA_CHECK.\n");
    printf("    CUDA_CHECK calls exit(). During process teardown the context may\n");
    printf("    already be gone, and cudaFree then legitimately returns an error.\n");
    printf("    Throwing or exiting from a destructor is worse than the leak it\n");
    printf("    would be reporting.\n");

    return verifySummary();
}
