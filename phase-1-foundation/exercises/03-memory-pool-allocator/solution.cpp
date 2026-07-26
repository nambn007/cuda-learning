// ============================================================
// 03 - A memory pool allocator  [SOLUTION]
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>  // placement new
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ============================================================
// BumpAllocator - an arena
// ============================================================
class BumpAllocator {
public:
    explicit BumpAllocator(size_t capacityBytes) : storage_(capacityBytes) {
        base_ = storage_.data();
        capacity_ = capacityBytes;
    }

    // Not copyable: two arenas sharing one buffer would hand out
    // the same addresses twice.
    BumpAllocator(const BumpAllocator&) = delete;
    BumpAllocator& operator=(const BumpAllocator&) = delete;

    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        size_t aligned = alignUp(offset_, alignment);
        if (aligned + size > capacity_) return nullptr;  // arena exhausted
        void* p = base_ + aligned;
        offset_ = aligned + size;
        return p;
    }

    // Freeing the entire arena is a single store. Note what it does
    // NOT do: it never touches the objects, so anything with a
    // non-trivial destructor must be destroyed explicitly first.
    void reset() { offset_ = 0; }

    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }

private:
    std::vector<unsigned char> storage_;
    unsigned char* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

// ============================================================
// PoolAllocator - fixed-size blocks, intrusive free list
// ============================================================
class PoolAllocator {
public:
    PoolAllocator(size_t blockSize, size_t blockCount) {
        // Each free block must be able to store one pointer.
        blockSize_ = alignUp(blockSize < sizeof(void*) ? sizeof(void*) : blockSize,
                             alignof(std::max_align_t));
        blockCount_ = blockCount;
        storage_.resize(blockSize_ * blockCount_);
        buildFreeList();
    }

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate() {
        if (!freeList_) return nullptr;  // pool exhausted
        void* head = freeList_;
        // The "next" pointer lives inside the free block itself,
        // so the free list costs zero extra memory.
        std::memcpy(&freeList_, head, sizeof(void*));
        return head;
    }

    void deallocate(void* p) {
        if (!p) return;
        std::memcpy(p, &freeList_, sizeof(void*));
        freeList_ = p;
    }

    // Rebuild the list: frees everything without touching objects.
    void reset() { buildFreeList(); }

    size_t blockSize() const { return blockSize_; }

private:
    void buildFreeList() {
        unsigned char* base = storage_.data();
        for (size_t i = 0; i + 1 < blockCount_; ++i) {
            void* next = base + (i + 1) * blockSize_;
            std::memcpy(base + i * blockSize_, &next, sizeof(void*));
        }
        void* null = nullptr;
        std::memcpy(base + (blockCount_ - 1) * blockSize_, &null, sizeof(void*));
        freeList_ = base;
    }

    std::vector<unsigned char> storage_;
    void* freeList_ = nullptr;
    size_t blockSize_ = 0;
    size_t blockCount_ = 0;
};

// ------------------------------------------------------------
// The same workload, three different allocators
// ------------------------------------------------------------
static long long runWithNew(size_t count) {
    std::vector<Node*> nodes(count);
    for (size_t i = 0; i < count; ++i)
        nodes[i] = new Node{static_cast<int>(i % 1000), 0.0f, nullptr, nullptr};
    long long sum = 0;
    for (size_t i = 0; i < count; ++i) sum += nodes[i]->key;
    for (size_t i = 0; i < count; ++i) delete nodes[i];
    return sum;
}

static long long runWithBump(BumpAllocator& arena, size_t count) {
    arena.reset();
    std::vector<Node*> nodes(count);
    for (size_t i = 0; i < count; ++i) {
        void* raw = arena.allocate(sizeof(Node), alignof(Node));
        // Placement new: construct the object in storage we own.
        nodes[i] = new (raw) Node{static_cast<int>(i % 1000), 0.0f, nullptr, nullptr};
    }
    long long sum = 0;
    for (size_t i = 0; i < count; ++i) sum += nodes[i]->key;
    // Node is trivially destructible, so a single reset() is all
    // the "free" this workload needs.
    arena.reset();
    return sum;
}

static long long runWithPool(PoolAllocator& pool, size_t count) {
    pool.reset();
    std::vector<Node*> nodes(count);
    for (size_t i = 0; i < count; ++i) {
        void* raw = pool.allocate();
        nodes[i] = new (raw) Node{static_cast<int>(i % 1000), 0.0f, nullptr, nullptr};
    }
    long long sum = 0;
    for (size_t i = 0; i < count; ++i) sum += nodes[i]->key;
    // Unlike the arena, a pool can free individual blocks.
    for (size_t i = 0; i < count; ++i) pool.deallocate(nodes[i]);
    return sum;
}

int main() {
    printBanner("Phase 1 / 03 - Memory pool allocator");
    printf("  Workload: allocate %zu nodes of %zu bytes, walk them, free them\n",
           kNodeCount, sizeof(Node));

    BumpAllocator arena(kNodeCount * sizeof(Node) + 4096);
    PoolAllocator pool(sizeof(Node), kNodeCount);

    printSection("Correctness");
    const long long expected = expectedChecksum(kNodeCount);
    reportCheck("new/delete checksum", runWithNew(kNodeCount) == expected);
    reportCheck("bump allocator checksum", runWithBump(arena, kNodeCount) == expected);
    reportCheck("pool allocator checksum", runWithPool(pool, kNodeCount) == expected);

    // A pool really does reuse blocks: free one, allocate one, and
    // you get the same address back.
    void* a = pool.allocate();
    pool.deallocate(a);
    void* b = pool.allocate();
    reportCheck("pool reuses freed blocks", a == b);
    pool.deallocate(b);

    // An exhausted arena must report failure rather than corrupt memory.
    BumpAllocator tiny(64);
    reportCheck("arena reports exhaustion", tiny.allocate(1024) == nullptr);

    printSection("Performance");
    ResultTable table;
    table.add("new / delete", timeCpuMs(5, [&] { runWithNew(kNodeCount); }));
    table.add("bump allocator", timeCpuMs(5, [&] { runWithBump(arena, kNodeCount); }));
    table.add("pool allocator", timeCpuMs(5, [&] { runWithPool(pool, kNodeCount); }));
    table.print("Allocation strategies");

    printf("\n  Why this matters for CUDA\n");
    printf("    cudaMalloc costs roughly 100-500 microseconds because it enters\n");
    printf("    the driver and synchronises the device. A kernel that runs in 50\n");
    printf("    microseconds is then dominated by its own allocation.\n");
    printf("    The fix is the same as here: reserve one large block up front and\n");
    printf("    sub-allocate. CUDA 11.2+ ships this as cudaMallocAsync backed by\n");
    printf("    a cudaMemPool_t; PyTorch and TensorFlow each ship their own.\n");

    return verifySummary();
}
