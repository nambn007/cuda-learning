// ============================================================
// 03 - A memory pool allocator  [STARTER]
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ============================================================
// TODO 1: BumpAllocator (also called an arena or linear allocator)
// ============================================================
// The simplest allocator that exists: one big buffer plus an
// offset. allocate() aligns the offset, returns the pointer, and
// moves the offset forward. There is no individual free - you
// throw the whole arena away at once with reset().
//
// That restriction is the point. Allocation becomes three
// instructions with no locking, no free list search and no
// fragmentation. Use it whenever objects share a lifetime, which
// is the common case in a per-frame or per-batch workload.
class BumpAllocator {
public:
    explicit BumpAllocator(size_t capacityBytes) {
        // TODO: allocate `capacityBytes` of raw storage and record
        // the capacity. std::malloc or a std::vector<unsigned char>
        // both work; the vector version is exception safe for free.
        (void)capacityBytes;
    }

    ~BumpAllocator() {
        // TODO: release the storage (nothing to do if you used a vector).
    }

    // Return a pointer to `size` bytes aligned to `alignment`, or
    // nullptr when the arena is exhausted.
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        // TODO:
        //   1. aligned = alignUp(offset_, alignment)
        //   2. if aligned + size > capacity_  -> return nullptr
        //   3. offset_ = aligned + size; return base_ + aligned
        (void)size;
        (void)alignment;
        return nullptr;
    }

    // Free everything at once. O(1), and it does not touch memory.
    void reset() {
        // TODO: rewind the offset to 0.
    }

    size_t used() const { return offset_; }

private:
    unsigned char* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

// ============================================================
// TODO 2: PoolAllocator (fixed-size blocks with a free list)
// ============================================================
// A bump allocator cannot free one object. A pool can, as long as
// every object has the same size.
//
// The trick: a free block holds no user data, so its first bytes
// are free real estate. Store the "next free block" pointer THERE.
// The free list therefore costs no extra memory at all.
//
//   allocate():   pop the head of the free list
//   deallocate(): push the block back onto the head
//
// Both are O(1) and both are just a couple of pointer writes.
class PoolAllocator {
public:
    PoolAllocator(size_t blockSize, size_t blockCount) {
        // A block must be large enough to hold the free-list
        // pointer we write into it while it is free.
        blockSize_ = alignUp(blockSize < sizeof(void*) ? sizeof(void*) : blockSize,
                             alignof(std::max_align_t));
        blockCount_ = blockCount;
        storage_.resize(blockSize_ * blockCount_);

        // TODO: thread every block onto the free list. Walk the
        // blocks and, for each one, write the address of the NEXT
        // block into its first sizeof(void*) bytes. The last block
        // stores nullptr. Then point freeList_ at block 0.
    }

    void* allocate() {
        // TODO: if freeList_ is null return nullptr; otherwise take
        // the head, read the "next" pointer stored inside it, make
        // that the new head, and return the old head.
        return nullptr;
    }

    void deallocate(void* p) {
        // TODO: write the current freeList_ into the first bytes of
        // p, then set freeList_ = p.
        (void)p;
    }

private:
    std::vector<unsigned char> storage_;
    void* freeList_ = nullptr;
    size_t blockSize_ = 0;
    size_t blockCount_ = 0;
};

// ------------------------------------------------------------
// Workload: build a linked list of nodes and walk it
// ------------------------------------------------------------
static long long buildAndWalkWithNew(size_t count) {
    std::vector<Node*> nodes(count);
    for (size_t i = 0; i < count; ++i) {
        nodes[i] = new Node{static_cast<int>(i % 1000), 0.0f, nullptr, nullptr};
    }
    long long sum = 0;
    for (size_t i = 0; i < count; ++i) sum += nodes[i]->key;
    for (size_t i = 0; i < count; ++i) delete nodes[i];
    return sum;
}

int main() {
    printBanner("Phase 1 / 03 - Memory pool allocator (starter)");

    printSection("Correctness");
    const long long expected = expectedChecksum(kNodeCount);

    // Baseline works out of the box.
    reportCheck("new/delete checksum", buildAndWalkWithNew(kNodeCount) == expected);

    // TODO 3: build the same list from a BumpAllocator and from a
    // PoolAllocator, verify the checksum matches, then benchmark
    // all three with timeCpuMs() and fill in a ResultTable.
    BumpAllocator bump(kNodeCount * sizeof(Node) + 1024);
    if (bump.allocate(sizeof(Node)) == nullptr) {
        printTodoNotice("implement BumpAllocator and PoolAllocator in main.cpp");
        return verifySummary();
    }

    return verifySummary();
}
