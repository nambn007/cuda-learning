#pragma once
// ============================================================
// reference.h - allocation workload shared by both variants
// ============================================================
// Why an allocator exercise in a CUDA curriculum?
//
// Because cudaMalloc is roughly a THOUSAND times more expensive
// than malloc. It synchronises the device, talks to the driver and
// may reorganise the GPU's virtual address space. A kernel that
// runs in 50 microseconds preceded by a 200 microsecond cudaMalloc
// is dominated by the allocation.
//
// The standard fix - on the CPU and on the GPU - is to allocate a
// big slab once and hand out pieces of it yourself. That is what
// cudaMallocAsync and its memory pools do since CUDA 11.2, what
// PyTorch's caching allocator does, and what you are about to
// write by hand.
// ============================================================

#include <cstddef>
#include <cstdint>
#include <vector>

#include "verify.h"

// A small node, the kind of object a tree or graph algorithm
// allocates by the million.
struct Node {
    int key;
    float value;
    Node* left;
    Node* right;
};

// Round `n` up to the next multiple of `alignment`.
// Alignment matters: an 8-byte double loaded from an address that
// is not a multiple of 8 is slower on x86 and a hard fault on some
// architectures. The bit trick works because alignment is always a
// power of two.
inline size_t alignUp(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

// The workload: allocate `count` nodes, link them into a list,
// walk the list to make sure the memory is really usable, then
// release everything.
inline constexpr size_t kNodeCount = 200000;
inline constexpr size_t kRounds = 20;

// Sum used as the correctness check - identical no matter which
// allocator produced the storage.
inline long long expectedChecksum(size_t count) {
    long long sum = 0;
    for (size_t i = 0; i < count; ++i) sum += static_cast<long long>(i % 1000);
    return sum;
}
