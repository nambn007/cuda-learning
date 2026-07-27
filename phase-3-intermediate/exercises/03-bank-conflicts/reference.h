#pragma once
// ============================================================
// reference.h - shared memory banks
// ============================================================
// Shared memory is not one flat block of storage. It is split into
// 32 BANKS, each 4 bytes wide, and consecutive 4-byte words land in
// consecutive banks:
//
//   address / 4 % 32  ->  bank number
//
//   word:  0   1   2  ...  31  32  33 ...
//   bank:  0   1   2  ...  31   0   1 ...
//
// Each bank serves ONE address per cycle. So for a warp's access:
//
//   32 threads, 32 different banks  -> one cycle, full speed
//   32 threads, all the SAME address -> one cycle (broadcast, free)
//   32 threads, k of them hitting different addresses in one bank
//     -> k cycles. This is a k-WAY BANK CONFLICT.
//
// The worst case is 32-way: every thread wants a different address
// in the same bank, and the access serialises completely.
//
// The classic trap is a 2D tile accessed by column:
//
//   __shared__ float tile[32][32];
//   tile[threadIdx.x][i]   // address = tid*32 + i
//                          // bank    = (tid*32 + i) % 32 = i % 32
//                          // -> every thread hits the SAME bank
//
// And the classic fix is one word of padding:
//
//   __shared__ float tile[32][33];
//   tile[threadIdx.x][i]   // address = tid*33 + i
//                          // bank    = (tid + i) % 32
//                          // -> all 32 banks, no conflict
//
// One extra column, 128 wasted bytes per tile, and the access goes
// from 32 cycles to 1. This exact trick appears again in exercise
// 05 (transpose) and exercise 04 (tiled matmul).
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kBanks = 32;
inline constexpr int kTile = 32;          // 32x32 tile
inline constexpr int kSharedWords = 1024; // power of two, so & works as %
inline constexpr int kIterations = 4096;  // enough shared traffic to dominate
inline constexpr int kBlocks = 1024;

// Strides to sweep. The conflict degree is gcd-based: with stride s
// the number of distinct banks touched is 32 / gcd(s, 32).
inline const std::vector<int> kStrides = {1, 2, 4, 8, 16, 32, 33};

inline int gcdInt(int a, int b) {
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Predicted number of cycles a warp's access takes.
inline int predictedWays(int stride) {
    if (stride % kBanks == 0) return kBanks;  // all in one bank
    return gcdInt(stride, kBanks);
}
