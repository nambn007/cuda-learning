// ============================================================
// 03 - Shared memory bank conflicts  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: stride through shared memory
// ------------------------------------------------------------
// Fill s[] cooperatively, barrier, then have thread `tid` start at
// (tid * stride) and read `iters` values, all threads advancing by
// 1 together so the offsets BETWEEN threads stay fixed.
//
//     int idx = (tid * stride) & (kSharedWords - 1);
//     for k in 0..iters:
//         acc += s[idx];
//         idx = (idx + 1) & (kSharedWords - 1);
//
// Write acc to out[] at the end, or the compiler deletes the loop.
// The array size is a power of two so the wrap is a mask, keeping
// integer work out of the measurement.
__global__ void sharedStride(float* out, int stride, int iters) {
    (void)out;
    (void)stride;
    (void)iters;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: a 2D tile, read by row and by column
// ------------------------------------------------------------
// Work out the bank for each BEFORE you write the code:
//
//   tile[i][tid]   address = i*32 + tid   bank = ?
//   tile[tid][i]   address = tid*32 + i   bank = ?
//
// One of them has every thread in the warp hitting the same bank.
__global__ void tileRowAccess(float* out, int iters) {
    (void)out;
    (void)iters;
    // TODO
}

__global__ void tileColumnAccess(float* out, int iters) {
    (void)out;
    (void)iters;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: the fix - one extra column
// ------------------------------------------------------------
//     __shared__ float tile[kTile][kTile + 1];
// Recompute the bank for tile[tid][i] with a row pitch of 33 and
// convince yourself it works before measuring it.
__global__ void tileColumnPadded(float* out, int iters) {
    (void)out;
    (void)iters;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: broadcast
// ------------------------------------------------------------
// Every thread reads the SAME address. Predict the cost first:
// is 32 threads wanting one address a 32-way conflict, or not?
__global__ void sharedBroadcast(float* out, int iters) {
    (void)out;
    (void)iters;
    // TODO
}

int main() {
    printBanner("Phase 3 / 03 - Bank conflicts (starter)");
    requireCudaDevice();

    printf("  %d banks of 4 bytes; bank = (address / 4) %% %d\n", kBanks, kBanks);
    printf("  Launch one warp per block so the effect is not diluted.\n");

    // --------------------------------------------------------
    // TODO 5: allocate the output, verify each kernel reads real
    //         data, then run the three experiments
    // --------------------------------------------------------
    // Experiment 1: sweep kStrides, print measured time next to
    //   predictedWays(stride) from reference.h.
    //   Before running: what do you expect for stride 33?
    //
    // Experiment 2: row vs column vs padded column.
    //   Before running: how much do you expect padding to recover?
    //
    // Experiment 3: broadcast.
    //
    // A warning about the model: the measured slowdown will come out
    // around HALF the predicted way count, because each loop
    // iteration also does an add, a mask and a branch that hide a
    // couple of extra shared-memory cycles. Predict the RATIO from
    // the model; measure the absolute cost.
    printSection("Experiments");
    printf("  TODO\n");

    printTodoNotice("implement the five kernels and the three experiments");
    return verifySummary();
}
