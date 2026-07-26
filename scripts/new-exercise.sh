#!/usr/bin/env bash
# ============================================================
# scripts/new-exercise.sh - scaffold a new exercise
# ============================================================
# Usage:
#   ./scripts/new-exercise.sh 3 07 warp-shuffle
#            ^phase           ^number ^slug
#
# Creates phase-3-*/exercises/07-warp-shuffle/ with the five files
# every exercise needs, pre-filled with the standard skeleton:
#
#   README.md  README.vi.md  main.cu  solution.cu  reference.h
#
# The CMake build picks the folder up automatically - there is no
# list to update.
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 3 ]]; then
    sed -n '2,18p' "$0"
    exit 1
fi

PHASE_NUM="$1"
EX_NUM="$2"
SLUG="$3"
TITLE="${4:-${SLUG//-/ }}"

PHASE_DIR="$(find "${REPO_ROOT}" -maxdepth 1 -type d -name "phase-${PHASE_NUM}-*" | head -1)"
if [[ -z "${PHASE_DIR}" ]]; then
    echo "No phase directory matching phase-${PHASE_NUM}-*" >&2
    exit 1
fi

SUBDIR="exercises"
[[ "${PHASE_NUM}" == "6" ]] && SUBDIR="projects"

EX_DIR="${PHASE_DIR}/${SUBDIR}/${EX_NUM}-${SLUG}"
if [[ -d "${EX_DIR}" ]]; then
    echo "Already exists: ${EX_DIR}" >&2
    exit 1
fi
mkdir -p "${EX_DIR}"

cat > "${EX_DIR}/README.md" <<EOF
<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# ${EX_NUM} - ${TITLE}

> Phase ${PHASE_NUM} | Difficulty: TODO | Estimated time: TODO

## Goal

TODO: one paragraph on what the learner will be able to do afterwards.

## Background

TODO: the concept, and why it matters on a GPU.

## Your task

Open \`main.cu\` and implement the parts marked \`TODO\`.

1. TODO
2. TODO

## Build and run

\`\`\`bash
cmake --build build --target p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_} -j
./build/bin/p${PHASE_NUM}/p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_}
\`\`\`

Reference solution: \`p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_}_sol\`.

## Expected output

TODO

## Key takeaways

- TODO

## Further reading

- TODO
EOF

cat > "${EX_DIR}/README.vi.md" <<EOF
<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# ${EX_NUM} - ${TITLE}

> Giai đoạn ${PHASE_NUM} | Độ khó: TODO | Thời gian ước tính: TODO

## Mục tiêu

TODO

## Kiến thức nền

TODO

## Nhiệm vụ của bạn

Mở \`main.cu\` và hoàn thành các phần đánh dấu \`TODO\`.

1. TODO

## Build và chạy

\`\`\`bash
cmake --build build --target p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_} -j
./build/bin/p${PHASE_NUM}/p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_}
\`\`\`

Lời giải tham chiếu: \`p${PHASE_NUM}_${EX_NUM}_${SLUG//-/_}_sol\`.

## Kết quả mong đợi

TODO

## Điểm cốt lõi

- TODO

## Đọc thêm

- TODO
EOF

cat > "${EX_DIR}/reference.h" <<'EOF'
#pragma once
// ============================================================
// reference.h - CPU reference implementation and problem setup
// ============================================================
// Shared by main.cu (starter) and solution.cu (reference), so the
// two are always measured against exactly the same golden result.
// ============================================================

#include "verify.h"

// TODO: golden implementation goes here.
EOF

cat > "${EX_DIR}/main.cu" <<EOF
// ============================================================
// ${EX_NUM} - ${TITLE}  [STARTER]
// ============================================================
// Implement every part marked TODO, then rebuild and run.
// ============================================================

#include "cuda_helper.h"
#include "reference.h"

int main() {
    printBanner("${TITLE} (starter)");
    requireCudaDevice();

    // TODO: implement the exercise.
    printTodoNotice("implement the kernel in main.cu");

    return verifySummary();
}
EOF

cat > "${EX_DIR}/solution.cu" <<EOF
// ============================================================
// ${EX_NUM} - ${TITLE}  [REFERENCE SOLUTION]
// ============================================================

#include "cuda_helper.h"
#include "reference.h"

int main() {
    printBanner("${TITLE}");
    requireCudaDevice();

    // TODO: reference implementation.

    return verifySummary();
}
EOF

echo "Created ${EX_DIR}"
echo "Files: README.md README.vi.md reference.h main.cu solution.cu"
echo "Reconfigure to pick it up:  cmake -S . -B build"
