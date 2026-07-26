#!/usr/bin/env bash
# ============================================================
# scripts/run-all.sh - run every reference solution as a test
# ============================================================
# Each solution verifies its own result against a CPU reference
# and exits non-zero on mismatch, so ctest doubles as a regression
# suite for the whole curriculum.
#
# Usage:
#   ./scripts/run-all.sh              # everything
#   ./scripts/run-all.sh p3           # one phase (p1..p6)
#   ./scripts/run-all.sh -R reduction # anything matching a regex
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "No build directory. Run ./scripts/build.sh first." >&2
    exit 1
fi

CTEST_ARGS=(--test-dir "${BUILD_DIR}" --output-on-failure)

if [[ $# -gt 0 ]]; then
    if [[ "$1" =~ ^p[1-6]$ ]]; then
        CTEST_ARGS+=(-L "$1")
        shift
    fi
    CTEST_ARGS+=("$@")
fi

ctest "${CTEST_ARGS[@]}"
