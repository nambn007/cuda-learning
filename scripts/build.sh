#!/usr/bin/env bash
# ============================================================
# scripts/build.sh - configure and build the whole curriculum
# ============================================================
# Usage:
#   ./scripts/build.sh                  # Release, auto-detected GPU
#   ./scripts/build.sh --debug          # -G, for cuda-gdb
#   ./scripts/build.sh --arch 86        # target one architecture
#   ./scripts/build.sh --arch portable  # fat binary, sm_70..sm_89
#   ./scripts/build.sh --solutions      # reference solutions only
#   ./scripts/build.sh --clean          # wipe build/ first
#   ./scripts/build.sh --optional       # include externally-dependent exercises
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

BUILD_TYPE="Release"
CUDA_ARCH="native"
TARGET="all"
CLEAN=0
OPTIONAL="OFF"
VERBOSE_PTXAS="OFF"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      BUILD_TYPE="Debug"; shift ;;
        --release)    BUILD_TYPE="Release"; shift ;;
        --arch)       CUDA_ARCH="$2"; shift 2 ;;
        --solutions)  TARGET="solutions"; shift ;;
        --starters)   TARGET="starters"; shift ;;
        --clean)      CLEAN=1; shift ;;
        --optional)   OPTIONAL="ON"; shift ;;
        --ptxas)      VERBOSE_PTXAS="ON"; shift ;;
        -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if ! command -v nvcc >/dev/null 2>&1; then
    cat >&2 <<'EOF'
nvcc was not found on this machine.

The supported way to build this repository is the Docker image:

    docker compose up -d --build
    docker compose exec cuda-dev bash
    ./scripts/build.sh

See docs/SETUP.md for a native (non-Docker) install instead.
EOF
    exit 1
fi

if [[ ${CLEAN} -eq 1 ]]; then
    echo ">> Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

echo ">> Configuring (${BUILD_TYPE}, CUDA_ARCH=${CUDA_ARCH})"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCUDA_ARCH="${CUDA_ARCH}" \
      -DCL_ENABLE_OPTIONAL="${OPTIONAL}" \
      -DCL_PTXAS_VERBOSE="${VERBOSE_PTXAS}" \
      ${NINJA_GENERATOR:+-G Ninja}

echo ">> Building target '${TARGET}'"
cmake --build "${BUILD_DIR}" --target "${TARGET}" -j "$(nproc)"

echo
echo ">> Done. Binaries are in ${BUILD_DIR}/bin/<phase>/"
echo ">> Run the full suite with: ./scripts/run-all.sh"
