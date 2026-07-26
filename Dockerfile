# ============================================================
# CUDA Learning - development environment
# ============================================================
# Base image ships nvcc, cuda-gdb, compute-sanitizer, ncu,
# cuobjdump and nvdisasm. We add the host toolchain, Nsight
# Systems and a few libraries used by the later phases.
#
# Build:
#   docker compose build
#   docker build -t cuda-learning-dev .
#
# The CUDA toolkit version only needs to be <= the version your
# driver supports (nvidia-smi prints it top right). 12.6 works on
# any driver from 560 upwards, including the 13.x series.
# ============================================================

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cuda-dev

LABEL org.opencontainers.image.title="CUDA Learning"
LABEL org.opencontainers.image.description="Development environment for the CUDA Learning curriculum"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# ============================================================
# System packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build toolchain
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    ccache \
    # Debugging and profiling on the host side
    gdb \
    valgrind \
    linux-tools-generic \
    strace \
    # Version control
    git \
    git-lfs \
    # Utilities
    wget \
    curl \
    unzip \
    vim \
    nano \
    less \
    htop \
    tree \
    bash-completion \
    # Python: plotting roofline charts, PyTorch extensions, Numba
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # OpenMP for the Phase 1 CPU baselines
    libomp-dev \
    # Image tooling: converts the PPM files the exercises write
    imagemagick \
    # OpenGL for the Phase 5 interop exercise
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglfw3-dev \
    libglew-dev \
    # Misc
    ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ============================================================
# Nsight Systems (timeline profiler)
# ============================================================
# The devel image already has Nsight Compute (`ncu`) for per-kernel
# analysis, but not Nsight Systems (`nsys`) for whole-application
# timelines. Phase 3 and Phase 4 need both. Set the build arg to 0
# to skip the ~500 MB download.
ARG INSTALL_NSIGHT_SYSTEMS=1
ARG CUDA_VERSION
RUN if [ "${INSTALL_NSIGHT_SYSTEMS}" = "1" ]; then \
        # The package name tracks the toolkit version, e.g.
        # cuda-nsight-systems-12-6. Fall back to the standalone
        # package name, and never fail the build if neither exists:
        # every exercise still works without nsys, it just cannot
        # produce a timeline.
        CU="$(echo "${CUDA_VERSION}" | cut -d. -f1-2 | tr '.' '-')" && \
        apt-get update && \
        { apt-get install -y --no-install-recommends "cuda-nsight-systems-${CU}" \
          || apt-get install -y --no-install-recommends nsight-systems-cli \
          || echo "WARNING: Nsight Systems is not available for this base image; nsys will be missing" ; } && \
        rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Skipping Nsight Systems install" ; \
    fi

# ============================================================
# Python packages (venv avoids the PEP 668 "externally managed" error)
# ============================================================
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    pillow \
    jupyter \
    && pip cache purge

# ============================================================
# CUDA environment
# ============================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# No architecture is hard-coded here on purpose. The CMake build
# detects the GPU actually present (CUDA_ARCH=native) so the same
# image works on a GTX 1660, an RTX 3060 or an H100. Override with
#   cmake -S . -B build -DCUDA_ARCH=86
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /workspace

COPY docker/scripts/verify-cuda.sh /usr/local/bin/verify-cuda
RUN chmod +x /usr/local/bin/verify-cuda

SHELL ["/bin/bash", "-c"]

# Login banner: reports the GPU that is actually visible.
RUN printf '%s\n' \
    'export PS1="\[\033[1;32m\]cuda-dev\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ "' \
    'echo "------------------------------------------------------------"' \
    'echo "  CUDA Learning - development environment"' \
    'echo "  nvcc   : $(nvcc --version | grep -oP "release \K[0-9.]+")"' \
    'echo "  cmake  : $(cmake --version | head -1 | awk "{print \$3}")"' \
    'echo "  python : $(python3 --version | awk "{print \$2}")"' \
    'if command -v nvidia-smi >/dev/null 2>&1; then' \
    '  echo "  GPU    : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | paste -sd", ")"' \
    'else' \
    '  echo "  GPU    : not visible (start the container with --gpus all)"' \
    'fi' \
    'echo "  Verify environment : verify-cuda"' \
    'echo "  Build everything   : ./scripts/build.sh"' \
    'echo "------------------------------------------------------------"' \
    >> /root/.bashrc

CMD ["/bin/bash"]
