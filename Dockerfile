# ============================================================
# CUDA Development Environment
# Multi-stage build for lean & fast dev container
# ============================================================
# Base: CUDA 12.6 (compatible with driver 580.x, backward compat)
# Target GPU: GTX 1660 SUPER (sm_75, Compute Capability 7.5)
# ============================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS cuda-dev

LABEL maintainer="nambn007"
LABEL description="CUDA Learning Development Environment"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# ============================================================
# System packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Debugging & profiling
    gdb \
    valgrind \
    linux-tools-generic \
    strace \
    # Version control
    git \
    # Utilities
    wget \
    curl \
    unzip \
    vim \
    nano \
    htop \
    tree \
    # Python (for plotting, scripting, PyCUDA later)
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # Image processing (for Phase 2 image exercises)
    libpng-dev \
    libjpeg-dev \
    # OpenGL (for Phase 5 CUDA-OpenGL interop)
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglfw3-dev \
    libglew-dev \
    # Misc
    ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ============================================================
# Python packages (in a venv to avoid PEP 668 issues)
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

# Default architecture for GTX 1660 SUPER (Turing, sm_75)
# Also include common architectures for broader compatibility
ENV CUDAARCHS="75"
ENV CMAKE_CUDA_ARCHITECTURES="75"

# ============================================================
# Working directory
# ============================================================
WORKDIR /workspace

# ============================================================
# Verification script
# ============================================================
COPY docker/scripts/verify-cuda.sh /usr/local/bin/verify-cuda
RUN chmod +x /usr/local/bin/verify-cuda

# ============================================================
# Default shell & entrypoint
# ============================================================
SHELL ["/bin/bash", "-c"]

# Custom prompt showing CUDA version
RUN echo 'export PS1="🔥 \[\033[1;32m\]cuda-dev\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ "' >> /root/.bashrc \
    && echo 'echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"' >> /root/.bashrc \
    && echo 'echo "  🚀 CUDA Development Environment"' >> /root/.bashrc \
    && echo 'echo "  📦 CUDA: $(nvcc --version | grep release | awk \"{print \\$6}\")"' >> /root/.bashrc \
    && echo 'echo "  🏗️  CMake: $(cmake --version | head -1 | awk \"{print \\$3}\")"' >> /root/.bashrc \
    && echo 'echo "  🐍 Python: $(python3 --version | awk \"{print \\$2}\")"' >> /root/.bashrc \
    && echo 'echo "  🎯 Target: sm_75 (GTX 1660 SUPER)"' >> /root/.bashrc \
    && echo 'echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"' >> /root/.bashrc

CMD ["/bin/bash"]
