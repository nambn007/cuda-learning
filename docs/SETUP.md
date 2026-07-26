<!-- Language: **English** | [Tiếng Việt](SETUP.vi.md) -->

# Setup

[← back to the repository](../README.md)

Two options: **Docker** (recommended — the toolchain, profilers and libraries are
pinned and identical for everyone) or a **native install**.

---

## Option 1 · Docker

### Prerequisites

- An NVIDIA GPU with a driver installed on the **host** (`nvidia-smi` must work).
- Docker Engine 19.03+ and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify the toolkit is wired up:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

If that prints your GPU, everything else will work.

### Start

```bash
docker compose up -d --build     # first build takes 10-20 min
docker compose exec cuda-dev bash

# inside the container
verify-cuda
./scripts/build.sh
./scripts/run-all.sh
```

The repository is bind-mounted at `/workspace`, so edit on the host with your
usual editor and build inside the container.

To skip the ~500 MB Nsight Systems download:

```bash
docker compose build --build-arg INSTALL_NSIGHT_SYSTEMS=0
```

### VS Code Dev Containers

Install the **Dev Containers** extension, open the folder, then
`Ctrl+Shift+P` → *Reopen in Container*. `.devcontainer/devcontainer.json` sets up
the CUDA, C++ and CMake extensions and points IntelliSense at `common/`.

---

## Option 2 · Native install

### Linux (Ubuntu 22.04 / 24.04)

```bash
# Driver, if you do not have one
sudo ubuntu-drivers autoinstall && sudo reboot

# CUDA Toolkit - follow the official installer for your distribution
# https://developer.nvidia.com/cuda-downloads
sudo apt install -y build-essential cmake ninja-build git

# Add CUDA to your PATH (adjust the version)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version
```

Then:

```bash
./scripts/build.sh
./scripts/run-all.sh
```

### Windows

Use **WSL2** with an Ubuntu distribution and follow the Linux instructions. The
Windows NVIDIA driver exposes the GPU to WSL2 automatically; do **not** install a
driver inside WSL. Native Windows with MSVC is not tested here.

### Optional tools

| Tool | Used by | Install |
|---|---|---|
| Nsight Systems (`nsys`) | Phase 3/16, Phase 4 | `apt install nsight-systems-cli` |
| Nsight Compute (`ncu`) | Phase 3/16 | ships with the CUDA toolkit |
| CUTLASS | Phase 5/10 | `git clone https://github.com/NVIDIA/cutlass` |
| PyTorch | Phase 5/11 | `pip install torch` |
| OpenGL headers | Phase 5/13 | `apt install libglfw3-dev libglew-dev` |

Build the optional exercises with `./scripts/build.sh --optional`.

---

## Build options

```bash
cmake -S . -B build -DCUDA_ARCH=native      # default: detect this machine's GPU
cmake -S . -B build -DCUDA_ARCH=86          # one architecture (faster builds)
cmake -S . -B build -DCUDA_ARCH=portable    # fat binary, sm_70..sm_89
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug   # -G, for cuda-gdb
cmake -S . -B build -DCL_PTXAS_VERBOSE=ON      # per-kernel register/shared usage
cmake -S . -B build -DCL_ENABLE_OPTIONAL=ON    # exercises with external deps
```

Handy targets:

```bash
cmake --build build --target solutions -j   # every reference solution
cmake --build build --target starters -j    # every starter
ctest --test-dir build -L p3                # run one phase
ctest --test-dir build -R reduction         # run by name
```

> **Never benchmark a Debug build.** `-G` disables most device optimisations and
> kernels run several times slower.

---

## Troubleshooting

**`nvcc: command not found`**
CUDA is not on your `PATH`. See the export lines above, or use Docker.

**`no kernel image is available for execution on the device`**
The binary was built for a different architecture. Rebuild with
`-DCUDA_ARCH=native`, or pass your compute capability explicitly
(`nvidia-smi --query-gpu=compute_cap --format=csv`).

**`CUDA driver version is insufficient for CUDA runtime version`**
The toolkit is newer than the driver. Either update the driver or use an older
toolkit — the driver version must be ≥ the toolkit version.

**`Failed to detect a default CUDA architecture`** during `cmake`
No GPU was visible at configure time. Use `-DCUDA_ARCH=portable`, or pass the
number directly.

**`nvidia-smi` works on the host but not in the container**
The NVIDIA Container Toolkit is missing or Docker was not restarted after
installing it. Test with the `docker run --gpus all ... nvidia-smi` command above.

**`ncu` reports `ERR_NVGPUCTRPERM`**
Profiling counters need elevated permissions. The compose file already adds
`cap_add: [SYS_ADMIN]`; on a native install, follow
[NVIDIA's guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm)
to allow non-root profiling.

**An exercise prints `[SKIP]`**
It needs hardware or a dependency you do not have — a second GPU, Tensor Cores,
CUTLASS. That is not a failure; the exercise says which requirement is unmet.

**A kernel gives wrong results only sometimes**
A race condition. Run `compute-sanitizer --tool racecheck ./your_binary`. For
out-of-range accesses, plain `compute-sanitizer ./your_binary`.

---

## Verifying a fresh setup

```bash
verify-cuda                                  # toolchain + a real kernel launch
./scripts/build.sh                           # everything compiles
./scripts/run-all.sh                         # everything produces correct results
```

All three clean means you are ready for
[Phase 1](../phase-1-foundation/README.md).
