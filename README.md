<div align="center">

# CUDA Learning

**A hands-on CUDA curriculum — from C++ fundamentals to GPU mastery**

**🇬🇧 English** · [🇻🇳 Tiếng Việt](README.vi.md)

[Roadmap](docs/ROADMAP.md) · [Setup](docs/SETUP.md) · [Glossary](docs/GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## What this is

Six phases taking you from "I know some C++" to writing GPU kernels you can defend
with a profiler. **33 exercises and 3 portfolio projects are implemented today**;
the full 63-exercise plan is specified in [docs/ROADMAP.md](docs/ROADMAP.md) and the
remaining folders can be scaffolded with one command.

Every exercise ships as **four files**:

| File | What it is |
|---|---|
| `README.md` / `README.vi.md` | the problem, the theory, and why it matters — in both languages |
| `main.cu` | a **starter** with `TODO`s for you to fill in |
| `solution.cu` | a **reference solution**, commented to explain *why*, not *what* |
| `reference.h` | the shared CPU golden result, so both are checked identically |

Every program **verifies its own result against a CPU reference and exits
non-zero on mismatch**, so the whole repository doubles as a regression suite:

```bash
ctest --test-dir build --output-on-failure
```

And every program **reports its own performance** in the units that matter —
GB/s, GFLOP/s, and speedup against a baseline. You are never asked to take a
claim on trust.

## Quick start

```bash
git clone <this-repo> && cd cuda-learning

# 1. Development environment (CUDA toolkit, Nsight, profilers)
docker compose up -d --build
docker compose exec cuda-dev bash

# 2. Check the GPU is really usable
verify-cuda

# 3. Build everything and run the reference solutions
./scripts/build.sh
./scripts/run-all.sh

# 4. Start learning
cat phase-1-foundation/exercises/01-matmul-naive-cpu/README.md
```

No Docker? See [docs/SETUP.md](docs/SETUP.md) for a native install.

## The six phases

| Phase | Topic | Exercises | Time | Level |
|---|---|---|---|---|
| [1](phase-1-foundation/) | **Foundation** — C++, caches, SIMD, roofline | 8 | 3–4 weeks | ⭐ |
| [2](phase-2-cuda-fundamentals/) | **CUDA Fundamentals** — kernels, memory, indexing | 12 | 4–6 weeks | ⭐⭐ |
| [3](phase-3-intermediate/) | **Intermediate** — coalescing, shared memory, parallel patterns | 7 of 16 | 6–8 weeks | ⭐⭐⭐ |
| [4](phase-4-advanced/) | **Advanced** — pinned memory, streams, graphs, atomics | 3 | 8–10 weeks | ⭐⭐⭐⭐ |
| [5](phase-5-expert/) | **Expert** — Thrust/CUB, cuBLAS, Tensor Cores | 3 | 6–8 weeks | ⭐⭐⭐⭐⭐ |
| [6](phase-6-mastery/) | **Mastery** — three portfolio projects | 3 projects | ongoing | 🏆 |

Full breakdown with per-exercise objectives: **[docs/ROADMAP.md](docs/ROADMAP.md)**.

**Where to start.** Solid on C++, caches and the roofline model? Jump to Phase 2.
Otherwise Phase 1 is not filler — every idea in it (row-major layout, cache lines,
SIMD lanes, arithmetic intensity) reappears on the GPU under a different name, and
Phase 3 assumes you already own them.

## How to work through an exercise

```bash
# 1. Read the problem
cat phase-2-cuda-fundamentals/exercises/03-vector-add/README.md

# 2. Fill in the TODOs in main.cu, then build and run
cmake --build build --target p2_03_vector_add -j
./build/bin/p2/p2_03_vector_add

# 3. Stuck, or want to compare? Read the reference solution
cmake --build build --target p2_03_vector_add_sol -j
./build/bin/p2/p2_03_vector_add_sol
```

Running an unfinished starter is safe: it prints what is still missing instead of
crashing.

Target names follow `p<phase>_<number>_<slug>`, with `_sol` for the reference
solution. Binaries land in `build/bin/p<phase>/`.

## Requirements

- **GPU:** NVIDIA, compute capability ≥ 5.0. Some Phase 5 exercises need ≥ 7.0
  (Tensor Cores) and skip themselves cleanly otherwise.
- **CUDA Toolkit:** ≥ 11.0, 12.x recommended. CUDA 13 works — affected code is
  version-guarded.
- **Host compiler:** GCC ≥ 9 or Clang ≥ 10, C++17.
- **CMake:** ≥ 3.20.

The build detects your GPU automatically (`-DCUDA_ARCH=native`). Override with
`-DCUDA_ARCH=86`, or `-DCUDA_ARCH=portable` for a fat binary.

Exercises needing more than one GPU, or an external dependency such as CUTLASS or
PyTorch, **skip themselves with an explanation** rather than failing the suite.

## Repository layout

```
common/              shared headers: error checking, timing, verification, PPM I/O
cmake/               exercise auto-discovery
docs/                roadmap, setup, glossary   (each with a .vi.md twin)
scripts/             build.sh, run-all.sh, new-exercise.sh
phase-N-*/
  README.md            phase overview and checklist
  exercises/NN-slug/   README.md, README.vi.md, main.cu, solution.cu, reference.h
phase-6-mastery/
  projects/slug/       spec, architecture, milestones, starter code
```

There is no central target list. Drop a folder with a `main.cu` into any
`exercises/` directory, re-run `cmake`, and it builds — see
[`scripts/new-exercise.sh`](scripts/new-exercise.sh).

## Bilingual by design

Every document exists twice: `X.md` (English) and `X.vi.md` (Tiếng Việt), with a
language switcher on the first line of each. **Code comments are English only**, so
that a single source file serves both audiences and stays diffable.

[docs/GLOSSARY.md](docs/GLOSSARY.md) maps the technical terms between the two
languages — useful when reading NVIDIA documentation after learning the concept in
Vietnamese.

## Current status

| Phase | Implemented | Verified on an RTX 3060 |
|---|---|---|
| 1 Foundation | 8 / 8 | all |
| 2 CUDA Fundamentals | 12 / 12 | all |
| 3 Intermediate | 7 / 16 | all 7 |
| 4 Advanced | 3 (of a larger plan) | 1 of 3 |
| 5 Expert | 3 (of a larger plan) | 2 of 3 |
| 6 Mastery | 3 projects | all 3 core modules |

**Everything present in the tree builds**, and every exercise marked verified passes
`ctest` on the reference machine. A handful of Phase 4/5 exercises are code-complete
but have not been run there yet — their READMEs say so and contain **no measured
numbers** rather than invented ones.

Deferred exercises are listed in each phase README and specified in
[docs/ROADMAP.md](docs/ROADMAP.md). `scripts/new-exercise.sh` scaffolds one in a
single command.

## License

MIT. Course materials referenced in the exercises belong to their respective
authors.
