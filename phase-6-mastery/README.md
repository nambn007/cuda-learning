<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 6 · Mastery

> **Ongoing · 🏆 · 5 projects** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## What this phase is for

Exercises have a known answer. Projects do not. This phase is where you find out
whether you can carry a GPU codebase from an empty directory to something you
would show an employer.

Each project ships:

- a **specification** — what it must do, and what "done" means
- an **architecture document** — the data layout, the kernel breakdown, the
  design decisions worth making early
- **milestones** — three to five stages, each independently runnable, so you
  always have something that works
- **evaluation criteria** — the performance numbers a good implementation hits
- a **worked core module** — one non-trivial kernel implemented and explained, so
  you start from a working foundation rather than a blank file

The rest is yours. That is the point.

## Projects

| Project | Difficulty | What you build | Key techniques |
|---|---|---|---|
| [gpu-ray-tracer](projects/gpu-ray-tracer/) | ⭐⭐⭐⭐ | A path tracer producing a noise-free image | BVH traversal, divergence management, RNG per thread, denoising |
| [mini-dl-framework](projects/mini-dl-framework/) | ⭐⭐⭐⭐⭐ | Train a small network end to end | Autograd, backward kernels, GEMM, operator fusion, memory pooling |
| [fluid-simulation](projects/fluid-simulation/) | ⭐⭐⭐⭐ | Real-time interactive fluid | Stencils, red-black solvers, spatial hashing, OpenGL interop |
| [gpu-database](projects/gpu-database/) | ⭐⭐⭐⭐ | Analytical queries over columnar data | Compaction, radix partitioning, hash joins, atomics |
| [llm-inference-engine](projects/llm-inference-engine/) | ⭐⭐⭐⭐⭐ | Serve a small transformer | KV-cache paging, attention kernels, batching, quantisation |

## How to choose

Pick the one whose *failure mode* you want to learn:

- **Ray tracer** — warp divergence and irregular memory. Every ray goes somewhere
  different.
- **DL framework** — API design under performance pressure, plus backward passes
  that must exactly mirror forward ones.
- **Fluid simulation** — sustained bandwidth and stencil optimisation, with a
  visual result that makes bugs obvious.
- **Database** — data-dependent parallelism. You cannot know the output size in
  advance.
- **LLM engine** — memory capacity as the binding constraint, and the current
  state of the art.

Two finished projects is the bar for this phase. One finished project beats five
started ones.

## Working method

1. **Make it correct first.** Write the CPU reference before the kernel; you will
   need it to debug.
2. **Profile before optimising.** Every project's evaluation criteria are
   expressed in profiler metrics for a reason.
3. **Keep a log.** What you tried, what it measured, what you concluded. This is
   what you actually show people — and it is what turns a project into a blog post
   or a conference talk.
4. **Compare against the best available.** cuBLAS, cuDNN, OptiX, vLLM. Knowing you
   are at 60% of the state of the art is worth more than not knowing.

## Beyond the projects

- Contribute to an NVIDIA open-source project (CUTLASS, cuDF, RAPIDS)
- Write up an optimisation you did, with numbers
- Answer questions on the NVIDIA developer forums
- Follow GTC sessions and the CUDA release notes; the hardware keeps moving
- Learn a second model — SYCL, HIP, or Triton — to see which CUDA ideas are
  fundamental and which are vendor-specific
