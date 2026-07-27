<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Project: GPU ray tracer

> **Phase 6 · Mastery** | ⭐⭐⭐⭐ | ✅ core module verified on an RTX 3060

## What you are building

A path tracer that renders a noise-free image of a non-trivial scene, fast enough
to iterate on.

## What ships with it

`main.cu` is a **working core module**, not a skeleton: spheres, per-thread RNG,
cosine-weighted diffuse bounces, gamma encoding, PPM output, and property-based
tests (non-uniform, non-black, deterministic). It renders 800×450 at 64 spp in
~3 ms on an RTX 3060 — 7.4 billion primary rays/second.

Everything past that is yours.

## Why this project

**Divergence.** Neighbouring pixels bounce in different directions and terminate
after different numbers of bounces, so threads in a warp do genuinely different
work. This is the hardest divergence problem in the curriculum — every other
exercise had a regular access pattern available if you looked for it. Here there
is not one.

## Milestones

| # | Deliverable | Introduces |
|---|---|---|
| 1 | Triangle meshes, load an `.obj` | Irregular data, indirection |
| 2 | BVH build (host) + traversal (device) | A stack in registers, the divergence problem in full |
| 3 | Materials: metal, dielectric, emissive | Branch-heavy shading, and what it costs |
| 4 | Russian-roulette termination, importance sampling | Variance reduction — fewer samples for the same quality |
| 5 | Wavefront restructuring | Sort rays by material to recover coherence |
| 6 | Denoising (À-Trous or similar) | 16 spp that looks like 1024 |

## Evaluation criteria

- **Correctness:** a Cornell box converges to the reference within a few percent.
- **Performance:** report Mrays/s, and `ncu`'s warp execution efficiency before and
  after milestone 5. If you cannot state the divergence cost in a number, you have
  not measured it.
- **Quality:** a side-by-side at 16 / 64 / 1024 spp, with and without denoising.

## Method

1. **Keep a CPU reference renderer.** Slow, obviously correct, and the only way to
   tell a rendering bug from a sampling bug.
2. **Look at the image after every change.** Property tests catch crashes; your
   eyes catch wrong.
3. **Profile before restructuring.** Milestone 5 is a large rewrite — do it only
   once you have measured what divergence is actually costing you.

## Reading

- *Ray Tracing in One Weekend* (Shirley) — start here if the maths is new
- *Physically Based Rendering* (Pharr, Jakob, Humphreys) — the reference
- Laine et al., *Megakernels Considered Harmful* (2013) — the case for milestone 5
- [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) — what you are
  competing with, and what RT cores do that you cannot
