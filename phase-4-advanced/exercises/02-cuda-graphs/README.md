<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 02 - CUDA graphs

> **Phase 4 · Advanced** | Difficulty: ⭐⭐⭐⭐ | Time: ~2 h | Prerequisites: [01](../01-pinned-and-streams/) | **Requires a GPU**
>
> ⚠️ **Not yet verified on the reference machine.** The code builds and the logic is
> reviewed, but this exercise has not been run on the RTX 3060, so this README
> deliberately contains **no measured numbers**. Run it and fill in your own.

## Goal

Remove the CPU from the inner loop. When a program launches dozens of small kernels
thousands of times, the driver — not the GPU — becomes the bottleneck.

## Background

Every kernel launch costs the CPU a few microseconds of driver work: validating
arguments, writing a command packet, ringing the doorbell. That is invisible next
to a 2 ms kernel and **dominant** next to a 5 µs one.

Deep learning inference, iterative solvers and physics steps all have the same
shape — dozens of small kernels, launched in the same order, thousands of times.
The GPU ends up idle between kernels, waiting for the CPU to catch up.

A **CUDA graph** records that sequence once as a DAG of nodes with dependencies,
and replays it with a **single** launch. The driver validates the work once, at
instantiation, instead of every time.

**Two ways to build one:**

| | |
|---|---|
| **Stream capture** | put a stream in capture mode, issue the work exactly as normal, end capture. Almost no code changes — the usual choice. |
| **Explicit API** | `cudaGraphAddKernelNode` etc. More work, but you control the dependency structure directly. |

**The catch:** a graph captures **pointers and values, not intent**. If the next
iteration should use a different buffer, replaying the same graph will happily rerun
on the *old* one — silently, with no error. Either keep the pointers stable
(double-buffer into fixed slots) or update the node parameters with
`cudaGraphExecKernelNodeSetParams`, which is far cheaper than re-instantiating.

## Your task

Open `main.cu`:

1. The baseline: `kKernelsPerIteration` launches in a stream, repeated
   `kIterations` times. Verify against `chainCPU()` first.
2. Capture the identical sequence with `cudaStreamBeginCapture` /
   `cudaStreamEndCapture` / `cudaGraphInstantiate`. Print the node count.
3. Time both and report **microseconds per launch**, not total milliseconds.
   Predict the per-launch cost before measuring.
4. Reproduce the pointer bug: capture a graph on one buffer, replay it expecting a
   second buffer to update, observe that nothing happens and nothing errors. Then
   fix it with `cudaGraphExecKernelNodeSetParams`.

## Build and run

```bash
cmake --build build --target p4_02_cuda_graphs -j
./build/bin/p4/p4_02_cuda_graphs
nsys profile --stats=true ./build/bin/p4/p4_02_cuda_graphs
```

## Expected output

The program prints a correctness section (both variants must match the CPU chain),
then a table comparing individual launches against graph replay, and a per-launch
cost in microseconds for each.

**Fill in your own numbers here** once you have run it. What to look for: the GPU
work is identical in both cases, so any difference is pure CPU-side launch
overhead. The gap should widen as you increase `kKernelsPerIteration` and shrink as
you make each kernel larger.

## Key takeaways

- **Launch overhead is a real cost**, measured in microseconds of *CPU* time per
  kernel, and it is invisible until your kernels get small.
- **A graph moves validation from every launch to one instantiation.**
- **Stream capture needs almost no code changes** — issue the work as usual between
  begin and end.
- **Graphs freeze pointers.** The most common bug is replaying with stale buffers,
  and it fails silently.
- Graphs pay off for **many small kernels in a repeated sequence**; they do nothing
  for a few large ones.

## Going further

- Sweep `kKernelsPerIteration` from 1 to 100. Where does the graph start to win?
- Grow the element count until each kernel takes milliseconds. Does the advantage
  disappear?
- Build the same graph with the explicit node API and compare the code.
- Capture a graph containing memcpys and events as well as kernels.
- Compare the timelines in `nsys` — the gaps between kernels are what the graph
  removes.
