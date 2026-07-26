<!-- Language: **English** | [Tiếng Việt](CONTRIBUTING.vi.md) -->

# Contributing

Fixes, new exercises and translation improvements are all welcome.

## Adding an exercise

```bash
./scripts/new-exercise.sh 3 17 wave-propagation "Wave propagation"
```

That creates `phase-3-intermediate/exercises/17-wave-propagation/` with the five
files an exercise needs. There is no central list to update — re-run
`cmake -S . -B build` and the new folder becomes two targets automatically.

### What an exercise must have

| File | Requirement |
|---|---|
| `README.md` | Goal, background, task, build/run, expected output, key takeaways, further reading |
| `README.vi.md` | The same content in Vietnamese, with the language switcher on line 1 |
| `reference.h` | The CPU golden result and the problem setup, shared by both variants |
| `main.cu` | Starter. Must **compile and run** with the `TODO`s unfilled, and print what is missing rather than crash |
| `solution.cu` | Reference solution. Must verify its own result and exit 0 |

Optional: `exercise.cmake` for extra libraries, compile options or a longer test
timeout.

### Rules that keep the repository coherent

**Every program verifies itself.** Compare against a CPU reference and return
`verifySummary()`. `ctest` is the repository's regression suite; an exercise that
cannot fail is not pulling its weight.

**Every program reports rates, not just times.** GB/s for memory-bound work,
GFLOP/s for compute-bound work, and a speedup against a stated baseline. Use
`ResultTable`.

**Comments are English only, in every file.** One source file serves both
audiences and stays diffable. Documentation is bilingual; code is not.

**Comments explain *why*.** `// increment i` is noise. `// four accumulators
because an FMA has ~4 cycles of latency` is the exercise.

**Never claim a result you have not measured.** Put real numbers from a real run
into the "Expected output" section, and say which GPU produced them. If a fix
turns out not to help on your hardware, say so and explain the conditions under
which it would — an honest negative result teaches more than a fabricated win.

**Skip, do not fail.** If an exercise needs two GPUs, Tensor Cores or an external
library, detect the situation and return `skipExercise("...")`. `[SKIP]` is not a
failure.

**Degrade gracefully across CUDA versions.** Guard version-specific APIs with
`#if CUDART_VERSION < 13000` or similar. The repository targets CUDA 11–13.

## Style

- 4-space indent, 100-column lines, `.clang-format` follows the Google style with
  those two changes.
- `d_` prefix for device pointers, `h_` for host pointers.
- Prefer `CUDA_CHECK(...)` around every runtime call. It costs nothing and saves
  hours.
- Grid-stride loops by default.

## Translations

The Vietnamese version is a **parallel document, not a machine translation**. Keep
technical terms in English where that is what a reader will search for (`warp`,
`coalescing`, `occupancy`) — [docs/GLOSSARY.md](docs/GLOSSARY.md) records the
agreed mapping. If you add a term, add it there too.

Both language versions must stay in sync. A pull request that changes `README.md`
without `README.vi.md` will be asked for the other half.

## Before opening a pull request

```bash
./scripts/build.sh --clean          # everything compiles
./scripts/run-all.sh                # everything passes
```

State which GPU and CUDA version you tested on.
