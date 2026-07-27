<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 09 - RGB to grayscale: AoS versus SoA

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1.5 h | Prerequisites: [04](../04-saxpy/), [06](../06-matrix-add-2d/) | **Requires a GPU**

## Goal

Work on real image data, and test a piece of GPU folklore — *"prefer Structure of
Arrays"* — by measuring it. The result contradicts the advice, and understanding
why gives you the correct rule instead of the slogan.

## Background

The arithmetic is one line (ITU-R BT.601 luma; the eye is far more sensitive to
green than to blue, so a plain average looks wrong):

```
gray = 0.299·R + 0.587·G + 0.114·B
```

**The layout question.** Images are stored interleaved — `RGBRGBRGB...` — an
**Array of Structures**. The alternative is three separate planes,
`RRRR...GGGG...BBBB` — a **Structure of Arrays**. Conventional advice says SoA is
faster on a GPU. Measure it here and the two are identical:

| Layout | What one warp reads | Sectors |
|---|---|---|
| AoS | 96 consecutive bytes | 3 |
| SoA | three runs of 32 bytes | 3 |

Same traffic, same instruction count, same speed. **This kernel needs all three
channels of every pixel**, so there is nothing for SoA to save.

**Where AoS really does hurt** is when you need *part* of each structure. Extract
just the red channel and AoS becomes a stride-3 read: a warp touches 96 bytes and
uses 32, throwing away two thirds of every sector. The exercise measures that case
too.

> **The rule, correctly stated:** AoS versus SoA matters in proportion to how much
> of each structure you actually read. Need every field? The layouts tie. Need one
> field out of many? SoA wins by roughly the ratio between them.

That is why a particle system with a 32-byte struct sees a large win from SoA when
a kernel only touches `position` — and why this grayscale kernel sees none.

**What does help here** is vectorising: give each thread four pixels, so twelve
byte-loads become three 4-byte loads. 4 pixels = 12 bytes = exactly three `uchar4`.
That is an **instruction-count** win, not a layout win, and it is worth ~1.3×.

**One more thing: do not compare images for exact equality.** `nvcc` contracts
`0.299f*r + 0.587f*g + 0.114f*b` into FMA instructions by default, rounding once
instead of twice. When the true value sits near `x.5`, host and device round to
different integers — about 1 pixel in 50,000 comes out one grey level apart. The
right test is "no pixel off by more than one level". Compile with `-fmad=false` if
you ever genuinely need bit-exact agreement.

## Your task

Open `main.cu` and write five kernels:

1. **`grayInterleaved()`** — AoS.
2. **`grayPlanar()`** — SoA.
3. **`grayInterleavedVec4()`** — AoS with `uchar4` loads. Work out which component
   of which `uchar4` is which channel; getting it wrong produces a visibly wrong
   image.
4. **`redFromInterleaved()` / `redFromPlanar()`** — the experiment that settles the
   layout question.

**Before you run it, write down your predictions** for (a) AoS vs SoA on
grayscale, (b) `uchar4` vs scalar, (c) AoS vs SoA on the red channel. At least one
of them will probably surprise you.

Verify with `checkImage()`, not `checkArrayExact()`.

## Build and run

```bash
cmake --build build --target p2_09_rgb_to_grayscale -j
cd build/testrun && ../bin/p2/p2_09_rgb_to_grayscale
```

It generates `input.ppm` if you do not supply one, and writes `output_gray.pgm`.
Open it — an image kernel that is subtly wrong usually *looks* wrong, which no
assertion gives you for free.

## Expected output

RTX 3060, 4096×2048:

```
--- Correctness ---
  [PASS] interleaved (AoS)              (max diff 1 level, 0.002% of pixels differ)
  [PASS] planar (SoA)                   (max diff 1 level, 0.002% of pixels differ)
  [PASS] interleaved with uchar4 loads  (max diff 1 level, 0.002% of pixels differ)

--- Performance ---
Variant                           Time (ms)      GB/s   Speedup
----------------------------------------------------------------
CPU (single thread)                  18.215      1.84     1.00x
GPU interleaved (AoS)                 0.142    236.38   128.32x
GPU interleaved, uchar4               0.106    315.08   171.04x
GPU planar (SoA)                      0.148    225.99   122.68x

--- When AoS really does hurt: reading one channel ---
red channel from AoS (stride 3)       0.136    123.19     1.00x
red channel from SoA (planar)         0.104    160.63     1.30x
```

**SoA is marginally *slower* than AoS for grayscale, and 1.30× faster for a single
channel.** Both numbers come from the same data on the same GPU.

## Key takeaways

- **Test the folklore.** "Prefer SoA" is a heuristic with a condition attached,
  and the condition is what matters.
- **Layout matters in proportion to the fraction of each structure you read.**
- **Instruction count is a separate axis from traffic.** `uchar4` moves identical
  bytes and is 1.3× faster.
- **FMA contraction makes host and device disagree by one bit**, which for an
  8-bit image means one grey level. Compare with a tolerance, or turn off `-fmad`.
- Image kernels have a free extra test: look at the output.

## Going further

- Convert your own photo: `convert photo.jpg input.ppm`, then rerun.
- Add a `uchar4` version of the planar kernel. Does SoA now beat AoS?
- Try a struct with 8 fields where the kernel reads one. How does the ratio scale?
- `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` on the two red
  kernels — confirm the 3:1 sector ratio.
