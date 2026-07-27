<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Project: LLM inference engine

> **Phase 6 · Mastery** | ⭐⭐⭐⭐⭐ | ✅ core module verified on an RTX 3060

## What you are building

An engine that serves a small transformer: batched decoding, a paged KV cache,
fused attention.

## What ships with it

`main.cu` implements the softmax at the heart of attention, **twice** — a
three-pass version and a single-pass **online** version — both verified against a
double-precision reference, including the property that actually matters
downstream: every row sums to 1.

Two things it demonstrates:

**Subtract the row maximum.** `exp(x)` overflows above `x = 88` in float, and
attention scores routinely exceed that. Subtracting the max is mathematically a
no-op and the difference between working and returning NaN. Try removing it.

**The online rescaling trick.**

```
m_new = max(m, x)
s_new = s·exp(m − m_new) + exp(x − m_new)
```

lets you maintain a correct softmax while seeing each row **once**. That is
precisely what makes FlashAttention possible: it computes attention over a sequence
whose score matrix is never held in memory at all.

## Why this project

**Memory capacity, not FLOPs, is the binding constraint.** Every milestone below is
about moving or storing less — which makes it the most current problem in the
curriculum, and the one where the state of the art moves fastest.

## Milestones

| # | Deliverable | Introduces |
|---|---|---|
| 1 | `QKᵀ` with the tiled GEMM from Phase 3/04 | Batched GEMM over heads |
| 2 | Fuse scores → softmax → `×V` into one kernel | Never materialise the score matrix |
| 3 | KV cache | Decode becomes memory-bound, not compute-bound |
| 4 | Paged attention | Sequences of different lengths sharing memory without fragmentation |
| 5 | Batched decoding | Many sequences, one step each; scheduling |
| 6 | INT8 or FP8 weights | Quantisation, and measuring the quality loss honestly |

## Evaluation criteria

- **Correctness:** logits match a PyTorch reference to 1e-3 for the same weights.
- **Performance:** tokens/second at batch 1 and batch 32; report the fraction of
  peak *bandwidth* reached during decode (it is memory bound — compare against
  Phase 3/01's numbers).
- **Memory:** peak VRAM against sequence length. Paged attention should make this
  roughly linear rather than quadratic in the number of concurrent sequences.
- **Quality (milestone 6):** perplexity before and after quantisation. A speedup
  reported without a quality number is not a result.

## Method

1. **Load real weights early** — a small GPT-2 or TinyLlama. Random weights hide
   numerical bugs that real distributions expose.
2. **Compare against PyTorch layer by layer**, not end to end. A wrong answer after
   12 blocks tells you nothing about which block.
3. **Measure bandwidth, not FLOPs**, during decode. Batch-1 decode reads the entire
   weight matrix to produce one token.

## Reading

- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (2023)
- [vLLM](https://github.com/vllm-project/vllm) — the reference implementation
- [llm.c](https://github.com/karpathy/llm.c) — a readable end-to-end starting point
