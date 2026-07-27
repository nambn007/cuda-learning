<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Project: Mini deep-learning framework

> **Phase 6 · Mastery** | ⭐⭐⭐⭐⭐ | ✅ core module verified on an RTX 3060

## What you are building

A framework that trains a small network end to end on the GPU: tensors, layers,
autograd, optimisers.

## What ships with it

`main.cu` is a **working linear layer** — forward (`Y = XW + b`) and all three
gradients (`dX = dY Wᵀ`, `dW = Xᵀ dY`, `db = Σ dY`) — checked against **numerical
differentiation**.

That check is the single most valuable tool in the project. A transposed index or a
missing sum shows up in seconds; without it, training just converges slightly worse
and never tells you why.

> **Rule for the whole project: every new layer gets a numerical gradient check
> before it is used in training.** There is no cheaper bug detector in machine
> learning.

## Why this project

**API design under performance pressure**, plus backward passes that must be the
*exact* derivative of the forward pass. Getting the maths right is half of it;
keeping the abstraction from costing you 3× is the other half.

## Milestones

| # | Deliverable | Introduces |
|---|---|---|
| 1 | `Tensor` with RAII, shapes, views | Phase 2/12 applied at scale |
| 2 | Swap the naive GEMM for Phase 3/04's, then cuBLAS | Measure both — you now know the gap |
| 3 | ReLU, softmax + cross-entropy, each gradient-checked | Fused elementwise, numerically stable softmax |
| 4 | An autograd tape | `backward()` generated, not written |
| 5 | SGD and Adam; train MNIST to >97% | The whole thing actually working |
| 6 | Fuse elementwise chains; a memory pool | Phase 1/03 and Phase 5's fusion, for real |

## Evaluation criteria

- **Correctness:** every layer passes a numerical gradient check to 1e-2 relative.
- **Convergence:** MNIST above 97% test accuracy.
- **Performance:** report time per epoch and your GEMM's percentage of cuBLAS.
  State how much the abstraction costs versus calling the kernels directly.

## Method

1. **Gradient-check first, optimise second.** A fast wrong gradient trains to a
   worse optimum and looks fine.
2. **Keep the naive kernels.** They are your reference when the fast ones disagree.
3. **Profile a whole epoch**, not a kernel. Framework overhead lives between
   kernels, and `nsys` is where you see it.

## Reading

- Karpathy, [micrograd](https://github.com/karpathy/micrograd) — autograd in 100 lines
- *Deep Learning* (Goodfellow et al.), chapter 6 — the backward-pass derivations
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — how a
  real one is structured
- Micikevicius et al., *Mixed Precision Training* (2018) — for when you add FP16
