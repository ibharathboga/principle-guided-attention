# E7 — Propagative Principle: Compute-Efficient PGA

> **Status**: Discussion / Pre-Experiment  
> **Date**: 2026-02-20  
> **Context**: Prior experiments e2 (PyTorch PGA), e5 (Token-Based), e6 (Strided)

---

## 1. Problem Statement

SVD is the dominant compute cost in PGA. In the current architecture, SVD is potentially computed **per token, per layer**. For a model with `T` tokens and `L` layers, this means `T × L` SVD decompositions per forward pass — a cost that scales poorly as models deepen.

The question: **Can we compute SVD once per token and propagate that same principle through all layers?**

## 2. Current Architecture (Per-Layer SVD)

In the general multi-layer PGA setup, each layer receives a new hidden representation and could recompute the principle from it:

```
x = embed(tokens)

For each layer l:
    P_l = SVD(x)          ← Recompute principle from current representation
    Q, K, V = project(x)
    Q, K, V = filter(Q, K, V, P_l)   ← Different P at each layer
    x = attention(Q, K, V) + x
    x = MLP(x) + x
```

Each layer sees a **transformed** version of the input. Recomputing SVD at every layer tries to capture the "principle" of that transformed representation.

**Cost**: `T × L` SVDs per forward pass.

## 3. Proposed: Propagative Principle (One SVD Per Token)

The principle is derived **once** from the raw observation (initial embedding) and carried through all layers unchanged:

```
x = embed(tokens)
P = SVD(x)               ← Compute principle ONCE from observation

For each layer l:
    Q, K, V = project(x)
    Q, K, V = filter(Q, K, V, P)    ← Same P at every layer
    x = attention(Q, K, V) + x
    x = MLP(x) + x
```

**Cost**: `T × 1` SVDs per forward pass.

## 4. Why "Propagative" and Not "Strided"

These are two orthogonal strategies operating on different axes:

```
             Tokens (T) →
           ┌──────────────────┐
Layer 0    │ P₀₀  P₀₁  P₀₂ ··│   ← Striding reduces across THIS axis (horizontal)
Layer 1    │ P₁₀  P₁₁  P₁₂ ··│
Layer 2    │ P₂₀  P₂₁  P₂₂ ··│   ← Propagative reduces across THIS axis (vertical)
Layer 3    │ P₃₀  P₃₁  P₃₂ ··│
           └──────────────────┘
                  Depth (L) ↓
```

| Strategy | Axis | What it collapses | SVD count |
|---|---|---|---|
| **Full** (baseline) | Neither | Nothing | `T × L` |
| **Strided** (e6) | Horizontal (tokens) | Nearby tokens share P | `T/s × L` |
| **Propagative** (e7) | Vertical (layers) | All layers share P | `T × 1` |
| **Both combined** | Both | Shared across tokens AND layers | `T/s × 1` |

## 5. Philosophical Justification

The propagative approach aligns with the core PGA premise:

> **The principle is a property of the observation, not the computation.**

When you read a sentence, the "topic" or "structural principle" is determined from the surface-level words. As your brain processes deeper semantics (analogous to deeper layers), the governing principle doesn't change — the same lens guides focus at every level of abstraction.

- **Layer 0**: Low-level features filtered by the principle
- **Layer 4**: Mid-level patterns, still governed by the same principle
- **Layer 12**: High-level abstractions, same guiding lens

The principle is discovered at observation time. The layers are just how deeply you refine your understanding *under* that principle.

## 6. Compute Impact

### Per Forward Pass (single sequence, length T)

| Model Config | Full PGA | Propagative PGA | Savings |
|---|---|---|---|
| 1 layer (e2) | T SVDs | T SVDs | 1× (no change) |
| 4 layers (e4-scale) | 4T SVDs | T SVDs | **4×** |
| 12 layers (GPT-2 small) | 12T SVDs | T SVDs | **12×** |
| 32 layers (LLaMA-7B) | 32T SVDs | T SVDs | **32×** |

The savings scale **linearly with model depth** — exactly where modern architectures are growing.

### Memory

- Full: Store `P` as `(B, T, D, D)` per layer, or recompute
- Propagative: Store `P` as `(B, T, D, D)` once, shared reference across layers

### Implementation

The propagative approach **eliminates recomputation entirely** inside the layer loop. `P` becomes a precomputed constant tensor passed to every layer — no different from how positional embeddings are computed once and reused.

## 7. What Could Be Lost

The counterargument to propagative P:

| Concern | Analysis |
|---|---|
| **Later layers see different representations** | True — but P is detached from gradients anyway. It's a filter, not a learned transformation. The *filter* being fixed doesn't prevent the *representations* from evolving. |
| **Diminishing relevance at deeper layers** | Possible — but the residual connections ensure the original signal (which P was derived from) persists through depth. |
| **Missing layer-specific structure** | Per-layer SVD could discover structure unique to intermediate representations. Propagative P trades this for consistency and cost. |

## 8. Hybrid: Propagative with Refresh

A middle ground exists — recompute P every `k` layers instead of every layer:

```
x = embed(tokens)
P = SVD(x)

For each layer l:
    if l % k == 0:            ← Refresh every k layers
        P = SVD(x.detach())
    ...use P...
```

Cost: `T × (L/k)` SVDs. This retains some depth-sensitivity while still saving compute.

## 9. Experiment Plan (E7)

To validate propagative PGA:

1. **Baseline**: Standard MicroGPT (no PGA)
2. **Full PGA**: SVD per token per layer (current e2 approach)
3. **Propagative PGA**: SVD once per token, shared across layers
4. **Metric**: Training loss, validation loss, wall-clock time
5. **Config**: Scale to 4 layers (`n_layer=4`) to make the distinction visible

> [!NOTE]
> With `n_layer=1` (current e2 config), propagative and full PGA are **identical**. The experiment requires depth > 1 to show any difference.

## 10. Summary

The propagative principle is not about reducing SVD across the token axis (that's striding). It's about recognizing that the **principle belongs to the observation**, and should be computed once at observation time, then carried through the entire depth of the network. The savings scale with model depth — the exact dimension that modern architectures are scaling along.
