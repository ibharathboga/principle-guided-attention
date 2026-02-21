# E12 Code Review — Logging, Mistakes, and Compromises

> Cross-referenced against all 5 Notion docs and the E12 source files.

---

## ✅ Logging — Mostly Solid

The logging in [pipeline.py](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py) is **good** for an experiment script. Here's what it covers and what it misses:

### What's Working
| Area | How | Line(s) |
|---|---|---|
| Timestamped output | `[HH:MM:SS] [MODEL]` prefix on every log | [L48-50](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L48-L50) |
| Config dump at start | Logs layers, embd, lr, steps, PGA hyperparams | [L200-205](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L200-L205) |
| Param count per model | `Initialized — X params` | [L112](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L112) |
| Train + Val loss every 100 steps | With rolling ETA and step time | [L149-154](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L149-L154) |
| Token store stats (live) | Vectors count + MB | [L145-147](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L145-L147) |
| NaN detection | Catches divergence early | [L131-133](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L131-L133) |
| Final summary + verdict | JSON dump, markdown report, plots | [L321-328](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L321-L328) |

### What's Missing

1. **No gradient norm logging.** The model uses `Adam` with no gradient clipping. There's no visibility into whether gradients are exploding or vanishing — you'd only discover a problem after it causes NaN.
2. **No SVD rank logging.** The adaptive energy threshold dynamically picks how many singular values to keep, but **we never log what rank was actually chosen.** Was it keeping 2 components? 10? This is critical data for understanding whether the energy threshold is working as intended.
3. **No SVD fallback logging.** The `except RuntimeError` in [compute_P_from_store_adaptive L287-288](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L287-L288) silently returns an identity matrix. If SVD fails repeatedly, you'd have **no idea** — the model would just quietly degrade to an unfiltered baseline.
4. **No per-head diagnostic.** Since a single `P` is shared across heads (the documented compromise), logging the effective rank or subspace overlap across heads could help quantify how much this compromise costs.

---

## ⚠️ Mistakes / Bugs

### 1. Energy Threshold Uses `<=` Instead of `<` — Off-by-One Bias
**File**: [models.py L295](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L295)

```python
keep_mask = cum_energy <= (ENERGY_THRESHOLD * total_energy)
```

With `<=` and `ENERGY_THRESHOLD = 0.90`, if a single component captures **exactly** 90% of the energy, `keep_mask` will be `[True, False, False, ...]` — keeping only 1 component. But cumulative energy at index 0 being exactly 90% means the first component alone captures 90%, so the mask is technically correct here. **However**, if the first component captures >90%, then `keep_mask[0] = False` for index 0 — but this is saved by the `keep_mask[:, 0] = True` override on line 298. The logic works but is fragile and non-obvious.

### 2. `_needs_full_rebuild` Attribute Used Before Being Set
**File**: [models.py L72 vs L74](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L72-L74)

```python
self._needs_full_rebuild = True   # L72: set inside the loop
# ...
if getattr(self, '_needs_full_rebuild', False):  # L74: safe but awkward
```

The `_needs_full_rebuild` attribute is never initialized in `__init__`. The `getattr` with default is a workaround, not a fix. It should be initialized in `__init__` alongside the other attributes.

### 3. `train_single_model` Ignores Its `model_cls_name` Argument
**File**: [pipeline.py L189-196](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L189-L196)

```python
def train_single_model(model_name, model_cls_name, vocab_size, train_data, val_data):
    from models import BaselineMicroGPT
    # ...
    model = BaselineMicroGPT(vocab_size)  # ← Always creates BaselineMicroGPT regardless of model_cls_name
```

The `model_cls_name` parameter is accepted but **completely ignored**. The function always instantiates `BaselineMicroGPT`. This is dead code / latent bug from a previous version that may have supported multiple model types via `ProcessPoolExecutor`. It works only because the Baseline is currently the only model trained in a subprocess.

### 4. Causal Mask Recomputed Every Layer, Every Step
**File**: [models.py L241-243](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L241-L243)

```python
att = att.masked_fill(
    torch.tril(torch.ones(T, T, device=idx_device)) == 0, float('-inf')
)
```

`torch.tril(torch.ones(T, T))` is recomputed at every layer of every step. Since `T = BLOCK_SIZE = 32` is constant, this should be a pre-computed buffer. This is a minor perf concern but a code-quality issue.

### 5. Storing Final-Layer Outputs Without `.detach()`
**File**: [models.py L369-370](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L369-L370)

```python
for b in range(B):
    self.token_store.store_vectors(idx[b], x[b])
```

While `store_vectors` does call `.detach()` on the passed vectors (line 57), calling it **inside `forward()`** before `loss.backward()` means `x` is still part of the computation graph at this point. The `detach()` inside `store_vectors` creates copies, so memory isn't leaked into the graph — but the ordering is subtle and warrants a comment.

### 6. `RMSNorm` Has No Learnable Scale
**File**: [models.py L31-38](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L31-L38)

Standard RMSNorm includes a learnable `weight` parameter (a gain vector). This implementation omits it. This reduces representational capacity — the model can only normalize magnitudes but cannot re-scale per-dimension. For a research prototype this is acceptable, but it's a deviation from the standard formulation.

---

## 🤝 Compromises

### 1. Single `P` Across All Heads (Documented ✅)
**Acknowledged in**: [README.md L24-31](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/README.md#L24-L31)

This is the biggest architectural compromise. Multi-head attention gains power from **diverse subspaces per head**. A single `P` forces all heads through the same geometric lens — defeating part of the purpose of multi-head attention. The README correctly documents the reasoning (computational cost).

### 2. `.detach()` on Query Projection Input
**File**: [models.py L361](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L361)

```python
x_proj = self.query_proj(x.detach())
```

The input to the MLP query projection is **detached** from the graph. This means gradients from the attention layers **cannot flow back through `query_proj`** — the MLP learns only from its own retrieval loss signal (which doesn't exist — there's no direct loss on retrieval quality). **The `query_proj` MLP can only learn via the main loss through its own parameters, but the signal is cut off by `.detach()`.** This means the MLP weights are updated, but the signal guiding their update is limited. This is an undocumented compromise.

> [!WARNING]
> Because `x.detach()` severs the gradient path, the `query_proj` MLP weights are still trained (they receive gradients from the main loss flowing back through `P` → `run_layers`). But the **input representation** `x` is treated as a constant, which means the embedding layer doesn't get gradient signal from the retrieval pathway. This is likely intentional (to avoid SVD-through backprop instability), but should be documented.

### 3. `BATCH_SIZE = 1` — No True Batching
**File**: [pipeline.py L36](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L36)

The batch size is hardcoded to 1. The token store retrieval loop is `O(B × T)` with Python-level iteration ([models.py L269-281](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/models.py#L269-L281)), so increasing `B` would cause severe slowdown. This is a practical compromise but limits the generality of the results.

### 4. CPU-Only Training
**File**: [pipeline.py L40](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/e12/pipeline.py#L40)

`DEVICE = 'cpu'` is hardcoded. The token store uses Python dicts and lists extensively, which wouldn't directly benefit from GPU. This is consistent but limits scaling.

### 5. Propagative SVD vs. Notion Spec
The Notion doc [PropagativeSvd.md](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/notion/PropagativeSvd.md) specifies SVD once **per token**, propagated across layers. The E12 code computes a `P` matrix of shape `(B, T, D, D)` — one per token per batch — and passes it to all layers. **This matches the spec.** ✅

### 6. PGA Filters Q/K/V in Notion but Only Q/K in E12
The Notion docs ([TokenBasedPga.md L59](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/notion/TokenBasedPga.md#L59), [TokenBased.bc1.md L55](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/notion/TokenBased.bc1.md#L55)) both say:
$$Attention(Q, K, V) \rightarrow Attention(QP, KP, VP)$$

But E12's README and code deliberately filter **only Q and K** — leaving V unfiltered. This is labeled as "Fix 1" and documented, but it's a **divergence from the Notion architecture spec**. The rationale (let the model attend to novel info outside the subspace) is sound, but the Notion docs haven't been updated to reflect this.

---

## Summary Verdict

| Category | Score | Notes |
|---|---|---|
| **Logging** | 7/10 | Good surface-level tracking. Missing gradient norms, SVD rank diagnostics, and SVD-failure visibility. |
| **Correctness** | 7/10 | Core PGA logic is implemented correctly. The `train_single_model` dead parameter, missing `_needs_full_rebuild` init, and `RMSNorm` without scale are code-quality issues. |
| **Compromises** | 8/10 | The single-P-per-head compromise is well-documented. The `.detach()` on query_proj and the Q/K-only divergence from Notion spec are undocumented. |
