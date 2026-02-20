# E10 Configuration Analysis — Compromises & Suggested Tweaks

> **Date**: 2026-02-20 17:49

---

## 1. Current Configuration (from `models.py`)

| Parameter | Baseline | E10-TokenStore | Notes |
|---|---|---|---|
| `N_LAYER` | 5 | 5 | Same |
| `N_EMBD` | 16 | 16 | **Very small** — limits expressiveness |
| `N_HEAD` | 4 | 4 | HEAD_DIM = 4 (D/H) |
| `BLOCK_SIZE` | 16 | 16 | Context window |
| `RANK` | — | 8 | SVD truncation rank |
| `RETRIEVE_K` | — | 8 | Vectors retrieved per token |
| `SIM_RATIO` | — | 0.7 | 70% similarity, 30% recency |
| Store size | — | Unbounded | Grows ~32 vectors/step |
| Params | 209,664 | 209,920 | Only +256 from `query_proj` |
| LR | 0.005 | 0.005 | Same |
| Steps | 5000 | 5000 | Same |

---

## 2. Compromises vs the Notion Spec

After cross-referencing `models.py` against all five notion documents, here are the **deviations and compromises** in the E10 implementation:

### ⚠️ Compromise 1: Rank = D/2 — Too Aggressive

The spec says P should be a **low-rank** projection that filters noise. But:

- `RANK = 8`, `N_EMBD = 16` → **rank is exactly D/2**
- The projection matrix P preserves 50% of the space
- This is barely filtering anything — P is almost full-rank

> The notion doc (`TokenBasedPga.properties.md`) says P should be **rank-deficient** with a large null space. At rank 8/16, the null space is only 8 dimensions — hardly a strong filter.

**Fix**: Try `RANK = 4` (25% of D) or even `RANK = 2` (12.5%). The spec intended an aggressive filter that enforces epistemic consistency.

---

### ⚠️ Compromise 2: Query Source — Using Raw Embeddings, Not Layer Outputs

The spec (`TokenBasedPga.md`, Stage 3) says:

> "Before processing a new token $x_t$, the model **queries its memory** to understand the current contextual regime."

The E10 code queries the store using `query_proj(embed(token))` — i.e. a **linear projection of the raw embedding**. But the spec's notion of "observation" (from `TokenBasedPga.view.md`) treats each token as an **Essence Vector** that captures the "moment." The raw embedding captures nothing about context — it's just a lookup table entry.

**What's stored** vs **what's queried**:
- **Stored**: Final-layer outputs (rich, contextual) ✅
- **Queried**: Projected raw embeddings (context-free) ❌

This mismatch means the retrieval is comparing a **shallow** query against **deep** stored representations. Cosine similarity between these spaces may not be meaningful.

**Fix**: Query with the *previous step's* final-layer output, or add a small MLP to bridge the gap.

---

### ⚠️ Compromise 3: No Energy Threshold — Fixed Rank Truncation

The notion doc (`TokenBasedPga.view.md` line 91) explicitly asks:

> "Would you like to define the heuristic for truncation? We could base it on a **fixed energy threshold** (e.g., preserving 95% of the singular value sum) or a hard-coded rank."

E10 uses **hard-coded rank** (RANK=8). This means:
- When the context matrix has clear structure (few dominant singular values), we're keeping noise dimensions
- When the context is truly high-rank, we might be throwing away signal

**Fix**: Use an adaptive energy threshold: keep enough singular vectors to capture 90-95% of total singular value mass. This would let P self-adjust.

---

### ⚠️ Compromise 4: Unbounded Store → Speed Collapse

The terminal output shows E10's step time growing from **17ms → 175ms** over 5000 steps as the store bloats from 1.6K → 161K vectors (9.86MB). The notion spec mentions a "FIFO or retrieval-based buffer" — it never intended an **unbounded** store.

**Why this hurts**:
- `_rebuild_cache()` re-stacks ALL vectors every time the cache is dirty
- Global cosine similarity scans the **entire** store (O(N) per query, N growing)
- The Python for-loop in `store_vectors()` appends one-by-one

**Fix**: Cap at 10K–20K vectors with LRU or importance-based eviction. Or use approximate nearest neighbor (FAISS/Annoy).

---

### ⚠️ Compromise 5: P Applied to Q, K, AND V

The original spec (`TokenBasedPga.md`, Stage 6) says:

> $Attention(Q, K, V) \rightarrow Attention(QP, KP, VP)$

E10 does exactly this (line 306-308 in `models.py`). However, filtering **V** is a strong constraint. The spec discusses this as "epistemic rigor," but in practice it means the model can only *output* information within the principal subspace. This prevents the model from learning representations that go *beyond* what the buffer has seen.

**Fix**: Try applying P to only Q and K (attention routing) while leaving V unfiltered (information content). This is a common modification in subspace attention literature.

---

### ⚠️ Compromise 6: No Per-Head Principles

The spec doesn't explicitly address multi-head attention. E10 applies the **same P** matrix to all heads:

```python
q = torch.matmul(q.unsqueeze(2), P).squeeze(2)  # Same P for all heads
```

This forces all 4 attention heads to focus on the same subspace, eliminating the multi-head diversity that makes transformers work.

**Fix**: Compute per-head P matrices (4 separate rank-2 projections instead of 1 rank-8 projection). Same total compute, much more expressive.

---

## 3. Configuration Tweaks (Ordered by Expected Impact)

### Tier 1: High Impact, Easy to Test

| Tweak | Current | Suggested | Why |
|---|---|---|---|
| **Reduce RANK** | 8 | 3–4 | Stronger geometric filter per spec |
| **Cap store size** | Unbounded | 10,000–20,000 | Stop speed collapse, force quality over quantity |
| **P on Q,K only** | Q, K, V | Q, K only | Let V carry full information |

### Tier 2: Medium Impact, Moderate Effort

| Tweak | Current | Suggested | Why |
|---|---|---|---|
| **Adaptive rank** | Fixed 8 | Energy threshold 90% | Self-adjusting filter strength |
| **Query with prev output** | Raw embedding | Previous step's final output | Match query space to stored space |
| **Per-head P** | 1 shared P | 4 separate P | Preserve head diversity |

### Tier 3: Architecture-Level

| Tweak | Current | Suggested | Why |
|---|---|---|---|
| **Scale N_EMBD** | 16 | 32–64 | More room for subspace filtering to help |
| **Increase BLOCK_SIZE** | 16 | 32–64 | More tokens per context |
| **LR schedule** | Constant 0.005 | Cosine/warmup | Better convergence |

---

## 4. The Core Tension

The fundamental tension in E10 is:

> **The model is too small for subspace filtering to shine.**

At `N_EMBD=16` and `RANK=8`, the projection matrix P preserves 50% of the space. That's barely filtering. The baseline already has such a small representational space that every dimension carries signal — there's no "noise" for PGA to remove.

**PGA's advantage will grow with model scale.** At `N_EMBD=64` and `RANK=8`, P would preserve only 12.5% of the space — a much more aggressive filter that can genuinely remove irrelevant dimensions while preserving the structural essence.

---

## 5. Recommended Next Experiment (E11)

A targeted experiment to test the top 3 tweaks:

```
E11 Config:
  - N_EMBD: 32 (2× current)
  - RANK: 4 (aggressive filter — 12.5% of D)
  - Store cap: 15,000 vectors
  - P applied to: Q and K only (V unfiltered)
  - Steps: 5000
  - Everything else unchanged
```

This preserves comparability with E10 while testing whether a stronger geometric filter + unconstrained value stream + capped store can beat the baseline on val loss.
