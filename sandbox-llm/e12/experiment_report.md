# E12 — The 4 Architectural Fixes

> **Date**: 2026-02-20 23:52

## Configuration

| Setting | Value |
|---|---|
| Dataset | enwik8 (100MB) |
| Layers | 5 |
| n_embd | 64 |
| Steps | 5000 |
| LR | 0.005 |
| SVD Energy Threshold | 0.9 |

## Fixes Implemented
1. **Scaled Space**: `N_EMBD=64` to provide geometric room for subspace filtering.
2. **Q/K Only**: P matrix applied only to queries and keys, Values unfiltered.
3. **MLP Query Proj**: Deep projection layer to match embedding space to context vector depth.
4. **Adaptive Propagative SVD**: SVD computed exactly once per forward pass, using dynamic cumulative energy summation instead of hard-coded rank constraints.

## Results

| Metric | Baseline | Prop PGA Buffer |
|---|---|---|
| Train Loss | 2.6820 | 2.7104 |
| Val Loss | 2.6246 | 2.7778 |
| Gap | -0.0574 | 0.0674 |
| Step Time | 18.1ms | 200.7ms |

## Verdict

**Val Loss Winner: Baseline** (2.6246)
**Generalization Winner: Baseline** (gap: -0.0574)
