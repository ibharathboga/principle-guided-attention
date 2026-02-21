# E13 — Principle Augmentation (Virtual Tokens)

> **Date**: 2026-02-21 09:52

## Configuration

| Setting | Value |
|---|---|
| Dataset | enwik8 (100MB) |
| Layers | 5 |
| n_embd | 64 |
| Steps | 2500 |
| LR | 0.005 |
| SVD Energy Threshold | 0.9 |

## Fixes Implemented
1. **Scaled Space**: `N_EMBD=64` to provide geometric room for subspace filtering.
2. **Q/K Only**: P matrix applied only to queries and keys, Values unfiltered.
3. **MLP Query Proj**: Deep projection layer to match embedding space to context vector depth.
4. **Adaptive Propagative SVD**: SVD computed exactly once per forward pass, using dynamic cumulative energy summation instead of hard-coded rank constraints.

## Results

| Metric | Baseline | Prop PGA Augment |
|---|---|---|
| Train Loss | 2.7203 | 2.7296 |
| Val Loss | 2.9130 | 2.7885 |
| Gap | 0.1927 | 0.0588 |
| Step Time | 18.5ms | 105.4ms |

## Architecture Notes
- Principles used: 16 max retrieved, flattened to ~10.7 virtual tokens.

## Verdict

**Val Loss Winner: PropPGAAugment** (2.7885)
**Generalization Winner: PropPGAAugment** (gap: 0.0588)
