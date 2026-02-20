# E9 — RETRO vs Feedback Buffer — Results

## Results (enwik8, 5000 steps)

| Model | Train | Val | Gap | Speed |
|:---|:---|:---|:---|:---|
| **Baseline** | 2.8760 | 2.7419 | -0.1341 | 16.3ms |
| **Prop-PGA (Window)** | 2.8486 | **2.7380** ✅ | -0.1106 | 20.8ms |
| **Prop-PGA+Buf (Final)** | 2.8543 | 2.8822 | +0.0278 | 25.6ms |
| **Raw-Obs-Buf** | 2.8591 | 2.7690 | -0.0900 | 25.5ms |
| **RETRO-Buf (Frozen)** | 2.8467 | 2.7896 | -0.0571 | 18.7ms |

## Rankings

**By Val Loss** (lower = better):
1. 🥇 Prop-PGA (Window) — 2.7380
2. 🥈 Baseline — 2.7419
3. 🥉 Raw-Obs-Buf — 2.7690
4. RETRO-Buf — 2.7896
5. Prop-PGA+Buf — 2.8822

**By Generalization** (gap closest to 0):
1. 🥇 Prop-PGA+Buf — +0.0278 (but overfitting!)
2. 🥈 RETRO-Buf — -0.0571
3. 🥉 Raw-Obs-Buf — -0.0900

## Key Findings

1. **Window-based PGA wins val loss** — The simplest SVD source (sliding window of adjacent tokens) generalizes best. No buffer needed.

2. **Raw-Obs-Buf > Prop-PGA+Buf** — Storing raw embeddings (same space for query AND keys) performs **much better** than storing final outputs with a `query_proj` bridge. Val: 2.7690 vs 2.8822. The vector space consistency hypothesis is confirmed.

3. **RETRO is middle-of-pack** — Frozen baseline outputs in buffer perform OK (2.7896) but don't beat the simpler approaches. The frozen vectors from a converged baseline are coherent but may not be maximally relevant to the PGA model's own learned representations.

4. **Feedback with raw observations is the best buffer strategy** — Raw-Obs-Buf is the only buffer model that comes close to the non-buffer approaches, validating the spec's philosophy: *"the principle is a property of the observation."*

## Conclusion

The buffer adds complexity but doesn't beat the simple sliding-window approach at this model/data scale. The **ranking of buffer strategies** is clear: Raw > RETRO > Final outputs. If a buffer is used, it should store observations, not thoughts.
