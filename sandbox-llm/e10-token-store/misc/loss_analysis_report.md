# E10 — Which Loss Matters More?

> **Date**: 2026-02-20 17:44

## Answer: Validation Loss

**Validation loss matters more** — almost always.

- **Training loss** tells you how well the model memorizes data it has already seen. You can always drive it lower by making the model bigger or training longer, but that doesn't mean it's *learning* anything useful.
- **Validation loss** tells you how well the model performs on **unseen data** — which is the whole point.

## Who's Really Winning?

With that lens, **Baseline (val 2.74)** is the real winner. Its lower validation loss means it actually predicts unseen text better than E9 or E10.

## The Generalization Gap

The gap (train − val) is a **secondary signal** — it tells you about *future potential*:

| Scenario | What it means |
|---|---|
| **Small positive gap** (E10: +0.025) | Model generalizes well, not overfitting — could keep improving with more training |
| **Large negative gap** (Baseline: −0.13) | Val is better than train (noisy training batches), but model may be near its ceiling |

E10-TokenStore's tighter gap suggests it has **more room to grow** if trained longer or scaled up. But *right now*, at step 5000, Baseline's val loss is simply better.

## Final Metrics

| Metric | Baseline | E9-Buffer | E10-TokenStore |
|---|---|---|---|
| **Train Loss** | 2.8760 | 2.8543 | **2.8313** ✅ |
| **Val Loss** | **2.7419** ✅ | 2.8822 | 2.8563 |
| **Train–Val Gap** | −0.1341 | +0.0278 | **+0.0249** ✅ |
| Step Time | **10.9 ms** | 14.7 ms | 70.2 ms |

## Bottom Line

> **Val loss is what you ship on.** Generalization gap tells you who might win the *next* race.

Baseline outperforms on the metric that matters, but E10's architecture has healthier learning dynamics — it just needs more steps (or a speed fix) to realize that potential.
