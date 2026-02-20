# EBC1 Experiment Walkthrough

## What Was Done

Ran the **Baseline MicroGPT vs PGA MicroGPT** experiment from `sandbox-llm/ebc1/` on **Shakespeare** data (40K lines) for **500 steps**.

### Fixes Applied

1. Copied `shakespeare.txt` from `e5-token/` into `ebc1/`
2. Fixed log format in both [microgpt_baseline.py](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/microgpt_baseline.py) and [microgpt_pga.py](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/microgpt_pga.py) — consistent `Step X/Y | Train Loss: Z` format with newlines every 10 steps
3. Created new [run_experiment.py](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/run_experiment.py) with parallel execution, log parsing, and matplotlib plotting

### How to Run

```bash
cd c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1
python run_experiment.py --steps 500        # or any step count
python run_experiment.py --steps 1000       # for longer training
```

## Results (500 steps, Shakespeare)

| Metric | Baseline | PGA | Winner |
|--------|----------|-----|--------|
| Execution Time | 254s | 368s | Baseline |
| Final Train Loss | **2.1446** | 2.1571 | Baseline (≈tie) |
| Final Val Loss | 2.6567 | **2.4502** | **PGA** |
| Val-Train Gap | 0.5121 | 0.2931 | **PGA** |

> [!IMPORTANT]
> PGA shows **better generalization**: 0.21 lower validation loss despite nearly identical training loss. Its smaller val-train gap (0.29 vs 0.51) suggests less overfitting.

## Loss Plot

![Loss plot showing train and validation loss curves for Baseline vs PGA MicroGPT](C:/Users/ibhar/.gemini/antigravity/brain/d911148b-65fb-4cf5-96b3-f8055db6cb22/loss_plot.png)

**Key observations:**
- Both training loss curves track closely and are noisy (expected with tiny model)
- Validation loss shows clear separation: PGA consistently lower from step ~100 onward
- PGA validation loss continues to drop at step 500, suggesting it could improve further with more steps

## Generated Text Samples

````carousel
**Baseline Samples:**
```
Sample 1: RONIUV:
Sample 2: Mhol har you tha
Sample 3: Ane s he shas hy
Sample 4: Whe irourt t lil
Sample 5: I lar sa theee t
```
<!-- slide -->
**PGA Samples:**
```
sample  1: PO:
sample  2: Wh s he f mo t e
sample  3: Mhe he hado we s
sample  4: And t an her tha
sample  5: Yer the the be t
```
````

Both models produce recognizable English-like fragments at 500 steps. PGA samples show slightly more coherent word beginnings ("And", "Yer the the").

## Output Files

- [loss_plot.png](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/loss_plot.png) — Steps vs Loss plot
- [results.txt](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/results.txt) — Text summary
- [baseline.log](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/baseline.log) / [pga.log](file:///c:/Users/ibhar/OneDrive/Desktop/attention-lab/sandbox-llm/ebc1/pga.log) — Raw training logs
