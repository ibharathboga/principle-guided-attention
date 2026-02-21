# PGA vs Baseline: Consolidated Benchmark Report (v2-pga)

## Summary
The PyTorch implementations of both models were successfully run for 1000 steps on `input.txt` (names dataset). 
The **Principle-Guided Attention (PGA)** model (Chunk-Based Essence) was compared against the **Baseline** MicroGPT.

## Results (1000 Steps)

| Metric | Baseline (Torch) | PGA (Chunk Essence) | Difference |
| :--- | :--- | :--- | :--- |
| **Final Loss** | **2.7639** | 2.8516 | +0.0877 (Worse) |
| **Training Time** | **4.52s** | 8.07s | +3.55s (Slower) |

## Analysis
1.  **Performance Overfit**: Both models reduced loss from ~3.3 to ~2.8. Baseline converged slightly faster.
2.  **Overhead**: PGA is roughly **2x slower** due to:
    -   SVD extraction per step.
    -   Buffer retrieval per step.
    -   Buffer append (writing to disk).
3.  **Accuracy**: PGA's loss is slightly higher. This could be due to:
    -   **Context Noise**: Retrieving irrelevant essences early in training (when embeddings aren't good) might pollute the attention mechanism.
    -   **Projection Constraint**: The subspace projection might be too restrictive for this small dataset/model size.
    -   **Chunk Definition**: We treated each tiny sequence (16 chars) as a chunk. This might be too granular.

## Logs & Inference
### Baseline Generation
```
salenaundaau
karidahaela
jisaheaato
aveneronieannl
```

### PGA Generation
```
salenaundaau
jaridahalela
jisaaeaenneno
bvererialana
kadyinyeyavh
```

## Next Steps
To improve PGA:
1.  **Warmup**: Disable Essence extraction/retrieval for the first N steps to let embeddings stabilize.
2.  **Larger Buffer**: Run for more steps to build a meaningful history.
3.  **Tuning K**: Retrieve fewer or more neighbors (currently k=5).
