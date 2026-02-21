# PGA vs Baseline: Guided Benchmark Report (v3-pga)

## Summary
The **Guided Principle-Guided Attention (PGA)** model was benchmarked against the **Baseline** MicroGPT for 1000 steps.
- **Guided Mechanism**: The PGA model used a frozen Baseline model (pre-trained 1k steps) to generate stable "Essence Vectors" for retrieval.
- **Goal**: Test if fixing the "Cold Start" problem (random essences at start) improves PGA performance.

## Results (1000 Steps)

| Metric | Baseline | PGA Guided | Difference |
| :--- | :--- | :--- | :--- |
| **Final Loss** | **2.7639** | 2.8537 | +0.0898 (Worse) |
| **Training Time** | **1.64s** | 3.30s | +1.66s (Slower) |

## Analysis
1.  **Hypothesis Invalidated**: Providing stable, high-quality essences from Step 0 **did not** significantly improve PGA performance compared to the unguided version (v2-pga loss was ~2.8516).
2.  **Implication**: The performance gap is likely **NOT** due to "polluted" buffers or random embeddings at the start.
3.  **Root Cause Candidates**:
    -   **Hard Projection**: Forcing $Q, K, V$ into a low-rank subspace ($k=5$) might be too destructive for a tiny model ($d=16$) where every dimension counts.
    -   **Chunk Granularity**: 16-token chunks might be too short to have a meaningful "Essence".
    -   **Retrieval Relevance**: The retrieved chunks might simply not be helpful for next-token prediction in this specific dataset (Names).

## Visual Comparison
Results stored in `comparison_log.csv`.

## Next Steps
-   **Soft PGA**: Instead of hard projection $x \cdot P$, try $x \cdot (I + \alpha P)$ to allow residual information.
-   **Larger Model**: PGA might only shine when $d$ is large (e.g., 768) and the "valid" subspace is truly sparse.
-   **Different Dataset**: Try Shakespeare where "context/topic" changes more meaningfully than in a list of names.

## Shakespeare Benchmark (v3-pga)
Running the same pipeline on `input_shakespeare.txt`.

| Metric | Baseline | PGA Guided | Difference |
| :--- | :--- | :--- | :--- |
| **Final Loss** | **2.1622** | 2.2394 | +0.0772 (Worse) |

- **Observation**: The gap persists even with a richer dataset. The PGA model generates somewhat more garbled text, possibly due to the subspace projection filtering out locally important syntax info.
- **Conclusion**: Hard projection ($x \cdot P$) is likely too aggressive for this scale/architecture.
- **Next Step Recommendation**: Implement **Soft PGA** (Residual Injection).

