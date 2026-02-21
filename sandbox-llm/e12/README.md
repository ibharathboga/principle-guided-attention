# E12: The 4 Architectural Fixes

This experiment implements the major architectural corrections necessary to give Principle-Guided Attention (PGA) a fair test against standard Self-Attention.

## The Model

In `models.py`, we implement `PropPGABuffer`. It incorporates the following 4 fixes over the E10/E11 runs:
1. **Scaled Space**: `N_EMBD` is scaled up to 64 to provide a high enough representational dimension where geometric subspace filtering can actually separate signal from noise.
2. **Q/K Only Filtering**: The projection matrix `P` is only applied to the Queries and Keys. The Values matrix is left unfiltered, ensuring the model can encode and attend to novel information outside of the current subspace.
3. **MLP Query Matching**: The buffer retrieval is queried not by a raw linear projection, but by a 2-layer MLP (`Linear -> RMSNorm -> ReLU -> Linear`). This provides the nonlinear depth necessary to map shallow token embeddings into the space of deep final-layer contextual vectors retrieved from the buffer.
4. **Adaptive Propagative SVD**: We compute Singular Value Decomposition exactly **once per token** (at the start) and propagate that `P` matrix down through all layers. Rather than a hard-coded rank, we use an **Adaptive Energy Threshold** (set to 0.90) that dynamically keeps the top singular vectors until they capture 90% of the variance.

## Logging

Logging is extremely robust in `pipeline.py`. It tracks and outputs:
- **Steps & Times**: Prints every `EVAL_INTERVAL` (100 steps) with rolling ETA, step times (ms), and memory profiling.
- **Store Capacity**: Logs the token vector store size, token diversity, and memory usage live.
- **Validation Loss**: Computes both Train and Val loss every 100 steps, running a 50-batch eval cycle.
- **Plot Generation**: At the end of 5000 steps, prints `e12_combined.png` tracking Train & Val curves against Baseline.
- **JSON & Verdict**: Generates `results.json` and `experiment_report.md` capturing raw loss data, gaps, and an official verdict of who won. 

---

## Compromise: Single `P` Across Heads

There is one notable compromise made in this E12 implementation: **We use a single shared projection matrix `P` across all attention heads.**

In standard Multi-Head Attention, each head projects data into a *different* subspace to look for different types of relationships (e.g., Head 1 tracks syntax, Head 2 tracks sentiment). By applying a single global `P` matrix to all queries and keys across all heads, we are forcing every head to route attention through the exact same geometric subspace. 

**Why we compromised here**: 
Implementing separate SVD calculations for *each head* of *every token* in a batched, parallel-training pipeline would cause a massive combinatorial explosion in computational cost and tensor shapes. To test if the absolute core theory works (Adaptive SVD -> Q/K routing) we settled on a single shared `P`, though this does sacrifice some of the representational diversity that multi-head attention naturally provides.
