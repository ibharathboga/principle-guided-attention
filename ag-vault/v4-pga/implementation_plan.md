# Implementation Plan: v4-pga (JIT Essence Recalculation)

## Problem with Previous Versions
- **v2-pga (Unguided)**: Essence vectors in the buffer became "stale" as the model's embedding weights ($W_E$) updated during training. The stored $v_{old}$ no longer represented the same semantic meaning in the new space $W_E'$, leading to "drift".
- **v3-pga (Guided)**: Used a frozen guide to fix the space, but this prevented the "Principles" from evolving with the model.

## Solution: JIT Recalculation
We will store the **source tokens** of the essence, and re-calculate the essence vector on-the-fly using the **current model's embeddings**.

### Data Flow
1.  **Observation**: Input Chunk -> Current Embeddings -> SVD -> Initial Essence ($e_t$).
2.  **Storage**: Store `{ "tokens": [token_ids], "cached_essence": e_t }` in the Buffer.
3.  **Retrieval**:
    - Query with current essence ($q$).
    - Find top-$k$ matches using `cached_essence` (Approximate search is fine).
    - **Load**: Retrieve the `token_ids` for these matches.
4.  **JIT Re-embedding**:
    - Pass retrieved `token_ids` through the **Current Model**'s embedding layer ($W_E$).
    - $X_{retrieved} = W_E(\text{tokens})$.
5.  **Re-SVD**:
    - Compute SVD on $X_{retrieved}$ -> obtain **Fresh Essence** ($e_{fresh}$).
6.  **Projection**:
    - Stack $e_{fresh}$ vectors -> Principle Matrix $P$.
    - Use $P$ to project $Q, K, V$ in the attention layer.
    - **Soft Projection**: Use residual injection: $x = x + \alpha \cdot (x \cdot P)$ to avoid information bottleneck.

## Components

### 1. [pga_jit.py](NEW)
A new training script based on `pga_torch.py` but with JIT logic.
- **EssenceBuffer**:
    - `add(tokens, vector)`: Stores tokens and the vector.
    - `retrieve(query_vec)`: Returns list of `tokens` (not vectors).
- **JIT Logic**:
    - Inside the training loop, after retrieval:
    - `fresh_embs = model.transformer.wte(retrieved_tokens) + model.transformer.wpe(...)`
    - `fresh_essences = [extract_essence(emb) for emb in fresh_embs]`
    - `P = ComputeProjection(fresh_essences)`

### 2. [microgpt_torch.py](REUSE)
- We will reuse the existing `microgpt_torch.py` which now has `get_embeddings` exposed.
- We might need to ensure `get_embeddings` works for a batch of retrieved chunks efficiently.

### 3. [pipeline.py](UPDATE)
- Update to run `v4-pga` benchmark.
- **Validation**:
    - Split `input.txt` into 90% Train / 10% Val.
    - Compute Validation Loss every 100 steps.
- **Logging**:
    - CSV columns: `step, train_loss, val_loss, essence_norm, projection_mag`.
    - `essence_norm`: Average L2 norm of essence vectors (check for collapse).
    - `projection_mag`: Average difference $\|x - xP\|$ (how much does P change the input?).

### 4. [plot_results.py](NEW)
- Visualizes:
    - Loss Curves (Train vs Val).
    - Essence Statistics (Norm/Variance over time).
    - Projection Impact.

## Verification
- **Metric**: Loss vs Baseline (Shakespeare).
- **Expectation**: JIT essences should provide "cleaner" signals that align with the model's current understanding, potentially beating the baseline.
