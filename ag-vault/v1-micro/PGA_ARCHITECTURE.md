# Principle-Guided Attention (PGA): Architecture & Workflow

## Core Concept
**Principle-Guided Attention (PGA)** is a mechanism that transforms standard Self-Attention into **Subspace-Regularized Attention**. Instead of attending to all tokens based merely on dot-product similarity, PGA projects queries and keys onto a **Principal Subspace** defined by the statistical "Essence" of the current context.

This acts as a geometric filter: it preserves signal that aligns with the "World Model" (Principle Basis) and discards orthogonal noise.

---

## The Workflow (Chunk-Based Essence)

### 1. The Observation (Text Chunk)
Instead of treating every token as an isolated observation, we treat a **Text Chunk** (e.g., a sentence or fixed window of $T$ tokens) as a single observation unit.
- **Input**: A sequence of tokens (Chunk).
- **Representation**: An Embedding Matrix $X_{chunk} \in \mathbb{R}^{T \times d}$.

### 2. Essence Extraction (SVD/PCA Compression)
Before storage, we distill the chunk into its core "Principle" or "Essence."
We apply **SVD (Singular Value Decomposition)** or PCA on the chunk's embedding matrix to find its principal direction of variance.

$$ X_{chunk} \xrightarrow{SVD} U \Sigma V^T $$

- **Essence Vector ($e_t$)**: The top right singular vector ($V^T[0]$). This is a $1 \times d$ vector that captures the dominant semantic direction of the entire chunk.
- **Storage**: We store this **1D Essence Vector** in the Observation Buffer, not the raw tokens.

### 3. Epistemic Memory (The Buffer)
The buffer now stores a timeline of these 1D Essence Vectors.
- **Structure**: A database or list of $N$ Essence Vectors ($N \times d$).

### 4. Contextual Retrieval
When a new query chunk arrives, we compute its Essence Vector $e_{query}$ and use it to search the database.
- **Query**: The computed Essence Vector of the current input.
- **Retrieval**: The system fetches top-$k$ similar Essence Vectors from history.

### 5. Principle Tensor Construction
We stack these retrieved Essence Vectors to form the **Principle Matrix** $X_{stack}$.
- **Stack**: $k \times d$ matrix.

### 6. Attention Gating (Subspace Projection)
We compute the **Projection Matrix** $P$ from this stack (using SVD again if needed to orthogonalize, or directly if the retrieved vectors form a good basis).
$$P = V_{stack}^T V_{stack}$$

The attention mechanism projects the current token embeddings onto this subspace:
$$Attention(Q, K, V) \rightarrow Attention(QP, KP, VP)$$

- **Result**: The model attends only to signals that align with the "Essence" of retrieved historical contexts.

---

## Summary
| Stage | Operation | Purpose |
| :--- | :--- | :--- |
| Stages | Operation | Purpose |
| :--- | :--- | :--- |
| **1. Chunk** | $X_{chunk}$ | Segment text into chunks. |
| **2. Extract** | $SVD(X_{chunk}) \to v_1$ | Compress 2D chunk to 1D Essence. |
| **3. Store/Retrieve** | $Buffer \leftrightarrow v_1$ | Store essence, retrieve similar essences. |
| **4. Stack** | $X_{stack}$ | Form Principle Matrix from history. |
| **5. Filter** | $x \cdot P$ | Project attention onto Principle Subspace. |
