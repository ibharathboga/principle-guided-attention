# Principle-Guided Attention (PGA): Architecture & Workflow

## Core Concept

**Principle-Guided Attention (PGA)** is a mechanism that transforms standard Self-Attention into **Subspace-Regularized Attention**. Instead of attending to all tokens based merely on dot-product similarity, PGA projects queries and keys onto a **Principal Subspace** defined by the statistical "Essence" of the current context.

This acts as a geometric filter: it preserves signal that aligns with the "World Model" (Principle Basis) and discards orthogonal noise.

---

## The Workflow

### 1. The Observation (Essence of the Moment)

Every token embedding $x_t$ is treated as an **Essence Vector** of that specific moment.

- **Input**: Token sequence.
- **Output**: Stream of essence vectors in $\mathbb{R}^d$.

### 2. Epistemic Memory (The Buffer)

We maintain an **Observation Buffer** (System 2 Memory) that stores recent and historical embeddings.

- **Function**: Acts as a reservoir of "valid states" the model has encountered.
- **Structure**: A FIFO or retrieval-based buffer of size $N \times d$.

### 3. Contextual Retrieval

Before processing a new token $x_t$, the model queries its memory to understand the current "contextual regime."

- **Query**: The current vector $x_t$.
- **Retrieval**: The system fetches $k$ vectors from the buffer that are:
  1.  **Similar** (Long-term semantic matches).
  2.  **Recent** (Short-term local context).

### 4. Principle Extraction (SVD on the Stack)

We stack these Essence Vectors to form the **Principle Tensor** (The Stack) $X_{stack}$.
_This stack represents the "World" as we currently know it._

We then distill this stack into a clean geometric basis using **Singular Value Decomposition (SVD)**.

$$ X\_{stack} = U \Sigma V^T $$

- **Input**: Principle Tensor (Stack of $k$ vectors).
- **Output (Basis)**: The top components of $V^T$. These are the "True Directions" of the subspace.

### 5. Subspace Projection (The Filter)

We construct a **Projection Matrix** $P$ from these Essence Vectors.
$$P = V_{top}^T V_{top}$$

This matrix projects any vector onto the subspace spanned by the Essence.

### 6. Attention Gating

Finally, the standard Transformer Attention mechanism is modified. The Query ($Q$), Key ($K$), and Value ($V$) projections are filtered through $P$ before interaction.

$$Attention(Q, K, V) \rightarrow Attention(QP, KP, VP)$$

- **Result**: The attention mechanism effectively becomes "blind" to information that does not lie within the Principal Subspace. It attends only to the _principled_ components of the signal.

---

## Summary

| Stage          | Operation         | Purpose                                |
| :------------- | :---------------- | :------------------------------------- |
| **1. Observe** | $x_t$             | Embed raw token.                       |
| **2. Recall**  | $Retrieve(x_t)$   | Fetch relevant history.                |
| **3. Distill** | $SVD(Stack)$      | Extract geometric "Truth" (Subspace).  |
| **4. Filter**  | $x \cdot P$       | Remove Out-of-Domain noise.            |
| **5. Attend**  | $Softmax(Q K^T)V$ | Standard attention on purified signal. |
