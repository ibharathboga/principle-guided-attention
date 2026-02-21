# Principle-Guided Attention (PGA) Overview

## Core Philosophy
The central thesis of the **Principle-Guided Attention (PGA)** architecture is that intelligence is not just about pattern matching, but about **Principle Discovery**.
- **Observations** are incomplete snapshots of reality.
- **Principles** are the invariant laws governing those observations.
- **Attention** should be a function of these principles, not just statistical correlation.

In this architecture, the **Principle** acts as a **Projection Matrix** that reshapes the Attention space, filtering out noise and amplifying signal based on logical relevance.

## Architecture Workflow

1.  **Input & Encoding**:
    -   Natural Language Query ($Q$) is vectorized into a high-dimensional state vector ($V_Q$).
    -   Entropy Check calculates "missing information."

2.  **Principle Extraction**:
    -   **Semantic Retrieval**: Pull relevant tensors ($T$) from the Observation Buffer.
    -   **Invariant Discovery**: Find the common function $f(x) = y$.
    -   **Principle Matrix ($P$)**: The discovered invariant is converted into a Transformation Matrix.

3.  **Modified QKV (The Implementation)**:
    -   Standard weights ($W$) are dynamically modified by $P$.
    -   $W' = W \cdot P$
    -   Attention scores represent **Logical Relevance**.

4.  **Integration & Decoding**:
    -   **Synthesized Essence Vector ($E$)**: The output of the attention cycle.
    -   **Verification**: Consistency check against the original Principle ($P$).
    -   **Decoding**: Project back to semantic space.

## Key Differentiator
Unlike standard LLMs where "beauty" might be associated with "bridge" due to training data frequency, PGA uses the **Principle of Gravity** to ensure the attention mechanism focused on "load," "tension," and "mass." Irrelevant concepts are mathematically suppressed.
