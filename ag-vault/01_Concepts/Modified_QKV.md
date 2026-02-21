# The Modified QKV Process (The Implementation)

In a standard Transformer, $Q$, $K$, and $V$ are created by learned, static weights ($W_Q, W_K, W_V$). In PGA, these weights are **dynamically altered by the Principle Matrix ($P$)**.

## 1. Dynamic Weight Modification
- The system receives the standard weights ($W$) and the Principle Matrix ($P$).
- **Change of Basis**: $W' = W \cdot P$
- This operation effectively rotates the vector space to align with the discovered principles.

## 2. Mathematical Logic
By multiplying by $P$:
- **Suppression**: Any data dimension that **contradicts** the principle (vectors orthogonal or opposing $P$) is mathematically suppressed (zeroed out or dampened).
- **Amplification**: Any data dimension that **aligns** with the principle (vectors parallel to $P$) is amplified.

## 3. Attention Score Calculation
- The Attention Score ($A$) is calculated using the modified weights.
- $A = \text{Softmax}(\frac{(Q \cdot P) \cdot (K \cdot P)^T}{\sqrt{d_k}})$
- The resulting score represents **Logical Relevance** rather than just statistical probability.
