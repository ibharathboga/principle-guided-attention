# E13: Principle Augmentation (True Principles)

## Philosophy: From Passive Filter to Active Agent

In previous runs (E10, E11), PGA was a **Passive Filter** (the "Cheatsheet"). It used SVD to identify a "topic" and hid noise. This limited the model.

**E13** implements **Active Augmentation**. We no longer just hide noise; we extract the core geometric Directions (Singular Vectors) from history and turn them into **Virtual Tokens** (Principle Tokens).

---

## 1. High-Level Comparison

| Feature | Passive Filtering (Old) | Active Augmentation (New) |
| :--- | :--- | :--- |
| **SVD Usage** | Mask matrix ($P$). | Basis Vectors ($V$) as Tokens. |
| **Action** | Blocks "Off-Topic" noise. | Provides "On-Topic" abstractions. |
| **Model Benefit** | Faster early convergence. | Higher final capacity/intelligence. |
| **Analogy** | Wearing earplugs to focus. | Having a mentor whisper advice in your ear. |

---

## 2. Low-Level Mechanics (The Math)

The following pseudocode outlines how the Attention Mechanism is modified in `models.py` to support Virtual Tokens.

```python
# A. Principle distilled via SVD from Buffer
# principles: Matrix [4, 64] (4 core concepts x 64 dimensions)
principles = SVD_Basis(Buffer_Context)[:4]

# B. Transformation through Layer Weights
# Principles are translated into the "language" of the current layer.
k_p = principles @ W_k  # Virtual Keys
v_p = principles @ W_v  # Virtual Values

# C. Extended Attention (Competition)
# The current word 'q' matches against Word Keys AND Principle Keys.
scores_local = q @ k_local  # Dot product with previous words
scores_p = q @ k_p          # Dot product with abstract principles

# Combine and Softmax
# The model decides: "Do I look at the text, or the principle?"
attention_weights = softmax(concat([scores_p, scores_local]))

# D. Final Injection
# The output is a mix of raw data and historical abstractions.
output = attention_weights @ concat([v_p, v_local])
```

## 3. Why This Works

1.  **Dynamic Routing**: The model isn't forced to use the principles. If the SVD finds junk, the attention weight for those virtual tokens will drop to 0.
2.  **Abstraction over Memorization**: By capping at 4 principles, the model cannot simply "copy" the buffer. It must learn to navigate the distilled geometric logic.
3.  **Recovering Lost Signal**: Unlike the $P$-matrix filter, which could only discard information, the $V$-vectors can *inject* information from the buffer that might have been missing from the local context window.
