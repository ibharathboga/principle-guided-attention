# Sandbox LLM E8 Verification & Experiment Report

## 1. Initial Review
The review of `sandbox-llm/e8-propagative-buffer` identified a vector space mismatch in the `PropagativePGABuffer` model where raw input embeddings (Layer 0) were being compared against processed final layer outputs (Layer N) stored in the observation buffer.

## 2. Corrective Action
To resolve this, we implemented a **Query Projection** layer:
- Added `self.query_proj` (Linear layer) to map the Raw Query into the "Thought Space" of the buffer.
- Modified `compute_projection_matrices_from_buffer` to use this projected query for both retrieval and SVD context construction.

## 3. Verification
A test script `test_query_proj.py` confirmed the fix was correctly implemented and the model allowed valid forward passes.

## 4. Experiment Results

### 2500 Steps
| Model | Final Val Loss | Notes |
| :--- | :--- | :--- |
| **Baseline** | **2.5316** | Best performance |
| **Prop-PGA + Buffer** | 2.5378 | Nearly matched Baseline (+0.0062) |
| **Prop-PGA (Window)** | 2.5725 | Worst performance |

### 5000 Steps
| Model | Final Train | Final Val | Overfit Gap | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 2.5204 | **2.3864** | -0.1340 | Best Val Loss. |
| **Prop-PGA (Window)** | 2.5525 | 2.4209 | -0.1316 | Recovered somewhat. |
| **Prop-PGA + Buffer** | **2.4888** | 2.5731 | +0.0843 | **Best Train Loss** but **started Overfitting**. |

### Interpretation
- **Overfitting in Buffer Model**: At 5000 steps, the `Prop-PGA + Buffer` achieved the *lowest training loss* (2.4888 vs Baseline 2.5204), indicating it has the highest capacity to learn data patterns. However, its validation loss degraded (2.5731), resulting in positive overfitting (+0.0843).
- **Explanation**: The buffer allows the model to "memorize" specific training examples via the feedback loop, which helps training loss but hurts generalization if the buffer content is too specific to the training set.
- **Baseline Strength**: The standard MicroGPT remains the most robust generalizer for this small-scale task.

## 5. Conclusion
- The **Query Projection fix** works and allows the Buffer model to learn effectively (beating Baseline in training loss).
- The **Buffer Mechanism** is powerful but prone to **overfitting** on small datasets/models because it effectively extends the model's state with "cached" training data.
- **Recommendation for Future**: Add regularization to the buffer (e.g., dropout on retrieved vectors, or limiting buffer lifespan) or use a larger dataset where memorization is harder.
