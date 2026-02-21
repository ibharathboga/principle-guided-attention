# Input & Encoding Layer

## 1. Vectorization
The system receives a natural language query ($Q$).
- **Process**: $Q$ is encoded into a high-dimensional state vector ($V_Q$).
- **Purpose**: To convert semantic meaning into a numerical format suitable for linear algebra operations.

## 2. Entropy Check
Before processing, the system evaluates the quality of the input.
- **Calculate Variance**: The system calculates the variance of $V_Q$.
- **Determine Missing Information**: High entropy or variance indicates ambiguity or lack of specific constraints.
- **Decision Gate**: If entropy is too high, the system may request clarification or default to a broader search strategy before proceeding to Principle Extraction.
