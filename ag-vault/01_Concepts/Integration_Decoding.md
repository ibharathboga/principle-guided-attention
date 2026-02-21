# Integration & Decoding

## 1. Output Generation
- The output of the Attention cycle is a **Synthesized Essence Vector ($E$)**.
- This vector represents the core meaning distilled through the Principle Filter.

## 2. Verification (Consistency Check)
- **Check**: Is $E$ consistent with the original Principle ($P$)?
- **Logic**: If $E \cdot P \approx 0$ (or below a threshold), it implies the output does not align with the governing principle.
- **Action**: Trigger a "Recalculation" (Recursive Logic) if a violation is detected. The system may need to refine $P$ or re-evaluate the input.

## 3. Formal Decoding
- **Projection**: $E$ is projected back into the semantic space (vocabulary) to generate the final output.
- **Constraint**: The decoder is constrained to use variables identified in $E$, ensuring the output "oozes clarity" and avoids irrelevant noise.
