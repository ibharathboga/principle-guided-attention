# Overfitting Analysis (2500 Steps)

## Configuration
- Data: `names.txt` (90/10 Split)
- Steps: 2500
- Eval Interval: 100 steps

## Final Metrics
| Model | Train Loss | Val Loss | Gap (Overfitting) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 2.2533 | 2.3359 | 0.0826 |
| **PGA** | 2.2636 | 2.3367 | 0.0731 |

## Analysis
- If Val Loss increases while Train Loss decreases, the model is overfitting.
- Comparison of Gap shows which model generalizes better.

## Verdict
Baseline achieved lower validation loss.
