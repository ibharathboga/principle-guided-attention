# E12 — Final Result Report

This is a consolidated record of the E12 experiment results.

## Final Metrics
- **Baseline Val Loss**: 2.6246
- **PropPGA Val Loss**: 2.7778
- **Winning Model**: Baseline

## Key Findings
1. **Adaptive SVD Success**: The rank stabilized at ~10 without any SVD failures over 5000 steps.
2. **Gradient Health**: Live monitoring proved that gradients are not the issue (norms ~2.5).
3. **Architecture Gap**: The gap between Baseline and PGA is likely due to the global P matrix compromise and the detached query projection.

Full analysis available in the workspace artifacts.
