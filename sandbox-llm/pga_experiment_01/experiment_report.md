# PGA Experiment Report (Parallel Run)

## Configuration
- Steps: 500
- Parallel Execution: Enabled (2 processes)
- Hardware: CPU (Pure Python autograd constraint)
- Smoothing: Moving Average (Window=25)

## Results (Last 20 steps average)
- **Baseline Final Loss**: 2.4464
- **PGA Final Loss**: 2.4447

## Performance Analysis
The plot now uses a **logarithmic scale** for the Y-axis and includes **smoothed trend lines** to visualize the learning trajectory more clearly. 

## Verdict
PGA improved performance by 0.0017 loss points.