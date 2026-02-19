# PGA Experiment Report (Parallel Run)

## Configuration
- Steps: 1000
- Parallel Execution: Enabled (2 processes)
- Hardware: CPU (Pure Python autograd constraint)
- Smoothing: Moving Average (Window=25)

## Results (Last 20 steps average)
- **Baseline Final Loss**: 2.3572
- **PGA Final Loss**: 2.3549

## Performance Analysis
The plot now uses a **logarithmic scale** for the Y-axis and includes **smoothed trend lines** to visualize the learning trajectory more clearly. 

## Verdict
PGA improved performance by 0.0023 loss points.