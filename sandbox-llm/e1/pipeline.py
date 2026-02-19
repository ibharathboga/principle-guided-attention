import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
import sys
import multiprocessing
import numpy as np

# Helper to load modules dynamically
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_baseline(steps):
    print(f"Starting Baseline ({steps} steps)...")
    try:
        baseline = load_module("microgpt_baseline.py", "baseline")
        losses = baseline.train(steps=steps)
        print("Baseline finished.")
        return losses
    except Exception as e:
        print(f"Baseline failed: {e}")
        return []

def run_pga(steps):
    print(f"Starting PGA ({steps} steps)...")
    try:
        pga = load_module("microgpt_pga.py", "pga")
        losses = pga.train(steps=steps)
        print("PGA finished.")
        return losses
    except Exception as e:
        print(f"PGA failed: {e}")
        return []

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()

if __name__ == "__main__":
    STEPS = 1000
    
    # Run in parallel
    with multiprocessing.Pool(processes=2) as pool:
        baseline_future = pool.apply_async(run_baseline, (STEPS,))
        pga_future = pool.apply_async(run_pga, (STEPS,))
        
        baseline_losses = baseline_future.get()
        pga_losses = pga_future.get()

    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Raw data (faint)
    plt.plot(baseline_losses, color='blue', alpha=0.15, label='Baseline (Raw)')
    plt.plot(pga_losses, color='red', alpha=0.15, label='PGA (Raw)')
    
    # Smoothed data (bold)
    # Using moving average for better trend visibility
    baseline_smooth = moving_average(baseline_losses, window_size=25)
    pga_smooth = moving_average(pga_losses, window_size=25)
    
    plt.plot(baseline_smooth, color='blue', linewidth=2, label='Baseline (Smoothed)')
    plt.plot(pga_smooth, color='red', linewidth=2, label='PGA (Smoothed)')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.title(f'MicroGPT: Baseline vs PGA Training Loss ({STEPS} steps)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.yscale('log') # Log scale as requested for better visibility of small differences
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300)
    print("Plot saved to comparison_plot.png")

    # Metrics
    final_baseline = np.mean(baseline_losses[-20:])
    final_pga = np.mean(pga_losses[-20:])

    # Report
    report_content = f"""# PGA Experiment Report (Parallel Run)

## Configuration
- Steps: {STEPS}
- Parallel Execution: Enabled (2 processes)
- Hardware: CPU (Pure Python autograd constraint)
- Smoothing: Moving Average (Window=25)

## Results (Last 20 steps average)
- **Baseline Final Loss**: {final_baseline:.4f}
- **PGA Final Loss**: {final_pga:.4f}

## Performance Analysis
The plot now uses a **logarithmic scale** for the Y-axis and includes **smoothed trend lines** to visualize the learning trajectory more clearly. 

## Verdict
"""
    if final_pga < final_baseline:
         report_content += f"PGA improved performance by {final_baseline - final_pga:.4f} loss points."
    else:
         report_content += "Baseline matched or outperformed PGA."

    with open("experiment_report.md", "w") as f:
        f.write(report_content)

    print("Report saved to experiment_report.md")
