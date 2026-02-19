import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
import sys
import os

# Helper to load modules dynamically
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

print("Loading Baseline...")
baseline = load_module("microgpt_baseline.py", "baseline")

print("Loading PGA...")
pga = load_module("microgpt_pga.py", "pga")

STEPS = 300

print(f"Starting Baseline Training ({STEPS} steps)...")
baseline_losses = baseline.train(steps=STEPS)

print(f"Starting PGA Training ({STEPS} steps)...")
pga_losses = pga.train(steps=STEPS)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(baseline_losses, label='Baseline', alpha=0.7)
plt.plot(pga_losses, label='PGA', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('MicroGPT: Baseline vs PGA Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plot.png')
print("Plot saved to comparison_plot.png")

# Report
report_content = f"""# PGA Experiment Report

## Configuration
- Steps: {STEPS}
- Embedding Dim: {baseline.n_embd}
- Heads: {baseline.n_head}
- Block Size: {baseline.block_size}

## Results
- Final Baseline Loss: {baseline_losses[-1]:.4f}
- Final PGA Loss: {pga_losses[-1]:.4f}

## Verdict
"""

if pga_losses[-1] < baseline_losses[-1]:
    report_content += "PGA outperformed the baseline."
else:
    report_content += "Baseline outperformed PGA (or equivalent)."

with open("experiment_report.md", "w") as f:
    f.write(report_content)

print("Report saved to experiment_report.md")
