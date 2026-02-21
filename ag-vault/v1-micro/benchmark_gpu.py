"""
GPU Parallel Benchmark: PyTorch Baseline vs PGA
"""

import subprocess
import os
import time
import sys
import re

def parse_logs(log_output):
    # Extract final loss
    if not log_output:
        return 0.0, 0.0, []
        
    lines = log_output.split('\n')
    
    final_train_loss = 0.0
    final_val_loss = 0.0
    
    # Regex for: Step 100/500 | Train Loss: 2.1234 | Val Loss: 2.3456
    # Note: Val Loss might not be in every line
    
    for line in lines:
        if "Train Loss:" in line:
            parts = line.split("|")
            for part in parts:
                if "Train Loss" in part:
                    try:
                        final_train_loss = float(part.split(":")[1].strip())
                    except: pass
                if "Val Loss" in part:
                    try:
                        final_val_loss = float(part.split(":")[1].strip())
                    except: pass
                    
    # Extract Samples
    samples = []
    capture = False
    for line in lines:
        if "Inference" in line:
            capture = True
            continue
        if capture and line.strip().startswith("Sample"):
            samples.append(line.strip())
            
    return final_train_loss, final_val_loss, samples

def main():
    steps = 500
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
        
    print("----------------------------------------------------------------")
    print(f"GPU Parallel Benchmark: Baseline vs PGA (Real SVD) ({steps} steps)")
    print("----------------------------------------------------------------")
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    base_log = "torch_baseline.log"
    pga_log = "torch_pga.log"
    
    # Start processes in parallel
    print(f"Starting Baseline PyTorch -> {base_log}...")
    base_file = open(base_log, "w")
    base_proc = subprocess.Popen(
        ["python", "-u", "baseline_torch.py", str(steps)],
        stdout=base_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    print(f"Starting PGA PyTorch (SVD) -> {pga_log}...")
    pga_file = open(pga_log, "w")
    pga_proc = subprocess.Popen(
        ["python", "-u", "pga_torch.py", str(steps)],
        stdout=pga_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    start_time = time.time()
    
    print("Both models running... Waiting for completion.")
    
    base_done = False
    pga_done = False
    
    while not base_done or not pga_done:
        if not base_done and base_proc.poll() is not None:
            base_done = True
            base_end_time = time.time()
            print(f"Baseline finished in {base_end_time - start_time:.2f}s")
            
        if not pga_done and pga_proc.poll() is not None:
            pga_done = True
            pga_end_time = time.time()
            print(f"PGA finished in {pga_end_time - start_time:.2f}s")
            
        time.sleep(1)
        
    base_file.close()
    pga_file.close()
    
    print("\nProcessing Results...")
    
    with open(base_log, "r") as f: base_out = f.read()
    with open(pga_log, "r") as f: pga_out = f.read()
    
    base_train, base_val, base_samples = parse_logs(base_out)
    pga_train, pga_val, pga_samples = parse_logs(pga_out)
    
    report = f"""
BENCHMARK RESULTS (PyTorch {steps} steps)
=======================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Baseline (PyTorch)
------------------
Execution Time: {base_end_time - start_time:.2f}s
Final Train Loss: {base_train:.4f}
Final Val Loss:   {base_val:.4f}
Samples:
{chr(10).join(base_samples[:3])} ...

PGA (PyTorch + SVD)
-------------------
Execution Time: {pga_end_time - start_time:.2f}s
Final Train Loss: {pga_train:.4f}
Final Val Loss:   {pga_val:.4f}
Samples:
{chr(10).join(pga_samples[:3])} ...

COMPARISON
----------
Train Loss Delta: {pga_train - base_train:.4f}
Val Loss Delta:   {pga_val - base_val:.4f}

"""
    print(report)
    with open("../comparison_results_torch.txt", "w") as f:
        f.write(report)
    print("Results saved to comparison_results_torch.txt")

if __name__ == "__main__":
    main()
