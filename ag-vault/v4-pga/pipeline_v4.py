import subprocess
import sys
import os
import argparse
import pandas as pd
import time

def run_command(cmd, description):
    print(f"\n[Pipeline] Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    start = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    print(f"[Pipeline] Finished: {description} ({time.time() - start:.2f}s)")
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()

    # Files
    baseline_log = "baseline_log.txt"
    pga_jit_log = "pga_jit_log.csv"
    
    # 1. Train Baseline
    cmd_baseline = [sys.executable, "microgpt_torch.py", "--steps", str(args.steps), "--log", baseline_log]
    run_command(cmd_baseline, "Training Baseline")

    # 2. Train JIT PGA
    cmd_pga = [sys.executable, "pga_jit.py", "--steps", str(args.steps), "--log", pga_jit_log]
    run_command(cmd_pga, "Training JIT PGA")
    
    # 3. Plot Results
    cmd_plot = [sys.executable, "plot_results.py", "--baseline", baseline_log, "--pga", pga_jit_log]
    run_command(cmd_plot, "Plotting Results")

if __name__ == "__main__":
    main()
