"""
EBC1 Experiment Runner: Baseline MicroGPT vs PGA MicroGPT
Runs both models in parallel on Shakespeare data and plots Steps vs Loss.
"""

import subprocess
import os
import sys
import time
import re
import argparse

def parse_loss_history(log_output):
    """
    Parse step-by-step loss from log output.
    Handles both formats:
      Baseline: "Step  100/1000 | Train Loss: 2.1234"
      PGA:      "step  100 / 1000 | train_loss 2.1234 | val_loss 2.3456"
    Returns: list of (step, train_loss) tuples, and list of (step, val_loss) tuples
    """
    train_losses = []
    val_losses = []
    
    for line in log_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Match step number - flexible pattern
        step_match = re.search(r'[Ss]tep\s+(\d+)', line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        
        # Match train loss - flexible pattern
        train_match = re.search(r'[Tt]rain[_ ][Ll]oss[:\s]+([0-9.]+)', line)
        if train_match:
            train_loss = float(train_match.group(1))
            train_losses.append((step, train_loss))
        
        # Match val loss
        val_match = re.search(r'[Vv]al[_ ][Ll]oss[:\s]+([0-9.]+)', line)
        if val_match:
            val_loss = float(val_match.group(1))
            val_losses.append((step, val_loss))
    
    return train_losses, val_losses


def extract_samples(log_output):
    """Extract generated samples from log output."""
    samples = []
    for line in log_output.split('\n'):
        line = line.strip()
        if line.lower().startswith('sample') or line.lower().startswith('sample '):
            samples.append(line)
    return samples


def main():
    parser = argparse.ArgumentParser(description="EBC1 Experiment: Baseline vs PGA")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--data", type=str, default="shakespeare.txt", help="Input data file")
    args = parser.parse_args()
    
    steps = args.steps
    data = args.data
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print(f"  EBC1 EXPERIMENT: Baseline vs PGA MicroGPT ({steps} steps)")
    print(f"  Dataset: {data}")
    print("=" * 70)
    
    # Verify dataset exists
    data_path = os.path.join(cwd, data)
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset '{data}' not found at {data_path}")
        sys.exit(1)
    
    # Log files
    base_log = os.path.join(cwd, "baseline.log")
    pga_log = os.path.join(cwd, "pga.log")
    
    # Start both models in parallel
    print(f"\n[1/3] Starting Baseline MicroGPT -> baseline.log")
    base_file = open(base_log, "w")
    base_proc = subprocess.Popen(
        ["python", "-u", "microgpt_baseline.py", "--steps", str(steps), "--data", data],
        stdout=base_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    print(f"[1/3] Starting PGA MicroGPT -> pga.log")
    pga_file = open(pga_log, "w")
    pga_proc = subprocess.Popen(
        ["python", "-u", "microgpt_pga.py", "--steps", str(steps), "--data", data],
        stdout=pga_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    start_time = time.time()
    print(f"\n[2/3] Both models running in parallel... (started at {time.strftime('%H:%M:%S')})")
    
    base_done = False
    pga_done = False
    base_end = pga_end = 0
    
    while not base_done or not pga_done:
        if not base_done and base_proc.poll() is not None:
            base_done = True
            base_end = time.time()
            elapsed = base_end - start_time
            print(f"  ✓ Baseline finished in {elapsed:.1f}s (exit code: {base_proc.returncode})")
        
        if not pga_done and pga_proc.poll() is not None:
            pga_done = True
            pga_end = time.time()
            elapsed = pga_end - start_time
            print(f"  ✓ PGA finished in {elapsed:.1f}s (exit code: {pga_proc.returncode})")
        
        time.sleep(2)
    
    base_file.close()
    pga_file.close()
    
    total_time = max(base_end, pga_end) - start_time
    print(f"\n  Total wall time: {total_time:.1f}s")
    
    # Read logs
    with open(base_log, "r") as f:
        base_out = f.read()
    with open(pga_log, "r") as f:
        pga_out = f.read()
    
    # Check for errors
    if base_proc.returncode != 0:
        print(f"\n⚠ Baseline exited with error code {base_proc.returncode}")
        print("Last 20 lines of baseline.log:")
        print('\n'.join(base_out.split('\n')[-20:]))
    
    if pga_proc.returncode != 0:
        print(f"\n⚠ PGA exited with error code {pga_proc.returncode}")
        print("Last 20 lines of pga.log:")
        print('\n'.join(pga_out.split('\n')[-20:]))
    
    # Parse results
    print(f"\n[3/3] Processing results and generating plot...")
    
    base_train, base_val = parse_loss_history(base_out)
    pga_train, pga_val = parse_loss_history(pga_out)
    
    base_samples = extract_samples(base_out)
    pga_samples = extract_samples(pga_out)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    if base_train:
        print(f"\n  Baseline MicroGPT:")
        print(f"    Execution Time:   {base_end - start_time:.1f}s")
        print(f"    Final Train Loss: {base_train[-1][1]:.4f}")
        if base_val:
            print(f"    Final Val Loss:   {base_val[-1][1]:.4f}")
        print(f"    Steps logged:     {len(base_train)}")
    else:
        print("\n  Baseline: No training loss data found in logs!")
    
    if pga_train:
        print(f"\n  PGA MicroGPT:")
        print(f"    Execution Time:   {pga_end - start_time:.1f}s")
        print(f"    Final Train Loss: {pga_train[-1][1]:.4f}")
        if pga_val:
            print(f"    Final Val Loss:   {pga_val[-1][1]:.4f}")
        print(f"    Steps logged:     {len(pga_train)}")
    else:
        print("\n  PGA: No training loss data found in logs!")
    
    if base_train and pga_train:
        delta = pga_train[-1][1] - base_train[-1][1]
        print(f"\n  COMPARISON:")
        print(f"    Train Loss Delta (PGA - Base): {delta:+.4f} {'(PGA better)' if delta < 0 else '(Baseline better)'}")
        if base_val and pga_val:
            val_delta = pga_val[-1][1] - base_val[-1][1]
            print(f"    Val Loss Delta (PGA - Base):   {val_delta:+.4f} {'(PGA better)' if val_delta < 0 else '(Baseline better)'}")
    
    # Generated samples
    if base_samples:
        print(f"\n  Baseline Samples (first 5):")
        for s in base_samples[:5]:
            print(f"    {s}")
    
    if pga_samples:
        print(f"\n  PGA Samples (first 5):")
        for s in pga_samples[:5]:
            print(f"    {s}")
    
    # --- PLOTTING ---
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'EBC1 Experiment: Baseline vs PGA MicroGPT ({steps} steps, Shakespeare)', 
                     fontsize=14, fontweight='bold')
        
        # --- Plot 1: Train Loss vs Steps ---
        ax1 = axes[0]
        if base_train:
            b_steps, b_losses = zip(*base_train)
            ax1.plot(b_steps, b_losses, label='Baseline', color='#2196F3', linewidth=1.5, alpha=0.8)
        if pga_train:
            p_steps, p_losses = zip(*pga_train)
            ax1.plot(p_steps, p_losses, label='PGA', color='#FF5722', linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Train Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # --- Plot 2: Val Loss vs Steps ---
        ax2 = axes[1]
        has_val_data = False
        if base_val:
            bv_steps, bv_losses = zip(*base_val)
            ax2.plot(bv_steps, bv_losses, label='Baseline (Val)', color='#2196F3', linewidth=2, marker='o', markersize=4)
            has_val_data = True
        if pga_val:
            pv_steps, pv_losses = zip(*pga_val)
            ax2.plot(pv_steps, pv_losses, label='PGA (Val)', color='#FF5722', linewidth=2, marker='s', markersize=4)
            has_val_data = True
        
        if has_val_data:
            ax2.set_xlabel('Steps', fontsize=12)
            ax2.set_ylabel('Validation Loss', fontsize=12)
            ax2.set_title('Validation Loss', fontsize=13)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0)
        else:
            ax2.text(0.5, 0.5, 'No validation data available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Validation Loss', fontsize=13)
        
        plt.tight_layout()
        
        plot_path = os.path.join(cwd, "loss_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Plot saved to: {plot_path}")
        
    except ImportError:
        print("\n  ⚠ matplotlib not installed. Skipping plot generation.")
        print("    Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n  ⚠ Plot generation failed: {e}")
    
    # Save text report
    report_path = os.path.join(cwd, "results.txt")
    with open(report_path, "w") as f:
        f.write(f"EBC1 Experiment Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps: {steps} | Dataset: {data}\n")
        f.write(f"{'='*50}\n\n")
        if base_train:
            f.write(f"Baseline: Final Train Loss = {base_train[-1][1]:.4f}\n")
            if base_val:
                f.write(f"          Final Val Loss   = {base_val[-1][1]:.4f}\n")
        if pga_train:
            f.write(f"PGA:      Final Train Loss = {pga_train[-1][1]:.4f}\n")
            if pga_val:
                f.write(f"          Final Val Loss   = {pga_val[-1][1]:.4f}\n")
        if base_train and pga_train:
            f.write(f"\nDelta (PGA - Base): {pga_train[-1][1] - base_train[-1][1]:+.4f}\n")
    
    print(f"  ✓ Report saved to: {report_path}")
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
