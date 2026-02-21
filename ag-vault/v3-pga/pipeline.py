import subprocess
import sys
import os
import argparse
import time

def run_command(cmd, description):
    print(f"\n[Pipeline] Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    start = time.time()
    # Run synchronously for the pipeline to work
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    duration = time.time() - start
    print(f"[Pipeline] Finished: {description} ({duration:.2f}s)")
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        sys.exit(1)

def get_final_loss(logfile):
    last_loss = None
    if os.path.exists(logfile):
        with open(logfile, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if ',' in line:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2 and 'step' not in parts[0]:
                            last_loss = float(parts[1])
                            break
                    except:
                        continue
    return last_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000, help="Number of training steps")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Files
    baseline_log = "baseline_log.txt"
    baseline_model = "baseline_model.pt"
    pga_log = "pga_guided_log.txt"
    
    # Phase 1: Train Baseline
    cmd_baseline = [
        sys.executable, "microgpt_torch.py", 
        "--steps", str(args.steps), 
        "--seed", str(args.seed),
        "--log", baseline_log,
        "--save_path", baseline_model
    ]
    run_command(cmd_baseline, "Training Baseline Model")
    
    # Check if model saved
    if not os.path.exists(baseline_model):
        print("Error: Baseline model not found. Aborting.")
        return

    # Phase 2: Train Guided PGA
    cmd_pga = [
        sys.executable, "pga_guided.py", 
        "--steps", str(args.steps), 
        "--seed", str(args.seed),
        "--log", pga_log,
        "--guide_path", baseline_model,
        "--buffer_file", "essences_guided.jsonl"
    ]
    run_command(cmd_pga, "Training Guided PGA Model")
    
    # Analysis
    loss_base = get_final_loss(baseline_log)
    loss_pga = get_final_loss(pga_log)
    
    print("\n" + "="*40)
    print(f"FINAL REPORT ({args.steps} steps)")
    print("="*40)
    print(f"Baseline Loss: {loss_base}")
    print(f"PGA Guided Loss: {loss_pga}")
    
    if loss_base and loss_pga:
        delta = loss_base - loss_pga
        print(f"Improvement: {delta:.4f}")
        if delta > 0:
            print("Verdict: PGA WINS! :rocket:")
        elif abs(delta) < 0.01:
            print("Verdict: TIE / MARGINAL")
        else:
            print("Verdict: BASELINE WINS")
            
    # Metrics Export for plotting
    # Combine logs into one CSV?
    # baseline_log.txt: step,loss
    # pga_guided_log.txt: step,loss
    # Create comparison.csv
    try:
        import pandas as pd
        df1 = pd.read_csv(baseline_log, names=['step', 'baseline_loss'], header=0)
        df2 = pd.read_csv(pga_log, names=['step', 'pga_loss'], header=0)
        merged = pd.merge(df1, df2, on='step')
        merged.to_csv("comparison_log.csv", index=False)
        print("Saved comparison_log.csv")
    except ImportError:
        print("Pandas not installed, skipping CSV merge.")
    except Exception as e:
        print(f"Could not merge logs: {e}")

if __name__ == "__main__":
    main()
