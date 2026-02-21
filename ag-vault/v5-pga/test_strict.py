import subprocess
import sys
import argparse
import time

def run_script(name, script, log):
    print(f"\n--- Running {name} ---")
    start = time.time()
    cmd = [sys.executable, script, "--steps", "1000", "--log", log]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}: {e}")
    print(f"Finished {name} in {time.time() - start:.2f}s")

def get_final_loss(log):
    try:
        with open(log, 'r') as f:
            lines = f.readlines()
            last = lines[-1].strip().split(',')
            return float(last[1])
    except:
        return 999.0

def main():
    run_script("Baseline Strict", "microgpt_strict.py", "baseline_strict_log.txt")
    run_script("PGA Strict", "pga_strict.py", "pga_strict_log.txt")
    
    l_base = get_final_loss("baseline_strict_log.txt")
    l_pga = get_final_loss("pga_strict_log.txt")
    
    print("\n=== v5-pga Results (Strict Architecture) ===")
    print(f"Baseline Loss: {l_base}")
    print(f"PGA Loss:      {l_pga}")
    print(f"Diff:          {l_base - l_pga:.4f}")

if __name__ == "__main__":
    main()
