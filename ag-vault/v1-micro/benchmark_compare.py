import subprocess
import sys
import re
import matplotlib.pyplot as plt

def run_script(script_name, steps=1000, data='input.txt'):
    print(f"Running {script_name} for {steps} steps on {data}...")
    # Pass args: python script.py --steps N --data path
    cmd = [sys.executable, script_name, '--steps', str(steps), '--data', data]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)
        return ""

def parse_baseline_output(output):
    # Baseline format:
    # Step 100/1000 | Train Loss: 2.34
    # [VALIDATION] Step 100 | Val Loss: 2.5
    
    train_points = []
    val_points = []
    
    lines = output.splitlines()
    for line in lines:
        # Train
        m_train = re.search(r"Step\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d\.]+)", line)
        if m_train:
            step = int(m_train.group(1))
            loss = float(m_train.group(2))
            train_points.append((step, loss))
            
        # Val
        m_val = re.search(r"\[VALIDATION\]\s+Step\s+(\d+)\s+\|\s+Val Loss:\s+([\d\.]+)", line)
        if m_val:
            step = int(m_val.group(1))
            loss = float(m_val.group(2))
            val_points.append((step, loss))
            
    return train_points, val_points

def parse_pga_output(output):
    # PGA format:
    # step  101 / 1000 | train_loss 2.34 | val_loss 2.5
    # or
    # step  101 / 1000 | train_loss 2.34
    
    train_points = []
    val_points = []
    
    lines = output.splitlines()
    for line in lines:
        # Combined Train/Val line in PGA
        if "val_loss" in line:
            m = re.search(r"step\s+(\d+)\s+/\s+\d+\s+\|\s+train_loss\s+([\d\.]+)\s+\|\s+val_loss\s+([\d\.]+)", line)
            if m:
                step = int(m.group(1))
                t_loss = float(m.group(2))
                v_loss = float(m.group(3))
                train_points.append((step, t_loss))
                val_points.append((step, v_loss))
        else:
            # Just train
            m = re.search(r"step\s+(\d+)\s+/\s+\d+\s+\|\s+train_loss\s+([\d\.]+)", line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                train_points.append((step, loss))

    return train_points, val_points

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--data', type=str, default='input.txt')
    args = parser.parse_args()
        
    # Run Baseline
    out_base = run_script("microgpt_baseline.py", args.steps, args.data)
    base_train, base_val = parse_baseline_output(out_base)
    
    # Run PGA
    out_pga = run_script("microgpt_pga.py", args.steps, args.data)
    pga_train, pga_val = parse_pga_output(out_pga)
    
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"{'Step':<10} | {'Base Val':<10} | {'PGA Val':<10} | {'Diff':<10}")
    print("-" * 46)
    
    # Align by step
    base_val_dict = dict(base_val)
    pga_val_dict = dict(pga_val)
    
    all_steps = sorted(set(base_val_dict.keys()) | set(pga_val_dict.keys()))
    
    import math
    for s in all_steps:
        bv = base_val_dict.get(s, float('nan'))
        pv = pga_val_dict.get(s, float('nan'))
        
        # Check for NaN
        bv_isnan = math.isnan(bv) if isinstance(bv, float) else False
        pv_isnan = math.isnan(pv) if isinstance(pv, float) else False
        
        diff = pv - bv if not (bv_isnan or pv_isnan) else 0
        
        # Simple string formatting
        bv_str = f"{bv:.4f}" if not isinstance(bv, float) or not (bv != bv) else "N/A" # bv != bv is NaN check
        pv_str = f"{pv:.4f}" if not isinstance(pv, float) or not (pv != pv) else "N/A"
        
        # manual float nan check
        import math
        if isinstance(bv, float) and math.isnan(bv): bv_str = "N/A"
        if isinstance(pv, float) and math.isnan(pv): pv_str = "N/A"
        
        diff_val = "N/A"
        if bv_str != "N/A" and pv_str != "N/A":
            diff_val = f"{pv - bv:.4f}"
            
        print(f"{s:<10} | {bv_str:<10} | {pv_str:<10} | {diff_val:<10}")

    # Summary
    if base_val and pga_val:
        final_base = base_val[-1][1]
        final_pga = pga_val[-1][1]
        print("-" * 46)
        print(f"Final Baseline Val Loss: {final_base:.4f}")
        print(f"Final PGA Val Loss:      {final_pga:.4f}")
        
        if final_pga < final_base:
            print("\nRESULT: PGA WINS (Lower Validation Loss)")
        else:
            print("\nRESULT: BASELINE WINS (Lower/Equal Validation Loss)")
            
    # Save chart if matplotlib available
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot Train
        # plt.plot(*zip(*base_train), label='Baseline Train', linestyle='--', alpha=0.5)
        # plt.plot(*zip(*pga_train), label='PGA Train', linestyle='--', alpha=0.5)
        
        # Plot Val
        if base_val: plt.plot(*zip(*base_val), label='Baseline Val', marker='o')
        if pga_val: plt.plot(*zip(*pga_val), label='PGA Val', marker='x')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Baseline vs PGA Validation Loss')
        plt.legend()
        plt.savefig('comparison_plot.png')
        print("\nPlot saved to comparison_plot.png")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    main()
