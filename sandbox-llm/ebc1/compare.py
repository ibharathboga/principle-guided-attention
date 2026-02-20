import subprocess
import os
import time
import sys

def parse_logs(log_output):
    # Extract final loss
    if not log_output:
        return 0.0, 0.0, []
        
    lines = log_output.split('\n')
    
    final_train_loss = 0.0
    final_val_loss = 0.0
    
    for line in lines:

        # Format: Step 100 / 1000 | Train Loss: 2.1234 | Val Loss: 2.3456
        if ("step" in line.lower()) and ("loss" in line.lower()):
            try:
                parts = line.split('|')
                # parts[1] -> " train_loss 2.1234 "
                # Adapting to "Train Loss: 2.1234"
                content = parts[1].strip()
                if ":" in content:
                    final_train_loss = float(content.split(":")[1])
                else:
                    final_train_loss = float(content.split()[1])
                
                if "Val Loss" in line or "val_loss" in line:
                    val_part = parts[2].strip()
                    if ":" in val_part:
                        final_val_loss = float(val_part.split(":")[1])
                    else:
                        final_val_loss = float(val_part.split()[1])
            except:
                pass
                
    # Extract Samples
    samples = []
    capture = False
    for line in lines:
        if "inference" in line:
            capture = True
            continue
        if capture and line.strip().startswith("sample"):
            samples.append(line.strip())
            
    return final_train_loss, final_val_loss, samples

def main():
    steps = 5000
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
        
    print("----------------------------------------------------------------")
    print(f"Parallel Benchmark: MicroGPT vs MicroGPT-PGA ({steps} steps)")
    print("----------------------------------------------------------------")
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    base_log = "microgpt_baseline_5k.log"
    pga_log = "microgpt_pga_5k.log"
    
    # Start processes in parallel
    print(f"Starting Baseline MicroGPT -> {base_log}...")
    base_file = open(base_log, "w")
    base_proc = subprocess.Popen(
        ["python", "-u", "microgpt_baseline.py", str(steps)],
        stdout=base_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    print(f"Starting PGA MicroGPT -> {pga_log}...")
    pga_file = open(pga_log, "w")
    pga_proc = subprocess.Popen(
        ["python", "-u", "pga_micro.py", str(steps)],
        stdout=pga_file,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True
    )
    
    start_time = time.time()
    
    # Wait for completion
    print("Both models running... Waiting for completion.")
    
    base_done = False
    pga_done = False
    
    # Poll
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
        
    # Close files
    base_file.close()
    pga_file.close()
    
    print("\nProcessing Results...")
    
    # Read logs
    with open(base_log, "r") as f: base_out = f.read()
    with open(pga_log, "r") as f: pga_out = f.read()
    
    base_train, base_val, base_samples = parse_logs(base_out)
    pga_train, pga_val, pga_samples = parse_logs(pga_out)
    
    # Report
    report = f"""
BENCHMARK RESULTS (Parallel {steps} steps)
=======================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Baseline (MicroGPT)
-------------------
Execution Time: {base_end_time - start_time:.2f}s
Final Train Loss: {base_train:.4f}
Final Val Loss:   {base_val:.4f}
Samples:
{chr(10).join(base_samples[:5])} ...

PGA (MicroGPT-PGA)
------------------
Execution Time: {pga_end_time - start_time:.2f}s
Final Train Loss: {pga_train:.4f}
Final Val Loss:   {pga_val:.4f}
Samples:
{chr(10).join(pga_samples[:5])} ...

COMPARISON
----------
Train Loss Delta (PGA - Base): {pga_train - base_train:.4f} 
Val Loss Delta (PGA - Base):   {pga_val - base_val:.4f} (Lower = Better Generalization)

Overfitting Check:
Base Gap (Val - Train): {base_val - base_train:.4f}
PGA Gap (Val - Train):  {pga_val - pga_train:.4f}
"""
    
    print(report)
    with open("../comparison_results_val.txt", "w") as f:
        f.write(report)
    print("Results saved to comparison_results_val.txt")

if __name__ == "__main__":
    main()
