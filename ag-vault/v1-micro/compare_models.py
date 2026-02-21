import subprocess
import os
import time
import re

def run_model(script_name, name, log_file):
    print(f"Running {name} ({script_name}) -> {log_file}...")
    start_time = time.time()
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            ["python", "-u", script_name],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        process.wait()
        
    end_time = time.time()
    print(f"{name} finished in {end_time - start_time:.2f} seconds.")
    
    # Read the log file for parsing
    try:
        with open(log_file, "r") as f:
            output = f.read()
    except:
        output = ""
        
    return output, end_time - start_time

def parse_logs(log_output):
    # Extract final loss
    # Pattern: "step 1000 / 1000 | loss 1.2345"
    if not log_output:
        return 0.0, []
        
    lines = log_output.split('\n')
    dataset_size = 0
    vocab_size = 0
    
    final_loss = 0.0
    for line in lines:
        if "num docs:" in line:
            dataset_size = int(line.split(":")[1].strip())
        if "vocab size:" in line:
            vocab_size = int(line.split(":")[1].strip())
        if "step" in line and "loss" in line:
            # step 1000 / 1000 | loss 2.8765
            try:
                parts = line.split('|')
                loss_part = parts[1].strip()
                final_loss = float(loss_part.split()[1])
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
            
    return final_loss, samples

def main():
    print("----------------------------------------------------------------")
    print("MicroGPT vs MicroGPT-PGA Benchmark")
    print("----------------------------------------------------------------")
    
    # 1. Run Baseline
    if not os.path.exists("microgpt.py"):
        print("Error: microgpt.py not found using relative path.")
        
    base_out, base_time = run_model("microgpt.py", "Baseline MicroGPT", "microgpt_log.txt")
    base_loss, base_samples = parse_logs(base_out)
    
    # 2. Run PGA
    pga_out, pga_time = run_model("microgpt_pga.py", "PGA MicroGPT", "pga_log.txt")
    pga_loss, pga_samples = parse_logs(pga_out)
    
    # 3. Report
    report = f"""
BENCHMARK RESULTS
=================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Baseline (MicroGPT)
-------------------
Execution Time: {base_time:.2f}s
Final Loss:     {base_loss:.4f}
Samples:
{chr(10).join(base_samples[:5])} ...

PGA (MicroGPT-PGA)
------------------
Execution Time: {pga_time:.2f}s
Final Loss:     {pga_loss:.4f}
Samples:
{chr(10).join(pga_samples[:5])} ...

COMPARISON
----------
Loss Delta (PGA - Base): {pga_loss - base_loss:.4f} (Lower is better)
Time Delta (PGA - Base): {pga_time - base_time:.2f}s

Analysis:
{'PGA achieved lower loss.' if pga_loss < base_loss else 'Baseline achieved lower loss.'}
Note: PGA has overhead due to SVD calculation at every step.
"""
    
    print(report)
    with open("../comparison_results.txt", "w") as f:
        f.write(report)
    print("Results saved to comparison_results.txt")

if __name__ == "__main__":
    main()
