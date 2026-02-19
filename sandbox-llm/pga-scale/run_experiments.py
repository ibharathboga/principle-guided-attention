
import subprocess
import concurrent.futures
import time
import sys
import os

def run_experiment(model, steps):
    print(f"Launching scaled experiment: {model} / {steps} steps")
    cmd = [
        sys.executable, 
        "-u", # Unbuffered output
        "train_scale.py", 
        "--model", model,
        "--steps", str(steps)
    ]
    
    log_filename = f"logs/log_scale_{model}_{steps}.txt"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_filename, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    print(f"Finished scaled: {model} / {steps} steps")

def main():
    experiments = [
        # ('baseline', 2500), # Already done
        ('pga', 2500),
    ]
    
    max_workers = 2 # Larger models, so fewer parallel jobs to avoid OOM/CPU choke
    print(f"Running {len(experiments)} scaled experiments with max_workers={max_workers}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model, steps in experiments:
            futures.append(executor.submit(run_experiment, model, steps))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
