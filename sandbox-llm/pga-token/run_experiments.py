
import subprocess
import concurrent.futures
import time
import sys
import os

def run_experiment(model, dataset, steps):
    print(f"Launching experiment: {model} / {dataset} / {steps} steps")
    cmd = [
        sys.executable, 
        "pipeline_benchmark.py", 
        "--model", model,
        "--dataset", dataset, 
        "--steps", str(steps)
    ]
    
    # We redirect output to a log file to avoid console clutter
    log_filename = f"logs/log_{model}_{dataset}_{steps}.txt"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_filename, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    print(f"Finished: {model} / {dataset} / {steps} steps")

def main():
    # We already ran PGA. Let's run Baseline.
    # If we want to re-run everything, we could.
    # But to save time, let's just run baseline for now.
    # The user asked "did you compare", so we need baseline data.
    
    experiments = []
    
    # Baseline Experiments
    for steps in [500, 1000, 1500, 2000]:
        experiments.append(('baseline', 'names', steps))
        experiments.append(('baseline', 'shakespeare', steps))
        
    # Re-run a few PGA if needed? No, we have the JSONs.
    
    max_workers = 4
    print(f"Running {len(experiments)} baseline experiments with max_workers={max_workers}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model, ds, steps in experiments:
            futures.append(executor.submit(run_experiment, model, ds, steps))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
