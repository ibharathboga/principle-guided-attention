import subprocess
import sys
import threading
import time
import os

def tail_logs(process, name, color_code):
    """Reads stdout from a process and prints it with a colored prefix."""
    for line in iter(process.stdout.readline, b''):
        line = line.decode('utf-8').strip()
        if line:
            # ANSI color codes: \033[36m (Cyan), \033[32m (Green)
            print(f"\033[{color_code}m[{name}] {line}\033[0m")
    process.stdout.close()

def main():
    steps = 2500
    baseline_log = "baseline_2500_log.txt"
    pga_log = "pga_strict_2500_log.txt"
    
    cmd_baseline = [sys.executable, "microgpt_strict.py", "--steps", str(steps), "--log", baseline_log, "--seed", "42"]
    cmd_pga = [sys.executable, "pga_strict.py", "--steps", str(steps), "--log", pga_log, "--seed", "42"]
    
    print(f"Starting Parallel Benchmark (v5-pga) for {steps} steps...")
    
    # Start processes
    # bufsize=1 means line buffered
    p_base = subprocess.Popen(cmd_baseline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    p_pga = subprocess.Popen(cmd_pga, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    
    # Start monitoring threads
    t1 = threading.Thread(target=tail_logs, args=(p_base, "Baseline", "36")) # Cyan
    t2 = threading.Thread(target=tail_logs, args=(p_pga, "PGA     ", "32")) # Green
    
    t1.start()
    t2.start()
    
    # Wait for completion
    p_base.wait()
    p_pga.wait()
    
    t1.join()
    t2.join()
    
    print("\nBenchmark Finished.")
    
    # Run Plotting
    print("Generating Plot...")
    subprocess.run([sys.executable, "plot_v5.py", "--baseline", baseline_log, "--pga", pga_log])

if __name__ == "__main__":
    main()
