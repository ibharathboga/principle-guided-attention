import subprocess
import time
import threading
import sys
import os

def tail_logs(process, name, color_code):
    """
    Reads stdout from a process and prints it with a prefix.
    """
    for line in iter(process.stdout.readline, b''):
        line = line.decode('utf-8').strip()
        if line:
            # Print with color
            print(f"\033[{color_code}m[{name}] {line}\033[0m")

def run_benchmark(steps=1000):
    print(f"Starting benchmark for {steps} steps...")
    
    # Define commands
    # Use python buffering 1 (line buffered) or -u (unbuffered) to see output immediately
    cmd_baseline = [sys.executable, "-u", "microgpt_torch.py", "--steps", str(steps), "--log", "baseline_log.txt"]
    cmd_pga = [sys.executable, "-u", "pga_torch.py", "--steps", str(steps), "--log", "pga_log.txt", "--buffer_file", "essences.jsonl"]
    
    # Start processes
    p_baseline = subprocess.Popen(cmd_baseline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    p_pga = subprocess.Popen(cmd_pga, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    
    # Start monitoring threads
    t1 = threading.Thread(target=tail_logs, args=(p_baseline, "Baseline", "36")) # Cyan
    t2 = threading.Thread(target=tail_logs, args=(p_pga, "PGA     ", "32")) # Green
    
    t1.start()
    t2.start()
    
    # Wait for completion
    p_baseline.wait()
    p_pga.wait()
    
    t1.join()
    t2.join()
    
    print("\nBenchmark completed.")
    analyze_results()

def analyze_results():
    print("\n--- Analysis ---")
    
    def get_final_loss(logfile):
        last_loss = None
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    try:
                        _, loss = last_line.split(',')
                        last_loss = float(loss)
                    except:
                        pass
        return last_loss

    loss_base = get_final_loss("baseline_log.txt")
    loss_pga = get_final_loss("pga_log.txt")
    
    print(f"Final Baseline Loss: {loss_base}")
    print(f"Final PGA Loss:      {loss_pga}")
    
    if loss_base and loss_pga:
        delta = loss_base - loss_pga
        print(f"Improvement:         {delta:.4f}")
        if delta > 0:
            print("Verdict: PGA Wins! :rocket:")
        else:
            print("Verdict: Baseline Wins.")

if __name__ == "__main__":
    # Ensure we are in the right directory or valid context
    if not os.path.exists("microgpt_torch.py"):
        print("Error: Script must be run in the folder containing microgpt_torch.py")
    else:
        run_benchmark(steps=1000)
