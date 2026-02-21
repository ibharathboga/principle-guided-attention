
import os
import sys

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("--- Starting Comparison ---")

print("\n1. Running Baseline MicroGPT...")
try:
    import microgpt
    baseline_losses = microgpt.train_and_inference(num_steps=1000)
    final_baseline_loss = baseline_losses[-1]
    print(f"Baseline Final Loss: {final_baseline_loss:.4f}")
except Exception as e:
    print(f"Error running Baseline: {e}")
    baseline_losses = []
    final_baseline_loss = float('inf')

print("\n2. Running PGA MicroGPT...")
try:
    import microgpt_pga
    pga_losses = microgpt_pga.train_and_inference(num_steps=1000)
    final_pga_loss = pga_losses[-1]
    print(f"PGA Final Loss: {final_pga_loss:.4f}")
except Exception as e:
    print(f"Error running PGA: {e}")
    pga_losses = []
    final_pga_loss = float('inf')

print("\n--- Results Summary ---")
print(f"Steps: 1000")
print(f"Baseline Loss: {final_baseline_loss:.4f}")
print(f"PGA Loss:      {final_pga_loss:.4f}")

delta = final_baseline_loss - final_pga_loss
if delta > 0:
    print(f"Result: PGA improved loss by {delta:.4f}")
else:
    print(f"Result: PGA degraded loss by {-delta:.4f}")

# Optional: Save results
with open("comparison_results.txt", "w") as f:
    f.write(f"Baseline Final Loss: {final_baseline_loss}\n")
    f.write(f"PGA Final Loss: {final_pga_loss}\n")
    f.write(f"Delta: {delta}\n")
