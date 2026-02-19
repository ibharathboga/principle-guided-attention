import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import os

from microgpt_pytorch import MicroGPT
from microgpt_pga_pytorch import MicroGPTPGA

# Configuration
STEPS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

def train_model(model_cls, name, seed, steps):
    # Set seed for this process
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"[{name}] Initialization on {DEVICE}...")
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
    
    # Deterministic Data
    vocab_size = 27
    data_len = steps * 20
    data = [(i % vocab_size) for i in range(data_len)]
    block_size = 16
    
    losses = []
    
    print(f"[{name}] Starting training...")
    model.train()
    for step in range(steps):
        # Batch preparation (Batch Size 1 for similarity to MicroGPT original, or larger?)
        # Let's use Batch Size 1 to match the 'equivalence' check context, 
        # but pure pytorch usually benefits from batching.
        # However, to keep it "micro", BS=1.
        
        start_idx = step % (len(data) - block_size - 1)
        chunk = data[start_idx : start_idx + block_size + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEVICE)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"[{name}] Step {step}: {loss.item():.4f}")
            
    return losses

def moving_average(data, window_size=25):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()

if __name__ == "__main__":
    # We can run sequentially or parallel. 
    # Since PyTorch might fight for resources if not managed, let's run sequentially 
    # to be safe on CPU, or parallel if we trust OS scheduling.
    # Given the small size, parallel is fine.
    
    # We need to wrap function calls for multiprocessing
    # But since models are different classes, we just call them.
    
    print("Running Baseline...")
    baseline_losses = train_model(MicroGPT, "Baseline", SEED, STEPS)
    
    print("Running PGA...")
    pga_losses = train_model(MicroGPTPGA, "PGA", SEED, STEPS)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    
    plt.plot(baseline_losses, color='blue', alpha=0.15, label='Baseline (Raw)')
    plt.plot(pga_losses, color='red', alpha=0.15, label='PGA (Raw)')
    
    baseline_smooth = moving_average(baseline_losses)
    pga_smooth = moving_average(pga_losses)
    
    plt.plot(baseline_smooth, color='blue', linewidth=2, label='Baseline (Smoothed)')
    plt.plot(pga_smooth, color='red', linewidth=2, label='PGA (Smoothed)')
    
    plt.xlabel('Step')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.title(f'PyTorch: Baseline vs PGA ({STEPS} Steps)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('comparison_pytorch.png', dpi=300)
    
    # Report
    final_baseline = np.mean(baseline_losses[-20:])
    final_pga = np.mean(pga_losses[-20:])
    
    report = f"""# PyTorch PGA Experiment Results

## Configuration
- Steps: {STEPS}
- Context Window: 16
- PGA Window: 8
- PGA Rank: 8
- Optimizer: Adam (lr=0.01)

## Results (Last 20 avg)
- **Baseline Final Loss**: {final_baseline:.5f}
- **PGA Final Loss**: {final_pga:.5f}
- **Difference**: {final_baseline - final_pga:.5f}

## Verdict
{"PGA Validated." if final_pga < final_baseline else "No reliable improvement."}
"""
    with open('experiment_report_pytorch.md', 'w') as f:
        f.write(report)
        
    print("Done. Report saved.")
