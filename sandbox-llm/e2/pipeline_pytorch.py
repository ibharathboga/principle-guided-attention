import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import urllib.request

from microgpt_pytorch import MicroGPT
from microgpt_pga_pytorch import MicroGPTPGA

# Configuration
STEPS = 2500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
BATCH_SIZE = 1 # Keep 1 for now to match microgpt exactly, but ideally larger
BLOCK_SIZE = 16

def load_data():
    if not os.path.exists("names.txt"):
        print("Downloading names.txt...")
        url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
        urllib.request.urlretrieve(url, "names.txt")
    
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s:i for i,s in enumerate(chars)}
    stoi['.'] = 26 # BOS/EOS token (MicroGPT uses 26 for BOS)
    # microgpt_original.py used: uchars = sorted(set("".join(docs))). BOS = len(uchars).
    # "names.txt" usually has lowercase letters.
    # We need to match the vocab size=27 of our model.
    # standard names.txt has 26 letters.
    
    # Let's create the dataset as a single long tensor of indices
    # We wrap each name with '.' (BOS/EOS)
    # Example: .anna.emma.
    
    # Replicating microgpt logic:
    # uchars = sorted(set("".join(docs)))
    # BOS = len(uchars)
    # vocab_size = len(uchars) + 1
    
    data = []
    # If using '.' as 26:
    for w in words:
        # MicroGPT uses BOS at start and end?
        # "tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]"
        w_idxs = [stoi[c] for c in w]
        data.extend([26] + w_idxs + [26]) # 26 is BOS
        
    # Split 90/10
    n = int(0.9 * len(data))
    train_data = torch.tensor(data[:n], dtype=torch.long)
    val_data = torch.tensor(data[n:], dtype=torch.long)
    
    print(f"Data Loaded. Vocab: {len(chars)+1}. Train size: {len(train_data)}. Val size: {len(val_data)}")
    return train_data, val_data

def get_batch(data):
    # Retrieve a batch of BATCH_SIZE
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
             X, Y = get_batch(data)
             logits, loss = model(X, Y)
             losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_model(model_cls, name, seed, steps, train_data, val_data):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"[{name}] Initialization on {DEVICE}...")
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
    
    train_losses_log = []
    val_losses_log = []
    
    print(f"[{name}] Starting training...")
    model.train()
    for step in range(steps):
        # Training Step
        xb, yb = get_batch(train_data)
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        
        # Periodic Eval and Logging
        if step % 100 == 0:
            losses = estimate_loss(model, train_data, val_data)
            train_losses_log.append(losses['train'])
            val_losses_log.append(losses['val'])
            print(f"[{name}] Step {step}: Train={losses['train']:.4f}, Val={losses['val']:.4f}")
            
    # Final eval
    losses = estimate_loss(model, train_data, val_data)
    train_losses_log.append(losses['train'])
    val_losses_log.append(losses['val'])
    
    return train_losses_log, val_losses_log

if __name__ == "__main__":
    train_data, val_data = load_data()
    
    print("Running Baseline...")
    base_train, base_val = train_model(MicroGPT, "Baseline", SEED, STEPS, train_data, val_data)
    
    print("Running PGA...")
    pga_train, pga_val = train_model(MicroGPTPGA, "PGA", SEED, STEPS, train_data, val_data)
    
    # Plotting
    steps_range = np.arange(0, STEPS + 101, 100) # 0, 100, ..., 2500 (plus final)
    # Adjust range to match log length
    if len(steps_range) != len(base_train):
        steps_range = np.linspace(0, STEPS, len(base_train))

    plt.figure(figsize=(12, 6))
    
    # Plot val losses mostly, train as dotted
    plt.plot(steps_range, base_train, 'b:', alpha=0.6, label='Baseline Train')
    plt.plot(steps_range, base_val, 'b-', linewidth=2, label='Baseline Val')
    
    plt.plot(steps_range, pga_train, 'r:', alpha=0.6, label='PGA Train')
    plt.plot(steps_range, pga_val, 'r-', linewidth=2, label='PGA Val')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Baseline vs PGA Overfitting Check ({STEPS} Steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('overfitting_check.png', dpi=300)
    
    # Report
    report = f"""# Overfitting Analysis (2500 Steps)

## Configuration
- Data: `names.txt` (90/10 Split)
- Steps: {STEPS}
- Eval Interval: 100 steps

## Final Metrics
| Model | Train Loss | Val Loss | Gap (Overfitting) |
| :--- | :--- | :--- | :--- |
| **Baseline** | {base_train[-1]:.4f} | {base_val[-1]:.4f} | {base_val[-1] - base_train[-1]:.4f} |
| **PGA** | {pga_train[-1]:.4f} | {pga_val[-1]:.4f} | {pga_val[-1] - pga_train[-1]:.4f} |

## Analysis
- If Val Loss increases while Train Loss decreases, the model is overfitting.
- Comparison of Gap shows which model generalizes better.

## Verdict
"""
    if pga_val[-1] < base_val[-1]:
         report += "PGA achieved lower validation loss.\n"
    else:
         report += "Baseline achieved lower validation loss.\n"
         
    with open('experiment_report_overfitting.md', 'w') as f:
        f.write(report)
        
    print("Done. Report saved.")
