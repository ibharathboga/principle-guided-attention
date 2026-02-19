
import os
import time
import json
import argparse
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import Scaled Models
from pga_scale import PGAMicroGPT, BLOCK_SIZE, VOCAB_SIZE, N_EMBD
from baseline_scale import BaselineMicroGPT

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        # We enforce a strict vocabulary.
        # Shakespeare file usually has ~65 chars.
        # We'll build the vocab dynamically from the file to ensure we cover everything.
        chars = sorted(list(set(text)))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        # Check alignment with model VOCAB_SIZE = 65
        # If dataset has fewer/more, we might need to adjust or pad.
        # Ideally, we should set VOCAB_SIZE in models dynamically, but for this script we hardcoded 65.
        # Let's hope tiny-shakespeare is consistent with standard char-rnn.
        
        self.vocab_size = len(chars)
        print(f"Dataset Vocab Size: {self.vocab_size}")
        
        self.data = []
        for ch in text:
            self.data.append(self.stoi[ch])
                
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_data():
    # Load Shakespeare from pga-token directory
    path = os.path.join(os.path.dirname(__file__), '../pga-token/shakespeare.txt')
    print(f"Loading data from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def train(args):
    torch.manual_seed(1337)
    
    # Data Preparation
    text = load_data()
    n = len(text)
    split_idx = int(n*0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = TextDataset(train_text, BLOCK_SIZE)
    val_dataset = TextDataset(val_text, BLOCK_SIZE)
    
    # Adjust VOCAB_SIZE hack
    # If vocab size mismatch, pytorch embedding will error or go OOB.
    # PGA/Baseline hardcoded 65.
    if train_dataset.vocab_size != VOCAB_SIZE:
        print(f"WARNING: Dataset vocab size {train_dataset.vocab_size} != Model vocab size {VOCAB_SIZE}")
        # We can't easily patch the class definition here without dynamic args.
        # Assuming typical shakespeare is 65.
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    if args.model == 'pga':
        model = PGAMicroGPT().to(device)
    elif args.model == 'baseline':
        model = BaselineMicroGPT().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")    
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    metrics = {
        'steps': [],
        'train_loss': [],
        'val_loss': [],
        'time': []
    }
    
    start_time = time.time()
    iter_train = iter(train_loader)
    
    print(f"Starting scaled training for {args.steps} steps on {args.model}...", flush=True)
    
    for step in range(args.steps):
        # Get Batch
        try:
            xb, yb = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            xb, yb = next(iter_train)
            
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward
        logits, loss = model(xb, yb)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step == args.steps - 1:
            # Validation
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for vb, vtar in val_loader:
                    vb, vtar = vb.to(device), vtar.to(device)
                    _, vloss = model(vb, vtar)
                    val_loss += vloss.item()
                    val_steps += 1
                    if val_steps > 20: break 
            
            if val_steps > 0:
                val_loss /= val_steps
            else:
                val_loss = 0.0
                
            model.train()
            
            print(f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}", flush=True)
            metrics['steps'].append(step)
            metrics['train_loss'].append(loss.item())
            metrics['val_loss'].append(val_loss)
            metrics['time'].append(time.time() - start_time)
            
    # Save Metrics
    os.makedirs('results', exist_ok=True)
    filename = f"results/metrics_scale_{args.model}_{args.steps}.json"
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Finished. Metrics saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['pga', 'baseline'])
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    
    train(args)
