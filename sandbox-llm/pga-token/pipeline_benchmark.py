
import os
import time
import json
import argparse
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import the model
from pga_token_impl import PGAMicroGPT, BLOCK_SIZE, VOCAB_SIZE, N_EMBD
from baseline_impl import BaselineMicroGPT

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        # We enforce the vocab of the model: lowercase a-z + space/special
        # 0 is reserved for 'unknown' or padding.
        # 1-26 are a-z.
        self.stoi = { ch:i+1 for i,ch in enumerate('abcdefghijklmnopqrstuvwxyz') }
        self.stoi[' '] = 0 # Space is 0
        self.stoi['\n'] = 0 # Newline is 0
        
        # Normalize text to this vocab
        self.data = []
        for ch in text.lower():
            if ch in self.stoi:
                self.data.append(self.stoi[ch])
            else:
                self.data.append(0) # Map everything else to 0
                
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_data(dataset_name):
    # Absolute paths based on user context
    base_path = r"c:\Users\ibhar\OneDrive\Desktop\attention-lab\sandbox-llm"
    if dataset_name == 'names':
        path = os.path.join(base_path, 'pga-e1', 'names.txt')
    elif dataset_name == 'shakespeare':
        path = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    print(f"Loading data from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def train(args):
    torch.manual_seed(1337)
    
    # Data Preparation
    text = load_data(args.dataset)
    n = len(text)
    split_idx = int(n*0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = TextDataset(train_text, BLOCK_SIZE)
    val_dataset = TextDataset(val_text, BLOCK_SIZE)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    if args.model == 'pga':
        model = PGAMicroGPT().to(device)
    elif args.model == 'baseline':
        model = BaselineMicroGPT().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    metrics = {
        'steps': [],
        'train_loss': [],
        'val_loss': [],
        'time': []
    }
    
    start_time = time.time()
    iter_train = iter(train_loader)
    
    print(f"Starting training for {args.steps} steps on {args.dataset}...")
    
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
                # Check random batches from validation
                for vb, vtar in val_loader:
                    vb, vtar = vb.to(device), vtar.to(device)
                    _, vloss = model(vb, vtar)
                    val_loss += vloss.item()
                    val_steps += 1
                    if val_steps > 10: break 
            
            if val_steps > 0:
                val_loss /= val_steps
            else:
                val_loss = 0.0
                
            model.train()
            
            print(f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}")
            metrics['steps'].append(step)
            metrics['train_loss'].append(loss.item())
            metrics['val_loss'].append(val_loss)
            metrics['time'].append(time.time() - start_time)
            
    # Save Metrics
    os.makedirs('results', exist_ok=True)
    filename = f"results/metrics_{args.model}_{args.dataset}_{args.steps}.json"
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Finished. Metrics saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['names', 'shakespeare'])
    parser.add_argument('--model', type=str, required=True, choices=['pga', 'baseline'])
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    
    if args.smoke_test:
        print("Running Smoke Test (10 steps)...")
        args.steps = 10
        
    train(args)
