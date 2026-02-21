"""
Scaled Baseline (v6).
Inherits from microgpt_strict.
Overrides Config for larger model.
"""

import argparse
import torch
import time
import os
import sys

# Add v5-pga to path to import microgpt_strict modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v5-pga'))

from microgpt_strict import GPT, Config, main as strict_main, BOS_TOKEN

# Override Config
class ScaleConfig(Config):
    n_layer = 4      # Deeper
    n_embd = 64      # Wider
    n_head = 4       # Same heads
    block_size = 64  # Longer context
    
# We need to reimplement main() because microgpt_strict.main() instantiates Config() directly
# or we can monkeypatch Config given Python's dynamic nature?
# Better to copy main logic to be safe and explicit.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="baseline_scale_log.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Load Data
    if os.path.exists("input.txt"):
        with open("input.txt", 'r') as f:
            docs = [line.strip() for line in f if line.strip()]
        uchars = sorted(list(set("".join(docs))))
        stoi = {ch:i for i,ch in enumerate(uchars)}
        BOS_TOKEN = len(uchars)
        vocab_size = len(uchars) + 1
    else:
        print("Error: input.txt not found")
        return

    config = ScaleConfig()
    config.vocab_size = vocab_size
    print(f"Scale Config: Layers={config.n_layer}, Embd={config.n_embd}, Block={config.block_size}")

    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) # Strict LR
    
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    
    log_file = open(args.log, "w")
    log_file.write("step,loss\n")
    
    start_time = time.time()
    
    for step in range(args.steps):
        doc = docs[step % len(docs)]
        tokens = [BOS_TOKEN] + [stoi[ch] for ch in doc] + [BOS_TOKEN]
        if len(tokens) - 1 > config.block_size:
             tokens = tokens[:config.block_size + 1]
        
        data = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0) 
        targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0)
        
        optimizer.zero_grad()
        logits, loss = model(data, targets)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            log_file.write(f"{step},{loss.item()}\n")
            log_file.flush()

    print(f"Finished in {time.time() - start_time:.2f}s")
    log_file.close()

if __name__ == "__main__":
    main()
