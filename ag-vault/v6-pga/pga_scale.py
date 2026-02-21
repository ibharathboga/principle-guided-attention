"""
Scaled PGA (v6).
Inherits from microgpt_strict but uses ScaleConfig.
JIT Essence Recalculation on 64-token chunks.
"""

import sys
import os
import argparse
import torch
import time
import random
import torch.nn as nn # Added import
from torch.nn import functional as F # Added import

# Add v5-pga to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v5-pga'))

from microgpt_strict import GPT, Config, RMSNorm, MLP, CausalSelfAttention, BOS_TOKEN
from pga_strict import extract_essence, JITEssenceBuffer, PGAGPTStrict

# Override Config
class ScaleConfig(Config):
    n_layer = 4      
    n_embd = 64      
    n_head = 4       
    block_size = 64  

# We need to redefine PGAGPTStrict to use the new Config or just reuse the class?
# The class uses `config` passed in `__init__`. So we just pass ScaleConfig.
# But we need the Training Loop to use the new config.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="pga_scale_log.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
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

    # Strict Init with Scaled Config
    model = PGAGPTStrict(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    buffer = JITEssenceBuffer(max_size=5000) # Larger buffer for larger context
    
    print(f"PGA Scale v6 Training ({args.steps} steps)...")
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
        
        # JIT Logic
        with torch.no_grad():
            curr_emb = model.get_embeddings(data)
            curr_essence = extract_essence(curr_emb.squeeze(0))
            
            retrieved = buffer.retrieve_tokens(curr_essence, k=5)
            
            fresh_essences = []
            for r_tok in retrieved:
                r_inp = r_tok.unsqueeze(0)
                r_emb = model.get_embeddings(r_inp).squeeze(0)
                fresh_essences.append(extract_essence(r_emb))
            
            P = None
            if fresh_essences:
                X_stack = torch.stack(fresh_essences)
                try:
                    _, _, Vh = torch.linalg.svd(X_stack, full_matrices=False)
                    P = Vh.T @ Vh
                except:
                    pass
            
            buffer.add(data.squeeze(0), curr_essence)

        optimizer.zero_grad()
        logits, loss = model(data, targets, P=P)
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
