"""
Strict JIT PGA (v5).
Inherits from microgpt_strict.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import time
import argparse
import random
import os

from microgpt_strict import GPT, Config, RMSNorm, MLP, CausalSelfAttention, BOS_TOKEN

# -----------------------------------------------------------------------------

class PGASelfAttentionStrict(CausalSelfAttention):
    def forward(self, x, P=None):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        if P is not None:
            # Soft PGA: q = q + tanh(alpha) * (q @ P)
            # For strict test, let's keep it simple: additive 
            # q = q + (q @ P)
            q_proj = q @ P
            q = q + q_proj

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class PGABlockStrict(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = PGASelfAttentionStrict(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, P=None):
        x = x + self.attn(self.ln_1(x), P=P)
        x = x + self.mlp(self.ln_2(x))
        return x

class PGAGPTStrict(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList([PGABlockStrict(config) for _ in range(config.n_layer)])
        self.apply(self._init_weights)
        
    def forward(self, idx, targets=None, P=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        x = self.transformer.ln_init(x) # Strict
        
        for block in self.transformer.h:
            x = block(x, P=P)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Re-use extract_essence from earlier
def extract_essence(x):
    chunk_centered = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(chunk_centered, full_matrices=False)
        essence = Vh[0, :] 
    except:
        essence = chunk_centered.mean(dim=0)
    return essence

# Buffer
class JITEssenceBuffer:
    def __init__(self, max_size=2000):
        self.data = [] 
        self.max_size = max_size

    def add(self, tokens, vector):
        entry = {'tokens': tokens.detach().cpu(), 'cached_vec': vector.detach().cpu()}
        self.data.append(entry)
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def retrieve_tokens(self, query_vec, k=5):
        if not self.data: return []
        device = query_vec.device
        history = torch.stack([d['cached_vec'] for d in self.data]).to(device)
        norm_history = F.normalize(history, p=2, dim=1)
        norm_query = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        scores = torch.mv(norm_history, norm_query)
        k = min(k, len(self.data))
        values, indices = torch.topk(scores, k)
        return [self.data[i]['tokens'].to(device) for i in indices]

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="pga_strict_log.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Load Data (Copied from microgpt_strict)
    if os.path.exists("input.txt"):
        with open("input.txt", 'r') as f:
            docs = [line.strip() for line in f if line.strip()]
        uchars = sorted(list(set("".join(docs))))
        stoi = {ch:i for i,ch in enumerate(uchars)}
        itos = {i:ch for i,ch in enumerate(uchars)}
        BOS_TOKEN = len(uchars)
        vocab_size = len(uchars) + 1
    else:
        print("Error: input.txt not found")
        return

    config = Config()
    config.vocab_size = vocab_size

    # Strict Init
    model = PGAGPTStrict(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) # Strict LR
    
    buffer = JITEssenceBuffer()
    
    print(f"PGA Strict v5 Training ({args.steps} steps)...")
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
        
        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            log_file.write(f"{step},{loss.item()}\n")
            log_file.flush()

    print(f"Finished in {time.time() - start_time:.2f}s")
    log_file.close()

if __name__ == "__main__":
    main()
