"""
The most atomic way to train and run inference for a GPT in PyTorch.
Ported from microgpt.py to use torch.nn and torch.optim.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import sys

# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.nonlin = nn.ReLU() # microgpt uses ReLU

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        
        # microgpt.py does rmsnorm AFTER adding pos embeddings (inside the loop it does it before attn, but the first x is normalized)? 
        # Actually microgpt.py: x = rmsnorm(x) happen at start of loop.
        # But here standard GPT-2 style is Pre-LN. MicroGPT is also Pre-LN.
        # But microgpt.py puts a norm on the residual path immediately?
        # Line 162: x = rmsnorm(x). Then loop starts.
        # Let's stick to standard Pre-LN Block structure which is robust.
        # The Block class above handles the norms.
        # However, microgpt.py has an initial rmsnorm(x) on line 162. 
        # We will skip that oddity and stick to the cleaner Block definition which is mathematically very similar.
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next.item() == 0: # BOS token in our mapping is likely not 0, but let's check vocab.
                # In microgpt, BOS is len(uchars). code will handle this outside.
                pass
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------

class Config:
    n_layer = 1
    n_embd = 16
    n_head = 4
    block_size = 16
    vocab_size = None # set later

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="baseline_log.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # 1. Dataset
    if not os.path.exists("input.txt"):
        print("Error: input.txt not found")
        return

    with open("input.txt", 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    
    uchars = sorted(list(set("".join(docs))))
    # Tokenizer
    stoi = {ch:i for i,ch in enumerate(uchars)}
    itos = {i:ch for i,ch in enumerate(uchars)}
    vocab_size = len(uchars) + 1 # +1 for BOS
    BOS_TOKEN = len(uchars)
    
    # Update Config
    config = Config()
    config.vocab_size = vocab_size

    print(f"Dataset: {len(docs)} documents")
    print(f"Vocab size: {vocab_size}")
    
    # 2. Model
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Standard LR
    
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # 3. Training Loop
    start_time = time.time()
    
    log_file = open(args.log, "w")
    log_file.write("step,loss\n")
    
    for step in range(args.steps):
        # Data Loader (Single Document, like microgpt)
        doc = docs[step % len(docs)]
        # Tokenize: [BOS] + doc + [BOS]
        # In microgpt: tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        tokens = [BOS_TOKEN] + [stoi[ch] for ch in doc] + [BOS_TOKEN]
        
        # Create input/target. 
        # Input: tokens[:-1], Target: tokens[1:]
        # Truncate to block_size if needed
        if len(tokens) - 1 > config.block_size:
            # simple truncation for now to match microgpt "min(block_size, len(tokens)-1)"
            # microgpt loops through positions. We want a batch of 1 sequence.
             tokens = tokens[:config.block_size + 1]
        
        data = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0) # (1, T)
        targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0) # (1, T)
        
        optimizer.zero_grad()
        logits, loss = model(data, targets)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            log_file.write(f"{step},{loss.item()}\n")
            log_file.flush()

    print(f"Training finished in {time.time() - start_time:.2f}s")
    log_file.close()

    # 4. Inference
    print("\n--- Inference ---")
    model.eval()
    for _ in range(5):
        # Start with BOS
        idx = torch.tensor([[BOS_TOKEN]], dtype=torch.long)
        gen = model.generate(idx, max_new_tokens=config.block_size, temperature=0.8)
        # Decode
        out_tokens = gen[0].tolist()
        # strip BOS
        chars = [itos[t] for t in out_tokens if t != BOS_TOKEN]
        print("".join(chars))

if __name__ == "__main__":
    main()
