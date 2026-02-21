"""
Strict PyTorch port of microgpt.py.
Matches architecture, initialization, and normalization exactly.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse

# -----------------------------------------------------------------------------
# Strict Modules
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.nonlin = nn.ReLU() # Strict: ReLU

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
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
            # Strict: Initial RMSNorm before blocks (mimics microgpt line 162-164)
            ln_init = RMSNorm(config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Strict: std=0.08 (mimics microgpt line 118)
        std = 0.08
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        # Strict: Apply Initial Norm
        x = self.transformer.ln_init(x)
        
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
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    @torch.no_grad()
    def get_embeddings(self, idx):
        # Helper for PGA - Strict Version
        # Must return x AFTER ln_init? 
        # In PGA, we want the "Observations" to be the input to the attention mechanism.
        # In standard GPT, attention input is ln_1(x).
        # In strict microgpt, x goes through ln_init before anything.
        # So "Raw Embeddings" might be too raw. 
        # Let's return x AFTER ln_init to be consistent with what the model sees.
        
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
             idx = idx[:, :self.config.block_size]
             t = self.config.block_size
             
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        x = self.transformer.ln_init(x) # Strict: Include this
        return x

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load_pretrained(cls, config, path):
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model

class Config:
    n_layer = 1
    n_embd = 16
    n_head = 4
    block_size = 16
    vocab_size = None 

# -----------------------------------------------------------------------------
# Main System
# -----------------------------------------------------------------------------

# Globals needed for import
BOS_TOKEN = None

if os.path.exists("input.txt"):
    with open("input.txt", 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    uchars = sorted(list(set("".join(docs))))
    stoi = {ch:i for i,ch in enumerate(uchars)}
    itos = {i:ch for i,ch in enumerate(uchars)}
    BOS_TOKEN = len(uchars)
    vocab_size = len(uchars) + 1
else:
    docs = []
    vocab_size = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="baseline_log.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    config = Config()
    config.vocab_size = vocab_size

    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # microgpt uses standard LR? yes 1e-3 implicitly often, microgpt.py uses 0.01 actually!
    # microgpt.py line 206: learning_rate = 0.01
    
    # Wait, microgpt.py uses 0.01?
    # Checked: "learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8"
    # PyTorch AdamW default is 1e-3. 
    # Strict: Use 1e-2 (0.01).
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 # Strict
    
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
        
        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            log_file.write(f"{step},{loss.item()}\n")

    print(f"Finished in {time.time() - start_time:.2f}s")
    log_file.close()

if __name__ == "__main__":
    main()
