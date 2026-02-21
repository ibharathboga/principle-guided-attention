"""
Standard PyTorch GPT Baseline
Efficient implementation for benchmarking against PGA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import math
import sys

# Set seed
torch.manual_seed(42)
random.seed(42)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Data Preparation ---
if not os.path.exists("input.txt"):
    import urllib.request
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

all_docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(all_docs)

split_idx = int(len(all_docs) * 0.9)
train_docs = all_docs[:split_idx]
val_docs = all_docs[split_idx:]

uchars = sorted(set("".join(all_docs)))
stoi = {ch:i for i,ch in enumerate(uchars)}
itos = {i:ch for i,ch in enumerate(uchars)}
vocab_size = len(uchars) + 1 # +1 for BOS
BOS_TOKEN = len(uchars)
stoi['<BOS>'] = BOS_TOKEN
itos[BOS_TOKEN] = '<BOS>'

print(f"Vocab size: {vocab_size}")

# --- Model Parameters ---
# Matching MicroGPT roughly
n_layer = 2  # Increased slightly for torch capacity
n_embd = 32 # Increased slightly
n_head = 4
block_size = 16
dropout = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class GBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[GBP() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == BOS_TOKEN:
                break
        return idx

# --- Training Loop ---
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
steps = 500
if len(sys.argv) > 1:
    steps = int(sys.argv[1])

print(f"Training Baseline PyTorch GPT for {steps} steps...")

# We train one doc at a time to mimic micro-batch nature of previous test, 
# or we can batch. Let's stick to single-doc stochastic gradient descent (SGD/Adam)
# for strict comparison parity with the micro-implementation logic, 
# though using GPUs usually invites batching. 
# To keep "Logic" consistent, we process 1 doc per step.

train_losses = []
val_losses = []

for step in range(steps):
    # Prepare batch (size 1)
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS_TOKEN] + [stoi[c] for c in doc] + [BOS_TOKEN]
    if len(tokens) > block_size + 1: 
        tokens = tokens[:block_size+1]
    
    # Needs at least 2 tokens
    if len(tokens) < 2: continue

    data = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    x = data[:, :-1]
    y = data[:, 1:]

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    
    if (step+1) % 100 == 0:
        model.eval()
        v_loss = 0
        with torch.no_grad():
            # Validate on 10 docs
            for i in range(10):
                vdoc = val_docs[i % len(val_docs)]
                vtokens = [BOS_TOKEN] + [stoi[c] for c in vdoc] + [BOS_TOKEN]
                if len(vtokens) > block_size + 1: vtokens = vtokens[:block_size+1]
                if len(vtokens) < 2: continue
                vdata = torch.tensor(vtokens, dtype=torch.long, device=device).unsqueeze(0)
                vx, vy = vdata[:, :-1], vdata[:, 1:]
                _, vl = model(vx, vy)
                v_loss += vl.item()
        v_loss /= 10
        val_losses.append(v_loss)
        print(f"Step {step+1}/{steps} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f}")
        model.train()
    else:
        print(f"Step {step+1}/{steps} | Train Loss: {loss.item():.4f}", end="\r")

print("\n--- Inference (Baseline PyTorch) ---")
model.eval()
for i in range(5):
    ctx = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=block_size)
    # decode
    out_list = out[0].tolist()
    # strip BOS
    cleaned = [x for x in out_list if x != BOS_TOKEN]
    name = "".join([itos[x] for x in cleaned])
    print(f"Sample {i+1}: {name}")
