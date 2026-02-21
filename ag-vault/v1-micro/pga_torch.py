"""
PGA PyTorch Implementation (Real SVD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import math
import sys

torch.manual_seed(42)
random.seed(42)

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
vocab_size = len(uchars) + 1 
BOS_TOKEN = len(uchars)
stoi['<BOS>'] = BOS_TOKEN
itos[BOS_TOKEN] = '<BOS>'

# --- Params ---
n_layer = 2
n_embd = 32
n_head = 4
block_size = 16

class ObservationBuffer:
    def __init__(self, capacity=20, rank_k=4):
        self.capacity = capacity
        self.rank_k = rank_k
        self.buffer = [] # List of tensors (T, C)

    def push(self, x):
        """x: tensor of shape (T, C)"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(x.detach()) # Store detached tensor

    def get_principle(self, current_T):
        """
        Computes Principle Matrix P (T, T) using SVD on buffered Grammar matrices.
        """
        if not self.buffer:
            return None
        
        # 1. Align observations to current_T
        # We only consider the sub-region that matches current sequence length
        # For simplicity in this micro-implementation, we crop/pad to block_size or current_T
        # Ideally, we compute P for the full context window.
        
        # Stack buffers: (N, T_buf, C)
        # We need to handle variable lengths if we process strictly by doc. 
        # But here we padded everything to block_size or similar??
        # Let's standardize on block_size for the "Principle" (Global Invariant).
        
        # Filter buffer for items that are at least current_T long
        valid_obs = [obs[:current_T, :] for obs in self.buffer if obs.size(0) >= current_T]
        if not valid_obs:
            return None
            
        # Stack: (N, T, C)
        X = torch.stack(valid_obs) 
        
        # 2. Compute Gram Matrices: G = X @ X.T -> (N, T, T)
        # This captures token-to-token correlations in the embedding space
        G = torch.bmm(X, X.transpose(1, 2))
        
        # 3. Average Gram Matrix (The "Common Structure")
        G_avg = torch.mean(G, dim=0) # (T, T)
        
        # 4. SVD: U, S, V = svd(G_avg)
        try:
            U, S, Vh = torch.linalg.svd(G_avg)
        except: 
            # SVD convergence failure fallback
            return None
            
        # 5. Low-Rank Reconstruction (The "Logic Filter")
        # Keep top k components
        k = min(self.rank_k, len(S))
        S_diag = torch.diag(S[:k])
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]
        
        P = U_k @ S_diag @ Vh_k # (T, T)
        
        # Normalize P strength to be compatible with attention logits scale
        # Attention logits are ~ sqrt(d_k). P should be a subtle bias.
        P = P / (P.std() + 1e-6) 
        
        return P

class PGACausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x, principle=None):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # --- PGA Injection ---
        if principle is not None:
            # principle is (T, T). Expand to (1, 1, T, T)
            # Additive bias: 'Masking' attention with Logic
            P_bias = principle.view(1, 1, T, T)
            att = att + (0.5 * P_bias) # Weighting factor 0.5
            
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class PGAGBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = PGACausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, principle=None):
        x = x + self.attn(self.ln1(x), principle=principle)
        x = x + self.mlp(self.ln2(x))
        return x

class PGAGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([PGAGBP() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.obs_buffer = ObservationBuffer()

    def forward(self, idx, targets=None, training_mode=False):
        B, T = idx.size()
        
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        
        # --- PGA: Principle Extraction & Application ---
        # 1. Extract Principle P from buffer history
        with torch.no_grad():
            P = self.obs_buffer.get_principle(T)
            if P is not None: 
                P = P.to(device)

        # 2. Store current observation for NEXT time (Feedback Loop)
        # We store the 'embedding state' representing the problem structure
        if training_mode:
            # Only buffer 1st element of batch for simplicity if B>1
            self.obs_buffer.push(x[0].clone())

        # 3. Pass P to blocks
        for block in self.blocks:
            x = block(x, principle=P)
            
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
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == BOS_TOKEN:
                break
        return idx

# --- Training Loop ---
model = PGAGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
steps = 500
if len(sys.argv) > 1:
    steps = int(sys.argv[1])

print(f"Training PGA PyTorch GPT (Real SVD) for {steps} steps...")

val_losses = []

for step in range(steps):
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS_TOKEN] + [stoi[c] for c in doc] + [BOS_TOKEN]
    if len(tokens) > block_size + 1: tokens = tokens[:block_size+1]
    if len(tokens) < 2: continue

    data = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x = data[:, :-1]
    y = data[:, 1:]

    # Pass training_mode=True to update buffer
    logits, loss = model(x, y, training_mode=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step+1) % 100 == 0:
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for i in range(10):
                vdoc = val_docs[i % len(val_docs)]
                vtokens = [BOS_TOKEN] + [stoi[c] for c in vdoc] + [BOS_TOKEN]
                if len(vtokens) > block_size + 1: vtokens = vtokens[:block_size+1]
                if len(vtokens) < 2: continue
                vdata = torch.tensor(vtokens, dtype=torch.long, device=device).unsqueeze(0)
                vx, vy = vdata[:, :-1], vdata[:, 1:]
                _, vl = model(vx, vy, training_mode=False)
                v_loss += vl.item()
        v_loss /= 10
        val_losses.append(v_loss)
        print(f"Step {step+1}/{steps} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f}")
        model.train()
    else:
        print(f"Step {step+1}/{steps} | Train Loss: {loss.item():.4f}", end="\r")

print("\n--- Inference (PGA PyTorch) ---")
model.eval()
for i in range(5):
    ctx = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=block_size)
    cleaned = [x for x in out[0].tolist() if x != BOS_TOKEN]
    name = "".join([itos[x] for x in cleaned])
    print(f"Sample {i+1}: {name}")
