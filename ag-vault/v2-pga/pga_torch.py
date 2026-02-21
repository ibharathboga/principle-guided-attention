"""
Principle-Guided Attention (PGA) Implementation in PyTorch.
Extends the baseline with Chunk-Based Essence Extraction and Subspace Projection.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import json

# -----------------------------------------------------------------------------
# PGA Components
# -----------------------------------------------------------------------------

class EssenceBuffer:
    def __init__(self, filename="essences.jsonl", max_size=1000):
        self.filename = filename
        self.max_size = max_size
        self.essences = [] # List[torch.Tensor]
        self.load()

    def load(self):
        self.essences = []
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        vec = torch.tensor(data['vector'])
                        self.essences.append(vec)
                    except:
                        pass
        # print(f"Loaded {len(self.essences)} essences from {self.filename}")

    def add(self, vector):
        # vector: (D,) tensor
        self.essences.append(vector.detach().cpu())
        # Append to file
        with open(self.filename, 'a') as f:
            f.write(json.dumps({'vector': vector.tolist()}) + '\n')
            
    def retrieve(self, query_vec, k=5):
        if not self.essences:
            return []
        
        # Stack essences: (N, D)
        # Check device of query_vec
        device = query_vec.device
        
        # Use simple slice for recent history context if N is small, 
        # but here we want similarity.
        # Let's convert list to tensor on correct device
        # Optimization: cache this if N is huge, but for now re-stacking is fine for microgpt scale
        history = torch.stack(self.essences).to(device) # (N, D)
        
        # Cosine similarity
        # query_vec: (D,)
        # sim = (history @ query_vec) / (|history| * |query_vec|)
        
        norm_history = F.normalize(history, p=2, dim=1)
        norm_query = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        
        scores = torch.mv(norm_history, norm_query) # (N,)
        
        # Get top k
        k = min(k, len(self.essences))
        values, indices = torch.topk(scores, k)
        
        return [history[i] for i in indices]

def extract_essence(x):
    """
    x: (B, T, D) - In microgpt B=1
    Returns: (D,) - The top right singular vector of the chunk
    """
    # Squeeze batch if B=1
    chunk = x.squeeze(0) # (T, D)
    
    # Center the data? PCA usually requires centering. SVD on raw data is also fine for "direction"
    # Let's center it to capture variance better
    chunk_centered = chunk - chunk.mean(dim=0, keepdim=True)
    
    # SVD
    # U, S, Vh = torch.linalg.svd(chunk_centered, full_matrices=False)
    # Vh is (D, D) or (T, D) depending on shape. 
    # torch.linalg.svd returns U, S, Vh. Vh is V^T.
    # The rows of Vh are the eigenvectors (principal components).
    # We want the first row.
    
    # Robust SVD handling
    try:
        _, _, Vh = torch.linalg.svd(chunk_centered, full_matrices=False)
        essence = Vh[0, :] # (D,)
    except:
        # Fallback if SVD fails (very rare)
        essence = chunk_centered.mean(dim=0)
        
    return essence

# -----------------------------------------------------------------------------
# Modified Model Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class PGASelfAttention(nn.Module):
    def __init__(self, config, buffer):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        self.buffer = buffer # Reference to EssenceBuffer
        self.k_retrieval = 5 # Number of essences to retrieve

    def forward(self, x):
        B, T, C = x.size() 
        
        # --- PGA LOGIC START ---
        # 1. Extract Essence of current chunk
        # Note: We do this PER PASS. 
        # During Inference (Generation), 'x' grows one token at a time.
        # This is tricky. Defining "Chunk" as the *current context window*.
        
        current_essence = extract_essence(x) # (C,)
        
        # 2. Retrieve Context
        # We find historical essences similar to the current one.
        retrieved_vecs = self.buffer.retrieve(current_essence, k=self.k_retrieval)
        
        # 3. Construct Projection Matrix P
        if retrieved_vecs:
            # Stack: (K, C)
            X_stack = torch.stack(retrieved_vecs)
            
            # We want an orthonormal basis for this stack.
            # SVD again on the stack to find the subspace.
            # X_stack = U S V^T. Basis is V^T.
            # Or just use the vectors themselves if K is small?
            # Let's do SVD to be cleaner and handle linear dependence.
            try:
                # Center the stack? Maybe not needed if they are directions.
                _, _, Vh_stack = torch.linalg.svd(X_stack, full_matrices=False)
                # Take top components. How many? All of them that are significant?
                # Let's take all K (or rank).
                V_top = Vh_stack # (K, C)
                
                # Projection Matrix P = V^T V
                # P is (C, C).
                # P projects onto the row space of V_top.
                # If V_top rows are orthonormal (which SVD gives), P = sum(v_i^T v_i)
                P = V_top.T @ V_top # (C, K) @ (K, C) -> (C, C)
                
            except:
                 P = torch.eye(C, device=x.device)
        else:
            P = torch.eye(C, device=x.device)
            
        # 4. Update Buffer (After retrieval, so we don't retrieve self immediately for the same step?)
        # Actually, for training stability, adding 'current' to buffer might be good or bad.
        # If we add it, we retrieve it next time.
        # We should add it *at the end of the step* or allow retrieval of self?
        # Let's add it *after* retrieval to simulate "predicting based on past".
        # We will add it EXPLICITLY in the training loop, NOT inside the forward pass of a layer,
        # to avoid adding it multiple times per forward/backward pass (which happens a lot).
        # WAIT. forward() is called multiple times? No, once per step usually.
        # But we have multiple layers? If we had multiple layers, we'd add multiple times?
        # PGA usually applies to the *first* layer or *all*?
        # Let's make it simple: The buffer is managed OUTSIDE or only in the first layer?
        # Or better: The Model manages the buffer update. The Layer just Reads.
        # So we pass P or do nothing here?
        # No, the layer calculates P.
        # We will expose `current_essence` or let the training loop handle `buffer.add`.
        
        # --- PGA LOGIC END ---

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Project Q, K, V onto P
        # Q: (B, T, C). P: (C, C).
        # Q' = Q @ P
        q = q @ P
        k = k @ P
        v = v @ P
        
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

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, buffer):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = PGASelfAttention(config, buffer) # Modified
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config, buffer):
        super().__init__()
        self.config = config
        self.buffer = buffer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, buffer) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
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
    def update_buffer_logic(self, idx):
        # Helper to extract and update buffer AFTER a step
        # Get embeddings
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        # Extract essence of this input chunk
        # Note: We use the input embeddings to represent the observation
        essence = extract_essence(x)
        self.buffer.add(essence)

# -----------------------------------------------------------------------------

class Config:
    n_layer = 1
    n_embd = 16
    n_head = 4
    block_size = 16
    vocab_size = None 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="pga_log.txt")
    parser.add_argument('--buffer_file', type=str, default="essences.jsonl")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # 1. Dataset
    if not os.path.exists("input.txt"):
        print("Error: input.txt not found")
        return

    with open("input.txt", 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    
    uchars = sorted(list(set("".join(docs))))
    stoi = {ch:i for i,ch in enumerate(uchars)}
    itos = {i:ch for i,ch in enumerate(uchars)}
    vocab_size = len(uchars) + 1 
    BOS_TOKEN = len(uchars)
    
    config = Config()
    config.vocab_size = vocab_size

    print(f"Dataset: {len(docs)} documents")
    print(f"Vocab size: {vocab_size}")
    
    # Reset Buffer File for new run? 
    # Maybe we want to persist across runs? 
    # For benchmark fairness, let's clear it or use a unique one.
    if os.path.exists(args.buffer_file):
        os.remove(args.buffer_file)
        
    buffer = EssenceBuffer(filename=args.buffer_file)
    model = GPT(config, buffer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    start_time = time.time()
    
    log_file = open(args.log, "w")
    log_file.write("step,loss\n")
    
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
        
        # Update Buffer with the chunk we just saw
        # This is where we simulate "Storing the experience"
        model.update_buffer_logic(data)
        
        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            log_file.write(f"{step},{loss.item()}\n")
            log_file.flush()

    print(f"Training finished in {time.time() - start_time:.2f}s")
    log_file.close()

    print("\n--- Inference ---")
    model.eval()
    for _ in range(5):
        idx = torch.tensor([[BOS_TOKEN]], dtype=torch.long)
        gen = model.generate(idx, max_new_tokens=config.block_size, temperature=0.8)
        out_tokens = gen[0].tolist()
        chars = [itos[t] for t in out_tokens if t != BOS_TOKEN]
        print("".join(chars))

if __name__ == "__main__":
    main()
