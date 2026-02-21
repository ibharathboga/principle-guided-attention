"""
Guided Principle-Guided Attention (PGA) Implementation.
Uses a frozen Baseline model to provide stable Essence Vectors for the Trainable Model.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import json
from microgpt_torch import GPT, Config, BOS_TOKEN # Import baseline class

# -----------------------------------------------------------------------------
# Re-use components
# -----------------------------------------------------------------------------

class EssenceBuffer:
    def __init__(self, filename="essences_guided.jsonl", max_size=1000):
        self.filename = filename
        self.max_size = max_size
        self.essences = [] 
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

    def add(self, vector):
        self.essences.append(vector.detach().cpu())
        with open(self.filename, 'a') as f:
            f.write(json.dumps({'vector': vector.tolist()}) + '\n')
            
    def retrieve(self, query_vec, k=5):
        if not self.essences:
            return []
        device = query_vec.device
        history = torch.stack(self.essences).to(device)
        norm_history = F.normalize(history, p=2, dim=1)
        norm_query = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        scores = torch.mv(norm_history, norm_query)
        k = min(k, len(self.essences))
        values, indices = torch.topk(scores, k)
        return [history[i] for i in indices]

def extract_essence(x):
    # x: (T, D) or (B, T, D)
    if x.dim() == 3:
        x = x.squeeze(0)
    chunk_centered = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(chunk_centered, full_matrices=False)
        essence = Vh[0, :] 
    except:
        essence = chunk_centered.mean(dim=0)
    return essence

# -----------------------------------------------------------------------------
# Guided Model
# -----------------------------------------------------------------------------

class PGAGuidedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, P=None):
        B, T, C = x.size() 
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # --- PGA Injection ---
        if P is not None:
            # P is (C, C). 
            # Project Q, K, V onto P
            q = q @ P
            k = k @ P
            v = v @ P
        # ---------------------

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

class GuidedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.Identity() # RMSNorm handled in microgpt style or standard? 
        # Wait, I need to match microgpt structure. 
        # MicroGPT Torch implementation uses RMSNorm class.
        from microgpt_torch import RMSNorm, MLP
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = PGAGuidedAttention(config) # Modified
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, P=None):
        x = x + self.attn(self.ln_1(x), P=P)
        x = x + self.mlp(self.ln_2(x))
        return x

class GuidedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        from microgpt_torch import RMSNorm
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GuidedBlock(config) for _ in range(config.n_layer)]),
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

    def forward(self, idx, targets=None, P=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x, P=P) # Pass P down
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # For inference, strictly we should use P too.
        # But where does P come from in inference? 
        # From the GUIDE model on the GENERATED context?
        # Yes.
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, P=None) # Disabling P for inference simplicity/speed for now, or TODO: Implement inference guidance
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# Main System
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--guide_path', type=str, required=True, help="Path to baseline model checkpoint")
    parser.add_argument('--log', type=str, default="pga_guided_log.txt")
    parser.add_argument('--buffer_file', type=str, default="essences_guided.jsonl")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Dataset
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

    # 1. Load Guide (Frozen)
    print(f"Loading guide from {args.guide_path}...")
    guide = GPT.load_pretrained(config, args.guide_path)
    guide.eval()
    for p in guide.parameters():
        p.requires_grad = False
    
    # 2. Init Trainable Model
    model = GuidedGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Buffer
    if os.path.exists(args.buffer_file):
        os.remove(args.buffer_file)
    buffer = EssenceBuffer(filename=args.buffer_file)
    
    print(f"PGA Guided Model training for {args.steps} steps...")
    
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
        
        # --- Guidance Step ---
        with torch.no_grad():
            # Get Guide Embeddings
            guide_emb = guide.get_embeddings(data) # (1, T, D)
            
            # Extract Essence
            essence = extract_essence(guide_emb)
            
            # Retrieve from buffer
            retrieved = buffer.retrieve(essence, k=5)
            
            # Construct P
            P = None
            if retrieved:
                X_stack = torch.stack(retrieved)
                try:
                    _, _, Vh_stack = torch.linalg.svd(X_stack, full_matrices=False)
                    V_top = Vh_stack 
                    P = V_top.T @ V_top # (C, C)
                except:
                    pass
            
            # Update buffer with THIS step's essence
            buffer.add(essence)
        # ---------------------
        
        optimizer.zero_grad()
        logits, loss = model(data, targets, P=P)
        loss.backward()
        optimizer.step()
        
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
