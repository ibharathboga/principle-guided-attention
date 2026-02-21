"""
JIT Principle-Guided Attention (PGA) Implementation (v4).
- Stores Tokens in Buffer.
- Retrieves Tokens -> Re-embeds with Current Model -> Computes Fresh Essence.
- Projects Attention using Fresh Essence.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import json
import random
# Using the copied microgpt_torch in current dir
from microgpt_torch import GPT, Config, BOS_TOKEN, RMSNorm, MLP, CausalSelfAttention

# -----------------------------------------------------------------------------
# JIT Components
# -----------------------------------------------------------------------------

class JITEssenceBuffer:
    def __init__(self, filename="essences_jit.jsonl", max_size=2000):
        self.filename = filename
        self.max_size = max_size
        self.data = [] # List of dicts: {'tokens': tensor, 'cached_vec': tensor}
        # We keep cached_vec for approximate retrieval
        
        # Clear file if exists for fresh start
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def add(self, tokens, vector):
        # tokens: (T,)
        # vector: (D,)
        entry = {'tokens': tokens.detach().cpu(), 'cached_vec': vector.detach().cpu()}
        self.data.append(entry)
        
        # Maintain max size (random drop or FIFO? FIFO is safer for now)
        if len(self.data) > self.max_size:
            self.data.pop(0)

        # Persistence (optional, maybe skip for speed in this test)
        # with open(self.filename, 'a') as f:
        #     f.write(json.dumps({'tokens': tokens.tolist(), 'vector': vector.tolist()}) + '\n')
            
    def retrieve_tokens(self, query_vec, k=5):
        if not self.data:
            return []
        
        device = query_vec.device
        
        # Stack cached vectors for search
        history = torch.stack([d['cached_vec'] for d in self.data]).to(device) # (N, D)
        
        norm_history = F.normalize(history, p=2, dim=1)
        norm_query = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        
        scores = torch.mv(norm_history, norm_query)
        k = min(k, len(self.data))
        values, indices = torch.topk(scores, k)
        
        # Return the TOKENS corresponding to top k
        return [self.data[i]['tokens'].to(device) for i in indices]

def extract_essence(x):
    # x: (T, D)
    chunk_centered = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(chunk_centered, full_matrices=False)
        essence = Vh[0, :] 
    except:
        essence = chunk_centered.mean(dim=0)
    return essence

# -----------------------------------------------------------------------------
# PGA Model Override
# -----------------------------------------------------------------------------

class PGASelfAttention(nn.Module):
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
        
        # --- Soft PGA Injection ---
        if P is not None:
            # P is (C, C)
            # Soft Projection: q = q + alpha * (q @ P)
            # Let's try explicit residual scaling. 
            # Ideally alpha is learned, but for now let's say alpha=1.0 (additive)
            # or we just project.
            # User suggested: q = q + alpha * (q @ P)
            # Let's stick to simple additive for now.
            
            q_proj = q @ P
            k_proj = k @ P
            v_proj = v @ P
            
            # Simple addition (Residual)
            q = q + q_proj
            k = k + k_proj
            v = v + v_proj
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

class PGABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = PGASelfAttention(config) # Use PGA Attention
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, P=None):
        x = x + self.attn(self.ln_1(x), P=P)
        x = x + self.mlp(self.ln_2(x))
        return x

class PGAGPT(GPT):
    def __init__(self, config):
        # We need to init parent, but override the transformer blocks
        super().__init__(config)
        # Re-build blocks with PGA
        self.transformer.h = nn.ModuleList([PGABlock(config) for _ in range(config.n_layer)])
        self.apply(self._init_weights)
        
    def forward(self, idx, targets=None, P=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x, P=P)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------
# Main & JIT Loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default="pga_jit_log.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Dataset
    if not os.path.exists("input.txt"):
        print("Error: input.txt not found")
        return
    with open("input.txt", 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    
    # Split Train/Val (90/10)
    random.shuffle(docs)
    split_idx = int(len(docs) * 0.9)
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]
    
    uchars = sorted(list(set("".join(docs))))
    stoi = {ch:i for i,ch in enumerate(uchars)}
    itos = {i:ch for i,ch in enumerate(uchars)}
    vocab_size = len(uchars) + 1 
    BOS_TOKEN = len(uchars)
    
    config = Config()
    config.vocab_size = vocab_size

    model = PGAGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    buffer = JITEssenceBuffer()
    
    print(f"PGA JIT Model training for {args.steps} steps...")
    
    log_file = open(args.log, "w")
    log_file.write("step,train_loss,val_loss,essence_norm,proj_mag\n")
    
    start_time = time.time()
    
    for step in range(args.steps):
        # 1. Get Data
        doc = train_docs[step % len(train_docs)]
        tokens = [BOS_TOKEN] + [stoi[ch] for ch in doc] + [BOS_TOKEN]
        if len(tokens) - 1 > config.block_size:
             tokens = tokens[:config.block_size + 1]
        
        data = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0) 
        targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0)
        
        # 2. Get Current Embeddings & Essence
        with torch.no_grad():
            curr_emb = model.get_embeddings(data) # (1, T, D)
            curr_essence = extract_essence(curr_emb.squeeze(0)) # (D,) JIT!
            essence_norm = curr_essence.norm().item()
            
            # 3. Retrieve TOKENS
            retrieved_tokens_list = buffer.retrieve_tokens(curr_essence, k=5)
            
            # 4. JIT Re-Calculate retrieved essences
            fresh_essences = []
            if retrieved_tokens_list:
                # Batchify ideally, but loop is fine for k=5
                for r_tok in retrieved_tokens_list:
                    # r_tok is (T_old,)
                    # Need to pass through current model wte/wpe
                    # But r_tok length might vary? 
                    # For now assume we deal with whatever size.
                    r_inp = r_tok.unsqueeze(0) # (1, T)
                    r_emb = model.get_embeddings(r_inp).squeeze(0) # (T, D)
                    fresh_essences.append(extract_essence(r_emb))
            
            # 5. Form Projection P
            P = None
            proj_mag = 0.0
            if fresh_essences:
                X_stack = torch.stack(fresh_essences) # (K, D)
                try:
                    _, _, Vh_stack = torch.linalg.svd(X_stack, full_matrices=False)
                    V_top = Vh_stack # All components or top? Let's use all for now as subspace
                    P = V_top.T @ V_top # (D, D) Projector
                except:
                    pass
            
            # 6. Store current (tokens, essence)
            # Note: storing tokens and *current* essence as cache key
            buffer.add(data.squeeze(0), curr_essence)

        # 7. Forward Pass with P
        model.train()
        optimizer.zero_grad()
        logits, loss = model(data, targets, P=P)
        loss.backward()
        optimizer.step()
        
        # Calc Projection Magnitude metric if P existed
        if P is not None:
             # Just an approximation: norm(x - xP) / norm(x) ?
             # Let's just log if P is None or not for now, or use a sample
             # proj_mag = ((curr_emb @ P).norm() / curr_emb.norm()).item()
             pass

        # 8. Validation & Logging
        if step % 50 == 0:
            val_loss = 0.0
            # Run a few val batches
            model.eval()
            with torch.no_grad():
                for idx in range(5):
                    v_doc = val_docs[idx % len(val_docs)]
                    v_tok = [BOS_TOKEN] + [stoi[ch] for ch in v_doc] + [BOS_TOKEN]
                    if len(v_tok) - 1 > config.block_size: v_tok = v_tok[:config.block_size+1]
                    v_data = torch.tensor(v_tok[:-1], dtype=torch.long).unsqueeze(0)
                    v_targ = torch.tensor(v_tok[1:], dtype=torch.long).unsqueeze(0)
                    _, v_l = model(v_data, v_targ, P=None) # Validate without P? Or with? 
                    # Ideally with P, but P depends on retrieval. 
                    # Let's validate without P for "Raw Model capability" or with P?
                    # The Baseline doesn't use P. To be fair, PGA should use P.
                    #But constructing P for val is slow. Let's validate RAW model to see if weights improved?
                    # No, we want to know if PGA improves prediction.
                    # We should disable P for speed/stability in simple val check, or enable if we want to test PGA generalization.
                    # Let's leave P=None for val to check "internal knowledge", and trust Training Loss for P effectiveness.
                    val_loss += v_l.item()
            val_loss /= 5
            
            print(f"Step {step:4d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Norm: {essence_norm:.2f}")
            log_file.write(f"{step},{loss.item()},{val_loss},{essence_norm},{proj_mag}\n")
            log_file.flush()

    print(f"Training finished in {time.time() - start_time:.2f}s")
    log_file.close()

if __name__ == "__main__":
    main()
