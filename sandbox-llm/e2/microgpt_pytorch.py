import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Strict Hyperparameters from microgpt.py
N_LAYER = 1
N_EMBD = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = 4 # n_embd // n_head
VOCAB_SIZE = 27 # 26 letters + 1 BOS

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # microgpt.py: scale = (ms + 1e-5)**-0.5; out = x * scale. 
        # It does NOT have a learnable weight parameter g.
        # Wait, let's double check microgpt.py:
        # def rmsnorm(x): ms = ...; scale = ...; return [xi * scale for xi in x]
        # Pure RMSNorm with no affine parameters.
        pass

    def forward(self, x):
        # x: (batch, seq, dim)
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        scale = (ms + self.eps).rsqrt()
        return x * scale

class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # Layers
        # In microgpt.py, weights are matrices. 
        # layer0.attn_wq is (n_embd, n_embd)
        # We use nn.Linear(bias=False) to match.
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_wq': nn.Linear(N_EMBD, N_EMBD, bias=False),
                'attn_wk': nn.Linear(N_EMBD, N_EMBD, bias=False),
                'attn_wv': nn.Linear(N_EMBD, N_EMBD, bias=False),
                'attn_wo': nn.Linear(N_EMBD, N_EMBD, bias=False),
                'mlp_fc1': nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
                'mlp_fc2': nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
            }) for _ in range(N_LAYER)
        ])
        
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.wte(idx) # (B, T, C)
        pos_emb = self.wpe(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        
        x = self.rmsnorm(x)
        
        for layer in self.layers:
            # 1) Attention
            x_residual = x
            x = self.rmsnorm(x)
            
            # Q, K, V
            q = layer['attn_wq'](x)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)
            
            # Reshape for multi-head
            # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
            k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            
            # Attention
            # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            
            # Apply Mask (Wait, microgpt.py doesn't seem to apply a causal mask in the attention loop?)
            # Let's check microgpt.py: 
            # `for t in range(len(k_h))` ... 
            # `attn_logits = [sum(q_h[j] * k_h[t][j] ...) for t in range(len(k_h))]`
            # Wait, `k_h` is *all previous keys*?
            # In `gpt()`: `keys[li].append(k)`. 
            # `keys` grows step by step. 
            # So at step `pos_id`, `keys` has length `pos_id + 1`.
            # This implicitly implements causal attention because we only attend to keys we've seen so far.
            # In PyTorch, since we process the whole sequence at once (usually), we need a mask.
            # BUT, microgpt.py processes token by token sequentially. 
            # If we want exact parity with the *training loop* in microgpt.py, we should adhere to that.
            # However, PyTorch is usually batched.
            # To prove equivalence, let's apply a Causal Mask.
            
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=idx.device)) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
            y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)
            
            x = layer['attn_wo'](y)
            x = x + x_residual
            
            # 2) MLP
            x_residual = x
            x = self.rmsnorm(x)
            x = layer['mlp_fc1'](x)
            x = F.relu(x)
            x = layer['mlp_fc2'](x)
            x = x + x_residual
            
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
        return logits, loss
