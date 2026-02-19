
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Strict Hyperparameters identical to PGA
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
    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        scale = (ms + self.eps).rsqrt()
        return x * scale

class BaselineMicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
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
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
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
            
            # (B, T, C) -> (B, n_head, T, head_dim)
            k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            
            # Standard Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            y = att @ v 
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
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
        return logits, loss
