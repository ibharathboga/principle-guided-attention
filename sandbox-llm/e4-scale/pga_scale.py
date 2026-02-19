
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Scaled Hyperparameters
N_LAYER = 4       # Increase depth
N_EMBD = 64       # Increase width
BLOCK_SIZE = 64   # Increase context
N_HEAD = 4
HEAD_DIM = 16     # n_embd // n_head
VOCAB_SIZE = 65   # Shakespeare has ~65 unique chars

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        scale = (ms + self.eps).rsqrt()
        return x * scale

class PrincipleEngine:
    @staticmethod
    @torch.no_grad()
    def compute_projection_matrix(buffer_stack):
        try:
            U, S, Vh = torch.linalg.svd(buffer_stack, full_matrices=False)
        except RuntimeError:
            B, K, C = buffer_stack.shape
            return torch.eye(C, device=buffer_stack.device).unsqueeze(0).repeat(B, 1, 1)

        # Truncation: Keep top 50% components
        target_rank = max(1, buffer_stack.shape[-1] // 2) 
        
        V_top = Vh[:, :target_rank, :]
        P = V_top.transpose(1, 2) @ V_top
        return P

class PGA_Attention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Compute Ps for each time step t
        Ps = []
        # Optimization: We can compute P every 'stride' steps to save compute if needed,
        # but for 2500 steps on GPU, per-token is fine for d=64.
        # WINDOW: Use full causal history up to BLOCK_SIZE
        
        # STRIDED SVD Optimization
        STRIDE = 4 
        last_P = None
        
        for t in range(T):
            if t % STRIDE == 0 or last_P is None:
                window_size = BLOCK_SIZE 
                start_idx = max(0, t - window_size + 1)
                curr_stack = x[:, start_idx : t+1, :] 
                
                # Compute and cache P
                last_P = PrincipleEngine.compute_projection_matrix(curr_stack) 
            
            # Use cached P for STRIDE steps
            Ps.append(last_P)
            
        Ps = torch.stack(Ps, dim=1) # (B, T, C, C)
        
        # Projects: q[b,t] = q[b,t] @ P[b,t]
        q_proj = torch.einsum('btc,btcd->btd', q, Ps)
        k_proj = torch.einsum('btc,btcd->btd', k, Ps)
        v_proj = torch.einsum('btc,btcd->btd', v, Ps)
        
        k = k_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        return y

class PGAMicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': PGA_Attention(N_EMBD, N_HEAD, BLOCK_SIZE),
                'mlp': nn.Sequential(
                    nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
                    nn.ReLU(),
                    nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
                )
            }) for _ in range(N_LAYER)
        ])
        
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.rmsnorm(x)
        
        for layer in self.layers:
            x_residual = x
            x = self.rmsnorm(x)
            x = layer['attn'](x)
            x = x + x_residual
            
            x_residual = x
            x = self.rmsnorm(x)
            x = layer['mlp'](x)
            x = x + x_residual
            
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
        return logits, loss
