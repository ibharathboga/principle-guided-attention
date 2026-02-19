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

# PGA Hyperparameters
WINDOW_SIZE = 8 # Look back window
RANK = 8       # Rank of principle subspace

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        scale = (ms + self.eps).rsqrt()
        return x * scale

class MicroGPTPGA(nn.Module):
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

    def compute_projection_matrices(self, x):
        """
        Computes the Projection Matrix P for every token in the sequence.
        x: (Batch, SeqLen, Dim)
        Returns P: (Batch, SeqLen, Dim, Dim)
        """
        B, T, D = x.shape
        P_stack = []
        
        # Stop Gradient for Principle Discovery
        x_detached = x.detach()
        
        # Iterate through sequence (Straightforward approach)
        for t in range(T):
            # Define context window: [max(0, t-window+1) : t+1]
            # We include the current token 't' in the context as the "Observation"
            # In the original python implementation we used the buffer + current.
            start = max(0, t - WINDOW_SIZE + 1)
            end = t + 1
            
            # Context: (B, WindowLen, D)
            context = x_detached[:, start:end, :]
            
            # Compute SVD for each batch element
            # Note: For very small batches/sequences, a loop over B might be readable, 
            # but torch.linalg.svd supports batching.
            # context shape: (B, L, D) -> svd -> U, S, Vh
            
            # If context is too small (e.g. at start), handled gracefully by SVD
            try:
                # Vh shape: (B, D, D) (if full_matrices=False and L >= D) 
                # or (B, L, D) (if L < D)
                # We need V (D, D) or (D, L). Vh is V.T.
                # So row 'i' of Vh is the i-th eigenvector.
                U, S, Vh = torch.linalg.svd(context, full_matrices=False)
            except RuntimeError:
                # Fallback implementation or identity if SVD fails
                P_t = torch.eye(D, device=x.device).unsqueeze(0).repeat(B, 1, 1)
                P_stack.append(P_t)
                continue
            
            # Select top k components
            # Vh: (B, min(L,D), D)
            # We want top components. They are the first rows of Vh.
            
            # Effective Rank
            k = min(RANK, S.shape[1])
            
            V_top = Vh[:, :k, :] # (B, k, D)
            
            # P = V.T @ V (if V is (D, k))
            # Here V_top is (k, D), effectively V^T in standard notation (rows are vectors).
            # So P = V_top.T @ V_top.
            # (B, D, k) @ (B, k, D) -> (B, D, D)
            
            P_t = torch.bmm(V_top.transpose(1, 2), V_top)
            P_stack.append(P_t)
            
        # Stack all steps
        P = torch.stack(P_stack, dim=1) # (B, T, D, D)
        return P

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.rmsnorm(x)
        
        # --- PGA Step ---
        # Compute Projection Matrices based on x (the embedding input)
        # In multi-layer setting, this might be re-computed per layer,
        # but for microgpt (1 layer), we do it once here or inside the layer loop.
        # Let's do it here based on embedding "observations".
        P = self.compute_projection_matrices(x)
        # ----------------
        
        for layer in self.layers:
            x_residual = x
            x = self.rmsnorm(x)
            
            # Standard QKV
            q = layer['attn_wq'](x) # (B, T, D)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)
            
            # --- PGA Projection ---
            # Apply P to Q, K, V
            # q: (B, T, D), P: (B, T, D, D)
            # q_new[b,t] = q[b,t] @ P[b,t]
            # (1, D) @ (D, D) -> (1, D)
            # unsqueeze q: (B, T, 1, D)
            # P: (B, T, D, D)
            # matmul: (B, T, 1, D) @ (B, T, D, D) -> (B, T, 1, D)
            
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)
            # ----------------------
            
            k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            
            # Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=idx.device)) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)
            
            x = layer['attn_wo'](y)
            x = x + x_residual
            
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
