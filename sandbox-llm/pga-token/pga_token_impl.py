
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Strict Hyperparameters identical to baseline for fair comparison
N_LAYER = 1
N_EMBD = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = 4 # n_embd // n_head
VOCAB_SIZE = 27 # 26 letters + 1 BOS (lowercase only + space/special)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        scale = (ms + self.eps).rsqrt()
        return x * scale

class PrincipleEngine:
    """
    Handles the compute of the Principle Matrix P via SVD on the Observation Buffer.
    """
    @staticmethod
    def compute_projection_matrix(buffer_stack):
        """
        buffer_stack: (B, K, C) - The stack of essence vectors (observations).
        Returns P: (B, C, C) - The projection matrix.
        """
        # SVD
        # torch.linalg.svd is batched.
        # U: (B, K, K), S: (B, min(K, C)), Vh: (B, C, C)
        # We want V (right singular vectors). Vh is V^T.
        # So the rows of Vh are the eigenvectors of X^T X.
        
        # Robust SVD handling
        try:
            U, S, Vh = torch.linalg.svd(buffer_stack, full_matrices=False)
        except RuntimeError:
            # Fallback for stability if SVD fails (rare with small matrices)
            B, K, C = buffer_stack.shape
            return torch.eye(C, device=buffer_stack.device).unsqueeze(0).repeat(B, 1, 1)

        # Truncation: Select top components using energy threshold or fixed rank.
        # Notion says: "Select top columns of V that correspond to largest singular values."
        # Vh has shape (B, R, C) where R = min(K, C).
        # The rows of Vh are the principal axes (if K >= C).
        # We want to form P = V_top^T V_top.
        # If we keep all, P is Identity (if rank is full).
        # We need to truncate to enforce "Subspace Regularization".
        # Let's keep top 50-75% energy or fixed rank.
        # For d=16, let's keep top dim // 2 = 8 components?
        # Or let's use a soft threshold relative to sum of S.
        
        # "Truncation step... fixed energy threshold... or hard-coded rank"
        # Let's use a fixed rank for consistent behavior in this micro-experiment.
        target_rank = max(1, buffer_stack.shape[-1] // 2) # Keep half dimensions (8/16)
        
        # Vh is (B, min(K,C), C). We want top 'target_rank' rows.
        V_top = Vh[:, :target_rank, :] # (B, r, C)
        
        # P = V^T V
        # (B, C, r) @ (B, r, C) -> (B, C, C)
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

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        # In PGA, we need to compute P for each time step or for the whole block based on causal history.
        # Since this is a "Micro" implementation, we can iterate to be explicit and correct 'Token-Based'.
        # However, for efficiency in PyTorch training, we usually batch.
        # But PGA effectively requires a different P for each token t, derived from x[:t].
        
        # Let's compute a tensor of Ps: (B, T, C, C)
        # This is expensive but necessary for "Token-Based" fidelity.
        
        # Construct stack for each t
        # Contexts: A list of tensors
        
        # Optimization: We can compute Q, K, V first, then project?
        # "The Query, Key, and Value projections are filtered through P"
        # Q_raw, K_raw, V_raw = self.c_attn(x).split(C, dim=2)
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2) # (B, T, C)
        
        # Now we need to project q[t], k[t], v[t] using P[t].
        # P[t] comes from x[:t+1] (inclusive of current observation? Doc says "Before processing... retrieves").
        # Doc: "Before processing a new token x_t... query memory... stack... SVD... P... Attention filtered".
        # If strictly causal: Context for x_t should probably include x_t itself if it's "Recalling" it? 
        # Notion: "Input: Token sequence... Output: Essence... 3. Contextual Retrieval... 4. SVD... 5. Filter... 6. Attention".
        # If we use x_t to verify x_t, it might just confirm itself. 
        # Usually "Context" means history x_{0:t}.
        # Let's assume Context = x_{0:t+1} (inclusive) for the "Principle of the moment".
        
        Ps = []
        for t in range(T):
            # Windowing: effectively "retrieval". 
            # We take last K tokens.
            window_size = 16 # Use full block size as lookback
            start_idx = max(0, t - window_size + 1)
            
            # Stack (B, current_window, C)
            curr_stack = x[:, start_idx : t+1, :] 
            
            # Compute P for this step
            P_t = PrincipleEngine.compute_projection_matrix(curr_stack) # (B, C, C)
            Ps.append(P_t)
            
        Ps = torch.stack(Ps, dim=1) # (B, T, C, C)
        
        # Apply Projection: Q' = Q P ?
        # Dimensions: Q is (B, T, C). P is (B, T, C, C).
        # We want to apply P_t to Q_t.
        # einsum 'b t c, b t c d -> b t d' ?? No.
        # P is symmetric CxC.
        # Vector v (1, C). v @ P -> (1, C).
        # So Q_t @ P_t.
        # Tensordot/Einsum: 
        # q: (b, t, c)
        # P: (b, t, c, c_out)
        # res: (b, t, c_out) -> sum_c (q_btc * P_btcc)
        
        q_proj = torch.einsum('btc,btcd->btd', q, Ps)
        k_proj = torch.einsum('btc,btcd->btd', k, Ps)
        v_proj = torch.einsum('btc,btcd->btd', v, Ps)
        
        # Proceed with standard attention on projected vectors
        # Reshape to (B, n_head, T, head_dim)
        k = k_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v_proj.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
            
            # ATTENTION with PGA
            x = layer['attn'](x)
            x = x + x_residual
            
            x_residual = x
            x = self.rmsnorm(x)
            
            # MLP
            x = layer['mlp'](x)
            x = x + x_residual
            
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
