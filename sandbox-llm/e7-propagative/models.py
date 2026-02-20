"""
E7 — Propagative PGA Experiment
Model Definitions: Baseline MicroGPT and Propagative PGA MicroGPT

Both models: 5 layers, n_embd=16, n_head=4, block_size=16
PGA adds zero learnable parameters — SVD is computed from embeddings once
and the same P is used across all 5 layers (propagative nature).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─── Hyperparameters ───
N_LAYER = 5
N_EMBD = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD  # 4

# PGA Hyperparameters
WINDOW_SIZE = 8   # Causal lookback window for SVD context
RANK = 8          # Rank of principle subspace


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt()


# ═══════════════════════════════════════════════════
#  Baseline MicroGPT — Standard Transformer, No PGA
# ═══════════════════════════════════════════════════

class BaselineMicroGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
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

        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.rmsnorm(tok_emb + pos_emb)

        for layer in self.layers:
            x_res = x
            x = self.rmsnorm(x)

            q = layer['attn_wq'](x)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)

            k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=idx.device)) == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)

            x = layer['attn_wo'](y) + x_res

            x_res = x
            x = self.rmsnorm(x)
            x = layer['mlp_fc1'](x)
            x = F.relu(x)
            x = layer['mlp_fc2'](x)
            x = x + x_res

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  Propagative PGA MicroGPT
#  SVD computed ONCE from embeddings, same P for all 5 layers
# ═══════════════════════════════════════════════════

class PropagativePGA(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
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

        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def compute_projection_matrices(self, x):
        """
        Compute P for each token from its causal context window.
        x: (B, T, D) — the embedding-level representation.
        Returns P: (B, T, D, D) — one P per token, to be reused across ALL layers.
        """
        B, T, D = x.shape
        P_stack = []

        # Detach: principle discovery is not part of the gradient graph
        x_det = x.detach()

        for t in range(T):
            start = max(0, t - WINDOW_SIZE + 1)
            context = x_det[:, start:t+1, :]  # (B, window_len, D)

            try:
                U, S, Vh = torch.linalg.svd(context, full_matrices=False)
            except RuntimeError:
                P_stack.append(torch.eye(D, device=x.device).unsqueeze(0).expand(B, -1, -1))
                continue

            k = min(RANK, S.shape[1])
            V_top = Vh[:, :k, :]                          # (B, k, D)
            P_t = torch.bmm(V_top.transpose(1, 2), V_top) # (B, D, D)
            P_stack.append(P_t)

        P = torch.stack(P_stack, dim=1)  # (B, T, D, D)
        return P

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.rmsnorm(tok_emb + pos_emb)

        # ─── PROPAGATIVE: Compute P ONCE from embeddings ───
        P = self.compute_projection_matrices(x)
        # P: (B, T, D, D) — fixed for all layers below
        # ───────────────────────────────────────────────────

        for layer in self.layers:
            x_res = x
            x = self.rmsnorm(x)

            q = layer['attn_wq'](x)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)

            # ─── PGA Projection: same P at every layer ───
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)
            # ──────────────────────────────────────────────

            k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=idx.device)) == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)

            x = layer['attn_wo'](y) + x_res

            x_res = x
            x = self.rmsnorm(x)
            x = layer['mlp_fc1'](x)
            x = F.relu(x)
            x = layer['mlp_fc2'](x)
            x = x + x_res

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
