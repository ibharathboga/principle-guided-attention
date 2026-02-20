"""
E9 — RETRO vs Feedback Buffer — 5-Way Comparison
Model Definitions:
  1. BaselineMicroGPT       — Standard Transformer (no PGA)
  2. PropagativePGA         — SVD once from embeddings, window-based
  3. PropagativePGABuffer   — SVD from buffer-retrieved (final outputs), feedback loop
  4. PropagativePGARawBuffer — SVD from buffer-retrieved (raw embeddings), feedback loop
  5. PropagativePGARetro    — SVD from frozen pre-populated buffer, no feedback

All models: 5 layers, n_embd=16, n_head=4, block_size=16
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
WINDOW_SIZE = 8
RANK = 8
BUFFER_CAPACITY = 512
RETRIEVE_K = 8


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt()


# ═══════════════════════════════════════════════════
#  Observation Buffer — FIFO + Cosine Retrieval
# ═══════════════════════════════════════════════════

class ObservationBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY, dim=N_EMBD, device='cpu'):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.buffer = torch.zeros(capacity, dim, device=device)
        self.ptr = 0
        self.count = 0

    def store(self, vectors):
        vectors = vectors.detach()
        for v in vectors:
            self.buffer[self.ptr] = v
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def retrieve(self, query, k=RETRIEVE_K):
        if self.count == 0:
            return query.unsqueeze(0)
        active = self.buffer[:self.count]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        sims = (query_norm @ active_norm.t()).squeeze(0)
        actual_k = min(k, self.count)
        topk_indices = sims.topk(actual_k).indices
        return active[topk_indices]

    def retrieve_hybrid(self, query, k=RETRIEVE_K):
        if self.count == 0:
            return query.unsqueeze(0)
        half_k = max(1, k // 2)
        active = self.buffer[:self.count]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        sims = (query_norm @ active_norm.t()).squeeze(0)
        sim_k = min(half_k, self.count)
        sim_indices = sims.topk(sim_k).indices
        similar = active[sim_indices]
        recent_k = min(k - sim_k, self.count)
        if self.count < self.capacity:
            recent_start = max(0, self.count - recent_k)
            recent = self.buffer[recent_start:self.count]
        else:
            indices = [(self.ptr - 1 - i) % self.capacity for i in range(recent_k)]
            recent = self.buffer[torch.tensor(indices, device=self.device)]
        combined = torch.cat([similar, recent], dim=0)
        return combined[:k]

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0
        self.count = 0

    def get_all_active(self):
        """Return all active vectors in the buffer."""
        return self.buffer[:self.count].clone()


# ═══════════════════════════════════════════════════
#  Shared: Transformer Block Logic
# ═══════════════════════════════════════════════════

def make_layers():
    return nn.ModuleList([
        nn.ModuleDict({
            'attn_wq': nn.Linear(N_EMBD, N_EMBD, bias=False),
            'attn_wk': nn.Linear(N_EMBD, N_EMBD, bias=False),
            'attn_wv': nn.Linear(N_EMBD, N_EMBD, bias=False),
            'attn_wo': nn.Linear(N_EMBD, N_EMBD, bias=False),
            'mlp_fc1': nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            'mlp_fc2': nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        }) for _ in range(N_LAYER)
    ])


def run_layers(layers, x, P, rmsnorm, idx_device):
    """Run transformer layers with optional PGA projection P."""
    B, T, _ = x.shape
    for layer in layers:
        x_res = x
        x = rmsnorm(x)

        q = layer['attn_wq'](x)
        k = layer['attn_wk'](x)
        v = layer['attn_wv'](x)

        if P is not None:
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)

        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        att = att.masked_fill(
            torch.tril(torch.ones(T, T, device=idx_device)) == 0, float('-inf')
        )
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)

        x = layer['attn_wo'](y) + x_res

        x_res = x
        x = rmsnorm(x)
        x = layer['mlp_fc1'](x)
        x = F.relu(x)
        x = layer['mlp_fc2'](x)
        x = x + x_res

    return x


def compute_P_from_buffer(obs_buffer, x_queries, rank=RANK, retrieve_k=RETRIEVE_K):
    """
    Build per-token projection matrices from buffer retrieval.
    x_queries: (B, T, D) — the vectors used to query the buffer
    Returns P: (B, T, D, D)
    """
    B, T, D = x_queries.shape
    P_stack = []
    x_det = x_queries.detach()

    for t in range(T):
        batch_Ps = []
        for b in range(B):
            query = x_det[b, t]
            retrieved = obs_buffer.retrieve_hybrid(query, k=retrieve_k)
            context = torch.cat([retrieved, query.unsqueeze(0)], dim=0)
            context = context.unsqueeze(0)

            try:
                U, S, Vh = torch.linalg.svd(context, full_matrices=False)
            except RuntimeError:
                batch_Ps.append(torch.eye(D, device=x_queries.device))
                continue

            k_rank = min(rank, S.shape[1])
            V_top = Vh[0, :k_rank, :]
            P_t = V_top.t() @ V_top
            batch_Ps.append(P_t)

        P_stack.append(torch.stack(batch_Ps))

    P = torch.stack(P_stack, dim=1)
    return P


# ═══════════════════════════════════════════════════
#  1. Baseline MicroGPT
# ═══════════════════════════════════════════════════

class BaselineMicroGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        x = run_layers(self.layers, x, None, self.rmsnorm, idx.device)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  2. Propagative PGA (Window) — same as e7
# ═══════════════════════════════════════════════════

class PropagativePGA(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)

    def compute_projection_matrices(self, x):
        B, T, D = x.shape
        P_stack = []
        x_det = x.detach()
        for t in range(T):
            start = max(0, t - WINDOW_SIZE + 1)
            context = x_det[:, start:t+1, :]
            try:
                U, S, Vh = torch.linalg.svd(context, full_matrices=False)
            except RuntimeError:
                P_stack.append(torch.eye(D, device=x.device).unsqueeze(0).expand(B, -1, -1))
                continue
            k = min(RANK, S.shape[1])
            V_top = Vh[:, :k, :]
            P_t = torch.bmm(V_top.transpose(1, 2), V_top)
            P_stack.append(P_t)
        return torch.stack(P_stack, dim=1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        P = self.compute_projection_matrices(x)
        x = run_layers(self.layers, x, P, self.rmsnorm, idx.device)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  3. Prop-PGA + Buffer (Feedback, Final Outputs)
#     Stores final-layer outputs, queries with query_proj(raw)
# ═══════════════════════════════════════════════════

class PropagativePGABuffer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.obs_buffer = ObservationBuffer(capacity=BUFFER_CAPACITY, dim=N_EMBD)
        self.query_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        # Query buffer with projected raw embeddings
        x_proj = self.query_proj(x.detach())
        P = compute_P_from_buffer(self.obs_buffer, x_proj)

        x = run_layers(self.layers, x, P, self.rmsnorm, idx.device)

        # Feedback: store final-layer outputs
        x_det = x.detach()
        for b in range(B):
            self.obs_buffer.store(x_det[b])

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  4. Raw Observation Buffer (Feedback, Raw Embeddings)
#     Stores rmsnorm(tok+pos), queries with rmsnorm(tok+pos)
#     No query_proj needed — same vector space
# ═══════════════════════════════════════════════════

class PropagativePGARawBuffer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.obs_buffer = ObservationBuffer(capacity=BUFFER_CAPACITY, dim=N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        # Query buffer with raw embeddings (same space as what's stored)
        P = compute_P_from_buffer(self.obs_buffer, x)

        x_raw = x.detach()  # Save raw for storage BEFORE layers modify x

        x = run_layers(self.layers, x, P, self.rmsnorm, idx.device)

        # Feedback: store RAW embeddings (not final outputs)
        for b in range(B):
            self.obs_buffer.store(x_raw[b])

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  5. RETRO Buffer (Frozen, Pre-populated from Baseline)
#     Buffer is pre-filled and NEVER updated during training
#     Queries with query_proj(raw)
# ═══════════════════════════════════════════════════

class PropagativePGARetro(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.obs_buffer = ObservationBuffer(capacity=BUFFER_CAPACITY, dim=N_EMBD)
        self.query_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def populate_buffer(self, vectors):
        """
        Pre-populate the buffer with frozen vectors from a trained baseline.
        vectors: (N, D) tensor — detached outputs from a pre-trained model.
        """
        self.obs_buffer.reset()
        self.obs_buffer.store(vectors.detach())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        # Query the frozen buffer with projected raw embeddings
        x_proj = self.query_proj(x.detach())
        P = compute_P_from_buffer(self.obs_buffer, x_proj)

        x = run_layers(self.layers, x, P, self.rmsnorm, idx.device)

        # NO feedback loop — buffer is frozen

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
