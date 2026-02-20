"""
E8 — Propagative PGA + Observation Buffer Experiment
Model Definitions:
  1. BaselineMicroGPT       — Standard Transformer (no PGA)
  2. PropagativePGA         — SVD once from embeddings, window-based (same as e7)
  3. PropagativePGABuffer   — SVD once from BUFFER-retrieved vectors, propagative

All models: 5 layers, n_embd=16, n_head=4, block_size=16
PGA adds zero learnable parameters.
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
WINDOW_SIZE = 8     # Causal lookback window (for window-based model)
RANK = 8            # Rank of principle subspace
BUFFER_CAPACITY = 512  # Observation buffer size
RETRIEVE_K = 8      # Number of vectors to retrieve from buffer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt()


# ═══════════════════════════════════════════════════
#  Observation Buffer — Persistent FIFO + Cosine Retrieval
# ═══════════════════════════════════════════════════

class ObservationBuffer:
    """
    Non-parametric epistemological memory.
    Stores essence vectors in a FIFO ring buffer and supports
    cosine-similarity-based retrieval.
    """
    def __init__(self, capacity=BUFFER_CAPACITY, dim=N_EMBD, device='cpu'):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.buffer = torch.zeros(capacity, dim, device=device)
        self.ptr = 0
        self.count = 0

    def store(self, vectors):
        """
        Store vectors into the buffer (FIFO eviction).
        vectors: (N, D) — detached tensor
        """
        vectors = vectors.detach()
        for v in vectors:
            self.buffer[self.ptr] = v
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def retrieve(self, query, k=RETRIEVE_K):
        """
        Retrieve top-k most similar vectors to query via cosine similarity.
        query: (D,) tensor
        Returns: (k, D) tensor of retrieved vectors
        """
        if self.count == 0:
            # Buffer empty — return query itself as fallback
            return query.unsqueeze(0)

        active = self.buffer[:self.count]  # (count, D)

        # Cosine similarity
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)  # (1, D)
        active_norm = F.normalize(active, dim=-1)              # (count, D)
        sims = (query_norm @ active_norm.t()).squeeze(0)       # (count,)

        actual_k = min(k, self.count)
        topk_indices = sims.topk(actual_k).indices
        return active[topk_indices]  # (k, D)

    def retrieve_hybrid(self, query, k=RETRIEVE_K):
        """
        Hybrid retrieval: half similar + half recent.
        This gives both long-term semantic matches and short-term context.
        query: (D,) tensor
        Returns: (k, D) tensor
        """
        if self.count == 0:
            return query.unsqueeze(0)

        half_k = max(1, k // 2)

        # Similar vectors
        active = self.buffer[:self.count]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        sims = (query_norm @ active_norm.t()).squeeze(0)

        sim_k = min(half_k, self.count)
        sim_indices = sims.topk(sim_k).indices
        similar = active[sim_indices]

        # Recent vectors (from the tail of the ring buffer)
        recent_k = min(k - sim_k, self.count)
        if self.count < self.capacity:
            recent_start = max(0, self.count - recent_k)
            recent = self.buffer[recent_start:self.count]
        else:
            indices = [(self.ptr - 1 - i) % self.capacity for i in range(recent_k)]
            recent = self.buffer[torch.tensor(indices, device=self.device)]

        # Combine and deduplicate by stacking
        combined = torch.cat([similar, recent], dim=0)
        # Take unique rows (simple approach: just take first k)
        return combined[:k]

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0
        self.count = 0


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
#  Propagative PGA (Window) — Same as e7
#  SVD computed ONCE from embeddings via sliding window
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

        P = torch.stack(P_stack, dim=1)
        return P

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.rmsnorm(tok_emb + pos_emb)

        P = self.compute_projection_matrices(x)

        for layer in self.layers:
            x_res = x
            x = self.rmsnorm(x)

            q = layer['attn_wq'](x)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)

            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)

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
#  Propagative PGA + Observation Buffer
#  SVD computed ONCE from BUFFER-RETRIEVED vectors
#  Same P used across all 5 layers (propagative)
#  Feedback loop: essence vectors stored back to buffer
# ═══════════════════════════════════════════════════

class PropagativePGABuffer(nn.Module):
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

        # The observation buffer — persists across forward calls
        self.obs_buffer = ObservationBuffer(capacity=BUFFER_CAPACITY, dim=N_EMBD)

        # Learnable projection to map Raw Observations (Query) -> Buffer Space (Keys)
        self.query_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def compute_projection_matrices_from_buffer(self, x):
        """
        For each token, query the observation buffer for similar vectors,
        stack them, run SVD, and build P.
        x: (B, T, D) — detached embeddings
        Returns P: (B, T, D, D)
        """
        B, T, D = x.shape
        P_stack = []
        x_det = x.detach()

        # Project the raw embeddings into "Thought Space" for retrieval consistency
        # x_det is (B, T, D). We project it to (B, T, D).
        x_proj = self.query_proj(x_det)

        for t in range(T):
            batch_Ps = []
            for b in range(B):
                # Use the projected query for retrieval
                query = x_proj[b, t]  # (D,)

                # Retrieve from buffer — hybrid: similar + recent
                retrieved = self.obs_buffer.retrieve_hybrid(query, k=RETRIEVE_K)  # (k, D)

                # Also include the projected query itself for self-context
                context = torch.cat([retrieved, query.unsqueeze(0)], dim=0)  # (k+1, D)
                context = context.unsqueeze(0)  # (1, k+1, D) for SVD

                try:
                    U, S, Vh = torch.linalg.svd(context, full_matrices=False)
                except RuntimeError:
                    batch_Ps.append(torch.eye(D, device=x.device))
                    continue

                k_rank = min(RANK, S.shape[1])
                V_top = Vh[0, :k_rank, :]        # (k_rank, D)
                P_t = V_top.t() @ V_top           # (D, D)
                batch_Ps.append(P_t)

            P_stack.append(torch.stack(batch_Ps))  # (B, D, D)

        P = torch.stack(P_stack, dim=1)  # (B, T, D, D)
        return P

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.rmsnorm(tok_emb + pos_emb)

        # ─── PROPAGATIVE + BUFFER: compute P once from buffer-retrieved context ───
        P = self.compute_projection_matrices_from_buffer(x)
        # P: (B, T, D, D) — fixed for all layers
        # ──────────────────────────────────────────────────────────────────────────

        for layer in self.layers:
            x_res = x
            x = self.rmsnorm(x)

            q = layer['attn_wq'](x)
            k = layer['attn_wk'](x)
            v = layer['attn_wv'](x)

            # PGA Projection: same P at every layer
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)

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

        # ─── FEEDBACK LOOP: store final representations back into buffer ───
        # Store the last layer's output (essence vectors) for each token
        x_det = x.detach()
        for b in range(B):
            self.obs_buffer.store(x_det[b])  # store (T, D) vectors
        # ──────────────────────────────────────────────────────────────────

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
