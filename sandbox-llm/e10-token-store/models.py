"""
E10 — Token-Keyed Store + Improved PGA Buffer — 3-Way Comparison
Model Definitions:
  1. BaselineMicroGPT        — Standard Transformer (no PGA)
  2. E9StylePGABuffer        — FIFO ring buffer, 50/50 retrieval (control from e9)
  3. ImprovedPGABuffer       — Token-keyed store, 70/30 retrieval, batched SVD

All models: 5 layers, n_embd=16, n_head=4, block_size=16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# ─── Hyperparameters ───
N_LAYER = 5
N_EMBD = 32
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD  # 4

# PGA Hyperparameters
RANK = 4
RETRIEVE_K = 8
WINDOW_SIZE = 8  # for E9-style buffer

# E9 buffer config
E9_BUFFER_CAPACITY = 512

# E10 retrieval split
SIM_RATIO = 0.7   # 70% similarity
REC_RATIO = 0.3   # 30% recency

# E10 store cap — max vectors stored per token_id
MAX_PER_TOKEN = 10_000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt()


# ═══════════════════════════════════════════════════
#  E9-Style Observation Buffer — FIFO + 50/50 Retrieval
# ═══════════════════════════════════════════════════

class ObservationBuffer:
    """FIFO ring buffer with 50/50 similarity/recency retrieval (from e9)."""

    def __init__(self, capacity=E9_BUFFER_CAPACITY, dim=N_EMBD, device='cpu'):
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


# ═══════════════════════════════════════════════════
#  E10 Token-Keyed Vector Store — Unbounded, Persistent
# ═══════════════════════════════════════════════════

class TokenKeyedStore:
    """
    Key:   token_id (int)
    Value: list of contextualized vectors (model outputs for that token)

    No eviction — unbounded growth.
    At 16-dim, 5000 steps × 16 tokens = 80K vectors ≈ 5MB.
    """

    def __init__(self, dim=N_EMBD, device='cpu'):
        self.dim = dim
        self.device = device
        self.store = {}        # token_id → list of (D,) tensors
        self.all_vectors = []  # flat list of all vectors for global search
        self.all_keys = []     # corresponding token_ids
        self._global_tensor = None  # cached tensor for cosine search
        self._dirty = True     # whether cache needs rebuild
        self.total_count = 0

    def store_vectors(self, token_ids, vectors):
        """
        Store contextualized vectors keyed by their token_ids.
        token_ids: (T,) int tensor — the input token ids
        vectors: (T, D) float tensor — the model's final-layer outputs
        """
        vectors = vectors.detach()
        token_ids = token_ids.detach()
        for i in range(len(token_ids)):
            tid = token_ids[i].item()
            v = vectors[i]
            if tid not in self.store:
                self.store[tid] = []
            self.store[tid].append(v)
            self.all_vectors.append(v)
            self.all_keys.append(tid)
            self.total_count += 1

            # Evict oldest vectors when a token exceeds the cap
            if len(self.store[tid]) > MAX_PER_TOKEN:
                n_evict = len(self.store[tid]) - MAX_PER_TOKEN
                self.store[tid] = self.store[tid][n_evict:]
                # Mark for full cache rebuild (evicted vectors still in flat lists)
                self._needs_full_rebuild = True

        # Rebuild flat lists if any eviction happened
        if getattr(self, '_needs_full_rebuild', False):
            self.all_vectors = []
            self.all_keys = []
            self.total_count = 0
            for tid_key, vecs in self.store.items():
                self.all_vectors.extend(vecs)
                self.all_keys.extend([tid_key] * len(vecs))
                self.total_count += len(vecs)
            self._needs_full_rebuild = False

        self._dirty = True

    def _rebuild_cache(self):
        """Rebuild the global tensor cache for cosine search."""
        if self.total_count == 0:
            self._global_tensor = None
        else:
            self._global_tensor = torch.stack(self.all_vectors)  # (N, D)
        self._dirty = False

    def retrieve(self, token_id, query, k=RETRIEVE_K):
        """
        Retrieve k vectors with same-token-id priority.
        1. Fetch all vectors for this exact token_id
        2. If < k, fill with global cosine similarity
        3. From combined pool: 70% similarity, 30% recency
        """
        if self.total_count == 0:
            return query.unsqueeze(0)

        k_sim = max(1, int(k * SIM_RATIO))    # 5-6
        k_rec = k - k_sim                      # 2-3

        # Step 1: same-token-id vectors
        same_token_vecs = self.store.get(token_id, [])

        if len(same_token_vecs) == 0:
            # No history for this token — fall back to global retrieval
            return self._global_retrieve(query, k, k_sim, k_rec)

        same_tensor = torch.stack(same_token_vecs)  # (M, D)

        if len(same_token_vecs) >= k:
            # Enough same-token vectors — do sim/recency split within them
            query_norm = F.normalize(query.unsqueeze(0), dim=-1)
            same_norm = F.normalize(same_tensor, dim=-1)
            sims = (query_norm @ same_norm.t()).squeeze(0)

            # Similarity portion
            actual_sim_k = min(k_sim, len(same_token_vecs))
            sim_indices = sims.topk(actual_sim_k).indices
            sim_vecs = same_tensor[sim_indices]

            # Recency portion (last k_rec vectors)
            actual_rec_k = min(k_rec, len(same_token_vecs))
            rec_vecs = same_tensor[-actual_rec_k:]

            combined = torch.cat([sim_vecs, rec_vecs], dim=0)
            return combined[:k]
        else:
            # Not enough same-token — get remaining from global pool
            remaining = k - len(same_token_vecs)
            global_vecs = self._global_similarity(query, remaining, exclude_token=token_id)
            combined = torch.cat([same_tensor, global_vecs], dim=0)
            return combined[:k]

    def _global_retrieve(self, query, k, k_sim, k_rec):
        """Full global retrieval when no same-token vectors exist."""
        if self._dirty:
            self._rebuild_cache()
        if self._global_tensor is None:
            return query.unsqueeze(0)

        n = self._global_tensor.shape[0]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        global_norm = F.normalize(self._global_tensor, dim=-1)
        sims = (query_norm @ global_norm.t()).squeeze(0)

        # Similarity portion
        actual_sim_k = min(k_sim, n)
        sim_indices = sims.topk(actual_sim_k).indices
        sim_vecs = self._global_tensor[sim_indices]

        # Recency portion
        actual_rec_k = min(k_rec, n)
        rec_vecs = self._global_tensor[-actual_rec_k:]

        combined = torch.cat([sim_vecs, rec_vecs], dim=0)
        return combined[:k]

    def _global_similarity(self, query, k, exclude_token=None):
        """Get top-k globally by cosine similarity, optionally excluding a token_id."""
        if self._dirty:
            self._rebuild_cache()
        if self._global_tensor is None or k <= 0:
            return query.unsqueeze(0).expand(max(1, k), -1)

        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        global_norm = F.normalize(self._global_tensor, dim=-1)
        sims = (query_norm @ global_norm.t()).squeeze(0)

        if exclude_token is not None:
            # Mask out vectors from the excluded token
            mask = torch.tensor(
                [1.0 if tid != exclude_token else 0.0 for tid in self.all_keys],
                device=self.device
            )
            sims = sims * mask + (1 - mask) * (-1e9)

        actual_k = min(k, self._global_tensor.shape[0])
        if actual_k <= 0:
            return query.unsqueeze(0)
        indices = sims.topk(actual_k).indices
        return self._global_tensor[indices]

    def save(self, path="token_store.pt"):
        """Persist the store to disk."""
        data = {}
        for tid, vecs in self.store.items():
            data[tid] = torch.stack(vecs)
        torch.save(data, path)

    def load(self, path="token_store.pt"):
        """Load store from disk."""
        if not os.path.exists(path):
            return False
        data = torch.load(path, weights_only=True)
        self.store = {}
        self.all_vectors = []
        self.all_keys = []
        self.total_count = 0
        for tid, tensor in data.items():
            tid = int(tid) if not isinstance(tid, int) else tid
            vecs = list(tensor)
            self.store[tid] = vecs
            self.all_vectors.extend(vecs)
            self.all_keys.extend([tid] * len(vecs))
            self.total_count += len(vecs)
        self._dirty = True
        return True

    def stats(self):
        """Return store statistics."""
        return {
            'total_vectors': self.total_count,
            'unique_tokens': len(self.store),
            'avg_per_token': self.total_count / max(1, len(self.store)),
            'memory_mb': (self.total_count * self.dim * 4) / (1024 * 1024),
        }


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


# ═══════════════════════════════════════════════════
#  E9-style: compute P from FIFO buffer (serial SVD)
# ═══════════════════════════════════════════════════

def compute_P_from_buffer_serial(obs_buffer, x_queries, rank=RANK, retrieve_k=RETRIEVE_K):
    """
    E9-style: serial per-token SVD.
    x_queries: (B, T, D)
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
#  E10: compute P from TokenKeyedStore (BATCHED SVD)
# ═══════════════════════════════════════════════════

def compute_P_from_store_batched(token_store, token_ids, x_queries, rank=RANK, retrieve_k=RETRIEVE_K):
    """
    E10-style: batched SVD computation.
    token_ids: (B, T) int tensor — the input token ids
    x_queries: (B, T, D) float tensor — projected queries
    Returns P: (B, T, D, D)
    """
    B, T, D = x_queries.shape
    x_det = x_queries.detach()

    # Build all context matrices
    contexts = []
    for b in range(B):
        for t in range(T):
            tid = token_ids[b, t].item()
            query = x_det[b, t]
            retrieved = token_store.retrieve(tid, query, k=retrieve_k)
            # Append query to retrieved set
            context = torch.cat([retrieved, query.unsqueeze(0)], dim=0)  # (k+1, D)
            # Pad or truncate to fixed size for batching
            target_size = retrieve_k + 1
            if context.shape[0] < target_size:
                pad = torch.zeros(target_size - context.shape[0], D, device=query.device)
                context = torch.cat([context, pad], dim=0)
            elif context.shape[0] > target_size:
                context = context[:target_size]
            contexts.append(context)

    # Stack into (B*T, k+1, D) and do ONE batched SVD
    contexts_batch = torch.stack(contexts)  # (B*T, k+1, D)

    try:
        U, S, Vh = torch.linalg.svd(contexts_batch, full_matrices=False)  # Batched!
    except RuntimeError:
        # Fallback: identity P
        return torch.eye(D, device=x_queries.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

    # Take top-rank components
    k_rank = min(rank, S.shape[-1])
    V_top = Vh[:, :k_rank, :]  # (B*T, rank, D)

    # Build projection matrices: P = V_top^T @ V_top
    P = torch.bmm(V_top.transpose(1, 2), V_top)  # (B*T, D, D)
    P = P.view(B, T, D, D)

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
#  2. E9-Style PGA Buffer (FIFO, 50/50, serial SVD)
# ═══════════════════════════════════════════════════

class E9StylePGABuffer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.obs_buffer = ObservationBuffer(capacity=E9_BUFFER_CAPACITY, dim=N_EMBD)
        self.query_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        # Query buffer with projected raw embeddings
        x_proj = self.query_proj(x.detach())
        P = compute_P_from_buffer_serial(self.obs_buffer, x_proj)

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
#  3. E10 Improved PGA Buffer (Token-Keyed, 70/30, Batched SVD)
# ═══════════════════════════════════════════════════

class ImprovedPGABuffer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.token_store = TokenKeyedStore(dim=N_EMBD)
        self.query_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        # Query token store with projected raw embeddings
        x_proj = self.query_proj(x.detach())
        P = compute_P_from_store_batched(self.token_store, idx, x_proj)

        x = run_layers(self.layers, x, P, self.rmsnorm, idx.device)

        # Feedback: store final-layer outputs keyed by token_id
        for b in range(B):
            self.token_store.store_vectors(idx[b], x[b])

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
