"""
E12 — The 4 Architectural Fixes
Model Definitions:
  1. BaselineMicroGPT        — Standard Transformer (scaled up)
  2. PropPGABuffer           — Token-keyed store, adaptive energy SVD, Q/K filtering only, MLP query projection

All models: 5 layers, n_embd=64, n_head=4, block_size=32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# ─── Hyperparameters ───
N_LAYER = 5
N_EMBD = 64
BLOCK_SIZE = 32
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD

# PGA Hyperparameters
ENERGY_THRESHOLD = 0.90
RETRIEVE_K = 16  # Increased to match larger scale
SIM_RATIO = 0.7
MAX_PER_TOKEN = 10_000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt() * self.weight


# ═══════════════════════════════════════════════════
#  E12 Token-Keyed Vector Store
# ═══════════════════════════════════════════════════

class TokenKeyedStore:
    def __init__(self, dim=N_EMBD, device='cpu'):
        self.dim = dim
        self.device = device
        self.store = {}
        self.all_vectors = []
        self.all_keys = []
        self._global_tensor = None
        self._dirty = True
        self._needs_full_rebuild = False # FIX: initialized
        self.total_count = 0

    def store_vectors(self, token_ids, vectors):
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

            if len(self.store[tid]) > MAX_PER_TOKEN:
                n_evict = len(self.store[tid]) - MAX_PER_TOKEN
                self.store[tid] = self.store[tid][n_evict:]
                self._needs_full_rebuild = True

        if self._needs_full_rebuild:
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
        if self.total_count == 0:
            self._global_tensor = None
        else:
            self._global_tensor = torch.stack(self.all_vectors)
        self._dirty = False

    def retrieve(self, token_id, query, k=RETRIEVE_K):
        if self.total_count == 0:
            return query.unsqueeze(0)

        k_sim = max(1, int(k * SIM_RATIO))
        k_rec = k - k_sim

        same_token_vecs = self.store.get(token_id, [])

        if len(same_token_vecs) == 0:
            return self._global_retrieve(query, k, k_sim, k_rec)

        same_tensor = torch.stack(same_token_vecs)

        if len(same_token_vecs) >= k:
            query_norm = F.normalize(query.unsqueeze(0), dim=-1)
            same_norm = F.normalize(same_tensor, dim=-1)
            sims = (query_norm @ same_norm.t()).squeeze(0)

            actual_sim_k = min(k_sim, len(same_token_vecs))
            sim_indices = sims.topk(actual_sim_k).indices
            sim_vecs = same_tensor[sim_indices]

            actual_rec_k = min(k_rec, len(same_token_vecs))
            rec_vecs = same_tensor[-actual_rec_k:]

            combined = torch.cat([sim_vecs, rec_vecs], dim=0)
            return combined[:k]
        else:
            remaining = k - len(same_token_vecs)
            global_vecs = self._global_similarity(query, remaining, exclude_token=token_id)
            combined = torch.cat([same_tensor, global_vecs], dim=0)
            return combined[:k]

    def _global_retrieve(self, query, k, k_sim, k_rec):
        if self._dirty:
            self._rebuild_cache()
        if self._global_tensor is None:
            return query.unsqueeze(0)

        n = self._global_tensor.shape[0]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        global_norm = F.normalize(self._global_tensor, dim=-1)
        sims = (query_norm @ global_norm.t()).squeeze(0)

        actual_sim_k = min(k_sim, n)
        sim_indices = sims.topk(actual_sim_k).indices
        sim_vecs = self._global_tensor[sim_indices]

        actual_rec_k = min(k_rec, n)
        rec_vecs = self._global_tensor[-actual_rec_k:]

        combined = torch.cat([sim_vecs, rec_vecs], dim=0)
        return combined[:k]

    def _global_similarity(self, query, k, exclude_token=None):
        if self._dirty:
            self._rebuild_cache()
        if self._global_tensor is None or k <= 0:
            return query.unsqueeze(0).expand(max(1, k), -1)

        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        global_norm = F.normalize(self._global_tensor, dim=-1)
        sims = (query_norm @ global_norm.t()).squeeze(0)

        if exclude_token is not None:
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
        data = {}
        for tid, vecs in self.store.items():
            data[tid] = torch.stack(vecs)
        torch.save(data, path)

    def load(self, path="token_store.pt"):
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
        return {
            'total_vectors': self.total_count,
            'unique_tokens': len(self.store),
            'avg_per_token': self.total_count / max(1, len(self.store)),
            'memory_mb': (self.total_count * self.dim * 4) / (1024 * 1024),
        }


# ═══════════════════════════════════════════════════
#  Transformer Blocks (Q/K Filtering Only)
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


def run_layers(layers, x, P, principles, rmsnorm, idx_device):
    """
    E13 — Principle Augmentation
    x: [B, T, D]
    P: [B, T, D, D] or None
    principles: [B, T, R, D] (Virtual Tokens from SVD)
    """
    B, T, D = x.shape
    
    # Pre-compute causal mask once per forward pass
    causal_mask = torch.tril(torch.ones(T, T, device=idx_device)) == 0
    
    for layer in layers:
        x_res = x
        x = rmsnorm(x)

        q = layer['attn_wq'](x)
        k = layer['attn_wk'](x)
        v = layer['attn_wv'](x)

        if P is not None:
            # Apply P only to Q and K (Not V)
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)

        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2) # [B, nh, T, hs]
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        # ─── Principle Augmentation (E13) ───
        if principles is not None:
            # principles: [B, T, R, D]
            R = principles.shape[2]
            k_p = layer['attn_wk'](principles) # [B, T, R, D]
            v_p = layer['attn_wv'](principles) # [B, T, R, D]
            
            # Reshape to heads
            k_p = k_p.view(B, T, R, N_HEAD, HEAD_DIM).permute(0, 3, 1, 2, 4) # [B, nh, T, R, hs]
            v_p = v_p.view(B, T, R, N_HEAD, HEAD_DIM).permute(0, 3, 1, 2, 4)
            
            # Dot product with Virtual Tokens
            # q: [B, nh, T, hs] @ k_p.T: [B, nh, T, hs, R] -> [B, nh, T, R]
            att_p = torch.matmul(q.unsqueeze(-2), k_p.transpose(-2, -1)).squeeze(-2)
            att_data = (q @ k.transpose(-2, -1))
            
            # Scale
            att_p = att_p * (1.0 / math.sqrt(HEAD_DIM))
            att_data = att_data * (1.0 / math.sqrt(HEAD_DIM))
            
            # Masking
            att_data = att_data.masked_fill(causal_mask, float('-inf'))
            
            # Combined Softmax (Competition)
            att_combined = torch.cat([att_p, att_data], dim=-1) # [B, nh, T, R + T]
            att_combined = F.softmax(att_combined, dim=-1)
            
            # Final sum from both worlds
            y_p = (att_combined[:, :, :, :R].unsqueeze(-2) @ v_p).squeeze(-2) # [B, nh, T, hs]
            y_data = att_combined[:, :, :, R:] @ v # [B, nh, T, hs]
            y = y_p + y_data
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            att = att.masked_fill(causal_mask, float('-inf'))
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
#  E12 Batched SVD with Adaptive Energy Threshold
# ═══════════════════════════════════════════════════

def compute_P_from_store_adaptive(token_store, token_ids, x_queries, retrieve_k=RETRIEVE_K):
    B, T, D = x_queries.shape
    x_det = x_queries.detach()

    contexts = []
    for b in range(B):
        for t in range(T):
            tid = token_ids[b, t].item()
            query = x_det[b, t]
            retrieved = token_store.retrieve(tid, query, k=retrieve_k)
            context = torch.cat([retrieved, query.unsqueeze(0)], dim=0)
            target_size = retrieve_k + 1
            if context.shape[0] < target_size:
                pad = torch.zeros(target_size - context.shape[0], D, device=query.device)
                context = torch.cat([context, pad], dim=0)
            elif context.shape[0] > target_size:
                context = context[:target_size]
            contexts.append(context)

    contexts_batch = torch.stack(contexts)

    try:
        U, S, Vh = torch.linalg.svd(contexts_batch, full_matrices=False)
    except RuntimeError:
        P = torch.eye(D, device=x_queries.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        principles = torch.eye(D, device=x_queries.device)[:1].unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        return P, principles, {'avg_rank': D, 'svd_fails': B * T}

    # FIX 4: Adaptive Energy Threshold
    total_energy = S.sum(dim=-1, keepdim=True)
    cum_energy = S.cumsum(dim=-1)
    keep_mask = cum_energy < (ENERGY_THRESHOLD * total_energy)
    keep_mask[:, 0] = True
    Vh_filtered = Vh * keep_mask.unsqueeze(-1)
    
    P = torch.bmm(Vh_filtered.transpose(1, 2), Vh_filtered)
    P = P.view(B, T, D, D)
    
    # E13 principles for augmentation
    principles = Vh_filtered.view(B, T, Vh_filtered.shape[1], D)
    
    avg_rank = keep_mask.sum(dim=-1).float().mean().item()
    return P, principles, {'avg_rank': avg_rank, 'svd_fails': 0}


# ═══════════════════════════════════════════════════
#  1. Baseline
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
        x = run_layers(self.layers, x, None, None, self.rmsnorm, idx.device)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ═══════════════════════════════════════════════════
#  2. E13 Propagative PGA with Augmentation
# ═══════════════════════════════════════════════════

class PropPGAAugment(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.layers = make_layers()
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.rmsnorm = RMSNorm(N_EMBD)
        self.token_store = TokenKeyedStore(dim=N_EMBD)

        # Deep MLP Projection
        self.query_proj = nn.Sequential(
            nn.Linear(N_EMBD, N_EMBD),
            RMSNorm(N_EMBD),
            nn.ReLU(),
            nn.Linear(N_EMBD, N_EMBD)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.rmsnorm(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))

        x_proj = self.query_proj(x.detach())
        
        # E13: Adaptive SVD with Principle Basis
        P, principles, svd_stats = compute_P_from_store_adaptive(self.token_store, idx, x_proj)
        self.last_svd_stats = svd_stats

        # Layers apply both Subspace Filtering and Principle Augmentation
        x = run_layers(self.layers, x, P, principles, self.rmsnorm, idx.device)

        if self.training:
            for b in range(B):
                self.token_store.store_vectors(idx[b], x[b])

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
