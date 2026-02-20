# ╔══════════════════════════════════════════════════════════════════╗
# ║  PGA Scaled Experiment — Google Colab Notebook                  ║
# ║  Copy each section into a separate Colab cell                   ║
# ║  Runtime → Change runtime type → T4 GPU                        ║
# ╚══════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════
# CELL 1: GPU Check & Setup
# ═══════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import zipfile
import urllib.request
from datetime import datetime, timedelta

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")


# ═══════════════════════════════════════════════════
# CELL 2: Configuration — SCALED UP
# ═══════════════════════════════════════════════════

SMOKE_TEST = False  # <--- Set to True for quick smoke test

# Model config — ~10M params (vs 210K in CPU experiments)
N_LAYER = 6
N_EMBD = 256
N_HEAD = 8
HEAD_DIM = N_EMBD // N_HEAD  # 32
BLOCK_SIZE = 128

# PGA config
WINDOW_SIZE = 32
RANK = 32          # At 256-dim, rank-32 keeps 12.5% of the space (real filtering)
BUFFER_CAPACITY = 16384   # ~8 batches of cross-batch memory
RETRIEVE_K = 16

# Training config
STEPS = 5000
LR = 3e-4          # Standard for scaled models (Adam + warmup territory)
BATCH_SIZE = 16     # GPU can handle larger batches
EVAL_INTERVAL = 250
EVAL_ITERS = 50
SEED = 42

if SMOKE_TEST:
    print("\n" + "="*50)
    print("⚠ SMOKE TEST MODE ENABLED")
    print("Running minimal steps to verify code correctness.")
    print("Set SMOKE_TEST = False for full experiment.")
    print("="*50 + "\n")
    STEPS = 20
    EVAL_INTERVAL = 10
    EVAL_ITERS = 2

print(f"Config: {N_LAYER}L × {N_EMBD}d × {N_HEAD}H, block={BLOCK_SIZE}")
print(f"PGA: window={WINDOW_SIZE}, rank={RANK}, buffer={BUFFER_CAPACITY}, k={RETRIEVE_K}")
print(f"Training: {STEPS} steps, lr={LR}, batch={BATCH_SIZE}")


# ═══════════════════════════════════════════════════
# CELL 3: Dataset — enwik8
# ═══════════════════════════════════════════════════

DATA_URL = "http://mattmahoney.net/dc/enwik8.zip"

if not os.path.exists("enwik8"):
    print("Downloading enwik8...")
    urllib.request.urlretrieve(DATA_URL, "enwik8.zip")
    with zipfile.ZipFile("enwik8.zip", 'r') as z:
        z.extractall(".")
    print("Done.")

text = open("enwik8", "r", encoding="utf-8", errors="replace").read()
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = [stoi[c] for c in text]
n = int(0.9 * len(data))
train_data = torch.tensor(data[:n], dtype=torch.long, device=DEVICE)
val_data = torch.tensor(data[n:], dtype=torch.long, device=DEVICE)

print(f"enwik8: {len(text):,} chars, vocab={VOCAB_SIZE}")
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")


# ═══════════════════════════════════════════════════
# CELL 4: Model Definitions
# ═══════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        ms = (x ** 2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt() * self.weight


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(N_EMBD)
        self.norm2 = RMSNorm(N_EMBD)
        self.attn_wq = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_wk = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_wv = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_wo = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.mlp_fc1 = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.mlp_fc2 = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)

    def forward(self, x, P=None):
        """x: (B, T, D), P: (B, T, D, D) or None"""
        B, T, D = x.shape
        x_res = x
        x_n = self.norm1(x)

        q = self.attn_wq(x_n)
        k = self.attn_wk(x_n)
        v = self.attn_wv(x_n)

        # PGA Projection
        if P is not None:
            q = torch.matmul(q.unsqueeze(2), P).squeeze(2)
            k = torch.matmul(k.unsqueeze(2), P).squeeze(2)
            v = torch.matmul(v.unsqueeze(2), P).squeeze(2)

        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        # Use PyTorch's efficient scaled_dot_product_attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, N_EMBD)

        x = self.attn_wo(y) + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = F.gelu(x)  # GELU instead of ReLU for scaled models
        x = self.mlp_fc2(x)
        x = x + x_res
        return x


# ─── Observation Buffer (GPU-compatible) ───

class ObservationBuffer:
    def __init__(self, capacity, dim, device):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.buffer = torch.zeros(capacity, dim, device=device)
        self.ptr = 0
        self.count = 0

    @torch.no_grad()
    def store(self, vectors):
        """vectors: (N, D)"""
        N = vectors.shape[0]
        if N >= self.capacity:
            self.buffer[:] = vectors[-self.capacity:]
            self.ptr = 0
            self.count = self.capacity
            return
        end = self.ptr + N
        if end <= self.capacity:
            self.buffer[self.ptr:end] = vectors
        else:
            first = self.capacity - self.ptr
            self.buffer[self.ptr:] = vectors[:first]
            self.buffer[:N - first] = vectors[first:]
        self.ptr = end % self.capacity
        self.count = min(self.count + N, self.capacity)

    @torch.no_grad()
    def retrieve_hybrid(self, queries, k):
        """
        Batched retrieval: queries (B*T, D) → returns (B*T, k, D)
        Half cosine-similar + half recent.
        """
        if self.count == 0:
            return queries.unsqueeze(1).expand(-1, k, -1)

        half_k = max(1, k // 2)
        active = self.buffer[:self.count]  # (count, D)

        # Cosine similarity: (BT, D) @ (D, count) → (BT, count)
        q_norm = F.normalize(queries, dim=-1)
        a_norm = F.normalize(active, dim=-1)
        sims = q_norm @ a_norm.t()

        sim_k = min(half_k, self.count)
        topk = sims.topk(sim_k, dim=-1)
        similar = active[topk.indices]  # (BT, sim_k, D)

        # Recent
        recent_k = min(k - sim_k, self.count)
        if self.count < self.capacity:
            recent = self.buffer[max(0, self.count - recent_k):self.count]
        else:
            indices = [(self.ptr - 1 - i) % self.capacity for i in range(recent_k)]
            recent = self.buffer[torch.tensor(indices, device=self.device)]
        # Expand recent for all queries
        recent = recent.unsqueeze(0).expand(queries.shape[0], -1, -1)

        combined = torch.cat([similar, recent], dim=1)
        return combined[:, :k, :]

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0
        self.count = 0


# ─── Batched P computation (GPU-efficient) ───

@torch.no_grad()
def compute_P_batched_buffer(obs_buffer, x, rank, retrieve_k):
    """
    Compute projection matrices from buffer retrieval — batched for GPU.
    x: (B, T, D)
    Returns P: (B, T, D, D)
    """
    B, T, D = x.shape
    queries = x.reshape(B * T, D)  # (BT, D)

    # Retrieve from buffer
    retrieved = obs_buffer.retrieve_hybrid(queries, k=retrieve_k)  # (BT, k, D)

    # Append query itself
    context = torch.cat([retrieved, queries.unsqueeze(1)], dim=1)  # (BT, k+1, D)

    # Batched SVD
    try:
        U, S, Vh = torch.linalg.svd(context, full_matrices=False)
    except RuntimeError:
        return torch.eye(D, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

    k_rank = min(rank, Vh.shape[1])
    V_top = Vh[:, :k_rank, :]  # (BT, k_rank, D)

    P = torch.bmm(V_top.transpose(1, 2), V_top)  # (BT, D, D)
    P = P.view(B, T, D, D)
    return P


@torch.no_grad()
def compute_P_window(x, window_size, rank):
    """
    Compute projection matrices from sliding window — batched.
    x: (B, T, D)
    Returns P: (B, T, D, D)
    """
    B, T, D = x.shape
    P_list = []

    for t in range(T):
        start = max(0, t - window_size + 1)
        context = x[:, start:t+1, :]  # (B, win, D)

        try:
            U, S, Vh = torch.linalg.svd(context, full_matrices=False)
        except RuntimeError:
            P_list.append(torch.eye(D, device=x.device).unsqueeze(0).expand(B, -1, -1))
            continue

        k = min(rank, S.shape[-1])
        V_top = Vh[:, :k, :]  # (B, k, D)
        P_t = torch.bmm(V_top.transpose(1, 2), V_top)  # (B, D, D)
        P_list.append(P_t)

    return torch.stack(P_list, dim=1)  # (B, T, D, D)


# ═══ Model 1: Baseline ═══

class BaselineMicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x, P=None)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


# ═══ Model 2: Propagative PGA (Window) ═══

class PropagativePGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        P = compute_P_window(x.detach(), WINDOW_SIZE, RANK)
        for block in self.blocks:
            x = block(x, P=P)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


# ═══ Model 3: Essence Buffer (stores post-layer outputs) ═══

class PropagativePGAEssenceBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.obs_buffer = ObservationBuffer(BUFFER_CAPACITY, N_EMBD, DEVICE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))

        P = compute_P_batched_buffer(self.obs_buffer, x.detach(), RANK, RETRIEVE_K)

        for block in self.blocks:
            x = block(x, P=P)

        # Feedback: store essence vectors (deep, contextually-enriched post-layer outputs)
        self.obs_buffer.store(x.detach().reshape(-1, N_EMBD))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


print("Models defined.")
print(f"Estimated params per model: ~{sum(p.numel() for p in BaselineMicroGPT().parameters()):,}")


# ═══════════════════════════════════════════════════
# CELL 5: Training Infrastructure
# ═══════════════════════════════════════════════════

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), device=DEVICE)
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS, device=DEVICE)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model, name):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    # Cosine LR schedule with warmup
    warmup_steps = min(500, STEPS // 10)

    train_losses, val_losses, eval_steps, step_times = [], [], [], []
    t_start = time.time()
    model.train()

    for step in range(STEPS):
        # LR warmup + cosine decay
        if step < warmup_steps:
            lr = LR * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, STEPS - warmup_steps)
            lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t_step = time.time()
        xb, yb = get_batch(train_data)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_times.append(time.time() - t_step)

        if torch.isnan(loss):
            print(f"⚠ NaN at step {step}!")
            break

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            eval_steps.append(step)

            elapsed = time.time() - t_start
            eta = timedelta(seconds=int((elapsed / (step + 1)) * (STEPS - step - 1)))
            ms = step_times[-1] * 1000

            print(
                f"[{name}] Step {step:>5}/{STEPS} | "
                f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | {ms:.1f}ms/step | ETA: {eta}"
            )

    # Final eval
    losses = estimate_loss(model)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])
    eval_steps.append(STEPS)
    total = time.time() - t_start
    avg_ms = np.mean(step_times) * 1000

    print(f"\n[{name}] DONE — {total:.1f}s | {avg_ms:.1f}ms/step | "
          f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")

    return {
        'name': name, 'param_count': params,
        'train_losses': train_losses, 'val_losses': val_losses,
        'eval_steps': eval_steps, 'total_time': total,
        'avg_step_ms': avg_ms,
        'final_train': losses['train'], 'final_val': losses['val'],
    }


# ═══════════════════════════════════════════════════
# CELL 6: Run All 3 Models (Sequential on GPU)
# ═══════════════════════════════════════════════════

random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(SEED)

print(f"\n{'#'*60}")
print(f"# PGA Scaled Experiment — {N_LAYER}L × {N_EMBD}d — enwik8")
print(f"# {STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
print(f"{'#'*60}\n")

results = {}

# 1. Baseline
torch.manual_seed(SEED)
results['Baseline'] = train_model(BaselineMicroGPT(), 'Baseline')

# 2. Prop-PGA (Window)
torch.manual_seed(SEED)
results['Prop-PGA'] = train_model(PropagativePGA(), 'Prop-PGA')

# 3. Essence-Buffer
torch.manual_seed(SEED)
results['Essence-Buf'] = train_model(PropagativePGAEssenceBuffer(), 'Essence-Buf')

print(f"\n{'='*60}")
print("ALL MODELS COMPLETE")
print(f"{'='*60}")


# ═══════════════════════════════════════════════════
# CELL 7: Plots & Report
# ═══════════════════════════════════════════════════

colors = {'Baseline': '#2196F3', 'Prop-PGA': '#FF5722', 'Essence-Buf': '#9C27B0'}
markers = {'Baseline': 'o', 'Prop-PGA': 's', 'Essence-Buf': 'D'}
order = ['Baseline', 'Prop-PGA', 'Essence-Buf']

# Combined plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for ax, key, title in [(ax1, 'train_losses', 'Training Loss'), (ax2, 'val_losses', 'Validation Loss')]:
    for name in order:
        r = results[name]
        ax.plot(r['eval_steps'], r[key], color=colors[name], linewidth=2,
                label=name, marker=markers[name], markersize=4)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'PGA Scaled — {N_LAYER}L × {N_EMBD}d × {N_HEAD}H | enwik8 | {STEPS} Steps',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('pga_scaled_results.png', dpi=150)
plt.show()

# Print results table
print(f"\n{'═'*60}")
print("RESULTS SUMMARY")
print(f"{'═'*60}")
print(f"{'Model':<16} {'Params':>10} {'Train':>10} {'Val':>10} {'Gap':>10} {'ms/step':>10}")
print(f"{'─'*66}")
for name in order:
    r = results[name]
    gap = r['final_val'] - r['final_train']
    print(f"{name:<16} {r['param_count']:>10,} {r['final_train']:>10.4f} {r['final_val']:>10.4f} {gap:>10.4f} {r['avg_step_ms']:>10.1f}")

best = min(order, key=lambda n: results[n]['final_val'])
print(f"\n🏆 Val Loss Winner: {best} ({results[best]['final_val']:.4f})")

gaps = {n: abs(results[n]['final_val'] - results[n]['final_train']) for n in order}
best_gen = min(gaps, key=gaps.get)
print(f"🏆 Generalization Winner: {best_gen} (gap: {gaps[best_gen]:.4f})")
