"""
E11 — Token-Keyed Store (capped) + Rank-4 PGA — 3-Way Comparison Pipeline
  1. Baseline MicroGPT
  2. E9-Style PGA Buffer (FIFO, 50/50, serial SVD) — control
  3. E11 Improved PGA Buffer (Token-Keyed, capped 10K/token, rank 4, batched SVD)

Dataset: enwik8 (100MB)
Usage: python pipeline.py
"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import time
import zipfile
import urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from models import (
    N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE as MODEL_BLOCK_SIZE,
    RANK, RETRIEVE_K, E9_BUFFER_CAPACITY, SIM_RATIO, MAX_PER_TOKEN
)

# ─── Configuration ───
STEPS = 5000
LR = 0.005
SEED = 42
BATCH_SIZE = 1
BLOCK_SIZE = MODEL_BLOCK_SIZE
EVAL_INTERVAL = 100
EVAL_ITERS = 50
DEVICE = 'cpu'

DATA_URL = "http://mattmahoney.net/dc/enwik8.zip"
DATA_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(DATA_DIR, "enwik8")
RESULTS_DIR = os.path.dirname(__file__)


def log(msg, model_name="SYSTEM"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{model_name}] {msg}", flush=True)


def load_data():
    if not os.path.exists(DATA_FILE):
        # Try copying from previous experiments
        for prev in ["e9-retro-feedback", "e8-buffer-enwik"]:
            prev_data = os.path.join(DATA_DIR, "..", prev, "enwik8")
            if os.path.exists(prev_data):
                import shutil
                shutil.copy2(prev_data, DATA_FILE)
                log(f"Copied enwik8 from {prev}")
                break
        else:
            zip_path = os.path.join(DATA_DIR, "enwik8.zip")
            if not os.path.exists(zip_path):
                log("Downloading enwik8...")
                urllib.request.urlretrieve(DATA_URL, zip_path)
            log("Extracting enwik8...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)

    text = open(DATA_FILE, "r", encoding="utf-8", errors="replace").read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    log(f"enwik8: {len(text):,} chars, {vocab_size} unique")

    data = [stoi[c] for c in text]
    n = int(0.9 * len(data))
    train_data = torch.tensor(data[:n], dtype=torch.long)
    val_data = torch.tensor(data[n:], dtype=torch.long)

    log(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    return train_data, val_data, vocab_size, itos


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model, model_name, train_data, val_data, store_save_path=None):
    """Train a model and return results dict."""
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Initialized — {param_count:,} params", model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.85, 0.99), eps=1e-8)

    train_losses, val_losses, step_times, eval_steps = [], [], [], []
    t_start = time.time()
    model.train()

    for step in range(STEPS):
        t_step = time.time()
        xb, yb = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        step_time = time.time() - t_step
        step_times.append(step_time)

        if torch.isnan(loss):
            log(f"⚠ NaN at step {step}!", model_name)
            break

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            eval_steps.append(step)

            elapsed = time.time() - t_start
            eta = str(timedelta(seconds=int((elapsed / (step + 1)) * (STEPS - step - 1))))

            # Log store stats for E10 model
            store_info = ""
            if hasattr(model, 'token_store'):
                stats = model.token_store.stats()
                store_info = f" | Store: {stats['total_vectors']} vecs, {stats['unique_tokens']} tokens, {stats['memory_mb']:.2f}MB"

            log(
                f"Step {step:>5}/{STEPS} | "
                f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                f"Step: {step_time*1000:.1f}ms | ETA: {eta}{store_info}",
                model_name
            )

    losses = estimate_loss(model, train_data, val_data)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])
    eval_steps.append(STEPS)

    total_time = time.time() - t_start
    avg_step = np.mean(step_times) * 1000

    # Save token store if applicable
    if store_save_path and hasattr(model, 'token_store'):
        model.token_store.save(store_save_path)
        stats = model.token_store.stats()
        log(f"Token store saved: {store_save_path} ({stats['total_vectors']} vectors, {stats['memory_mb']:.2f}MB)", model_name)

    log(
        f"DONE — {total_time:.1f}s | {avg_step:.1f}ms/step | "
        f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f}",
        model_name
    )

    result = {
        'name': model_name,
        'train_losses': train_losses, 'val_losses': val_losses,
        'eval_steps': eval_steps, 'total_time': total_time,
        'avg_step_ms': avg_step, 'param_count': param_count,
        'final_train': losses['train'], 'final_val': losses['val'],
    }

    if hasattr(model, 'token_store'):
        result['store_stats'] = model.token_store.stats()

    return result


def train_single_model(model_name, model_cls_name, vocab_size, train_data, val_data):
    """Wrapper for parallel execution — instantiates and trains a model."""
    from models import BaselineMicroGPT, E9StylePGABuffer

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cls_map = {
        "BaselineMicroGPT": BaselineMicroGPT,
        "E9StylePGABuffer": E9StylePGABuffer,
    }
    model = cls_map[model_cls_name](vocab_size)
    return train_model(model, model_name, train_data, val_data)


def run_experiment():
    log("=" * 60)
    log("E11 — Token-Keyed Store (capped) + Rank-4 PGA — 3-Way")
    log(f"Config: {N_LAYER}L, embd={N_EMBD}, lr={LR}, steps={STEPS}")
    log(f"E9 Buffer: cap={E9_BUFFER_CAPACITY}, k={RETRIEVE_K}")
    log(f"E11 Store: cap={MAX_PER_TOKEN}/token, rank={RANK}, 70/30 split, batched SVD, k={RETRIEVE_K}")
    log(f"Dataset: enwik8, block={BLOCK_SIZE}")
    log("=" * 60)

    train_data, val_data, vocab_size, itos = load_data()
    log(f"Vocab: {vocab_size}")

    t_total = time.time()

    # ─── Train Baseline and E9-Buffer in parallel, E10 in main process ───
    log("═══ Training 3 models (Baseline + E9 parallel, E10 in main) ═══")

    # E10 must run in main process (TokenKeyedStore uses lists/dicts, not picklable across processes easily)
    from models import ImprovedPGABuffer

    # Start parallel workers for Baseline and E9
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        f1 = executor.submit(train_single_model, "Baseline", "BaselineMicroGPT", vocab_size, train_data, val_data)
        futures[f1] = "Baseline"
        f2 = executor.submit(train_single_model, "E9-Buffer", "E9StylePGABuffer", vocab_size, train_data, val_data)
        futures[f2] = "E9-Buffer"

        # Train E10 in main process
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        e10_model = ImprovedPGABuffer(vocab_size)
        store_path = os.path.join(RESULTS_DIR, "token_store.pt")
        e10_result = train_model(e10_model, "E11-TokenStore", train_data, val_data, store_save_path=store_path)

        results = {"E11-TokenStore": e10_result}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log(f"ERROR in {name}: {e}")
                raise

    total_wall = time.time() - t_total
    log(f"Total wall time: {total_wall:.1f}s")

    # ─── Plotting ───
    log("Generating plots...")

    colors = {
        'Baseline': '#2196F3',
        'E9-Buffer': '#FF5722',
        'E11-TokenStore': '#4CAF50',
    }
    markers = {
        'Baseline': 'o',
        'E9-Buffer': 's',
        'E11-TokenStore': '^',
    }
    order = ['Baseline', 'E9-Buffer', 'E11-TokenStore']

    # Combined plot
    plt.figure(figsize=(16, 9))
    for name in order:
        r = results[name]
        c = colors[name]
        m = markers[name]
        plt.plot(r['eval_steps'], r['train_losses'],
                 color=c, linestyle=':', alpha=0.4, linewidth=1.5, label=f'{name} Train')
        plt.plot(r['eval_steps'], r['val_losses'],
                 color=c, linewidth=2.5, label=f'{name} Val', marker=m, markersize=3)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(
        f'E11 — Token-Keyed Store (capped) vs E9 Buffer vs Baseline\n'
        f'{N_LAYER}L | embd={N_EMBD} | rank={RANK} | lr={LR} | enwik8 | {STEPS} Steps',
        fontsize=13
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, 'e11_combined.png')
    plt.savefig(combined_path, dpi=300)
    log(f"Combined plot: {combined_path}")

    # Side-by-side train/val
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for ax, key, title in [(ax1, 'train_losses', 'Training'), (ax2, 'val_losses', 'Validation')]:
        for name in order:
            r = results[name]
            ax.plot(r['eval_steps'], r[key],
                    color=colors[name], linewidth=2, label=name,
                    marker=markers[name], markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title} Loss')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'E11 — 3-Way Comparison ({N_LAYER}L, embd={N_EMBD}, rank={RANK}, enwik8)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    split_path = os.path.join(RESULTS_DIR, 'e11_results.png')
    plt.savefig(split_path, dpi=300)
    log(f"Split plot: {split_path}")

    # ─── Verdict ───
    all_models = [(name, results[name]) for name in order]
    best_val = min(all_models, key=lambda x: x[1]['final_val'])
    gaps = {name: r['final_val'] - r['final_train'] for name, r in all_models}
    best_gen = min(gaps.items(), key=lambda x: abs(x[1]))

    report = f"""# E11 — Token-Keyed Store (capped) + Rank-4 PGA

> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Configuration

| Setting | Value |
|---|---|
| Dataset | enwik8 (100MB) |
| Layers | {N_LAYER} |
| n_embd | {N_EMBD} |
| Steps | {STEPS} |
| LR | {LR} |
| Batch Size | {BATCH_SIZE} |
| SVD Rank | {RANK} |
| Retrieve K | {RETRIEVE_K} |
| Store Cap | {MAX_PER_TOKEN}/token |

## Models

| | Baseline | E9-Buffer | E11-TokenStore |
|---|---|---|---|
| Params | {results['Baseline']['param_count']:,} | {results['E9-Buffer']['param_count']:,} | {results['E11-TokenStore']['param_count']:,} |
| Buffer Type | — | FIFO ring (512) | Token-keyed dict (cap {MAX_PER_TOKEN}/token) |
| SVD | — | Serial (16/step) | Batched (1 call/step) |
| Retrieval | — | 50/50 sim/recent | 70/30 sim/recent, same-token priority |
| Feedback | — | Final outputs | Final outputs (keyed by token_id) |
| Persistence | — | ❌ | ✅ token_store.pt |

## Results

| Metric | Baseline | E9-Buffer | E11-TokenStore |
|---|---|---|---|
| Train Loss | {results['Baseline']['final_train']:.4f} | {results['E9-Buffer']['final_train']:.4f} | {results['E11-TokenStore']['final_train']:.4f} |
| Val Loss | {results['Baseline']['final_val']:.4f} | {results['E9-Buffer']['final_val']:.4f} | {results['E11-TokenStore']['final_val']:.4f} |
| Gap | {gaps['Baseline']:.4f} | {gaps['E9-Buffer']:.4f} | {gaps['E11-TokenStore']:.4f} |
| Step Time | {results['Baseline']['avg_step_ms']:.1f}ms | {results['E9-Buffer']['avg_step_ms']:.1f}ms | {results['E11-TokenStore']['avg_step_ms']:.1f}ms |
| Total | {results['Baseline']['total_time']:.1f}s | {results['E9-Buffer']['total_time']:.1f}s | {results['E11-TokenStore']['total_time']:.1f}s |

## Verdict

**Val Loss Winner: {best_val[0]}** ({best_val[1]['final_val']:.4f})
**Generalization Winner: {best_gen[0]}** (gap: {best_gen[1]:.4f})

![E11 Combined](e11_combined.png)
"""

    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Report: {report_path}")

    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({name: results[name] for name in order}, f, indent=2)
    log(f"Data: {json_path}")

    log("=" * 60)
    log(f"VERDICT — Val Loss Winner: {best_val[0]}")
    log(f"VERDICT — Generalization Winner: {best_gen[0]}")
    log("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_experiment()
