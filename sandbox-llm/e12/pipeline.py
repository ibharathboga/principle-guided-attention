"""
E12 — The 4 Architectural Fixes Pipeline
  1. Baseline MicroGPT
  2. Prop PGA Buffer (Adaptive SVD, Q/K filtering, MLP query matching)

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
    ENERGY_THRESHOLD, RETRIEVE_K, MAX_PER_TOKEN
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
        for prev in ["e10-token-store", "e9-retro-feedback", "e8-buffer-enwik"]:
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

    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
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
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Initialized — {param_count:,} params", model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.85, 0.99), eps=1e-8)

    train_losses, val_losses, step_times, eval_steps = [], [], [], []
    svd_ranks, svd_fails = [], 0
    t_start = time.time()
    model.train()

    for step in range(STEPS):
        t_step = time.time()
        xb, yb = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        optimizer.step()
        
        if hasattr(model, 'last_svd_stats'):
            svd_ranks.append(model.last_svd_stats['avg_rank'])
            svd_fails += model.last_svd_stats['svd_fails']

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

            store_info = ""
            if hasattr(model, 'token_store'):
                stats = model.token_store.stats()
                store_info = f" | Store: {stats['total_vectors']} vecs, {stats['memory_mb']:.2f}MB"

            svd_info = ""
            if len(svd_ranks) > 0:
                recent_avg = sum(svd_ranks[-EVAL_INTERVAL:]) / min(len(svd_ranks), EVAL_INTERVAL)
                svd_info = f" | Rank: {recent_avg:.1f} (Fail: {svd_fails})"

            log(
                f"Step {step:>5}/{STEPS} | "
                f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                f"Norm: {grad_norm:.2f} | "
                f"Step: {step_time*1000:.1f}ms | ETA: {eta}{store_info}{svd_info}",
                model_name
            )

    losses = estimate_loss(model, train_data, val_data)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])
    eval_steps.append(STEPS)

    total_time = time.time() - t_start
    avg_step = np.mean(step_times) * 1000

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
    import models
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_cls = getattr(models, model_cls_name)
    model = model_cls(vocab_size)
    return train_model(model, model_name, train_data, val_data)


def run_experiment():
    log("=" * 60)
    log("E12 — The 4 Architectural Fixes")
    log(f"Config: {N_LAYER}L, embd={N_EMBD}, lr={LR}, steps={STEPS}")
    log(f"E12 PGA: cap={MAX_PER_TOKEN}/token, adaptive_energy={ENERGY_THRESHOLD}, MLP Query, Q/K Only")
    log(f"Dataset: enwik8, block={BLOCK_SIZE}")
    log("=" * 60)

    train_data, val_data, vocab_size, itos = load_data()
    log(f"Vocab: {vocab_size}")

    t_total = time.time()

    log("═══ Training 2 models (Baseline in parallel, PropPGABuffer in main) ═══")

    from models import PropPGABuffer

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {}
        f1 = executor.submit(train_single_model, "Baseline", "BaselineMicroGPT", vocab_size, train_data, val_data)
        futures[f1] = "Baseline"

        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        pga_model = PropPGABuffer(vocab_size)
        store_path = os.path.join(RESULTS_DIR, "token_store.pt")
        pga_result = train_model(pga_model, "PropPGABuffer", train_data, val_data, store_save_path=store_path)

        results = {"PropPGABuffer": pga_result}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log(f"ERROR in {name}: {e}")
                raise

    total_wall = time.time() - t_total
    log(f"Total wall time: {total_wall:.1f}s")

    log("Generating plots...")

    colors = {
        'Baseline': '#2196F3',
        'PropPGABuffer': '#4CAF50',
    }
    markers = {
        'Baseline': 'o',
        'PropPGABuffer': '^',
    }
    order = ['Baseline', 'PropPGABuffer']

    plt.figure(figsize=(10, 6))
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
        f'E12 — The 4 Architectural Fixes\n'
        f'{N_LAYER}L | embd={N_EMBD} | energy={ENERGY_THRESHOLD} | enwik8 | {STEPS} Steps',
        fontsize=13
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, 'e12_combined.png')
    plt.savefig(combined_path, dpi=300)
    log(f"Combined plot: {combined_path}")

    all_models = [(name, results[name]) for name in order]
    best_val = min(all_models, key=lambda x: x[1]['final_val'])
    gaps = {name: r['final_val'] - r['final_train'] for name, r in all_models}
    best_gen = min(gaps.items(), key=lambda x: abs(x[1]))

    report = f"""# E12 — The 4 Architectural Fixes

> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Configuration

| Setting | Value |
|---|---|
| Dataset | enwik8 (100MB) |
| Layers | {N_LAYER} |
| n_embd | {N_EMBD} |
| Steps | {STEPS} |
| LR | {LR} |
| SVD Energy Threshold | {ENERGY_THRESHOLD} |

## Fixes Implemented
1. **Scaled Space**: `N_EMBD=64` to provide geometric room for subspace filtering.
2. **Q/K Only**: P matrix applied only to queries and keys, Values unfiltered.
3. **MLP Query Proj**: Deep projection layer to match embedding space to context vector depth.
4. **Adaptive Propagative SVD**: SVD computed exactly once per forward pass, using dynamic cumulative energy summation instead of hard-coded rank constraints.

## Results

| Metric | Baseline | Prop PGA Buffer |
|---|---|---|
| Train Loss | {results['Baseline']['final_train']:.4f} | {results['PropPGABuffer']['final_train']:.4f} |
| Val Loss | {results['Baseline']['final_val']:.4f} | {results['PropPGABuffer']['final_val']:.4f} |
| Gap | {gaps['Baseline']:.4f} | {gaps['PropPGABuffer']:.4f} |
| Step Time | {results['Baseline']['avg_step_ms']:.1f}ms | {results['PropPGABuffer']['avg_step_ms']:.1f}ms |

## Verdict

**Val Loss Winner: {best_val[0]}** ({best_val[1]['final_val']:.4f})
**Generalization Winner: {best_gen[0]}** (gap: {best_gen[1]:.4f})
"""
    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Report: {report_path}")

    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({name: results[name] for name in order}, f, indent=2)

    log("=" * 60)
    log(f"VERDICT — Val Loss: {best_val[0]}")
    log(f"VERDICT — Generalization: {best_gen[0]}")
    log("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_experiment()
