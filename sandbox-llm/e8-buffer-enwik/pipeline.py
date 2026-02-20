"""
E8 — Propagative PGA + Observation Buffer — enwik8 Dataset
3-way parallel comparison:
  1. Baseline MicroGPT
  2. Propagative PGA (window-based, same as e7)
  3. Propagative PGA + Observation Buffer (full PGA spec)

Dataset: enwik8 (100MB raw Wikipedia XML — character-level)
Usage: python pipeline.py
"""

import torch
import torch.nn as nn
import random
import numpy as np
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
    WINDOW_SIZE, RANK, BUFFER_CAPACITY, RETRIEVE_K
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
        zip_path = os.path.join(DATA_DIR, "enwik8.zip")
        if not os.path.exists(zip_path):
            log(f"Downloading enwik8 to {zip_path}...")
            urllib.request.urlretrieve(DATA_URL, zip_path)
            log("Download complete.")

        log("Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        log("Extraction complete.")

    text = open(DATA_FILE, "r", encoding="utf-8", errors="replace").read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    log(f"enwik8 loaded: {len(text):,} chars, {vocab_size} unique chars")

    data = [stoi[c] for c in text]
    n = int(0.9 * len(data))
    train_data = torch.tensor(data[:n], dtype=torch.long)
    val_data = torch.tensor(data[n:], dtype=torch.long)

    log(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
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


def train_single_model(model_name, model_cls_name, vocab_size, train_data, val_data):
    """Train a single model. Runs in a separate process."""
    from models import BaselineMicroGPT, PropagativePGA, PropagativePGABuffer

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if model_cls_name == "BaselineMicroGPT":
        model = BaselineMicroGPT(vocab_size)
    elif model_cls_name == "PropagativePGA":
        model = PropagativePGA(vocab_size)
    elif model_cls_name == "PropagativePGABuffer":
        model = PropagativePGABuffer(vocab_size)
    else:
        raise ValueError(f"Unknown model: {model_cls_name}")

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Initialized — {param_count:,} parameters", model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.85, 0.99), eps=1e-8)

    train_losses = []
    val_losses = []
    step_times = []
    eval_steps = []

    t_start = time.time()
    model.train()

    for step in range(STEPS):
        t_step_start = time.time()

        xb, yb = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        step_time = time.time() - t_step_start
        step_times.append(step_time)

        if torch.isnan(loss):
            log(f"⚠ NaN loss at step {step}! Stopping.", model_name)
            break

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            eval_steps.append(step)

            elapsed = time.time() - t_start
            avg_step_time = elapsed / (step + 1)
            remaining_steps = STEPS - step - 1
            eta_seconds = avg_step_time * remaining_steps
            eta = str(timedelta(seconds=int(eta_seconds)))

            log(
                f"Step {step:>5}/{STEPS} | "
                f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                f"Step: {step_time*1000:.1f}ms | ETA: {eta}",
                model_name
            )

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])
    eval_steps.append(STEPS)

    total_time = time.time() - t_start
    avg_step = np.mean(step_times) * 1000

    log(
        f"DONE — Total: {total_time:.1f}s | "
        f"Avg step: {avg_step:.1f}ms | "
        f"Final Train: {losses['train']:.4f} | Final Val: {losses['val']:.4f}",
        model_name
    )

    return {
        'name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'eval_steps': eval_steps,
        'total_time': total_time,
        'avg_step_ms': avg_step,
        'param_count': param_count,
        'final_train': losses['train'],
        'final_val': losses['val'],
    }


def run_parallel():
    """Run all three models in parallel."""
    log("=" * 60)
    log("E8 — Propagative PGA + Observation Buffer — enwik8")
    log(f"Config: {N_LAYER} layers, n_embd={N_EMBD}, lr={LR}, steps={STEPS}")
    log(f"Buffer: capacity={BUFFER_CAPACITY}, retrieve_k={RETRIEVE_K}")
    log(f"Dataset: enwik8, block_size={BLOCK_SIZE}")
    log("=" * 60)

    train_data, val_data, vocab_size, itos = load_data()

    log(f"Vocab size: {vocab_size}")
    log("Starting parallel training (3 models)...")
    log("-" * 60)

    t_total_start = time.time()

    models_to_run = [
        ("Baseline", "BaselineMicroGPT"),
        ("Prop-PGA", "PropagativePGA"),
        ("Prop-PGA+Buf", "PropagativePGABuffer"),
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {}
        for name, cls_name in models_to_run:
            f = executor.submit(
                train_single_model,
                name, cls_name,
                vocab_size, train_data, val_data
            )
            futures[f] = name

        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log(f"ERROR in {name}: {e}")
                raise

    total_wall = time.time() - t_total_start
    log("-" * 60)
    log(f"All 3 models complete. Total wall time: {total_wall:.1f}s")

    baseline = results["Baseline"]
    pga_win = results["Prop-PGA"]
    pga_buf = results["Prop-PGA+Buf"]

    # ─── Plotting ───
    log("Generating plots...")

    C_BASE = '#2196F3'
    C_PGA = '#FF5722'
    C_BUF = '#4CAF50'

    # Combined plot
    plt.figure(figsize=(14, 8))

    plt.plot(baseline['eval_steps'], baseline['train_losses'],
             color=C_BASE, linestyle=':', alpha=0.5, linewidth=1.5, label='Baseline Train')
    plt.plot(pga_win['eval_steps'], pga_win['train_losses'],
             color=C_PGA, linestyle=':', alpha=0.5, linewidth=1.5, label='Prop-PGA Train')
    plt.plot(pga_buf['eval_steps'], pga_buf['train_losses'],
             color=C_BUF, linestyle=':', alpha=0.5, linewidth=1.5, label='Prop-PGA+Buf Train')

    plt.plot(baseline['eval_steps'], baseline['val_losses'],
             color=C_BASE, linewidth=2.5, label='Baseline Val', marker='o', markersize=3)
    plt.plot(pga_win['eval_steps'], pga_win['val_losses'],
             color=C_PGA, linewidth=2.5, label='Prop-PGA Val', marker='s', markersize=3)
    plt.plot(pga_buf['eval_steps'], pga_buf['val_losses'],
             color=C_BUF, linewidth=2.5, label='Prop-PGA+Buf Val', marker='^', markersize=3)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(
        f'E8 enwik8 — Baseline vs Prop-PGA vs Prop-PGA+Buffer\n'
        f'{N_LAYER} Layers | lr={LR} | enwik8 | {STEPS} Steps | Buffer={BUFFER_CAPACITY}',
        fontsize=13
    )
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    combined_path = os.path.join(RESULTS_DIR, 'e8_enwik_combined.png')
    plt.savefig(combined_path, dpi=300)
    log(f"Combined plot saved: {combined_path}")

    # Side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, loss_key, title in [
        (ax1, 'train_losses', 'Training Loss'),
        (ax2, 'val_losses', 'Validation Loss')
    ]:
        ax.plot(baseline['eval_steps'], baseline[loss_key],
                color=C_BASE, linewidth=2, label='Baseline', marker='o', markersize=3)
        ax.plot(pga_win['eval_steps'], pga_win[loss_key],
                color=C_PGA, linewidth=2, label='Prop-PGA', marker='s', markersize=3)
        ax.plot(pga_buf['eval_steps'], pga_buf[loss_key],
                color=C_BUF, linewidth=2, label='Prop-PGA+Buf', marker='^', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'E8 enwik8 — 3-Way Comparison ({N_LAYER} Layers, lr={LR}, enwik8)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    split_path = os.path.join(RESULTS_DIR, 'e8_enwik_results.png')
    plt.savefig(split_path, dpi=300)
    log(f"Split plot saved: {split_path}")

    # ─── Verdict Report ───
    all_models = [
        ('Baseline', baseline),
        ('Prop-PGA', pga_win),
        ('Prop-PGA+Buf', pga_buf),
    ]

    best = min(all_models, key=lambda x: x[1]['final_val'])
    winner_name = best[0]

    base_gap = baseline['final_val'] - baseline['final_train']
    pga_gap = pga_win['final_val'] - pga_win['final_train']
    buf_gap = pga_buf['final_val'] - pga_buf['final_train']
    best_gen = min([('Baseline', abs(base_gap)), ('Prop-PGA', abs(pga_gap)), ('Prop-PGA+Buf', abs(buf_gap))],
                   key=lambda x: x[1])[0]

    report = f"""# E8 — Propagative PGA + Observation Buffer Report (enwik8)

> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Configuration

| Setting | Value |
|---|---|
| Dataset | **enwik8** (100MB Wikipedia XML) |
| Layers | {N_LAYER} |
| n_embd | {N_EMBD} |
| n_head | {N_HEAD} |
| block_size | {BLOCK_SIZE} |
| Vocab Size | {vocab_size} |
| Learning Rate | {LR} |
| Steps | {STEPS} |
| Batch Size | {BATCH_SIZE} |
| Optimizer | Adam (betas=0.85/0.99) |
| PGA Window | {WINDOW_SIZE} |
| PGA Rank | {RANK} |
| Buffer Capacity | {BUFFER_CAPACITY} |
| Buffer Retrieve K | {RETRIEVE_K} |
| Execution | Parallel (3 processes) |

## Model Specs

| | Baseline | Prop-PGA (Window) | Prop-PGA+Buffer |
|---|---|---|---|
| Parameters | {baseline['param_count']:,} | {pga_win['param_count']:,} | {pga_buf['param_count']:,} |
| Extra Learnable | — | 0 | 256 (query_proj) |
| SVD source | — | Sliding window ({WINDOW_SIZE}) | Buffer retrieval ({RETRIEVE_K}) |
| P reuse | — | All {N_LAYER} layers | All {N_LAYER} layers |
| Persistent memory | — | ❌ | ✅ ({BUFFER_CAPACITY} capacity) |
| Feedback loop | — | ❌ | ✅ (stores essence vectors) |

## Results

| Metric | Baseline | Prop-PGA | Prop-PGA+Buf |
|---|---|---|---|
| Final Train Loss | {baseline['final_train']:.4f} | {pga_win['final_train']:.4f} | {pga_buf['final_train']:.4f} |
| Final Val Loss | {baseline['final_val']:.4f} | {pga_win['final_val']:.4f} | {pga_buf['final_val']:.4f} |
| Overfit Gap | {base_gap:.4f} | {pga_gap:.4f} | {buf_gap:.4f} |
| Avg Step Time | {baseline['avg_step_ms']:.1f}ms | {pga_win['avg_step_ms']:.1f}ms | {pga_buf['avg_step_ms']:.1f}ms |
| Total Time | {baseline['total_time']:.1f}s | {pga_win['total_time']:.1f}s | {pga_buf['total_time']:.1f}s |

## Verdict

**Loss Winner: {winner_name}** (lowest validation loss)
**Generalization Winner: {best_gen}** (smallest overfit gap)

## Dataset Notes

enwik8 is ~90× larger than TinyShakespeare with ~3× more unique characters.
This tests whether the observation buffer provides genuine retrieval value
rather than memorizing the training set.

![E8 enwik8 Combined Results](e8_enwik_combined.png)
"""

    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Report saved: {report_path}")

    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': baseline,
            'pga_window': pga_win,
            'pga_buffer': pga_buf,
            'config': {
                'steps': STEPS, 'lr': LR, 'n_layer': N_LAYER, 'n_embd': N_EMBD,
                'block_size': BLOCK_SIZE, 'vocab_size': vocab_size,
                'buffer_capacity': BUFFER_CAPACITY, 'retrieve_k': RETRIEVE_K,
                'dataset': 'enwik8',
                'wall_time': total_wall,
            }
        }, f, indent=2)
    log(f"Raw data saved: {json_path}")

    log("=" * 60)
    log(f"VERDICT — Loss Winner: {winner_name}")
    log(f"VERDICT — Generalization Winner: {best_gen}")
    log("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_parallel()
