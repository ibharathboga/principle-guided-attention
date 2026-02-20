"""
E7 — Propagative PGA Experiment Pipeline
Parallel training of Baseline vs Propagative PGA on Shakespeare data.

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
import urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add e7 dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from models import N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE as MODEL_BLOCK_SIZE, WINDOW_SIZE, RANK

# ─── Configuration ───
STEPS = 2500
LR = 0.05
SEED = 42
BATCH_SIZE = 1
BLOCK_SIZE = MODEL_BLOCK_SIZE
EVAL_INTERVAL = 100
EVAL_ITERS = 50
DEVICE = 'cpu'  # keep CPU for fair comparison and compatibility

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
RESULTS_DIR = os.path.dirname(__file__)


def log(msg, model_name="SYSTEM"):
    """Timestamped logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{model_name}] {msg}", flush=True)


def load_data():
    """Download Shakespeare and prepare train/val splits."""
    if not os.path.exists(DATA_FILE):
        log(f"Downloading Shakespeare to {DATA_FILE}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    text = open(DATA_FILE, "r", encoding="utf-8").read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    log(f"Shakespeare loaded: {len(text):,} chars, {vocab_size} unique chars")

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
    """
    Train a single model. Designed to run in a separate process.
    Returns dict with train/val loss logs and timing info.
    """
    # Import here to avoid pickle issues with multiprocessing
    from models import BaselineMicroGPT, PropagativePGA

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Instantiate model
    if model_cls_name == "BaselineMicroGPT":
        model = BaselineMicroGPT(vocab_size)
    else:
        model = PropagativePGA(vocab_size)

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

        # Check for NaN
        if torch.isnan(loss):
            log(f"⚠ NaN loss at step {step}! Stopping.", model_name)
            break

        # Periodic evaluation
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
    """Run both models in parallel processes."""
    log("=" * 60)
    log("E7 — Propagative PGA Experiment")
    log(f"Config: {N_LAYER} layers, n_embd={N_EMBD}, lr={LR}, steps={STEPS}")
    log(f"Dataset: Shakespeare, block_size={BLOCK_SIZE}")
    log("=" * 60)

    train_data, val_data, vocab_size, itos = load_data()

    log(f"Vocab size: {vocab_size}")
    log("Starting parallel training...")
    log("-" * 60)

    t_total_start = time.time()

    # Run both models in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                train_single_model,
                "Baseline", "BaselineMicroGPT",
                vocab_size, train_data, val_data
            ): "Baseline",
            executor.submit(
                train_single_model,
                "Prop-PGA", "PropagativePGA",
                vocab_size, train_data, val_data
            ): "Prop-PGA",
        }

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
    log(f"Both models complete. Total wall time: {total_wall:.1f}s")

    baseline = results["Baseline"]
    pga = results["Prop-PGA"]

    # ─── Plotting ───
    log("Generating plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Train Loss
    ax1.plot(baseline['eval_steps'], baseline['train_losses'], 'b-', linewidth=2, label='Baseline Train', marker='o', markersize=3)
    ax1.plot(pga['eval_steps'], pga['train_losses'], 'r-', linewidth=2, label='Prop-PGA Train', marker='s', markersize=3)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss: Baseline vs Propagative PGA')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val Loss
    ax2.plot(baseline['eval_steps'], baseline['val_losses'], 'b-', linewidth=2, label='Baseline Val', marker='o', markersize=3)
    ax2.plot(pga['eval_steps'], pga['val_losses'], 'r-', linewidth=2, label='Prop-PGA Val', marker='s', markersize=3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss: Baseline vs Propagative PGA')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'E7 — Propagative PGA (5 Layers, lr={LR}, Shakespeare)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, 'e7_results.png')
    plt.savefig(plot_path, dpi=300)
    log(f"Plot saved: {plot_path}")

    # ─── Combined plot (both train + val on one graph) ───
    plt.figure(figsize=(12, 7))
    plt.plot(baseline['eval_steps'], baseline['train_losses'], 'b:', alpha=0.6, linewidth=1.5, label='Baseline Train')
    plt.plot(baseline['eval_steps'], baseline['val_losses'], 'b-', linewidth=2.5, label='Baseline Val', marker='o', markersize=3)
    plt.plot(pga['eval_steps'], pga['train_losses'], 'r:', alpha=0.6, linewidth=1.5, label='Prop-PGA Train')
    plt.plot(pga['eval_steps'], pga['val_losses'], 'r-', linewidth=2.5, label='Prop-PGA Val', marker='s', markersize=3)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'E7 — Baseline vs Propagative PGA\n5 Layers | lr={LR} | Shakespeare | {STEPS} Steps', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    combined_path = os.path.join(RESULTS_DIR, 'e7_combined.png')
    plt.savefig(combined_path, dpi=300)
    log(f"Combined plot saved: {combined_path}")

    # ─── Verdict Report ───
    base_gap = baseline['final_val'] - baseline['final_train']
    pga_gap = pga['final_val'] - pga['final_train']

    if pga['final_val'] < baseline['final_val']:
        verdict = "✅ Propagative PGA achieved LOWER validation loss — PGA wins."
        winner = "Propagative PGA"
    elif pga['final_val'] > baseline['final_val']:
        verdict = "❌ Baseline achieved lower validation loss — Baseline wins."
        winner = "Baseline"
    else:
        verdict = "🟡 Tied — both models achieved identical validation loss."
        winner = "Tie"

    if pga_gap < base_gap:
        gen_verdict = "Propagative PGA generalizes better (smaller train-val gap)."
    else:
        gen_verdict = "Baseline generalizes better (smaller train-val gap)."

    speed_ratio = pga['total_time'] / baseline['total_time']
    if speed_ratio < 0.95:
        speed_verdict = f"Prop-PGA was {1/speed_ratio:.2f}× faster than Baseline."
    elif speed_ratio > 1.05:
        speed_verdict = f"Prop-PGA was {speed_ratio:.2f}× slower than Baseline."
    else:
        speed_verdict = "Both models had similar training speed."

    report = f"""# E7 — Propagative PGA Experiment Report

> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Configuration

| Setting | Value |
|---|---|
| Dataset | Shakespeare (Tiny) |
| Layers | 5 |
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
| Execution | Parallel (2 processes) |

## Model Specs

| | Baseline | Propagative PGA |
|---|---|---|
| Parameters | {baseline['param_count']:,} | {pga['param_count']:,} |
| Extra Learnable Params | — | 0 (SVD is compute-only) |
| SVD calls/forward | 0 | T={BLOCK_SIZE} (once from embeddings) |
| P reuse | — | Same P across all 5 layers |

## Results

| Metric | Baseline | Propagative PGA | Delta |
|---|---|---|---|
| Final Train Loss | {baseline['final_train']:.4f} | {pga['final_train']:.4f} | {pga['final_train'] - baseline['final_train']:.4f} |
| Final Val Loss | {baseline['final_val']:.4f} | {pga['final_val']:.4f} | {pga['final_val'] - baseline['final_val']:.4f} |
| Overfitting Gap | {base_gap:.4f} | {pga_gap:.4f} | {pga_gap - base_gap:.4f} |
| Avg Step Time | {baseline['avg_step_ms']:.1f}ms | {pga['avg_step_ms']:.1f}ms | {pga['avg_step_ms'] - baseline['avg_step_ms']:.1f}ms |
| Total Time | {baseline['total_time']:.1f}s | {pga['total_time']:.1f}s | — |

## Verdict

**Winner: {winner}**

- **Loss**: {verdict}
- **Generalization**: {gen_verdict}
- **Speed**: {speed_verdict}

## Propagative Nature

The key innovation tested: SVD is computed **once per token from the embedding layer**, and the resulting projection matrix P is **reused identically across all 5 layers**. This means:

- The "principle" is a property of the **observation**, not the intermediate computation
- SVD cost = T (not T × L) — **5× compute savings** vs full per-layer PGA
- No extra parameters — PGA is pure compute overhead

![E7 Combined Results](e7_combined.png)
"""

    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Report saved: {report_path}")

    # Save raw results as JSON
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({'baseline': baseline, 'pga': pga, 'config': {
            'steps': STEPS, 'lr': LR, 'n_layer': N_LAYER, 'n_embd': N_EMBD,
            'block_size': BLOCK_SIZE, 'vocab_size': vocab_size,
            'wall_time': total_wall,
        }}, f, indent=2)
    log(f"Raw data saved: {json_path}")

    log("=" * 60)
    log(f"VERDICT: {winner}")
    log("=" * 60)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()

    run_parallel()

