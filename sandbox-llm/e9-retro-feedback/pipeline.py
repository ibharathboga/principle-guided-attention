"""
E9 — RETRO vs Feedback Buffer — 5-Way Comparison Pipeline
  Phase 1 (parallel): Baseline, Prop-PGA, Prop-PGA+Buffer, Raw-Obs-Buffer
  Phase 2 (sequential): Pre-populate buffer from trained Baseline → train RETRO-Buffer

Dataset: enwik8 (100MB)
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
RETRO_PREFILL_BATCHES = 512  # How many batches to run through baseline to fill buffer

DATA_URL = "http://mattmahoney.net/dc/enwik8.zip"
DATA_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(DATA_DIR, "enwik8")
RESULTS_DIR = os.path.dirname(__file__)


def log(msg, model_name="SYSTEM"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{model_name}] {msg}", flush=True)


def load_data():
    if not os.path.exists(DATA_FILE):
        # Try copying from e8-buffer-enwik
        e8_data = os.path.join(DATA_DIR, "..", "e8-buffer-enwik", "enwik8")
        if os.path.exists(e8_data):
            import shutil
            shutil.copy2(e8_data, DATA_FILE)
            log("Copied enwik8 from e8-buffer-enwik")
        else:
            zip_path = os.path.join(DATA_DIR, "enwik8.zip")
            if not os.path.exists(zip_path):
                log(f"Downloading enwik8...")
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


def train_model(model, model_name, train_data, val_data):
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

            log(
                f"Step {step:>5}/{STEPS} | "
                f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                f"Step: {step_time*1000:.1f}ms | ETA: {eta}",
                model_name
            )

    losses = estimate_loss(model, train_data, val_data)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])
    eval_steps.append(STEPS)

    total_time = time.time() - t_start
    avg_step = np.mean(step_times) * 1000

    log(
        f"DONE — {total_time:.1f}s | {avg_step:.1f}ms/step | "
        f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f}",
        model_name
    )

    return {
        'name': model_name,
        'train_losses': train_losses, 'val_losses': val_losses,
        'eval_steps': eval_steps, 'total_time': total_time,
        'avg_step_ms': avg_step, 'param_count': param_count,
        'final_train': losses['train'], 'final_val': losses['val'],
    }


def train_single_model(model_name, model_cls_name, vocab_size, train_data, val_data):
    """Wrapper for parallel execution — instantiates and trains a model."""
    from models import (BaselineMicroGPT, PropagativePGA,
                        PropagativePGABuffer, PropagativePGARawBuffer)

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cls_map = {
        "BaselineMicroGPT": BaselineMicroGPT,
        "PropagativePGA": PropagativePGA,
        "PropagativePGABuffer": PropagativePGABuffer,
        "PropagativePGARawBuffer": PropagativePGARawBuffer,
    }
    model = cls_map[model_cls_name](vocab_size)
    return train_model(model, model_name, train_data, val_data)


def train_retro_model(vocab_size, train_data, val_data, baseline_state_dict):
    """Phase 2: Train RETRO model using frozen baseline outputs."""
    from models import BaselineMicroGPT, PropagativePGARetro

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_name = "RETRO-Buf"

    # Load trained baseline
    log("Loading trained baseline for RETRO buffer population...", model_name)
    baseline = BaselineMicroGPT(vocab_size)
    baseline.load_state_dict(baseline_state_dict)
    baseline.eval()

    # Collect baseline outputs to pre-populate the buffer
    log(f"Collecting {RETRO_PREFILL_BATCHES} batches of baseline outputs...", model_name)
    all_outputs = []
    with torch.no_grad():
        for i in range(RETRO_PREFILL_BATCHES):
            xb, _ = get_batch(train_data)
            logits, _ = baseline(xb)
            # Get the hidden states before lm_head by running embedding + layers
            x = baseline.rmsnorm(baseline.wte(xb) + baseline.wpe(torch.arange(BLOCK_SIZE)))
            from models import run_layers
            x = run_layers(baseline.layers, x, None, baseline.rmsnorm, xb.device)
            all_outputs.append(x.detach().view(-1, N_EMBD))  # (B*T, D)

    all_outputs = torch.cat(all_outputs, dim=0)  # (N, D)
    # Take last BUFFER_CAPACITY vectors
    if all_outputs.shape[0] > BUFFER_CAPACITY:
        all_outputs = all_outputs[-BUFFER_CAPACITY:]
    log(f"Collected {all_outputs.shape[0]} vectors for RETRO buffer", model_name)

    # Create RETRO model and populate buffer
    retro_model = PropagativePGARetro(vocab_size)
    retro_model.populate_buffer(all_outputs)
    log(f"Buffer populated: {retro_model.obs_buffer.count} vectors", model_name)

    # Train
    return train_model(retro_model, model_name, train_data, val_data)


def run_experiment():
    log("=" * 60)
    log("E9 — RETRO vs Feedback Buffer — 5-Way Comparison")
    log(f"Config: {N_LAYER}L, embd={N_EMBD}, lr={LR}, steps={STEPS}")
    log(f"Buffer: cap={BUFFER_CAPACITY}, k={RETRIEVE_K}")
    log(f"Dataset: enwik8, block={BLOCK_SIZE}")
    log("=" * 60)

    train_data, val_data, vocab_size, itos = load_data()
    log(f"Vocab: {vocab_size}")

    t_total = time.time()

    # ─── Phase 1: Train 4 models in parallel ───
    log("═══ Phase 1: Training 4 models in parallel ═══")

    phase1_models = [
        ("Baseline", "BaselineMicroGPT"),
        ("Prop-PGA", "PropagativePGA"),
        ("Prop-PGA+Buf", "PropagativePGABuffer"),
        ("Raw-Obs-Buf", "PropagativePGARawBuffer"),
    ]

    # We need baseline state dict for Phase 2, so train baseline in-process
    # and the other 3 in parallel
    from models import BaselineMicroGPT
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    baseline_model = BaselineMicroGPT(vocab_size)

    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit non-baseline models
        futures = {}
        for name, cls_name in phase1_models[1:]:  # Skip baseline
            f = executor.submit(
                train_single_model, name, cls_name,
                vocab_size, train_data, val_data
            )
            futures[f] = name

        # Train baseline in main process (we need its weights)
        baseline_result = train_model(baseline_model, "Baseline", train_data, val_data)
        baseline_state_dict = baseline_model.state_dict()

        results = {"Baseline": baseline_result}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log(f"ERROR in {name}: {e}")
                raise

    phase1_time = time.time() - t_total
    log(f"Phase 1 complete: {phase1_time:.1f}s")

    # ─── Phase 2: Train RETRO model ───
    log("═══ Phase 2: Training RETRO model (frozen baseline buffer) ═══")
    t_phase2 = time.time()

    retro_result = train_retro_model(vocab_size, train_data, val_data, baseline_state_dict)
    results["RETRO-Buf"] = retro_result

    phase2_time = time.time() - t_phase2
    total_wall = time.time() - t_total
    log(f"Phase 2 complete: {phase2_time:.1f}s")
    log(f"Total wall time: {total_wall:.1f}s")

    # ─── Plotting ───
    log("Generating plots...")

    colors = {
        'Baseline': '#2196F3',
        'Prop-PGA': '#FF5722',
        'Prop-PGA+Buf': '#4CAF50',
        'Raw-Obs-Buf': '#9C27B0',
        'RETRO-Buf': '#FF9800',
    }
    markers = {
        'Baseline': 'o',
        'Prop-PGA': 's',
        'Prop-PGA+Buf': '^',
        'Raw-Obs-Buf': 'D',
        'RETRO-Buf': 'v',
    }
    order = ['Baseline', 'Prop-PGA', 'Prop-PGA+Buf', 'Raw-Obs-Buf', 'RETRO-Buf']

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
        f'E9 — 5-Way Buffer Comparison\n'
        f'{N_LAYER}L | embd={N_EMBD} | lr={LR} | enwik8 | {STEPS} Steps',
        fontsize=13
    )
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, 'e9_combined.png')
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
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'E9 — 5-Way Comparison ({N_LAYER}L, enwik8)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    split_path = os.path.join(RESULTS_DIR, 'e9_results.png')
    plt.savefig(split_path, dpi=300)
    log(f"Split plot: {split_path}")

    # ─── Verdict ───
    all_models = [(name, results[name]) for name in order]
    best_val = min(all_models, key=lambda x: x[1]['final_val'])
    gaps = {name: r['final_val'] - r['final_train'] for name, r in all_models}
    best_gen = min(gaps.items(), key=lambda x: abs(x[1]))

    report = f"""# E9 — RETRO vs Feedback Buffer — 5-Way Comparison

> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Configuration

| Setting | Value |
|---|---|
| Dataset | enwik8 (100MB) |
| Layers | {N_LAYER} |
| n_embd | {N_EMBD} |
| Steps | {STEPS} |
| LR | {LR} |
| Buffer | {BUFFER_CAPACITY} capacity, k={RETRIEVE_K} |
| RETRO prefill | {RETRO_PREFILL_BATCHES} batches |

## Models

| | Baseline | Prop-PGA | Prop-PGA+Buf | Raw-Obs-Buf | RETRO-Buf |
|---|---|---|---|---|---|
| Params | {results['Baseline']['param_count']:,} | {results['Prop-PGA']['param_count']:,} | {results['Prop-PGA+Buf']['param_count']:,} | {results['Raw-Obs-Buf']['param_count']:,} | {results['RETRO-Buf']['param_count']:,} |
| SVD src | — | Window | Buffer (final) | Buffer (raw) | Buffer (frozen) |
| Feedback | — | — | ✅ Final outputs | ✅ Raw embeddings | ❌ Frozen |
| query_proj | — | — | ✅ | ❌ | ✅ |

## Results

| Metric | Baseline | Prop-PGA | Prop-PGA+Buf | Raw-Obs-Buf | RETRO-Buf |
|---|---|---|---|---|---|
| Train Loss | {results['Baseline']['final_train']:.4f} | {results['Prop-PGA']['final_train']:.4f} | {results['Prop-PGA+Buf']['final_train']:.4f} | {results['Raw-Obs-Buf']['final_train']:.4f} | {results['RETRO-Buf']['final_train']:.4f} |
| Val Loss | {results['Baseline']['final_val']:.4f} | {results['Prop-PGA']['final_val']:.4f} | {results['Prop-PGA+Buf']['final_val']:.4f} | {results['Raw-Obs-Buf']['final_val']:.4f} | {results['RETRO-Buf']['final_val']:.4f} |
| Gap | {gaps['Baseline']:.4f} | {gaps['Prop-PGA']:.4f} | {gaps['Prop-PGA+Buf']:.4f} | {gaps['Raw-Obs-Buf']:.4f} | {gaps['RETRO-Buf']:.4f} |
| Step Time | {results['Baseline']['avg_step_ms']:.1f}ms | {results['Prop-PGA']['avg_step_ms']:.1f}ms | {results['Prop-PGA+Buf']['avg_step_ms']:.1f}ms | {results['Raw-Obs-Buf']['avg_step_ms']:.1f}ms | {results['RETRO-Buf']['avg_step_ms']:.1f}ms |
| Total | {results['Baseline']['total_time']:.1f}s | {results['Prop-PGA']['total_time']:.1f}s | {results['Prop-PGA+Buf']['total_time']:.1f}s | {results['Raw-Obs-Buf']['total_time']:.1f}s | {results['RETRO-Buf']['total_time']:.1f}s |

## Verdict

**Val Loss Winner: {best_val[0]}** ({best_val[1]['final_val']:.4f})
**Generalization Winner: {best_gen[0]}** (gap: {best_gen[1]:.4f})

![E9 Combined](e9_combined.png)
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
