"""Parse terminal_output.txt and generate steps-vs-loss comparison plots."""
import re, os, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

LOG = os.path.join(os.path.dirname(__file__), "terminal_output.txt")
OUT_DIR = os.path.dirname(os.path.dirname(__file__))  # e10-token-store/

# ── Parse ────────────────────────────────────────────────────────────────
pattern = re.compile(
    r"\[(?P<model>Baseline|E9-Buffer|E10-TokenStore)\]\s+"
    r"Step\s+(?P<step>\d+)/\d+\s+\|\s+"
    r"Train:\s+(?P<train>[\d.]+)\s+\|\s+"
    r"Val:\s+(?P<val>[\d.]+)"
)

data = {"Baseline": [], "E9-Buffer": [], "E10-TokenStore": []}
seen = set()

with open(LOG, encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            model = m.group("model")
            step = int(m.group("step"))
            key = (model, step)
            if key in seen:          # skip duplicate lines in the log
                continue
            seen.add(key)
            data[model].append({
                "step": step,
                "train": float(m.group("train")),
                "val":   float(m.group("val")),
            })

for v in data.values():
    v.sort(key=lambda d: d["step"])

# ── Styling ──────────────────────────────────────────────────────────────
COLORS = {
    "Baseline":       "#6366f1",   # indigo
    "E9-Buffer":      "#f59e0b",   # amber
    "E10-TokenStore": "#10b981",   # emerald
}
DASH = {
    "Baseline":       "-",
    "E9-Buffer":      "--",
    "E10-TokenStore": "-.",
}

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "font.family":      "sans-serif",
    "font.size":        11,
})

# ── Figure 1 : Train + Val side-by-side ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=130)

for model, pts in data.items():
    steps  = [p["step"]  for p in pts]
    trains = [p["train"] for p in pts]
    vals   = [p["val"]   for p in pts]
    c, ls  = COLORS[model], DASH[model]

    ax1.plot(steps, trains, color=c, linestyle=ls, linewidth=1.8, label=model, alpha=0.9)
    ax2.plot(steps, vals,   color=c, linestyle=ls, linewidth=1.8, label=model, alpha=0.9)

for ax, title in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.8)
    ax.grid(True, linewidth=0.4)
    ax.set_xlim(left=0)

fig.suptitle("E10 — Steps vs Loss  (Baseline · E9-Buffer · E10-TokenStore)",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
path1 = os.path.join(OUT_DIR, "e10_steps_vs_loss.png")
fig.savefig(path1, bbox_inches="tight")
print(f"Saved  {path1}")

# ── Figure 2 : Combined overlay ─────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 7), dpi=130)

for model, pts in data.items():
    steps  = [p["step"]  for p in pts]
    trains = [p["train"] for p in pts]
    vals   = [p["val"]   for p in pts]
    c, ls  = COLORS[model], DASH[model]

    ax.plot(steps, trains, color=c, linestyle=ls, linewidth=1.8,
            label=f"{model} train", alpha=0.85)
    ax.plot(steps, vals,   color=c, linestyle=":",  linewidth=1.4,
            label=f"{model} val", alpha=0.65)

ax.set_title("E10 — All Curves Overlaid", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.legend(ncol=2, framealpha=0.8, fontsize=10)
ax.grid(True, linewidth=0.4)
ax.set_xlim(left=0)
fig2.tight_layout()
path2 = os.path.join(OUT_DIR, "e10_steps_vs_loss_overlay.png")
fig2.savefig(path2, bbox_inches="tight")
print(f"Saved  {path2}")

# ── Print summary table ─────────────────────────────────────────────────
print("\n── Final metrics ──────────────────────────────")
print(f"{'Model':<18} {'Train':>8} {'Val':>8} {'Gap':>8} {'Points':>6}")
for model, pts in data.items():
    if pts:
        last = pts[-1]
        gap = last["train"] - last["val"]
        print(f"{model:<18} {last['train']:>8.4f} {last['val']:>8.4f} {gap:>+8.4f} {len(pts):>6}")
print()
