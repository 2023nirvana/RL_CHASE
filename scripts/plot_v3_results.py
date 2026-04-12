#!/usr/bin/env python3
"""V3 Online PPO 六组实验对比可视化"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

EXPS = [
    ("v3_bppo_A",  "BPPO-A (conservative)", "#1f77b4", "--"),
    ("v3_bppo_B",  "BPPO-B (aggressive)",   "#1f77b4", "-"),
    ("v3_iql_A",   "IQL-A  (conservative)",  "#ff7f0e", "--"),
    ("v3_iql_B",   "IQL-B  (aggressive)",    "#ff7f0e", "-"),
    ("v3_td3bc_A", "TD3BC-A (conservative)", "#2ca02c", "--"),
    ("v3_td3bc_B", "TD3BC-B (aggressive)",   "#2ca02c", "-"),
]

def load(name):
    p = CKPT / name / "history.json"
    with open(p) as f:
        return json.load(f)

def smooth(y, w=5):
    if len(y) < w:
        return np.array(y)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

# ── Load data ──
data = {}
for eid, label, color, ls in EXPS:
    data[eid] = load(eid)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("V3 Online PPO — 6 Experiments Comparison (500K steps)", fontsize=15, fontweight="bold")

metrics = [
    ("train_sr",     "Train Success Rate",  axes[0, 0], True),
    ("train_rew",    "Train Reward",         axes[0, 1], False),
    ("train_ep_len", "Train Episode Length", axes[0, 2], False),
    ("train_r_event",   "Train Event Reward",   axes[1, 0], False),
    ("train_r_shaping", "Train Shaping Reward", axes[1, 1], False),
    ("train_r_sg_obs",  "Train Subgoal Penalty", axes[1, 2], False),
]

for key, title, ax, is_pct in metrics:
    for eid, label, color, ls in EXPS:
        h = data[eid]
        steps = [e["steps"] / 1000 for e in h]
        vals  = [e[key] for e in h]
        if is_pct:
            vals = [v * 100 for v in vals]
        ax.plot(steps, smooth(vals), label=label, color=color,
                linestyle=ls, linewidth=1.8, alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Steps (K)")
    ax.grid(True, alpha=0.3)
    if is_pct:
        ax.set_ylabel("%")
        ax.set_ylim(50, 100)

axes[0, 0].legend(fontsize=8, loc="lower right")
plt.tight_layout(rect=[0, 0, 1, 0.95])

out = ROOT / "visualization" / "v3_training_curves.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# ── Eval SR bar chart (from log files) ──
import re

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("V3 Eval Performance (best checkpoint)", fontsize=14, fontweight="bold")

labels = []
best_srs = []
best_rews = []
colors_bar = []

for eid, label, color, ls in EXPS:
    log_path = ROOT / "logs" / f"{eid}.log"
    srs, rews = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\[Eval\].*sr=([0-9.]+)%.*rew=([0-9.-]+)", line)
            if not m:
                m = re.search(r"\[Eval\].*rew=([0-9.-]+).*sr=([0-9.]+)%", line)
                if m:
                    rews.append(float(m.group(1)))
                    srs.append(float(m.group(2)))
            else:
                srs.append(float(m.group(1)))
                rews.append(float(m.group(2)))
    labels.append(label.split("(")[0].strip())
    best_srs.append(max(srs) if srs else 0)
    best_rews.append(max(rews) if rews else 0)
    colors_bar.append(color)

x = np.arange(len(labels))
bars1 = axes2[0].bar(x, best_srs, color=colors_bar, alpha=0.8, edgecolor="black")
axes2[0].set_ylabel("Success Rate (%)")
axes2[0].set_title("Best Eval Success Rate")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(labels, rotation=25, ha="right")
axes2[0].set_ylim(80, 105)
axes2[0].axhline(y=95, color="red", linestyle="--", alpha=0.5, label="95% line")
for bar, v in zip(bars1, best_srs):
    axes2[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")

bars2 = axes2[1].bar(x, best_rews, color=colors_bar, alpha=0.8, edgecolor="black")
axes2[1].set_ylabel("Reward")
axes2[1].set_title("Best Eval Reward")
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(labels, rotation=25, ha="right")
for bar, v in zip(bars2, best_rews):
    axes2[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out2 = ROOT / "visualization" / "v3_eval_comparison.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")

# ── Loss curves ──
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))
fig3.suptitle("V3 Loss & Training Dynamics", fontsize=14, fontweight="bold")

loss_metrics = [
    ("actor_loss",  "Actor (Policy) Loss",  axes3[0]),
    ("critic_loss", "Critic (Value) Loss",   axes3[1]),
    ("entropy",     "Entropy",               axes3[2]),
]

for key, title, ax in loss_metrics:
    for eid, label, color, ls in EXPS:
        h = data[eid]
        steps = [e["steps"] / 1000 for e in h]
        vals  = [e[key] for e in h]
        ax.plot(steps, smooth(vals), label=label, color=color,
                linestyle=ls, linewidth=1.5, alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Steps (K)")
    ax.grid(True, alpha=0.3)

axes3[0].legend(fontsize=7, loc="upper right")
plt.tight_layout(rect=[0, 0, 1, 0.93])
out3 = ROOT / "visualization" / "v3_loss_curves.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")

print("Done!")
