#!/usr/bin/env python3
"""Compare tabular RL policy with classical pursuit baselines on the same environment.

Outputs under --out-dir:
- comparison_metrics.csv
- comparison_summary.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from tabular_pursuit_qlearning import PursuitEvasionTabularEnv


PolicyFn = Callable[[PursuitEvasionTabularEnv, tuple], int]


@dataclass
class EvalResult:
    policy: str
    capture_rate: float
    terrain_capture_rate: float
    boundary_capture_rate: float
    avg_steps: float
    avg_capture_steps: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tabular RL with classical baselines")
    p.add_argument("--result-dir", type=str, required=True, help="Directory containing q_table.pkl")
    p.add_argument("--episodes", type=int, default=400, help="Evaluation episodes per policy")
    p.add_argument("--seed", type=int, default=2026, help="Random seed")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <result-dir>/comparison)")
    return p.parse_args()


def load_q_table(result_dir: str) -> Dict[tuple, np.ndarray]:
    qtable_path = os.path.join(result_dir, "q_table.pkl")
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(f"Missing file: {qtable_path}")
    with open(qtable_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "q_table" in obj:
        return obj["q_table"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported q_table.pkl format")


def angle_to_action(rel_angle: float, n_actions: int = 8) -> int:
    """Map relative angle in [-pi, pi] to discrete action index."""
    step = 2.0 * math.pi / float(n_actions)
    idx = int(np.round(rel_angle / step)) % n_actions
    return idx


def make_rl_policy(q_table: Dict[tuple, np.ndarray]) -> PolicyFn:
    def rl_policy(env: PursuitEvasionTabularEnv, s: tuple) -> int:
        q = q_table.get(s)
        if q is None:
            return 0
        return int(np.argmax(q[: env.n_actions]))

    return rl_policy


def pure_pursuit_policy(env: PursuitEvasionTabularEnv, s: tuple) -> int:
    del s
    # action=0 means heading directly toward evader bearing.
    return 0


def lead_pursuit_policy(env: PursuitEvasionTabularEnv, s: tuple) -> int:
    del s
    rel = env.e - env.p
    dist = float(np.linalg.norm(rel))
    if dist < 1e-6:
        return 0

    # Predict short-horizon evader displacement and aim at intercept point.
    tau = np.clip(dist / max(env.v_max_p, 1e-6), 0.3, 1.8)
    target = env.e + env.ve * tau
    desired = math.atan2(float(target[1] - env.p[1]), float(target[0] - env.p[0]))
    alpha = math.atan2(float(rel[1]), float(rel[0]))
    rel_angle = (desired - alpha + math.pi) % (2.0 * math.pi) - math.pi
    return angle_to_action(rel_angle, env.n_actions)


def random_policy(env: PursuitEvasionTabularEnv, s: tuple) -> int:
    del s
    return random.randrange(env.n_actions)


def evaluate_policy(policy_name: str, policy_fn: PolicyFn, episodes: int, seed: int) -> EvalResult:
    random.seed(seed)
    np.random.seed(seed)

    wins = 0
    terrain = 0
    boundary = 0
    steps_all: List[int] = []
    steps_cap: List[int] = []

    for _ in range(episodes):
        env = PursuitEvasionTabularEnv(seed=random.randint(0, 1_000_000))
        s = env.reset()
        done = False
        info = {}

        while not done:
            a = policy_fn(env, s)
            s, _, done, info = env.step(a)

        steps_all.append(env.step_n)
        captured = bool(info.get("captured", False))
        if captured:
            wins += 1
            steps_cap.append(env.step_n)
            if info.get("near_terrain", False):
                terrain += 1
            if info.get("near_boundary", False):
                boundary += 1

    cap_rate = wins / max(1, episodes)
    terrain_rate = terrain / max(1, wins)
    boundary_rate = boundary / max(1, wins)

    return EvalResult(
        policy=policy_name,
        capture_rate=cap_rate,
        terrain_capture_rate=terrain_rate,
        boundary_capture_rate=boundary_rate,
        avg_steps=float(np.mean(steps_all)) if steps_all else 0.0,
        avg_capture_steps=float(np.mean(steps_cap)) if steps_cap else 0.0,
    )


def save_csv(results: List[EvalResult], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "capture_rate",
            "terrain_capture_rate",
            "boundary_capture_rate",
            "avg_steps",
            "avg_capture_steps",
        ])
        for r in results:
            writer.writerow([
                r.policy,
                f"{r.capture_rate:.6f}",
                f"{r.terrain_capture_rate:.6f}",
                f"{r.boundary_capture_rate:.6f}",
                f"{r.avg_steps:.3f}",
                f"{r.avg_capture_steps:.3f}",
            ])


def save_plot(results: List[EvalResult], out_png: str) -> None:
    policies = [r.policy for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    vals = [r.capture_rate for r in results]
    ax.bar(policies, vals, color=["#2E7D32", "#1565C0", "#6A1B9A", "#EF6C00"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Capture Rate")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    vals_t = [r.terrain_capture_rate for r in results]
    vals_b = [r.boundary_capture_rate for r in results]
    x = np.arange(len(policies))
    w = 0.36
    ax.bar(x - w / 2, vals_t, width=w, label="terrain", color="#2E7D32")
    ax.bar(x + w / 2, vals_b, width=w, label="boundary", color="#C62828")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Capture Type Breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[2]
    vals = [r.avg_steps for r in results]
    ax.bar(policies, vals, color=["#455A64", "#607D8B", "#78909C", "#90A4AE"])
    ax.set_title("Average Episode Steps")
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.result_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    q_table = load_q_table(args.result_dir)
    policies: List[tuple[str, PolicyFn]] = [
        ("RL_Greedy", make_rl_policy(q_table)),
        ("PurePursuit", pure_pursuit_policy),
        ("LeadPursuit", lead_pursuit_policy),
        ("Random", random_policy),
    ]

    results: List[EvalResult] = []
    for i, (name, fn) in enumerate(policies):
        r = evaluate_policy(name, fn, episodes=args.episodes, seed=args.seed + i * 101)
        results.append(r)
        print(
            f"[{name}] capture={r.capture_rate:.2%}, terrain={r.terrain_capture_rate:.2%}, "
            f"boundary={r.boundary_capture_rate:.2%}, avg_steps={r.avg_steps:.1f}, "
            f"avg_cap_steps={r.avg_capture_steps:.1f}"
        )

    out_csv = os.path.join(out_dir, "comparison_metrics.csv")
    out_png = os.path.join(out_dir, "comparison_summary.png")
    save_csv(results, out_csv)
    save_plot(results, out_png)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
