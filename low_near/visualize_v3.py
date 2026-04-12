#!/usr/bin/env python3
"""V3 小目标到达可视化：轨迹 GIF + 误差曲线 + 统计指标图。

用法：
    python low_near/visualize_v3.py --exp-dir low_near/exp1_baseline
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous
from training.train_2d_v3_continuous import GaussianActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="best")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--gif-episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_model(exp_dir, checkpoint_name, device):
    ckpt_path = os.path.join(exp_dir, "checkpoints", f"{checkpoint_name}.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden_cfg = ckpt.get("config", {}).get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64])
    model = GaussianActorCritic(obs_dim, act_dim, hidden=tuple(hidden_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def rollout(model, env, device, deterministic=True):
    obs, info = env.reset()
    traj_pos = [env.inner.dynamics.eta[:2].copy()]
    traj_target = info["target_pos"]
    errors = []
    headings = []
    rewards = []
    done = False
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(x, deterministic=deterministic)
        action_np = action[0].cpu().numpy()
        obs, reward, term, trunc, info = env.step(action_np)
        traj_pos.append(env.inner.dynamics.eta[:2].copy())
        errors.append(info.get("pos_error", float("nan")))
        headings.append(info.get("heading_error", float("nan")))
        rewards.append(reward)
        done = term or trunc

    return {
        "traj_pos": np.array(traj_pos),
        "target": np.array(traj_target),
        "errors": np.array(errors),
        "headings": np.array(headings),
        "rewards": np.array(rewards),
        "success": info.get("termination_reason") == "success",
        "reason": info.get("termination_reason", "unknown"),
        "steps": len(errors),
    }


def save_trajectory_gif(episode, out_path, fps=20):
    traj = episode["traj_pos"]
    target = episode["target"]
    n = len(traj)

    fig, ax = plt.subplots(figsize=(6, 6))
    margin = 0.3
    all_pts = np.vstack([traj, target.reshape(1, 2)])
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("V3 Small-Target Reach")

    ax.plot(target[0], target[1], "r*", markersize=18, label="Target")
    line, = ax.plot([], [], "b-", lw=2, alpha=0.8, label="AUV")
    dot = ax.scatter([], [], c="blue", s=80, zorder=5)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="upper right")

    def update(i):
        line.set_data(traj[:i+1, 0], traj[:i+1, 1])
        dot.set_offsets([traj[i, 0], traj[i, 1]])
        err = episode["errors"][min(i, len(episode["errors"])-1)] if i < len(episode["errors"]) else episode["errors"][-1]
        status = "SUCCESS" if episode["success"] and i == n - 1 else "running"
        text.set_text(f"step={i} err={err:.4f}m {status}")
        return [line, dot, text]

    ani = FuncAnimation(fig, update, frames=n, interval=int(1000/fps), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_statistics_plot(episodes, out_path):
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    final_errors = [e["errors"][-1] if len(e["errors"]) > 0 else float("nan") for e in episodes]
    step_counts = [e["steps"] for e in episodes]
    total_rewards = [float(np.sum(e["rewards"])) for e in episodes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.bar(["Success", "Fail"], [successes/n, 1-successes/n], color=["green", "red"], alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_title(f"Success Rate: {successes}/{n} = {successes/n:.1%}")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.hist(final_errors, bins=20, color="steelblue", alpha=0.75)
    ax.axvline(0.1, color="red", ls="--", label="threshold=0.1m")
    ax.set_title("Final Position Error Distribution")
    ax.set_xlabel("error (m)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.hist(step_counts, bins=20, color="orange", alpha=0.75)
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("steps")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.hist(total_rewards, bins=20, color="purple", alpha=0.75)
    ax.set_title("Total Reward Distribution")
    ax.set_xlabel("reward")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_error_curves(episodes, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for i, ep in enumerate(episodes[:8]):
        color = "green" if ep["success"] else "red"
        ax.plot(ep["errors"], color=color, alpha=0.7, lw=1.5, label=f"ep{i+1}" if i < 4 else None)
    ax.axhline(0.1, color="black", ls="--", alpha=0.5, label="threshold")
    ax.set_title("Position Error Over Time")
    ax.set_xlabel("step")
    ax.set_ylabel("error (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    for i, ep in enumerate(episodes[:8]):
        color = "green" if ep["success"] else "red"
        ax.plot(ep["headings"], color=color, alpha=0.7, lw=1.5)
    ax.set_title("Heading Error Over Time")
    ax.set_xlabel("step")
    ax.set_ylabel("heading error (rad)")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    exp_dir = args.exp_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Loading model from {exp_dir} ...")
    model = load_model(exp_dir, args.checkpoint, device)

    env = LowLevelAUV2DEnvV3Continuous(seed=args.seed)

    n_episodes = args.episodes
    print(f"Running {n_episodes} evaluation episodes ...")
    episodes = []
    for i in range(n_episodes):
        ep = rollout(model, env, device, deterministic=True)
        episodes.append(ep)
        tag = "OK" if ep["success"] else "FAIL"
        err = ep["errors"][-1] if len(ep["errors"]) > 0 else float("nan")
        print(f"  ep {i+1:3d}/{n_episodes}: {tag}  err={err:.4f}m  steps={ep['steps']}")

    successes = sum(1 for e in episodes if e["success"])
    print(f"\nSuccess rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}")

    # --- save GIFs for up to 3 best and 3 worst episodes ---
    sorted_eps = sorted(episodes, key=lambda e: e["errors"][-1] if len(e["errors"]) > 0 else 999)
    gif_dir = os.path.join(exp_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    for i, ep in enumerate(sorted_eps[:min(args.gif_episodes, len(sorted_eps))]):
        out = os.path.join(gif_dir, f"best_{i+1}.gif")
        save_trajectory_gif(ep, out, fps=args.fps)
        print(f"  Saved {out}")

    for i, ep in enumerate(reversed(sorted_eps[-min(args.gif_episodes, len(sorted_eps)):])):
        out = os.path.join(gif_dir, f"worst_{i+1}.gif")
        save_trajectory_gif(ep, out, fps=args.fps)
        print(f"  Saved {out}")

    # --- statistics ---
    save_statistics_plot(episodes, os.path.join(exp_dir, "eval_statistics.png"))
    print(f"  Saved eval_statistics.png")

    save_error_curves(episodes, os.path.join(exp_dir, "eval_error_curves.png"))
    print(f"  Saved eval_error_curves.png")

    # --- summary text ---
    final_errors = [e["errors"][-1] for e in episodes if len(e["errors"]) > 0]
    with open(os.path.join(exp_dir, "eval_summary.txt"), "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Success Rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}\n")
        f.write(f"Mean Final Error: {np.mean(final_errors):.4f} m\n")
        f.write(f"Median Final Error: {np.median(final_errors):.4f} m\n")
        f.write(f"Min Final Error: {np.min(final_errors):.4f} m\n")
        f.write(f"Max Final Error: {np.max(final_errors):.4f} m\n")
        f.write(f"Mean Steps: {np.mean([e['steps'] for e in episodes]):.1f}\n")
        f.write(f"Mean Reward: {np.mean([np.sum(e['rewards']) for e in episodes]):.2f}\n")

    print(f"  Saved eval_summary.txt")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
