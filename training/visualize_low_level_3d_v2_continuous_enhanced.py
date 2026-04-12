"""
3D V2 连续低层增强可视化
=========================

输出内容：
- 每回合详细图（3D轨迹、XY轨迹+朝向、深度/距离曲线）
- 多个回合 GIF（XY 视角）
- 聚合统计图
- 训练曲线（优先使用 training_metrics.csv/json）
- summary.json
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_3d_v2_continuous import LowLevelAUV3DEnvV2Continuous


class GaussianActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 64)):
        super().__init__()
        actor_layers = []
        prev = obs_dim
        for h in hidden:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*actor_layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        critic_layers = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.critic = nn.Sequential(*critic_layers, nn.Linear(prev, 1))

    def get_action(self, x, deterministic=True):
        mu = self.mu_head(self.actor_body(x))
        if deterministic:
            z = mu
        else:
            log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = torch.exp(log_std).expand_as(mu)
            z = torch.distributions.Normal(mu, std).rsample()
        return torch.tanh(z)


def _load_model(ckpt_path: str, device: torch.device) -> GaussianActorCritic:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", 10))
    act_dim = int(ckpt.get("act_dim", 4))

    state = ckpt["model"]
    hidden = []
    idx = 0
    while True:
        k = f"actor_body.{idx}.weight"
        if k not in state:
            break
        hidden.append(int(state[k].shape[0]))
        idx += 2
    if not hidden:
        hidden = [128, 64]

    model = GaussianActorCritic(obs_dim, act_dim, hidden=tuple(hidden)).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _pick_ckpt(ckpt: Optional[str], ckpt_dir: Optional[str]) -> str:
    if ckpt is not None:
        return ckpt
    if ckpt_dir is None:
        raise ValueError("Need --ckpt or --ckpt-dir")
    for name in ["best.pth", "final.pth", "latest.pth"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _run_episode(
    env: LowLevelAUV3DEnvV2Continuous,
    model: GaussianActorCritic,
    device: torch.device,
    max_steps: int,
    min_z_gap: float = 0.0,
    max_reset_tries: int = 20,
):
    obs, info = env.reset()
    for _ in range(max_reset_tries):
        init_pos = np.array(info.get("init_pos", env.inner.dynamics.eta[:3]), dtype=np.float64)
        target_pos = np.array(info.get("target_pos", env.inner.target_state[:3]), dtype=np.float64)
        if abs(float(target_pos[2] - init_pos[2])) >= float(min_z_gap):
            break
        obs, info = env.reset()
    target = np.array(info.get("target_pos", env.inner.target_state[:3]), dtype=np.float64)

    positions = [env.inner.dynamics.eta[:3].copy()]
    yaws = [float(env.inner.dynamics.eta[5])]
    speeds = [float(np.linalg.norm(env.inner.dynamics.nu[:3]))]
    rewards = []
    distances = [float(np.linalg.norm(positions[-1] - target))]

    done = False
    steps = 0
    termination_reason = "timeout"

    while (not done) and steps < max_steps:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = model.get_action(x, deterministic=True)[0].cpu().numpy()

        obs, reward, term, trunc, step_info = env.step(action)
        done = bool(term or trunc)
        steps += 1

        positions.append(env.inner.dynamics.eta[:3].copy())
        yaws.append(float(env.inner.dynamics.eta[5]))
        speeds.append(float(np.linalg.norm(env.inner.dynamics.nu[:3])))
        rewards.append(float(reward))
        distances.append(float(np.linalg.norm(positions[-1] - target)))

        if done:
            termination_reason = step_info.get("termination_reason", "unknown")

    success = termination_reason == "success"
    ep_return = float(np.sum(rewards))

    return {
        "positions": np.array(positions),
        "yaws": np.array(yaws),
        "speeds": np.array(speeds),
        "rewards": np.array(rewards),
        "distances": np.array(distances),
        "target": target,
        "steps": int(steps),
        "return": ep_return,
        "success": bool(success),
        "termination_reason": termination_reason,
    }


def _plot_episode_detail(ep: Dict, out_path: str, z_visual_scale: float = 1.8):
    p = ep["positions"]
    yaws = ep["yaws"]
    target = ep["target"]
    d = ep["distances"]
    r = ep["rewards"]
    success = ep["success"]

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(p[:, 0], p[:, 1], p[:, 2], lw=2)
    ax1.scatter([p[0, 0]], [p[0, 1]], [p[0, 2]], c="green", s=60, label="start")
    ax1.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], c="blue", s=60, label="end")
    ax1.scatter([target[0]], [target[1]], [target[2]], c="red", marker="*", s=120, label="target")
    ax1.set_title("3D Trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    x_span = max(1e-6, float(np.max(p[:, 0]) - np.min(p[:, 0])))
    y_span = max(1e-6, float(np.max(p[:, 1]) - np.min(p[:, 1])))
    z_min = float(min(np.min(p[:, 2]), target[2]))
    z_max = float(max(np.max(p[:, 2]), target[2]))
    z_span = max(1e-6, z_max - z_min)
    z_pad = max(0.15, 0.1 * z_span)
    ax1.set_zlim(z_min - z_pad, z_max + z_pad)
    ax1.set_box_aspect((x_span, y_span, max(0.2, z_span * z_visual_scale)))
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(p[:, 0], p[:, 1], lw=2)
    ax2.scatter([p[0, 0]], [p[0, 1]], c="green", s=60)
    ax2.scatter([p[-1, 0]], [p[-1, 1]], c="blue", s=60)
    ax2.scatter([target[0]], [target[1]], c="red", marker="*", s=120)
    step = max(1, len(p) // 20)
    for i in range(0, len(p), step):
        dx = 0.25 * np.cos(yaws[i])
        dy = 0.25 * np.sin(yaws[i])
        arr = FancyArrowPatch((p[i, 0], p[i, 1]), (p[i, 0] + dx, p[i, 1] + dy), mutation_scale=8)
        ax2.add_patch(arr)
    ax2.add_patch(Circle((target[0], target[1]), 0.5, fill=False, linestyle="--", alpha=0.6))
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("XY + heading")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.arange(len(d)), d, lw=2, label="distance")
    ax3.axhline(0.5, ls="--", alpha=0.7, label="success radius")
    ax3.set_title("Distance to target")
    ax3.set_xlabel("step")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    if len(r) > 0:
        ax4.plot(np.arange(1, len(r) + 1), r, lw=1.6, label="reward")
    ax4.set_title(f"Reward ({'Success' if success else 'Fail'})")
    ax4.set_xlabel("step")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    fig.suptitle(f"3D Continuous Episode | return={ep['return']:.2f} | steps={ep['steps']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_3d_gif(
    ep: Dict,
    out_path: str,
    fps: int = 10,
    frame_step: int = 4,
    z_visual_scale: float = 1.8,
    camera_elev: float = 24.0,
    camera_azim: float = 40.0,
):
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
    except Exception:
        return

    p = ep["positions"]
    target = ep["target"]

    idxs = list(range(0, len(p), max(1, frame_step)))
    if idxs[-1] != len(p) - 1:
        idxs.append(len(p) - 1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([target[0]], [target[1]], [target[2]], c="red", marker="*", s=120, label="target")

    x_min = float(min(np.min(p[:, 0]), target[0]) - 1.0)
    x_max = float(max(np.max(p[:, 0]), target[0]) + 1.0)
    y_min = float(min(np.min(p[:, 1]), target[1]) - 1.0)
    y_max = float(max(np.max(p[:, 1]), target[1]) + 1.0)
    z_min = float(min(np.min(p[:, 2]), target[2]) - 0.2)
    z_max = float(max(np.max(p[:, 2]), target[2]) + 0.2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=float(camera_elev), azim=float(camera_azim))
    ax.set_box_aspect((
        max(1e-6, x_max - x_min),
        max(1e-6, y_max - y_min),
        max(0.2, (z_max - z_min) * z_visual_scale),
    ))

    line, = ax.plot([], [], [], lw=2, color="tab:blue")
    point, = ax.plot([], [], [], "o", color="tab:blue")
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        title.set_text("")
        return line, point, title

    def update(fi):
        i = idxs[fi]
        line.set_data(p[:i + 1, 0], p[:i + 1, 1])
        line.set_3d_properties(p[:i + 1, 2])
        point.set_data([p[i, 0]], [p[i, 1]])
        point.set_3d_properties([p[i, 2]])
        title.set_text(f"step {i}/{len(p)-1}")
        return line, point, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(idxs), interval=max(1, 1000 // max(1, fps)), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _plot_aggregate(episodes: List[Dict], out_path: str):
    returns = np.array([e["return"] for e in episodes], dtype=np.float64)
    lengths = np.array([e["steps"] for e in episodes], dtype=np.float64)
    success = np.array([1.0 if e["success"] else 0.0 for e in episodes], dtype=np.float64)
    finals = np.array([e["distances"][-1] for e in episodes], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(returns, bins=min(12, len(returns)), alpha=0.8)
    ax.set_title("Return Distribution")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.hist(lengths, bins=min(12, len(lengths)), alpha=0.8)
    ax.set_title("Episode Length Distribution")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.hist(finals, bins=min(12, len(finals)), alpha=0.8)
    ax.axvline(0.5, ls="--", alpha=0.7, label="success radius")
    ax.set_title("Final Distance Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.bar(["success_rate"], [float(success.mean()) if len(success) else 0.0])
    ax.set_ylim(0, 1.0)
    ax.set_title("Success Rate")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_training_metrics(log_dir: str) -> Optional[Dict[str, np.ndarray]]:
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    json_path = os.path.join(log_dir, "training_metrics.json")

    rows = None
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

    if not rows:
        return None

    def to_arr(key):
        vals = []
        for r in rows:
            v = r.get(key, None)
            if v in (None, "", "null"):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
        return np.array(vals, dtype=np.float64)

    return {
        "steps": to_arr("steps"),
        "train_reward": to_arr("train_mean_reward"),
        "train_loss": to_arr("train_loss"),
        "train_success": to_arr("train_success_rate"),
        "eval_reward": to_arr("eval_mean_reward"),
        "eval_success": to_arr("eval_success_rate"),
    }


def _plot_training_curves(metrics: Dict[str, np.ndarray], out_path: str):
    steps = metrics["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(steps, metrics["train_reward"], label="train")
    valid = ~np.isnan(metrics["eval_reward"])
    if np.any(valid):
        ax.plot(steps[valid], metrics["eval_reward"][valid], label="eval")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, metrics["train_loss"], color="tab:orange")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(steps, metrics["train_success"], label="train")
    valid = ~np.isnan(metrics["eval_success"])
    if np.any(valid):
        ax.plot(steps[valid], metrics["eval_success"][valid], label="eval")
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(np.arange(1, len(metrics["train_reward"]) + 1), metrics["train_reward"])
    ax.set_title("Reward vs Update")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization for 3D V2 continuous policy")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--target-distance", type=float, default=12.0)
    parser.add_argument("--z-min", type=float, default=2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--gif-count", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gif-frame-step", type=int, default=4)
    parser.add_argument("--gif-camera-elev", type=float, default=24.0)
    parser.add_argument("--gif-camera-azim", type=float, default=40.0)
    parser.add_argument("--min-z-gap", type=float, default=0.8)
    parser.add_argument("--z-visual-scale", type=float, default=1.8)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--train-log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = _pick_ckpt(args.ckpt, args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, device)

    env = LowLevelAUV3DEnvV2Continuous(
        target_distance=args.target_distance,
        z_range=(args.z_min, args.z_max),
        max_steps=args.max_steps,
        enable_domain_randomization=False,
    )

    episodes = []
    for i in range(args.episodes):
        ep = _run_episode(
            env,
            model,
            device,
            max_steps=args.max_steps,
            min_z_gap=args.min_z_gap,
        )
        episodes.append(ep)
        if i == 0:
            _plot_episode_detail(
                ep,
                os.path.join(out_dir, "episode_1_detailed.png"),
                z_visual_scale=args.z_visual_scale,
            )

    env.close()

    success_rate = float(np.mean([1.0 if e["success"] else 0.0 for e in episodes])) if episodes else 0.0
    return_mean = float(np.mean([e["return"] for e in episodes])) if episodes else 0.0
    return_std = float(np.std([e["return"] for e in episodes])) if episodes else 0.0
    length_mean = float(np.mean([e["steps"] for e in episodes])) if episodes else 0.0

    best_success_eps = [e for e in episodes if e["success"]]
    if best_success_eps:
        best_ep = max(best_success_eps, key=lambda x: x["return"])
    else:
        best_ep = max(episodes, key=lambda x: x["return"])
    _plot_episode_detail(
        best_ep,
        os.path.join(out_dir, "episode_best_success_detailed.png"),
        z_visual_scale=args.z_visual_scale,
    )

    _plot_aggregate(episodes, os.path.join(out_dir, "aggregate_stats.png"))

    for i in range(min(args.gif_count, len(episodes))):
        _save_3d_gif(
            episodes[i],
            os.path.join(out_dir, f"episode_{i+1}.gif"),
            fps=args.fps,
            frame_step=args.gif_frame_step,
            z_visual_scale=args.z_visual_scale,
            camera_elev=args.gif_camera_elev,
            camera_azim=args.gif_camera_azim,
        )

    summary = {
        "episodes": int(args.episodes),
        "success_rate": success_rate,
        "return_mean": return_mean,
        "return_std": return_std,
        "length_mean": length_mean,
        "ckpt": ckpt_path,
        "ckpt_dir": args.ckpt_dir,
        "train_log_dir": args.train_log_dir,
        "has_true_training_curve": False,
    }

    if args.train_log_dir:
        metrics = _load_training_metrics(args.train_log_dir)
        if metrics is not None:
            _plot_training_curves(metrics, os.path.join(out_dir, "training_curves_from_log.png"))
            summary["has_true_training_curve"] = True

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Enhanced 3D continuous visualization done.")
    print(f"Output dir: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
