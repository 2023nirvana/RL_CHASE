"""
2D V2 连续动作结果增强可视化
=============================

功能：
- 多 episode 回放评估（成功率/回报/步长）
- 详细单回合图（轨迹+朝向箭头、动作分量、速度/角速度、奖励、航向误差）
- 成功/失败轨迹对比图与统计图
- GIF 动画（带朝向箭头）
- 训练过程代理图：best/latest/final checkpoint 评估曲线（当缺少 loss 历史时）
"""

import os
import sys
import argparse
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v2_continuous import LowLevelAUV2DEnvV2Continuous


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

    def forward(self, x):
        mu = self.mu_head(self.actor_body(x))
        v = self.critic(x).squeeze(-1)
        return mu, v

    def act_deterministic(self, x):
        mu, _ = self.forward(x)
        return torch.tanh(mu)


@dataclass
class EpisodeTraj:
    pos: np.ndarray
    target: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    yaw_rate: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    distances: np.ndarray
    heading_error: np.ndarray
    target_bearing: np.ndarray
    success: bool
    termination_reason: str

    @property
    def ep_return(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def ep_len(self) -> int:
        return int(len(self.rewards))


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", 8))
    act_dim = int(ckpt.get("act_dim", 3))
    hidden = tuple(ckpt.get("config", {}).get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64]))
    model = GaussianActorCritic(obs_dim, act_dim, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _wrap_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def run_episode(model, env: LowLevelAUV2DEnvV2Continuous, device: torch.device, deterministic=True) -> EpisodeTraj:
    obs, info = env.reset()

    pos = [env.inner.dynamics.eta[:2].copy()]
    target = [env.inner.target_state[:2].copy()]
    yaw = [float(env.inner.dynamics.eta[5])]
    speed = [float(np.linalg.norm(env.inner.dynamics.nu[:2]))]
    yaw_rate = [float(env.inner.dynamics.nu[5])]
    distances = [float(np.linalg.norm(pos[-1] - target[-1]))]
    actions = []
    rewards = []
    heading_error = []
    target_bearing = []

    done = False
    term_reason = ""
    success = False

    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = model.act_deterministic(x)[0].cpu().numpy()
            else:
                action = model.act_deterministic(x)[0].cpu().numpy()

        obs, rew, term, trunc, step_info = env.step(action)

        p = env.inner.dynamics.eta[:2].copy()
        g = env.inner.target_state[:2].copy()
        y = float(env.inner.dynamics.eta[5])
        v = float(np.linalg.norm(env.inner.dynamics.nu[:2]))
        yr = float(env.inner.dynamics.nu[5])
        d = float(np.linalg.norm(p - g))

        bearing = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
        he = float(_wrap_pi(bearing - y))

        pos.append(p)
        target.append(g)
        yaw.append(y)
        speed.append(v)
        yaw_rate.append(yr)
        distances.append(d)
        heading_error.append(he)
        target_bearing.append(bearing)

        actions.append(np.asarray(step_info.get("action_continuous", action), dtype=np.float32))
        rewards.append(float(rew))

        done = bool(term or trunc)
        if done:
            term_reason = str(step_info.get("termination_reason", "unknown"))
            success = term_reason == "success"

    return EpisodeTraj(
        pos=np.asarray(pos),
        target=np.asarray(target),
        yaw=np.asarray(yaw),
        speed=np.asarray(speed),
        yaw_rate=np.asarray(yaw_rate),
        actions=np.asarray(actions) if len(actions) else np.zeros((0, 3), dtype=np.float32),
        rewards=np.asarray(rewards) if len(rewards) else np.zeros((0,), dtype=np.float32),
        distances=np.asarray(distances),
        heading_error=np.asarray(heading_error) if len(heading_error) else np.zeros((0,), dtype=np.float32),
        target_bearing=np.asarray(target_bearing) if len(target_bearing) else np.zeros((0,), dtype=np.float32),
        success=success,
        termination_reason=term_reason,
    )


def plot_episode_detailed(traj: EpisodeTraj, save_path: str, world_size: float, success_radius: float, arrow_stride: int = 8):
    pos, goal, yaw = traj.pos, traj.target, traj.yaw
    steps = np.arange(traj.ep_len)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], color="tab:blue", lw=1.8, label="AUV")
    ax.plot(goal[:, 0], goal[:, 1], color="tab:red", lw=1.2, alpha=0.5, label="Target")
    ax.scatter(pos[0, 0], pos[0, 1], c="tab:blue", s=70, marker="o", label="Start")
    ax.scatter(pos[-1, 0], pos[-1, 1], c="tab:blue", s=70, marker="x", label="End")
    ax.scatter(goal[0, 0], goal[0, 1], c="tab:red", s=85, marker="*", label="Goal")
    ax.add_patch(Circle((goal[0, 0], goal[0, 1]), success_radius, fill=False, ec="tab:red", ls="--", lw=1.3))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, ec="black", lw=1.2))

    ids = np.arange(0, len(pos), max(1, arrow_stride))
    u = np.cos(yaw[ids])
    v = np.sin(yaw[ids])
    ax.quiver(pos[ids, 0], pos[ids, 1], u, v, angles='xy', scale_units='xy', scale=8.0, color='tab:green', alpha=0.8, width=0.004)

    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title(f"Trajectory + Heading ({'Success' if traj.success else traj.termination_reason})")

    ax = axes[0, 1]
    ax.plot(np.arange(len(traj.distances)), traj.distances, lw=2, color="tab:purple", label="distance")
    ax.axhline(success_radius, ls="--", lw=1.2, color="tab:red", label="success radius")
    ax.set_title("Distance to Goal")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.arange(len(traj.speed)), traj.speed, lw=2, color="tab:blue", label="|v_xy|")
    ax.plot(np.arange(len(traj.yaw_rate)), np.abs(traj.yaw_rate), lw=1.8, color="tab:orange", label="|yaw_rate|")
    ax.set_title("Speed / Yaw-rate")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    if traj.actions.shape[0] > 0:
        ax.plot(steps, traj.actions[:, 0], lw=1.6, label="a_surge")
        ax.plot(steps, traj.actions[:, 1], lw=1.6, label="a_sway")
        ax.plot(steps, traj.actions[:, 2], lw=1.6, label="a_yaw")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Continuous Actions")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(steps, traj.rewards, lw=1.5, color="tab:green", label="reward")
    ax.set_title("Per-step Reward")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    if len(traj.heading_error) > 0:
        ax.plot(steps, traj.heading_error, lw=1.8, color="tab:red", label="heading error")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Heading Error (rad)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(f"Return={traj.ep_return:.2f}, Length={traj.ep_len}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_gif_with_heading(traj: EpisodeTraj, save_path: str, world_size: float, success_radius: float, fps: int = 12, frame_step: int = 6):
    pos = traj.pos
    target = traj.target
    yaw = traj.yaw
    dist = traj.distances

    idxs = list(range(0, len(pos), max(1, frame_step)))
    if idxs[-1] != len(pos) - 1:
        idxs.append(len(pos) - 1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, ec="black", lw=1.8))
    suc_circle = Circle((target[0, 0], target[0, 1]), success_radius, fill=False, ec="tab:red", ls="--", lw=1.2, alpha=0.8)
    ax.add_patch(suc_circle)

    line, = ax.plot([], [], color="tab:blue", lw=1.6, alpha=0.8)
    p_dot, = ax.plot([], [], "bo", markersize=8)
    g_dot, = ax.plot([target[0, 0]], [target[0, 1]], "r*", markersize=12)
    heading_quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=8.0, color='tab:green', width=0.005)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("2D V2 Continuous - Trajectory + Heading")

    def init():
        line.set_data([], [])
        p_dot.set_data([], [])
        txt.set_text("")
        return line, p_dot, g_dot, heading_quiver, txt

    def animate(i):
        fi = idxs[i]
        line.set_data(pos[:fi + 1, 0], pos[:fi + 1, 1])
        p_dot.set_data([pos[fi, 0]], [pos[fi, 1]])
        suc_circle.center = (target[fi, 0], target[fi, 1])
        g_dot.set_data([target[fi, 0]], [target[fi, 1]])

        nonlocal heading_quiver
        try:
            heading_quiver.remove()
        except Exception:
            pass
        heading_quiver = ax.quiver(
            [pos[fi, 0]], [pos[fi, 1]],
            [np.cos(yaw[fi])], [np.sin(yaw[fi])],
            angles='xy', scale_units='xy', scale=5.0,
            color='tab:green', width=0.006,
        )

        txt.set_text(f"step={fi}\\ndist={dist[fi]:.3f}")
        return line, p_dot, g_dot, heading_quiver, txt

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(idxs), interval=1000 // max(1, fps), blit=True)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()


def plot_success_failure_compare(trajs: List[EpisodeTraj], save_path: str, world_size: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    succ = [t for t in trajs if t.success]
    fail = [t for t in trajs if not t.success]

    for t in succ[:20]:
        axes[0].plot(t.pos[:, 0], t.pos[:, 1], alpha=0.45, lw=1.2)
    axes[0].set_title(f"Successful Trajectories ({len(succ)})")

    for t in fail[:20]:
        axes[1].plot(t.pos[:, 0], t.pos[:, 1], alpha=0.45, lw=1.2)
    axes[1].set_title(f"Failed Trajectories ({len(fail)})")

    for ax in axes:
        ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, ec="black", lw=1.2))
        ax.set_xlim(-world_size - 1, world_size + 1)
        ax.set_ylim(-world_size - 1, world_size + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_aggregate_stats(trajs: List[EpisodeTraj], save_path: str):
    returns = np.array([t.ep_return for t in trajs], dtype=np.float32)
    lengths = np.array([t.ep_len for t in trajs], dtype=np.float32)
    success = np.array([1.0 if t.success else 0.0 for t in trajs], dtype=np.float32)

    max_len = int(max([len(t.distances) for t in trajs]))
    dist_mat = np.full((len(trajs), max_len), np.nan, dtype=np.float32)
    for i, t in enumerate(trajs):
        dist_mat[i, :len(t.distances)] = t.distances
    dist_mean = np.nanmean(dist_mat, axis=0)
    dist_p25 = np.nanpercentile(dist_mat, 25, axis=0)
    dist_p75 = np.nanpercentile(dist_mat, 75, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.hist(returns, bins=12, color="tab:blue", alpha=0.8)
    ax.set_title("Episode Return Distribution")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.hist(lengths, bins=12, color="tab:orange", alpha=0.8)
    ax.set_title("Episode Length Distribution")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    x = np.arange(len(dist_mean))
    ax.plot(x, dist_mean, lw=2, color="tab:purple", label="mean distance")
    ax.fill_between(x, dist_p25, dist_p75, color="tab:purple", alpha=0.2, label="IQR")
    ax.set_title("Distance Curve Envelope")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    rate = float(np.mean(success))
    ax.bar(["success", "failure"], [rate, 1.0 - rate], color=["tab:green", "tab:red"], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"N={len(trajs)}, success={np.mean(success):.2%}, return_mean={np.mean(returns):.2f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_checkpoint(ckpt_path: str, target_distance: float, episodes: int, device: torch.device) -> Dict[str, float]:
    model = load_model(ckpt_path, device)
    env = LowLevelAUV2DEnvV2Continuous(target_distance=target_distance, enable_domain_randomization=False)
    trajs = [run_episode(model, env, device, deterministic=True) for _ in range(episodes)]
    env.close()

    return {
        "return_mean": float(np.mean([t.ep_return for t in trajs])),
        "len_mean": float(np.mean([t.ep_len for t in trajs])),
        "success_rate": float(np.mean([1.0 if t.success else 0.0 for t in trajs])),
    }


def plot_checkpoint_proxy_curve(ckpt_dir: str, target_distance: float, out_path: str, device: torch.device):
    names = ["best", "latest", "final"]
    paths = [os.path.join(ckpt_dir, f"{n}.pth") for n in names]
    valid = [(n, p) for n, p in zip(names, paths) if os.path.exists(p)]
    if not valid:
        return

    metrics = {"name": [], "return": [], "success": [], "len": []}
    for name, path in valid:
        m = evaluate_checkpoint(path, target_distance=target_distance, episodes=8, device=device)
        metrics["name"].append(name)
        metrics["return"].append(m["return_mean"])
        metrics["success"].append(m["success_rate"])
        metrics["len"].append(m["len_mean"])

    x = np.arange(len(metrics["name"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].plot(x, metrics["return"], marker="o", lw=2)
    axes[0].set_xticks(x, metrics["name"])
    axes[0].set_title("Return (proxy)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, metrics["success"], marker="o", lw=2, color="tab:green")
    axes[1].set_xticks(x, metrics["name"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Success Rate (proxy)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, metrics["len"], marker="o", lw=2, color="tab:orange")
    axes[2].set_xticks(x, metrics["name"])
    axes[2].set_title("Episode Length (proxy)")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Training-process proxy from checkpoints (no per-update loss log found)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _load_training_metrics(log_dir: str) -> Optional[Dict[str, np.ndarray]]:
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    json_path = os.path.join(log_dir, "training_metrics.json")

    rows = None
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        return None

    if rows is None or len(rows) == 0:
        return None

    def to_arr(key: str):
        vals = []
        for r in rows:
            v = r.get(key, np.nan)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        return np.array(vals, dtype=np.float64)

    return {
        "steps": to_arr("steps"),
        "update": to_arr("update"),
        "train_mean_reward": to_arr("train_mean_reward"),
        "train_loss": to_arr("train_loss"),
        "train_success_rate": to_arr("train_success_rate"),
        "eval_mean_reward": to_arr("eval_mean_reward"),
        "eval_success_rate": to_arr("eval_success_rate"),
    }


def plot_training_curves_from_log(log_dir: str, out_path: str) -> bool:
    m = _load_training_metrics(log_dir)
    if m is None:
        return False

    steps = m["steps"]
    updates = m["update"]
    train_rew = m["train_mean_reward"]
    train_loss = m["train_loss"]
    train_sr = m["train_success_rate"]
    eval_rew = m["eval_mean_reward"]
    eval_sr = m["eval_success_rate"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(steps, train_rew, lw=1.8, label="train_mean_reward")
    valid = ~np.isnan(eval_rew)
    if np.any(valid):
        ax.plot(steps[valid], eval_rew[valid], lw=1.8, label="eval_mean_reward")
    ax.set_title("Reward Curve (from training log)")
    ax.set_xlabel("timesteps")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, train_loss, lw=1.8, color="tab:orange")
    ax.set_title("Loss Curve (from training log)")
    ax.set_xlabel("timesteps")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(steps, train_sr, lw=1.8, label="train_success_rate")
    valid = ~np.isnan(eval_sr)
    if np.any(valid):
        ax.plot(steps[valid], eval_sr[valid], lw=1.8, label="eval_success_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate Curve (from training log)")
    ax.set_xlabel("timesteps")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(updates, train_rew, lw=1.8, label="reward by update")
    ax.set_title("Reward vs Update")
    ax.set_xlabel("update")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization for 2D V2 continuous policy")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path (final.pth etc.)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="directory containing best/latest/final")
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--gif-count", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif-frame-step", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--train-log-dir", type=str, default=None, help="Directory containing training_metrics.csv/json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.ckpt is None:
        if args.ckpt_dir is None:
            raise ValueError("Please provide --ckpt or --ckpt-dir")
        default_final = os.path.join(args.ckpt_dir, "final.pth")
        default_latest = os.path.join(args.ckpt_dir, "latest.pth")
        args.ckpt = default_final if os.path.exists(default_final) else default_latest

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.dirname(args.ckpt)

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", "low_level_2d_v2_cont_enhanced")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    env = LowLevelAUV2DEnvV2Continuous(target_distance=args.target_distance, enable_domain_randomization=False)
    world_size = 10.0
    success_radius = 0.2

    trajs: List[EpisodeTraj] = []
    for _ in range(args.episodes):
        trajs.append(run_episode(model, env, device, deterministic=True))
    env.close()

    # best success / worst fail for detail
    succ = [t for t in trajs if t.success]
    fail = [t for t in trajs if not t.success]

    if succ:
        best = sorted(succ, key=lambda t: t.ep_return, reverse=True)[0]
        plot_episode_detailed(best, os.path.join(out_dir, "episode_best_success_detailed.png"), world_size, success_radius)
    if fail:
        hard = sorted(fail, key=lambda t: t.ep_return)[0]
        plot_episode_detailed(hard, os.path.join(out_dir, "episode_hard_failure_detailed.png"), world_size, success_radius)

    # also plot first episode
    plot_episode_detailed(trajs[0], os.path.join(out_dir, "episode_1_detailed.png"), world_size, success_radius)

    plot_success_failure_compare(trajs, os.path.join(out_dir, "success_failure_trajectories.png"), world_size)
    plot_aggregate_stats(trajs, os.path.join(out_dir, "aggregate_stats.png"))

    # gifs: prefer successes then failures
    gif_candidates = succ + fail
    for i, t in enumerate(gif_candidates[:max(0, args.gif_count)]):
        gif_path = os.path.join(out_dir, f"episode_{i+1}.gif")
        save_gif_with_heading(t, gif_path, world_size, success_radius, fps=args.fps, frame_step=args.gif_frame_step)

    # training proxy plot from checkpoints
    train_log_dir = args.train_log_dir
    if train_log_dir is None:
        # 默认按实验名推断日志目录
        exp_name = os.path.basename(os.path.normpath(args.ckpt_dir))
        train_log_dir = os.path.join(PROJECT_ROOT, "visualization", "logs", exp_name)

    true_curve_path = os.path.join(out_dir, "training_curves_from_log.png")
    has_true_curve = plot_training_curves_from_log(train_log_dir, true_curve_path)
    if not has_true_curve:
        proxy_path = os.path.join(out_dir, "training_proxy_checkpoint_curve.png")
        plot_checkpoint_proxy_curve(args.ckpt_dir, args.target_distance, proxy_path, device)

    summary = {
        "episodes": len(trajs),
        "success_rate": float(np.mean([1.0 if t.success else 0.0 for t in trajs])),
        "return_mean": float(np.mean([t.ep_return for t in trajs])),
        "return_std": float(np.std([t.ep_return for t in trajs])),
        "length_mean": float(np.mean([t.ep_len for t in trajs])),
        "ckpt": args.ckpt,
        "ckpt_dir": args.ckpt_dir,
        "train_log_dir": train_log_dir,
        "has_true_training_curve": has_true_curve,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Enhanced visualization done.")
    print(f"Output dir: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
