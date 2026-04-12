"""
Visualize 2D low-level target-reaching (V2) results.

Generates trajectory plot, distance curve, speed/acceleration curves, and GIF.
"""
import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v2 import LowLevelAUV2DEnvV2


def build_actor(obs_dim=8, hidden_sizes=(128, 64), num_actions=6, device="cpu"):
    layers = []
    prev = obs_dim
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, num_actions))
    model = torch.nn.Sequential(*layers).to(device)
    return model


def load_actor(ckpt_path: str, device: torch.device, hidden_sizes=(128, 64)):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = build_actor(obs_dim=8, hidden_sizes=hidden_sizes, num_actions=6, device=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def compute_acceleration(speeds: np.ndarray, dt: float) -> np.ndarray:
    if len(speeds) < 2:
        return np.zeros_like(speeds)
    acc = np.diff(speeds) / max(dt, 1e-6)
    return np.append(acc, acc[-1] if len(acc) else 0.0)


def run_episode(actor, env: LowLevelAUV2DEnvV2, device: torch.device, deterministic=True):
    obs, info = env.reset()
    pursuer_pos = [env.inner.dynamics.eta[:2].copy()]
    target_pos = [env.inner.target_state[:2].copy()]
    pursuer_speeds = []
    distances = [float(np.linalg.norm(pursuer_pos[0] - target_pos[0]))]
    rewards = []
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(x)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = torch.distributions.Categorical(logits=logits).sample().item()
        obs, reward, term, trunc, info = env.step(action)
        nu = env.inner.dynamics.nu
        pursuer_speed = float(np.sqrt(nu[0] ** 2 + nu[1] ** 2))
        pursuer_speeds.append(pursuer_speed)
        pursuer_pos.append(env.inner.dynamics.eta[:2].copy())
        target_pos.append(env.inner.target_state[:2].copy())
        dist = float(np.linalg.norm(pursuer_pos[-1] - target_pos[-1]))
        distances.append(dist)
        rewards.append(reward)
        done = term or trunc
    success = info.get("termination_reason") == "success"
    return {
        "pursuer_pos": np.array(pursuer_pos),
        "target_pos": np.array(target_pos),
        "pursuer_speeds": np.array(pursuer_speeds),
        "distances": np.array(distances),
        "rewards": np.array(rewards),
        "success": success,
    }


def plot_episode(traj: dict, save_path: str, world_size: float, catch_radius: float,
                 dt: float, pursuer_v_max: float, pursuer_a_max: float):
    p = traj["pursuer_pos"]
    t = traj["target_pos"]
    d = traj["distances"]
    p_speed = traj["pursuer_speeds"]
    success = traj["success"]

    p_acc = compute_acceleration(p_speed, dt)

    steps_vel = np.arange(len(p_speed))
    steps_dist = np.arange(len(d))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(p[:, 0], p[:, 1], "b-", lw=1.5, label="Pursuer", alpha=0.8)
    ax.plot(t[:, 0], t[:, 1], "r-", lw=1.5, label="Target", alpha=0.8)
    ax.scatter(p[0, 0], p[0, 1], c="blue", s=80, marker="o")
    ax.scatter(p[-1, 0], p[-1, 1], c="blue", s=80, marker="x")
    ax.scatter(t[0, 0], t[0, 1], c="red", s=80, marker="o")
    ax.scatter(t[-1, 0], t[-1, 1], c="red", s=80, marker="x")
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size,
                           fill=False, edgecolor="k", lw=1.5))
    circle = Circle((t[-1, 0], t[-1, 1]), catch_radius, fill=False,
                    edgecolor="green", linestyle="--", label="success_radius")
    ax.add_patch(circle)
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("2D Target Reach (" + ("Success" if success else "Timeout") + ")")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps_dist, d, "g-", lw=2, label="Distance")
    ax.axhline(y=catch_radius, color="gray", ls="--", label=f"success_radius={catch_radius}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.set_title("Distance to target")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps_vel, p_speed, "b-", lw=2, label="Pursuer |v|")
    ax.axhline(y=pursuer_v_max, color="blue", ls="--", alpha=0.5, label=f"v_max={pursuer_v_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed |v|")
    ax.legend(fontsize=8)
    ax.set_title("Speed")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_vel, p_acc, "b-", lw=2, label="Pursuer |a|")
    ax.axhline(y=pursuer_a_max, color="blue", ls="--", alpha=0.5, label=f"a_max={pursuer_a_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration |a|")
    ax.legend(fontsize=8)
    ax.set_title("Acceleration")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_gif(traj: dict, save_path: str, world_size: float, success_radius: float, fps: int = 15):
    p_pos = np.array(traj["pursuer_pos"])
    t_pos = np.array(traj["target_pos"])
    n_frames = len(p_pos)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size,
                           fill=False, edgecolor="k", lw=2))
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6)
    t_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6)
    p_pt, = ax.plot([], [], "bo", markersize=10)
    t_pt, = ax.plot([], [], "ro", markersize=8)
    circle = Circle((0, 0), success_radius, fill=False, edgecolor="green", linestyle="--", alpha=0.5)
    ax.add_patch(circle)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.set_title("2D Target Reach")

    def init():
        p_line.set_data([], [])
        t_line.set_data([], [])
        p_pt.set_data([], [])
        t_pt.set_data([], [])
        txt.set_text("")
        return p_line, t_line, p_pt, t_pt, txt

    def animate(i):
        p_line.set_data(p_pos[: i + 1, 0], p_pos[: i + 1, 1])
        t_line.set_data(t_pos[: i + 1, 0], t_pos[: i + 1, 1])
        p_pt.set_data([p_pos[i, 0]], [p_pos[i, 1]])
        t_pt.set_data([t_pos[i, 0]], [t_pos[i, 1]])
        circle.center = (t_pos[i, 0], t_pos[i, 1])
        d = np.linalg.norm(p_pos[i] - t_pos[i])
        txt.set_text(f"Step: {i}\nDist: {d:.2f}")
        return p_line, t_line, p_pt, t_pt, txt

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()


def _parse_target_distance(value: float) -> Tuple[float, float]:
    return (value, value)


def main():
    parser = argparse.ArgumentParser(description="Visualize low-level 2D V2 target reach")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--find-success", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_actor(args.ckpt, device)

    env = LowLevelAUV2DEnvV2(target_distance=args.target_distance)
    dt = env.inner.dt
    world_size = 10.0
    success_radius = 0.3
    pursuer_v_max = 1.5
    pursuer_a_max = 2.0

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", "low_level_2d_v2")
    os.makedirs(out_dir, exist_ok=True)

    max_attempts = args.episodes if not args.find_success else max(args.episodes * 5, args.episodes)
    saved = 0
    attempts = 0
    while saved < args.episodes and attempts < max_attempts:
        attempts += 1
        traj = run_episode(actor, env, device, deterministic=True)
        if args.find_success and not traj.get("success", False):
            continue
        ep_id = saved + 1
        plot_path = os.path.join(out_dir, f"ll2d_ep{ep_id}_traj_vel_acc.png")
        plot_episode(traj, plot_path, world_size, success_radius, dt, pursuer_v_max, pursuer_a_max)
        if not args.no_gif:
            gif_path = os.path.join(out_dir, f"ll2d_ep{ep_id}.gif")
            save_gif(traj, gif_path, world_size, success_radius, fps=args.gif_fps)
        print(f"Saved episode {ep_id} to {out_dir}")
        saved += 1

    if saved == 0:
        print("No successful episodes found.")


if __name__ == "__main__":
    main()
