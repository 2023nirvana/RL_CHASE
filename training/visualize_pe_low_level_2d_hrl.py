"""
2D HRL 追逃可视化：高层策略 + 固定低层策略
=========================================

生成：轨迹图、距离曲线、速度曲线、加速度曲线与 GIF。
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_pursuit_evasion import (
    LowLevelAUV2DPursuitEvasionEnv,
)


def _build_actor(obs_dim: int, num_actions: int, hidden_sizes=(256, 128, 64)) -> torch.nn.Module:
    layers = []
    prev_dim = obs_dim
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev_dim, h))
        layers.append(torch.nn.ReLU())
        prev_dim = h
    layers.append(torch.nn.Linear(prev_dim, num_actions))
    return torch.nn.Sequential(*layers)


def _get_low_level_obs(
    pursuer_pos: np.ndarray,
    pursuer_vel: np.ndarray,
    pursuer_yaw: float,
    pursuer_yaw_rate: float,
    target_pos: np.ndarray,
) -> np.ndarray:
    pos_error_world = target_pos - pursuer_pos
    distance = np.linalg.norm(pos_error_world)
    target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
    heading_error = np.arctan2(
        np.sin(target_angle - pursuer_yaw),
        np.cos(target_angle - pursuer_yaw),
    )
    cos_yaw = np.cos(pursuer_yaw)
    sin_yaw = np.sin(pursuer_yaw)
    pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
    pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
    obs = np.array(
        [
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            pursuer_vel[0] / 2.0,
            pursuer_vel[1] / 2.0,
            pursuer_yaw_rate / 2.0,
            distance / 5.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, 10.0)


def _high_action_to_waypoint(
    action: int,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    waypoint_distance: float,
    world_size: float,
) -> np.ndarray:
    if action == 0:
        return evader_pos.copy()
    angle = (action - 1) * (2 * np.pi / 8)
    offset = waypoint_distance * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    target = pursuer_pos + offset
    return np.clip(target, -world_size, world_size)


def _compute_acceleration(speeds: np.ndarray, dt: float) -> np.ndarray:
    if len(speeds) < 2:
        return np.zeros_like(speeds)
    acc = np.diff(speeds) / dt
    return np.append(acc, acc[-1] if len(acc) else 0.0)


def load_actor(ckpt_path: str, obs_dim: int, n_actions: int, hidden_sizes, device: torch.device, key: str = "actor"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_actor(obs_dim, n_actions, hidden_sizes=hidden_sizes).to(device)
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def run_episode(
    high_actor,
    low_actor,
    env: LowLevelAUV2DPursuitEvasionEnv,
    device: torch.device,
    low_level_steps: int,
    waypoint_distance: float,
    deterministic: bool = True,
):
    obs, info = env.reset()
    pursuer_pos = [env.inner.dynamics.eta[:2].copy()]
    evader_pos = [env.evader_pos.copy()]
    pursuer_speeds = []
    evader_speeds = []
    distances = [float(np.linalg.norm(pursuer_pos[0] - evader_pos[0]))]
    rewards = []

    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = high_actor(x)
            if deterministic:
                high_action = torch.argmax(logits, dim=-1).item()
            else:
                high_action = torch.distributions.Categorical(logits=logits).sample().item()

        pursuer_pos_np = env.inner.dynamics.eta[:2].copy()
        pursuer_vel_np = env.inner.dynamics.nu[:2].copy()
        pursuer_yaw = float(env.inner.dynamics.eta[5])
        pursuer_yaw_rate = float(env.inner.dynamics.nu[5])
        evader_pos_np = env.evader_pos.copy()

        subgoal = _high_action_to_waypoint(
            int(high_action),
            pursuer_pos_np,
            evader_pos_np,
            waypoint_distance,
            env.world_size,
        )

        for _ in range(low_level_steps):
            low_obs = _get_low_level_obs(
                pursuer_pos_np,
                pursuer_vel_np,
                pursuer_yaw,
                pursuer_yaw_rate,
                subgoal,
            )
            with torch.no_grad():
                low_logits = low_actor(torch.tensor(low_obs, dtype=torch.float32, device=device).unsqueeze(0))
                if deterministic:
                    low_action = torch.argmax(low_logits, dim=-1).item()
                else:
                    low_action = torch.distributions.Categorical(logits=low_logits).sample().item()
            obs, reward, term, trunc, info = env.step(int(low_action))
            rewards.append(reward)

            pursuer_pos_np = env.inner.dynamics.eta[:2].copy()
            pursuer_vel_np = env.inner.dynamics.nu[:2].copy()
            pursuer_yaw = float(env.inner.dynamics.eta[5])
            pursuer_yaw_rate = float(env.inner.dynamics.nu[5])

            pursuer_speed = float(np.linalg.norm(pursuer_vel_np))
            evader_speed = float(np.linalg.norm(env.evader_vel))
            pursuer_speeds.append(pursuer_speed)
            evader_speeds.append(evader_speed)

            pursuer_pos.append(pursuer_pos_np)
            evader_pos.append(env.evader_pos.copy())
            distances.append(float(np.linalg.norm(pursuer_pos_np - env.evader_pos)))

            done = term or trunc
            if done:
                break

    return {
        "pursuer_pos": np.array(pursuer_pos),
        "evader_pos": np.array(evader_pos),
        "pursuer_speeds": np.array(pursuer_speeds),
        "evader_speeds": np.array(evader_speeds),
        "distances": np.array(distances),
        "rewards": np.array(rewards),
        "success": info.get("termination_reason") == "success",
        "steps": len(rewards),
    }


def plot_episode(traj: dict, save_path: str, world_size: float, catch_radius: float, dt: float,
                 pursuer_v_max: float, evader_v_max: float, pursuer_a_max: float, evader_a_max: float):
    p = traj["pursuer_pos"]
    e = traj["evader_pos"]
    d = traj["distances"]
    p_speed = traj["pursuer_speeds"]
    e_speed = traj["evader_speeds"]
    success = traj["success"]

    p_acc = _compute_acceleration(p_speed, dt)
    e_acc = _compute_acceleration(e_speed, dt)

    steps_vel = np.arange(len(p_speed))
    steps_dist = np.arange(len(d))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(p[:, 0], p[:, 1], "b-", lw=1.5, label="Pursuer", alpha=0.8)
    ax.plot(e[:, 0], e[:, 1], "r-", lw=1.5, label="Evader", alpha=0.8)
    ax.scatter(p[0, 0], p[0, 1], c="blue", s=80, marker="o", zorder=5)
    ax.scatter(p[-1, 0], p[-1, 1], c="blue", s=80, marker="x", zorder=5)
    ax.scatter(e[0, 0], e[0, 1], c="red", s=80, marker="o", zorder=5)
    ax.scatter(e[-1, 0], e[-1, 1], c="red", s=80, marker="x", zorder=5)
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size,
                           fill=False, edgecolor="k", lw=1.5))
    circle = Circle((e[-1, 0], e[-1, 1]), catch_radius, fill=False,
                    edgecolor="green", linestyle="--", label="catch_radius")
    ax.add_patch(circle)
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("HRL Trajectory (" + ("Caught" if success else "Timeout") + ")")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps_dist, d, "g-", lw=2, label="Distance")
    ax.axhline(y=catch_radius, color="gray", ls="--", label=f"catch_radius={catch_radius}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.set_title("Distance to evader")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps_vel, p_speed, "b-", lw=2, label="Pursuer |v|", alpha=0.9)
    ax.plot(steps_vel, e_speed, "r-", lw=2, label="Evader |v|", alpha=0.9)
    ax.axhline(y=pursuer_v_max, color="blue", ls="--", alpha=0.5, label=f"P v_max={pursuer_v_max:.2f}")
    ax.axhline(y=evader_v_max, color="red", ls="--", alpha=0.5, label=f"E v_max={evader_v_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed |v|")
    ax.legend(fontsize=8)
    ax.set_title("Speed (Pursuer vs Evader)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_vel, p_acc, "b-", lw=2, label="Pursuer |a|", alpha=0.9)
    ax.plot(steps_vel, e_acc, "r-", lw=2, label="Evader |a|", alpha=0.9)
    ax.axhline(y=pursuer_a_max, color="blue", ls="--", alpha=0.5, label=f"P a_max={pursuer_a_max:.2f}")
    ax.axhline(y=evader_a_max, color="red", ls="--", alpha=0.5, label=f"E a_max={evader_a_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration |a|")
    ax.legend(fontsize=8)
    ax.set_title("Acceleration (Pursuer vs Evader)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_gif(traj: dict, save_path: str, world_size: float, catch_radius: float, fps: int = 15, every_n: int = 2):
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
    except ImportError:
        print("GIF skipped: PillowWriter not available")
        return

    p_pos = np.array(traj["pursuer_pos"])
    e_pos = np.array(traj["evader_pos"])
    n_frames = len(p_pos)
    indices = list(range(0, n_frames, every_n))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size,
                           fill=False, edgecolor="k", lw=2))
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6)
    e_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6)
    p_pt, = ax.plot([], [], "bo", markersize=10)
    e_pt, = ax.plot([], [], "ro", markersize=8)
    catch_circle = Circle((0, 0), catch_radius, fill=False, edgecolor="green", linestyle="--", alpha=0.5)
    ax.add_patch(catch_circle)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.set_title("HRL Pursuit-Evasion")

    def init():
        p_line.set_data([], [])
        e_line.set_data([], [])
        p_pt.set_data([], [])
        e_pt.set_data([], [])
        txt.set_text("")
        return p_line, e_line, p_pt, e_pt, txt

    def animate(i):
        idx = indices[i]
        p_line.set_data(p_pos[: idx + 1, 0], p_pos[: idx + 1, 1])
        e_line.set_data(e_pos[: idx + 1, 0], e_pos[: idx + 1, 1])
        p_pt.set_data([p_pos[idx, 0]], [p_pos[idx, 1]])
        e_pt.set_data([e_pos[idx, 0]], [e_pos[idx, 1]])
        d = np.linalg.norm(p_pos[idx] - e_pos[idx])
        catch_circle.center = (e_pos[idx, 0], e_pos[idx, 1])
        txt.set_text(f"Step: {idx}\nDist: {d:.2f}")
        return p_line, e_line, p_pt, e_pt, txt

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices),
                                   interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()


def _parse_init_distance(s: str) -> Tuple[float, float]:
    if not s:
        return (2.0, 6.0)
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        return (2.0, 6.0)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (2.0, 6.0)


def main():
    parser = argparse.ArgumentParser(description="Visualize HRL 2D pursuit-evasion")
    parser.add_argument("--high-ckpt", type=str, required=True)
    parser.add_argument("--low-ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--init-distance-range", type=str, default="2,6")
    parser.add_argument("--world-size", type=float, default=10.0)
    parser.add_argument("--catch-radius", type=float, default=0.5)
    parser.add_argument("--max-episode-seconds", type=float, default=90.0)
    parser.add_argument("--curriculum-stage", type=int, default=0)
    parser.add_argument("--low-level-steps", type=int, default=5)
    parser.add_argument("--waypoint-distance", type=float, default=2.0)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--find-success", action="store_true", help="Search for successful episodes")
    parser.add_argument("--low-level-hidden", type=int, nargs=2, default=[128, 64])
    parser.add_argument("--high-level-hidden", type=int, nargs=3, default=[256, 128, 64])
    args = parser.parse_args()

    init_distance_range = _parse_init_distance(args.init_distance_range)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LowLevelAUV2DPursuitEvasionEnv(
        init_distance_range=init_distance_range,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        max_episode_seconds=args.max_episode_seconds,
        curriculum_stage=args.curriculum_stage,
    )

    high_actor = load_actor(
        args.high_ckpt,
        obs_dim=env.observation_space.shape[0],
        n_actions=9,
        hidden_sizes=tuple(args.high_level_hidden),
        device=device,
        key="high_actor",
    )
    low_actor = load_actor(
        args.low_ckpt,
        obs_dim=8,
        n_actions=6,
        hidden_sizes=tuple(args.low_level_hidden),
        device=device,
        key="actor",
    )

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", "pe2d_hrl_vis")
    os.makedirs(out_dir, exist_ok=True)

    dt = env.inner.dt
    pursuer_v_max = env.pursuer_v_max
    evader_v_max = env.evader_v_max
    pursuer_a_max = env.pursuer_a_max
    evader_a_max = env.evader_a_max

    max_attempts = args.episodes if not args.find_success else max(args.episodes * 5, args.episodes)
    saved = 0
    attempts = 0
    while saved < args.episodes and attempts < max_attempts:
        attempts += 1
        traj = run_episode(
            high_actor,
            low_actor,
            env,
            device,
            low_level_steps=args.low_level_steps,
            waypoint_distance=args.waypoint_distance,
            deterministic=True,
        )
        if args.find_success and not traj.get("success", False):
            continue
        ep_id = saved + 1
        plot_path = os.path.join(out_dir, f"hrl_ep{ep_id}_traj_vel_acc.png")
        plot_episode(
            traj,
            plot_path,
            world_size=args.world_size,
            catch_radius=args.catch_radius,
            dt=dt,
            pursuer_v_max=pursuer_v_max,
            evader_v_max=evader_v_max,
            pursuer_a_max=pursuer_a_max,
            evader_a_max=evader_a_max,
        )
        print(f"Saved plot: {plot_path}")

        if not args.no_gif:
            gif_path = os.path.join(out_dir, f"hrl_ep{ep_id}.gif")
            save_gif(
                traj,
                gif_path,
                world_size=args.world_size,
                catch_radius=args.catch_radius,
                fps=args.gif_fps,
            )
            print(f"Saved gif: {gif_path}")
        saved += 1

    if saved == 0:
        print("No successful episodes found.")


if __name__ == "__main__":
    main()
