#!/usr/bin/env python3
"""Generate expert-policy pursuit visuals: dynamic GIF + static PNG."""

import os
import sys
import math
import argparse

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed


def draw_world(ax, env):
    half = env.half
    margin = env.boundary_margin

    ax.plot([-half, half, half, -half, -half],
            [-half, -half, half, half, -half],
            "k-", linewidth=1.8, alpha=0.85)

    ax.fill_between([-half, half], [half - margin, half - margin], [half, half], color="red", alpha=0.08)
    ax.fill_between([-half, half], [-half, -half], [-half + margin, -half + margin], color="red", alpha=0.08)
    ax.fill_betweenx([-half + margin, half - margin], [-half, -half], [-half + margin, -half + margin], color="red", alpha=0.08)
    ax.fill_betweenx([-half + margin, half - margin], [half - margin, half - margin], [half, half], color="red", alpha=0.08)

    for obs in env.obstacles:
        rect = Rectangle((obs.cx - obs.w / 2, obs.cy - obs.h / 2),
                         obs.w, obs.h, facecolor="gray", edgecolor="black", alpha=0.55)
        ax.add_patch(rect)

    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def expert_predict_action(env):
    """Simple predictive pursuit expert."""
    eta = env.auv_env.inner.dynamics.eta
    p = np.array([eta[0], eta[1]])
    e = env.e
    ve = env.ve

    e_pred = e + ve * 2.0
    e_pred = np.clip(e_pred, -env.half + 1, env.half - 1)

    direction = e_pred - p
    dist = np.linalg.norm(direction)
    if dist > 1e-6:
        action = direction / max(dist, env.subgoal_range)
    else:
        action = np.zeros(2, dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def rollout_expert(env, seed=42):
    env.rng = np.random.default_rng(seed)
    obs = env.reset()

    eta = env.auv_env.inner.dynamics.eta
    traj_p = [np.array([eta[0], eta[1]], dtype=np.float32)]
    traj_e = [env.e.copy().astype(np.float32)]
    vp_list = [env._get_pursuer_vel().astype(np.float32)]
    ve_list = [env.ve.copy().astype(np.float32)]
    distances = [float(np.linalg.norm(traj_p[0] - traj_e[0]))]

    subgoals = []
    sg_origins = []

    done = False
    info = {}
    while not done:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]], dtype=np.float32)

        action = expert_predict_action(env)

        dx = float(action[0]) * env.subgoal_range
        dy = float(action[1]) * env.subgoal_range
        sg = np.array([p[0] + dx, p[1] + dy], dtype=np.float32)
        sg = np.clip(sg, -env.half + 0.5, env.half - 0.5)

        sg_origins.append(p.copy())
        subgoals.append(sg.copy())

        obs, reward, done, info = env.step(action)

        eta = env.auv_env.inner.dynamics.eta
        traj_p.append(np.array([eta[0], eta[1]], dtype=np.float32))
        traj_e.append(env.e.copy().astype(np.float32))
        vp_list.append(env._get_pursuer_vel().astype(np.float32))
        ve_list.append(env.ve.copy().astype(np.float32))
        distances.append(float(np.linalg.norm(traj_p[-1] - traj_e[-1])))

    return {
        "p": np.array(traj_p),
        "e": np.array(traj_e),
        "vp": np.array(vp_list),
        "ve": np.array(ve_list),
        "distance": np.array(distances),
        "subgoals": np.array(subgoals) if subgoals else np.zeros((0, 2), dtype=np.float32),
        "sg_origins": np.array(sg_origins) if sg_origins else np.zeros((0, 2), dtype=np.float32),
        "captured": bool(info.get("captured", False)),
        "collision": bool(info.get("collision", False)),
        "steps": int(info.get("steps", 0)),
        "seed": seed,
    }


def find_good_episode(env, base_seed=0, max_search=100):
    best = None
    for i in range(max_search):
        trace = rollout_expert(env, seed=base_seed + i)
        if trace["captured"]:
            if best is None or trace["steps"] >= best["steps"]:
                best = trace
                if trace["steps"] >= 20:
                    break
    if best is None:
        best = rollout_expert(env, seed=base_seed)
    return best


def save_static_png(trace, env, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_world(ax, env)

    p = trace["p"]
    e = trace["e"]
    sg = trace["subgoals"]
    origins = trace["sg_origins"]

    ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2.0, alpha=0.85, label="Pursuer")
    ax.plot(e[:, 0], e[:, 1], "r-", linewidth=2.0, alpha=0.85, label="Evader")

    if len(sg) > 0:
        ax.scatter(sg[:, 0], sg[:, 1], c=np.arange(len(sg)), cmap="viridis", s=26,
                   alpha=0.75, zorder=5, edgecolor="black", linewidth=0.25, label="Subgoals")
        stride = max(1, len(sg) // 20)
        for i in range(0, len(sg), stride):
            ax.annotate("", xy=sg[i], xytext=origins[i],
                        arrowprops=dict(arrowstyle="->", color="green", alpha=0.45, lw=1.0))

    ax.scatter(p[0, 0], p[0, 1], c="blue", s=120, marker="o", edgecolor="white", zorder=7, label="P Start")
    ax.scatter(e[0, 0], e[0, 1], c="red", s=120, marker="o", edgecolor="white", zorder=7, label="E Start")
    ax.scatter(p[-1, 0], p[-1, 1], c="blue", s=180, marker="*", edgecolor="white", zorder=7)
    ax.scatter(e[-1, 0], e[-1, 1], c="red", s=180, marker="*", edgecolor="white", zorder=7)

    catch_circle = plt.Circle((p[-1, 0], p[-1, 1]), env.catch_radius,
                              fill=False, color="green", linewidth=2.0, linestyle="--")
    ax.add_patch(catch_circle)

    status = "CAPTURED" if trace["captured"] else "MISSED"
    ax.set_title(
        f"Expert Episode (seed={trace['seed']}): {status}, steps={trace['steps']}\n"
        f"subgoal_range={env.subgoal_range}, v_e={env.v_max_e}, a_e={env.a_max_e}",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved PNG: {save_path}")


def save_dynamics_gif(trace, env, save_path, fps=8, stride=1):
    p = trace["p"]
    e = trace["e"]
    vp = trace["vp"]
    ve = trace["ve"]
    dist = trace["distance"]
    sg = trace["subgoals"]
    origins = trace["sg_origins"]

    n = p.shape[0]
    t_axis = np.arange(n)
    speed_p = np.linalg.norm(vp, axis=1)
    speed_e = np.linalg.norm(ve, axis=1)

    frames = list(range(0, n, max(1, stride)))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.3)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_speed = fig.add_subplot(gs[1, 1])

    draw_world(ax_world, env)
    ax_world.set_title("Expert Pursuit Replay")

    p_line, = ax_world.plot([], [], "b-", linewidth=2, alpha=0.85, label="Pursuer")
    e_line, = ax_world.plot([], [], "r-", linewidth=2, alpha=0.85, label="Evader")
    p_dot = ax_world.scatter([], [], c="blue", s=90, zorder=6)
    e_dot = ax_world.scatter([], [], c="red", s=90, zorder=6)
    sg_hist = ax_world.scatter([], [], c=[], cmap="viridis", s=22, alpha=0.7, zorder=5)
    sg_cur = ax_world.scatter([], [], c="gold", s=120, marker="*", edgecolor="black", zorder=7, label="Current subgoal")
    sg_arrow = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                               color="green", alpha=0.65)
    txt = ax_world.text(0.02, 0.98, "", transform=ax_world.transAxes, va="top", fontsize=10)
    ax_world.legend(loc="upper right")

    line_dist, = ax_dist.plot([], [], color="black", linewidth=2.0, label="distance(p,e)")
    cursor_d = ax_dist.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_dist.axhline(env.catch_radius, linestyle=":", color="green", linewidth=1.2,
                    label=f"catch_r={env.catch_radius}")
    ax_dist.set_xlim(0, n - 1)
    ax_dist.set_ylim(0, max(float(np.max(dist)), 2.0) * 1.1)
    ax_dist.set_title("Distance to Evader")
    ax_dist.set_xlabel("High-level Step")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="upper right")

    line_vp, = ax_speed.plot([], [], color="royalblue", linewidth=2.0, label="|v_pursuer|")
    line_ve, = ax_speed.plot([], [], color="crimson", linewidth=2.0, label="|v_evader|")
    cursor_s = ax_speed.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_speed.set_xlim(0, n - 1)
    vmax = max(float(np.max(speed_p)), float(np.max(speed_e)), 0.1)
    ax_speed.set_ylim(0, vmax * 1.25)
    ax_speed.set_title("Speed Profile")
    ax_speed.set_xlabel("High-level Step")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(loc="upper right")

    def update(i):
        t = frames[i]

        p_hist = p[:t + 1]
        e_hist = e[:t + 1]
        p_line.set_data(p_hist[:, 0], p_hist[:, 1])
        e_line.set_data(e_hist[:, 0], e_hist[:, 1])
        p_dot.set_offsets([p[t, 0], p[t, 1]])
        e_dot.set_offsets([e[t, 0], e[t, 1]])

        if t > 0 and len(sg) > 0:
            sg_hist_xy = sg[:t]
            sg_hist.set_offsets(sg_hist_xy)
            sg_hist.set_array(np.arange(len(sg_hist_xy)))
        else:
            sg_hist.set_offsets(np.zeros((0, 2)))
            sg_hist.set_array(np.array([]))

        if t > 0 and len(sg) >= t:
            cur_sg = sg[t - 1]
            cur_o = origins[t - 1]
            sg_cur.set_offsets([cur_sg[0], cur_sg[1]])
            dxy = cur_sg - cur_o
            sg_arrow.set_offsets([[cur_o[0], cur_o[1]]])
            sg_arrow.set_UVC(dxy[0], dxy[1])
        else:
            sg_cur.set_offsets(np.zeros((0, 2)))
            sg_arrow.set_offsets([[0, 0]])
            sg_arrow.set_UVC(0, 0)

        line_dist.set_data(t_axis[:t + 1], dist[:t + 1])
        line_vp.set_data(t_axis[:t + 1], speed_p[:t + 1])
        line_ve.set_data(t_axis[:t + 1], speed_e[:t + 1])
        cursor_d.set_xdata([t, t])
        cursor_s.set_xdata([t, t])

        status = "CAPTURE" if trace["captured"] and t == n - 1 else "RUN"
        txt.set_text(
            f"step={t:03d}  d={dist[t]:.2f}  |v_p|={speed_p[t]:.2f}  |v_e|={speed_e[t]:.2f}  {status}"
        )

        return [p_line, e_line, p_dot, e_dot, sg_hist, sg_cur, sg_arrow, txt,
                line_dist, line_vp, line_ve, cursor_d, cursor_s]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)
    ani.save(save_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print(f"Saved GIF: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="expert_subgoal_visuals")
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--max_search", type=int, default=120)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--subgoal_range", type=float, default=2.0)
    parser.add_argument("--v_max_e", type=float, default=0.65)
    parser.add_argument("--a_max_e", type=float, default=0.10)
    parser.add_argument("--evader_mode", type=str, default="medium", choices=["simple", "medium", "hard"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_path = os.path.join(PROJECT_ROOT, "low_near/v4_tight/checkpoints/best.pth")
    nn_policy = load_v4_nn(nn_path, device)

    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=30.0,
        catch_radius=1.0,
        max_steps=6000,
        subgoal_range=args.subgoal_range,
        low_steps=50,
        seed=42,
        evader_mode=args.evader_mode,
    )

    env.v_max_e = args.v_max_e
    env.a_max_e = args.a_max_e

    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.run_name, "visuals")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Searching expert episode... (evader_mode={args.evader_mode})")
    trace = find_good_episode(env, base_seed=args.base_seed, max_search=args.max_search)
    status = "CAPTURED" if trace["captured"] else "MISSED"
    print(f"Selected episode: seed={trace['seed']}, {status}, steps={trace['steps']}")

    png_path = os.path.join(save_dir, "pursuit_episode_expert.png")
    gif_path = os.path.join(save_dir, "pursuit_with_dynamics_expert.gif")

    save_static_png(trace, env, png_path)
    save_dynamics_gif(trace, env, gif_path, fps=args.fps, stride=args.stride)

    print("Done.")
    print(f"Output dir: {save_dir}")


if __name__ == "__main__":
    main()
