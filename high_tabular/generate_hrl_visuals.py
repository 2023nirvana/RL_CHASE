#!/usr/bin/env python3
"""Generate GIF and PNG visualizations for HRL pursuit-evasion results."""

import os, sys, math
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from high_tabular.hrl_v3_real_auv_fixed import (
    PursuitEnvRealAUVFixed, HighLevelPolicy2D, load_v4_nn
)


def draw_world(ax, env):
    half = env.half
    margin = env.boundary_margin
    # boundary lines
    ax.plot([-half, half, half, -half, -half],
            [-half, -half, half, half, -half], "k--", linewidth=1.0, alpha=0.6)
    # red boundary zones
    ax.fill_between([-half, half], [half-margin, half-margin], [half, half], color="red", alpha=0.08)
    ax.fill_between([-half, half], [-half, -half], [-half+margin, -half+margin], color="red", alpha=0.08)
    ax.fill_betweenx([-half+margin, half-margin], [-half, -half], [-half+margin, -half+margin], color="red", alpha=0.08)
    ax.fill_betweenx([-half+margin, half-margin], [half-margin, half-margin], [half, half], color="red", alpha=0.08)
    # obstacles
    for obs in env.obstacles:
        rect = Rectangle((obs.cx - obs.w/2, obs.cy - obs.h/2),
                          obs.w, obs.h, color="gray", alpha=0.5)
        ax.add_patch(rect)
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def rollout(env, actor, device, seed=42):
    """Rollout one episode, collect full trajectory data."""
    env.rng = np.random.default_rng(seed)
    obs = env.reset()

    eta = env.auv_env.inner.dynamics.eta
    traj_p = [np.array([eta[0], eta[1]])]
    traj_e = [env.e.copy()]
    vp_list = [env._get_pursuer_vel()]
    ve_list = [env.ve.copy()]
    subgoals = []
    distances = [np.linalg.norm(traj_p[0] - traj_e[0])]

    done = False
    info = {}
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = actor(obs_t)
            action = mu.cpu().numpy().flatten()
            action = np.clip(action, -1, 1)

            # record subgoal
            eta = env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            sg = np.array([p[0] + action[0]*env.subgoal_range,
                           p[1] + action[1]*env.subgoal_range])
            subgoals.append(sg.copy())

            obs, reward, done, info = env.step(action)

            eta = env.auv_env.inner.dynamics.eta
            traj_p.append(np.array([eta[0], eta[1]]))
            traj_e.append(env.e.copy())
            vp_list.append(env._get_pursuer_vel())
            ve_list.append(env.ve.copy())
            distances.append(np.linalg.norm(traj_p[-1] - traj_e[-1]))

    return {
        "p": np.array(traj_p),
        "e": np.array(traj_e),
        "vp": np.array(vp_list),
        "ve": np.array(ve_list),
        "subgoals": np.array(subgoals) if subgoals else np.zeros((0, 2)),
        "distance": np.array(distances),
        "captured": info.get("captured", False),
        "collision": info.get("collision", False),
        "steps": info.get("steps", 0),
        "seed": seed,
    }


def find_capture_episode(env, actor, device, base_seed=0, max_search=100):
    """Search for a successful capture episode."""
    best = None
    for i in range(max_search):
        trace = rollout(env, actor, device, seed=base_seed + i)
        if trace["captured"]:
            if best is None or trace["steps"] > 15:  # prefer non-trivial episodes
                best = trace
                if trace["steps"] >= 20:
                    break
    return best


def save_static_png(trace, env, save_path):
    """Save a static trajectory PNG (like the tabular version)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_world(ax, env)

    p, e = trace["p"], trace["e"]
    sg = trace["subgoals"]

    ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2, alpha=0.8, label="Pursuer")
    ax.plot(e[:, 0], e[:, 1], "r-", linewidth=2, alpha=0.8, label="Evader")

    if len(sg) > 0:
        ax.scatter(sg[:, 0], sg[:, 1], c="green", s=25, alpha=0.4, zorder=4, label="Subgoals")

    # start/end markers
    ax.scatter(p[0, 0], p[0, 1], c="blue", s=120, marker="o", zorder=5, edgecolor="white", label="P Start")
    ax.scatter(e[0, 0], e[0, 1], c="red", s=120, marker="o", zorder=5, edgecolor="white", label="E Start")
    ax.scatter(p[-1, 0], p[-1, 1], c="blue", s=180, marker="*", zorder=5, edgecolor="white")
    ax.scatter(e[-1, 0], e[-1, 1], c="red", s=180, marker="*", zorder=5, edgecolor="white")

    # catch radius
    circle = plt.Circle((p[-1, 0], p[-1, 1]), env.catch_radius,
                         fill=False, color="green", linewidth=2, linestyle="--")
    ax.add_patch(circle)

    # direction arrows
    n = len(p)
    for i in range(0, n-1, max(1, n//10)):
        ax.annotate("", xy=p[i+1], xytext=p[i],
                     arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5))
    for i in range(0, len(e)-1, max(1, len(e)//10)):
        ax.annotate("", xy=e[min(i+1, len(e)-1)], xytext=e[i],
                     arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))

    status = "CAPTURED" if trace["captured"] else ("COLLISION" if trace["collision"] else "MISSED")
    ax.set_title(f"HRL Episode (seed={trace['seed']}): {status}, steps={trace['steps']}", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PNG: {save_path}")


def save_dynamics_gif(trace, env, save_path, fps=10, stride=1):
    """Save a dual-panel GIF with world view + distance/speed plots."""
    p, e = trace["p"], trace["e"]
    vp, ve = trace["vp"], trace["ve"]
    dist = trace["distance"]
    n = p.shape[0]

    speed_p = np.linalg.norm(vp, axis=1)
    speed_e = np.linalg.norm(ve, axis=1)
    t_axis = np.arange(n)

    frames = list(range(0, n, max(1, stride)))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.28)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    draw_world(ax_world, env)
    ax_world.set_title("HRL Pursuit-Evasion Replay")
    p_line, = ax_world.plot([], [], "b-", linewidth=2, alpha=0.85, label="Pursuer")
    e_line, = ax_world.plot([], [], "r-", linewidth=2, alpha=0.85, label="Evader")
    p_dot = ax_world.scatter([], [], c="blue", s=90, zorder=5)
    e_dot = ax_world.scatter([], [], c="red", s=90, zorder=5)
    p_quiver = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="blue", alpha=0.75)
    e_quiver = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="red", alpha=0.75)
    world_text = ax_world.text(0.02, 0.98, "", transform=ax_world.transAxes, va="top", fontsize=10)
    ax_world.legend(loc="upper right")

    # distance plot
    line_dist, = ax_top.plot([], [], color="black", linewidth=2.0, label="distance(p,e)")
    cursor_top = ax_top.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_top.axhline(env.catch_radius, linestyle=":", color="green", linewidth=1, label=f"catch_r={env.catch_radius}")
    ax_top.set_xlim(0, n - 1)
    ax_top.set_ylim(0, max(float(np.max(dist)), 2.0) * 1.1)
    ax_top.set_title("Distance to Evader")
    ax_top.set_xlabel("High-level Step")
    ax_top.set_ylabel("Distance")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right")

    # speed plot
    line_vp, = ax_bot.plot([], [], color="royalblue", linewidth=2.0, label="|v_pursuer|")
    line_ve, = ax_bot.plot([], [], color="crimson", linewidth=2.0, label="|v_evader|")
    cursor_bot = ax_bot.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_bot.set_xlim(0, n - 1)
    y_bot_max = max(float(np.max(speed_p)), float(np.max(speed_e)), 0.1)
    ax_bot.set_ylim(0, y_bot_max * 1.2)
    ax_bot.set_title("Speed Profile")
    ax_bot.set_xlabel("High-level Step")
    ax_bot.set_ylabel("Speed (m/s)")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc="upper right")

    def update(i):
        t = frames[i]
        p_hist = p[:t+1]
        e_hist = e[:t+1]

        p_line.set_data(p_hist[:, 0], p_hist[:, 1])
        e_line.set_data(e_hist[:, 0], e_hist[:, 1])
        p_dot.set_offsets([p[t, 0], p[t, 1]])
        e_dot.set_offsets([e[t, 0], e[t, 1]])
        p_quiver.set_offsets([[p[t, 0], p[t, 1]]])
        e_quiver.set_offsets([[e[t, 0], e[t, 1]]])

        # scale velocity arrows for visibility
        scale = 5.0
        p_quiver.set_UVC(vp[t, 0]*scale, vp[t, 1]*scale)
        e_quiver.set_UVC(ve[t, 0]*scale, ve[t, 1]*scale)

        line_dist.set_data(t_axis[:t+1], dist[:t+1])
        line_vp.set_data(t_axis[:t+1], speed_p[:t+1])
        line_ve.set_data(t_axis[:t+1], speed_e[:t+1])
        cursor_top.set_xdata([t, t])
        cursor_bot.set_xdata([t, t])

        status = "CAPTURE" if trace["captured"] and t == n - 1 else "RUN"
        world_text.set_text(f"step={t:03d}  d={dist[t]:.2f}  status={status}")

        return [p_line, e_line, p_dot, e_dot, p_quiver, e_quiver,
                world_text, line_dist, line_vp, line_ve, cursor_top, cursor_bot]

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=int(1000 / max(1, fps)), blit=False)
    ani.save(save_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print(f"Saved GIF: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_path = os.path.join(PROJECT_ROOT, "low_near/v4_tight/checkpoints/best.pth")
    nn_policy = load_v4_nn(nn_path, device)

    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy, device=device,
        world_size=30.0, catch_radius=1.0, max_steps=6000,
        subgoal_range=0.5, low_steps=50, seed=42,
    )

    # Load trained actor
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/3d_v3_mapfix_smoke/best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = HighLevelPolicy2D(13, 2, hidden=(128, 128)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"Loaded actor from {ckpt_path}, capture_rate={ckpt.get('capture_rate', '?')}%")

    save_dir = os.path.join(PROJECT_ROOT, "checkpoints/3d_v3_mapfix_smoke/visuals")
    os.makedirs(save_dir, exist_ok=True)

    # Find a good capture episode
    print("Searching for capture episode...")
    trace = find_capture_episode(env, actor, device, base_seed=0, max_search=200)
    if trace is None:
        print("No capture found, using seed=42 anyway")
        trace = rollout(env, actor, device, seed=42)

    status = "CAPTURED" if trace["captured"] else "MISSED"
    print(f"Selected episode: seed={trace['seed']}, {status}, steps={trace['steps']}")

    # Save PNG
    save_static_png(trace, env, os.path.join(save_dir, "pursuit_episode.png"))

    # Save GIF (with dynamics panel)
    save_dynamics_gif(trace, env, os.path.join(save_dir, "pursuit_with_dynamics.gif"),
                      fps=8, stride=1)

    print(f"\nAll visuals saved to: {save_dir}")


if __name__ == "__main__":
    main()
