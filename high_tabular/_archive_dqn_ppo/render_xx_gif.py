#!/usr/bin/env python3
"""Render GIFs for the 2D pursuit-evasion setup in xx.py."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from xx import PursuitEvasionObstacle, policy_lead, policy_pure_pursuit


def make_cfg():
    return dict(
        v_max_p=1.0,
        v_max_e=1.5,
        a_max_p=2.0,
        a_max_e=0.5,
        dt=0.1,
        K=10,
        N_max=100,
        L=15.0,
        obstacles=[
            {"cx": 0, "cy": 0, "w": 10, "h": 2.5},
            {"cx": -6, "cy": 6, "w": 3, "h": 3},
            {"cx": 7, "cy": -5, "w": 3, "h": 4},
        ],
        D_c=1.0,
        D_escape=28.0,
        l_high=5.0,
        n_actions=8,
        n_rays=8,
        ray_max=15.0,
        evader_policy="smart_avoid",
    )


def rollout_frames(env, policy_fn, seed=42):
    obs = env.reset(seed=seed)
    frames = []
    frames.append((env.p_p.copy(), env.p_e.copy(), env.info.copy() if hasattr(env, "info") else {}))

    while not env.done:
        a = policy_fn(obs, env)
        obs, _, _, _, info = env.step(a)
        frames.append((env.p_p.copy(), env.p_e.copy(), info.copy()))
    return frames


def render_gif(frames, env, out_path, title, fps=8):
    fig, ax = plt.subplots(figsize=(7, 7))

    for obs_rect in env.obstacles:
        rect = patches.Rectangle(
            (obs_rect.x_min, obs_rect.y_min),
            obs_rect.w,
            obs_rect.h,
            linewidth=2,
            edgecolor="black",
            facecolor="gray",
            alpha=0.6,
        )
        ax.add_patch(rect)

    border = patches.Rectangle(
        (-env.L, -env.L),
        2 * env.L,
        2 * env.L,
        fill=False,
        edgecolor="navy",
        lw=2,
        ls="--",
    )
    ax.add_patch(border)

    pursuer_line, = ax.plot([], [], "b-", lw=1.8, label="Pursuer")
    evader_line, = ax.plot([], [], "r-", lw=1.8, label="Evader")
    pursuer_dot, = ax.plot([], [], "bo", ms=5)
    evader_dot, = ax.plot([], [], "ro", ms=5)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.set_xlim(-env.L - 1, env.L + 1)
    ax.set_ylim(-env.L - 1, env.L + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title(title)

    traj_p = []
    traj_e = []

    def update(i):
        p_p, p_e, info = frames[i]
        traj_p.append(p_p)
        traj_e.append(p_e)

        tp = np.array(traj_p)
        te = np.array(traj_e)

        pursuer_line.set_data(tp[:, 0], tp[:, 1])
        evader_line.set_data(te[:, 0], te[:, 1])
        pursuer_dot.set_data([p_p[0]], [p_p[1]])
        evader_dot.set_data([p_e[0]], [p_e[1]])

        cap = info.get("captured", False)
        esc = info.get("escaped", False)
        if cap:
            status = "CAPTURED"
        elif esc:
            status = "ESCAPED"
        else:
            status = f"RUNNING (step={info.get('step_n', i)})"
        status_text.set_text(status)

        return pursuer_line, evader_line, pursuer_dot, evader_dot, status_text

    anim = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / fps), blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


def main():
    cfg = make_cfg()

    env1 = PursuitEvasionObstacle(cfg)
    frames1 = rollout_frames(env1, policy_lead, seed=42)
    render_gif(
        frames1,
        env1,
        out_path="high_tabular/xx_lead_pursuit.gif",
        title="xx.py | Lead Pursuit",
        fps=8,
    )

    env2 = PursuitEvasionObstacle(cfg)
    frames2 = rollout_frames(env2, policy_pure_pursuit, seed=42)
    render_gif(
        frames2,
        env2,
        out_path="high_tabular/xx_pure_pursuit.gif",
        title="xx.py | Pure Pursuit",
        fps=8,
    )

    print("Saved: high_tabular/xx_lead_pursuit.gif")
    print("Saved: high_tabular/xx_pure_pursuit.gif")


if __name__ == "__main__":
    main()
