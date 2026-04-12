#!/usr/bin/env python3
"""Generate GIF and dynamics visualizations for tabular pursuit results.

This script loads a saved tabular Q-table and reproduces richer artifacts:
- pursuit_episode.gif
- pursuit_with_dynamics.gif
- dynamics_analysis.png

It is designed to work with outputs produced by tabular_pursuit_qlearning.py.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tabular_pursuit_qlearning import PursuitEvasionTabularEnv


@dataclass
class EpisodeTrace:
    p: np.ndarray
    e: np.ndarray
    vp: np.ndarray
    ve: np.ndarray
    d_obs_e: np.ndarray
    d_boundary_e: np.ndarray
    distance: np.ndarray
    captured: bool
    near_terrain: bool
    near_boundary: bool
    steps: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rich visuals for tabular pursuit results")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory containing q_table.pkl")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for episode replay")
    parser.add_argument("--search-episodes", type=int, default=40, help="Episodes to search for a good demo")
    parser.add_argument("--capture-search-episodes", type=int, default=320, help="Episodes to search for capture case gallery")
    parser.add_argument("--captures-per-band", type=int, default=2, help="How many short/mid/long capture gifs to export")
    parser.add_argument("--analysis-episodes", type=int, default=60, help="Episodes used for dynamics analysis")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    parser.add_argument("--stride", type=int, default=2, help="Frame downsample stride for GIF")
    return parser.parse_args()


def load_q_table(result_dir: str) -> Dict[tuple, np.ndarray]:
    qtable_path = os.path.join(result_dir, "q_table.pkl")
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(f"Missing file: {qtable_path}")

    with open(qtable_path, "rb") as f:
        obj = pickle.load(f)

    # Supports both raw q-table dict and wrapped payload format.
    if isinstance(obj, dict) and "q_table" in obj:
        return obj["q_table"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported q_table.pkl format")


def greedy_action(q_table: Dict[tuple, np.ndarray], state: tuple, n_actions: int) -> int:
    q = q_table.get(state)
    if q is None:
        return 0
    return int(np.argmax(q[:n_actions]))


def rollout_episode(q_table: Dict[tuple, np.ndarray], seed: int) -> EpisodeTrace:
    env = PursuitEvasionTabularEnv(seed=seed)
    state = env.reset()

    p_list: List[np.ndarray] = [env.p.copy()]
    e_list: List[np.ndarray] = [env.e.copy()]
    vp_list: List[np.ndarray] = [env.vp.copy()]
    ve_list: List[np.ndarray] = [env.ve.copy()]
    d_obs_e: List[float] = [env._nearest_obstacle_dist(env.e)]
    d_boundary_e: List[float] = [env._boundary_dist(env.e)]
    distance: List[float] = [float(np.linalg.norm(env.e - env.p))]

    done = False
    info = {}
    while not done:
        action = greedy_action(q_table, state, env.n_actions)
        state, _, done, info = env.step(action)
        p_list.append(env.p.copy())
        e_list.append(env.e.copy())
        vp_list.append(env.vp.copy())
        ve_list.append(env.ve.copy())
        d_obs_e.append(env._nearest_obstacle_dist(env.e))
        d_boundary_e.append(env._boundary_dist(env.e))
        distance.append(float(np.linalg.norm(env.e - env.p)))

    return EpisodeTrace(
        p=np.array(p_list),
        e=np.array(e_list),
        vp=np.array(vp_list),
        ve=np.array(ve_list),
        d_obs_e=np.array(d_obs_e),
        d_boundary_e=np.array(d_boundary_e),
        distance=np.array(distance),
        captured=bool(info.get("captured", False)),
        near_terrain=bool(info.get("near_terrain", False)),
        near_boundary=bool(info.get("near_boundary", False)),
        steps=len(p_list) - 1,
        seed=seed,
    )


def choose_demo_episode(q_table: Dict[tuple, np.ndarray], base_seed: int, search_episodes: int) -> EpisodeTrace:
    traces: List[EpisodeTrace] = []
    for k in range(search_episodes):
        traces.append(rollout_episode(q_table, seed=base_seed + k))

    captures = [t for t in traces if t.captured]
    if captures:
        # Prefer capture near terrain, then shorter successful episodes.
        captures.sort(key=lambda t: (not t.near_terrain, t.steps, float(np.mean(t.distance))))
        return captures[0]

    # Fallback: choose the episode that gets closest.
    traces.sort(key=lambda t: float(np.min(t.distance)))
    return traces[0]


def choose_intent_episode(traces: List[EpisodeTrace]) -> EpisodeTrace:
    """Pick a case that best reflects terrain-aware pursuit intent."""
    if not traces:
        raise ValueError("No traces provided")

    captures = [t for t in traces if t.captured]
    if not captures:
        traces.sort(key=lambda t: float(np.min(t.distance)))
        return traces[0]

    def intent_score(t: EpisodeTrace) -> float:
        # Prefer captures near terrain, away from boundary, with clear pursuit compression.
        start_d = float(t.distance[0])
        min_d = float(np.min(t.distance))
        mean_obs = float(np.mean(t.d_obs_e))
        score = 0.0
        score += 6.0 if t.near_terrain else 0.0
        score += 2.0 if not t.near_boundary else -2.5
        score += max(0.0, start_d - min_d)
        score += max(0.0, 4.0 - mean_obs) * 0.6
        score += min(float(t.steps), 140.0) * 0.01
        return score

    captures.sort(key=intent_score, reverse=True)
    return captures[0]


def select_capture_gallery(traces: List[EpisodeTrace], per_band: int) -> List[tuple[str, EpisodeTrace]]:
    captures = [t for t in traces if t.captured]
    if not captures:
        return []

    captures.sort(key=lambda t: t.steps)
    steps = np.array([t.steps for t in captures], dtype=np.float32)
    q1 = float(np.quantile(steps, 0.33))
    q2 = float(np.quantile(steps, 0.66))

    bands = {
        "short": [t for t in captures if t.steps <= q1],
        "mid": [t for t in captures if q1 < t.steps <= q2],
        "long": [t for t in captures if t.steps > q2],
    }

    def rank_key(t: EpisodeTrace):
        return (
            not t.near_terrain,
            t.near_boundary,
            float(np.mean(t.d_obs_e)),
            t.steps,
        )

    selected: List[tuple[str, EpisodeTrace]] = []
    seen_seeds = set()
    idx = 1
    for band_name in ["short", "mid", "long"]:
        pool = sorted(bands[band_name], key=rank_key)
        take = 0
        for t in pool:
            if t.seed in seen_seeds:
                continue
            selected.append((f"capture_{band_name}_{idx}_seed{t.seed}.gif", t))
            seen_seeds.add(t.seed)
            idx += 1
            take += 1
            if take >= per_band:
                break

    return selected


def draw_world(ax: plt.Axes, env: PursuitEvasionTabularEnv) -> None:
    for obs in env.obstacles:
        rect = plt.Rectangle((obs.x_min, obs.y_min), obs.w, obs.h, color="gray", alpha=0.5)
        ax.add_patch(rect)

    m = env.boundary_margin
    L = env.L
    ax.fill_between([-L, L], [L - m, L - m], [L, L], color="red", alpha=0.08)
    ax.fill_between([-L, L], [-L, -L], [-L + m, -L + m], color="red", alpha=0.08)
    ax.fill_betweenx([-L + m, L - m], [-L, -L], [-L + m, -L + m], color="red", alpha=0.08)
    ax.fill_betweenx([-L + m, L - m], [L - m, L - m], [L, L], color="red", alpha=0.08)

    ax.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L], "k--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(-L - 1, L + 1)
    ax.set_ylim(-L - 1, L + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def save_episode_gif(trace: EpisodeTrace, out_path: str, fps: int, stride: int, with_dynamics: bool) -> None:
    env = PursuitEvasionTabularEnv(seed=trace.seed)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_world(ax, env)
    ax.set_title("Greedy Policy Replay")

    p_line, = ax.plot([], [], "b-", linewidth=2, alpha=0.8, label="Pursuer")
    e_line, = ax.plot([], [], "r-", linewidth=2, alpha=0.8, label="Evader")
    p_dot = ax.scatter([], [], c="blue", s=80, zorder=5)
    e_dot = ax.scatter([], [], c="red", s=80, zorder=5)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="upper right")

    p_quiver = None
    e_quiver = None
    if with_dynamics:
        p_quiver = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="blue", alpha=0.8)
        e_quiver = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="red", alpha=0.8)

    n = trace.p.shape[0]
    frames = list(range(0, n, max(1, stride)))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    def update(i: int):
        t = frames[i]
        p_hist = trace.p[: t + 1]
        e_hist = trace.e[: t + 1]

        p_line.set_data(p_hist[:, 0], p_hist[:, 1])
        e_line.set_data(e_hist[:, 0], e_hist[:, 1])
        p_dot.set_offsets([trace.p[t, 0], trace.p[t, 1]])
        e_dot.set_offsets([trace.e[t, 0], trace.e[t, 1]])

        status = "CAPTURE" if trace.captured and t == n - 1 else "RUN"
        text.set_text(
            f"step={t:03d}  d={trace.distance[t]:.2f}\n"
            f"d_obs_e={trace.d_obs_e[t]:.2f}  d_bnd_e={trace.d_boundary_e[t]:.2f}\n"
            f"status={status}"
        )

        artists = [p_line, e_line, p_dot, e_dot, text]
        if with_dynamics and p_quiver is not None and e_quiver is not None:
            p_quiver.set_offsets([[trace.p[t, 0], trace.p[t, 1]]])
            e_quiver.set_offsets([[trace.e[t, 0], trace.e[t, 1]]])
            p_quiver.set_UVC(trace.vp[t, 0], trace.vp[t, 1])
            e_quiver.set_UVC(trace.ve[t, 0], trace.ve[t, 1])
            artists.extend([p_quiver, e_quiver])
        return artists

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def save_dual_panel_dynamics_gif(trace: EpisodeTrace, out_path: str, fps: int, stride: int) -> None:
    env = PursuitEvasionTabularEnv(seed=trace.seed)
    n = trace.p.shape[0]
    frames = list(range(0, n, max(1, stride)))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    speed_p = np.linalg.norm(trace.vp, axis=1)
    speed_e = np.linalg.norm(trace.ve, axis=1)
    t_axis = np.arange(n)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.28)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    draw_world(ax_world, env)
    ax_world.set_title("Pursuit-Evasion Replay")
    p_line, = ax_world.plot([], [], "b-", linewidth=2, alpha=0.85, label="Pursuer")
    e_line, = ax_world.plot([], [], "r-", linewidth=2, alpha=0.85, label="Evader")
    p_dot = ax_world.scatter([], [], c="blue", s=90, zorder=5)
    e_dot = ax_world.scatter([], [], c="red", s=90, zorder=5)
    p_quiver = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="blue", alpha=0.75)
    e_quiver = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0, color="red", alpha=0.75)
    world_text = ax_world.text(0.02, 0.98, "", transform=ax_world.transAxes, va="top", fontsize=10)
    ax_world.legend(loc="upper right")

    line_dist, = ax_top.plot([], [], color="black", linewidth=2.0, label="distance(p,e)")
    line_obs, = ax_top.plot([], [], color="forestgreen", linewidth=1.8, label="d_obs_e")
    line_bnd, = ax_top.plot([], [], color="darkorange", linewidth=1.8, label="d_bnd_e")
    cursor_top = ax_top.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_top.axhline(1.0, color="black", linestyle=":", alpha=0.45, linewidth=1.0)
    ax_top.set_xlim(0, n - 1)
    y_top_max = max(float(np.max(trace.distance)), float(np.max(trace.d_obs_e)), float(np.max(trace.d_boundary_e)), 1.5)
    ax_top.set_ylim(0, y_top_max * 1.1)
    ax_top.set_title("Geometry Metrics")
    ax_top.set_xlabel("Step")
    ax_top.set_ylabel("Distance")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right")

    line_vp, = ax_bot.plot([], [], color="royalblue", linewidth=2.0, label="|v_p|")
    line_ve, = ax_bot.plot([], [], color="crimson", linewidth=2.0, label="|v_e|")
    cursor_bot = ax_bot.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_bot.set_xlim(0, n - 1)
    y_bot_max = max(float(np.max(speed_p)), float(np.max(speed_e)), 0.6)
    ax_bot.set_ylim(0, y_bot_max * 1.2)
    ax_bot.set_title("Speed Profile")
    ax_bot.set_xlabel("Step")
    ax_bot.set_ylabel("Speed")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc="upper right")

    def update(i: int):
        t = frames[i]
        p_hist = trace.p[: t + 1]
        e_hist = trace.e[: t + 1]

        p_line.set_data(p_hist[:, 0], p_hist[:, 1])
        e_line.set_data(e_hist[:, 0], e_hist[:, 1])
        p_dot.set_offsets([trace.p[t, 0], trace.p[t, 1]])
        e_dot.set_offsets([trace.e[t, 0], trace.e[t, 1]])
        p_quiver.set_offsets([[trace.p[t, 0], trace.p[t, 1]]])
        e_quiver.set_offsets([[trace.e[t, 0], trace.e[t, 1]]])
        p_quiver.set_UVC(trace.vp[t, 0], trace.vp[t, 1])
        e_quiver.set_UVC(trace.ve[t, 0], trace.ve[t, 1])

        line_dist.set_data(t_axis[: t + 1], trace.distance[: t + 1])
        line_obs.set_data(t_axis[: t + 1], trace.d_obs_e[: t + 1])
        line_bnd.set_data(t_axis[: t + 1], trace.d_boundary_e[: t + 1])
        line_vp.set_data(t_axis[: t + 1], speed_p[: t + 1])
        line_ve.set_data(t_axis[: t + 1], speed_e[: t + 1])
        cursor_top.set_xdata([t, t])
        cursor_bot.set_xdata([t, t])

        status = "CAPTURE" if trace.captured and t == n - 1 else "RUN"
        world_text.set_text(
            f"step={t:03d} d={trace.distance[t]:.2f} status={status}\n"
            f"d_obs_e={trace.d_obs_e[t]:.2f} d_bnd_e={trace.d_boundary_e[t]:.2f}"
        )

        return [
            p_line,
            e_line,
            p_dot,
            e_dot,
            p_quiver,
            e_quiver,
            world_text,
            line_dist,
            line_obs,
            line_bnd,
            line_vp,
            line_ve,
            cursor_top,
            cursor_bot,
        ]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def save_dynamics_analysis(traces: List[EpisodeTrace], out_path: str) -> None:
    if not traces:
        return

    captures = np.array([1 if t.captured else 0 for t in traces], dtype=np.float32)
    steps = np.array([t.steps for t in traces], dtype=np.float32)
    min_dist = np.array([float(np.min(t.distance)) for t in traces], dtype=np.float32)
    mean_obs = np.array([float(np.mean(t.d_obs_e)) for t in traces], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    cap_rate = float(np.mean(captures))
    ax.bar(["Captured", "Escaped"], [cap_rate, 1.0 - cap_rate], color=["green", "crimson"], alpha=0.75)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Capture Rate")
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[0, 1]
    cap_steps = steps[captures > 0.5]
    esc_steps = steps[captures < 0.5]
    if len(cap_steps) > 0:
        ax.hist(cap_steps, bins=15, alpha=0.7, label="Captured", color="green")
    if len(esc_steps) > 0:
        ax.hist(esc_steps, bins=15, alpha=0.7, label="Escaped", color="red")
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.scatter(mean_obs, min_dist, c=captures, cmap="RdYlGn", alpha=0.8)
    ax.set_title("Mean Evader-Obstacle Dist vs Min Pursuit Dist")
    ax.set_xlabel("Mean d_obs_e")
    ax.set_ylabel("Min distance(p, e)")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    for i, t in enumerate(traces[:8]):
        alpha = 0.9 if i < 3 else 0.45
        color = "green" if t.captured else "red"
        ax.plot(t.distance, color=color, alpha=alpha, linewidth=1.5)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1, label="capture_dist=1.0")
    ax.set_title("Distance-to-Evader Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("distance")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "captures"), exist_ok=True)

    q_table = load_q_table(args.result_dir)
    print(f"Loaded Q-table states: {len(q_table)}")

    demo = choose_demo_episode(q_table, base_seed=args.seed, search_episodes=args.search_episodes)
    tag = "captured" if demo.captured else "closest"
    print(f"Selected demo episode seed={demo.seed}, steps={demo.steps}, tag={tag}")

    gallery_pool = [rollout_episode(q_table, seed=args.seed + 5000 + i) for i in range(args.capture_search_episodes)]
    gallery = select_capture_gallery(gallery_pool, per_band=max(1, args.captures_per_band))
    if gallery:
        print(f"Selected {len(gallery)} capture cases for gallery")
    else:
        print("No captured episodes found for gallery in current search range")

    intent_demo = choose_intent_episode(gallery_pool)
    print(
        f"Intent case seed={intent_demo.seed}, steps={intent_demo.steps}, "
        f"near_terrain={intent_demo.near_terrain}, near_boundary={intent_demo.near_boundary}"
    )

    gif_basic = os.path.join(args.result_dir, "pursuit_episode.gif")
    gif_dyn = os.path.join(args.result_dir, "pursuit_with_dynamics.gif")
    save_episode_gif(intent_demo, gif_basic, fps=args.fps, stride=args.stride, with_dynamics=False)
    save_dual_panel_dynamics_gif(intent_demo, gif_dyn, fps=args.fps, stride=args.stride)
    print(f"Saved: {gif_basic}")
    print(f"Saved: {gif_dyn}")

    captures_dir = os.path.join(args.result_dir, "captures")
    for name, trace in gallery:
        out_path = os.path.join(captures_dir, name)
        save_episode_gif(trace, out_path, fps=args.fps, stride=args.stride, with_dynamics=False)
        print(f"Saved: {out_path}")

    # Save a static summary trajectory image to match other result folders.
    fig, ax = plt.subplots(figsize=(7, 7))
    env = PursuitEvasionTabularEnv(seed=intent_demo.seed)
    draw_world(ax, env)
    ax.plot(intent_demo.p[:, 0], intent_demo.p[:, 1], "b-", linewidth=2, label="Pursuer")
    ax.plot(intent_demo.e[:, 0], intent_demo.e[:, 1], "r-", linewidth=2, label="Evader")
    ax.scatter(intent_demo.p[0, 0], intent_demo.p[0, 1], c="blue", s=90, marker="o")
    ax.scatter(intent_demo.e[0, 0], intent_demo.e[0, 1], c="red", s=90, marker="o")
    ax.scatter(intent_demo.p[-1, 0], intent_demo.p[-1, 1], c="blue", s=120, marker="*")
    ax.scatter(intent_demo.e[-1, 0], intent_demo.e[-1, 1], c="red", s=120, marker="*")
    ax.set_title(f"Representative Episode (seed={intent_demo.seed}, captured={intent_demo.captured})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    static_episode = os.path.join(args.result_dir, "episodes", "representative_episode.png")
    plt.savefig(static_episode, dpi=160)
    plt.close(fig)
    print(f"Saved: {static_episode}")

    traces = [rollout_episode(q_table, seed=args.seed + 1000 + i) for i in range(args.analysis_episodes)]
    analysis_path = os.path.join(args.result_dir, "dynamics_analysis.png")
    save_dynamics_analysis(traces, analysis_path)
    print(f"Saved: {analysis_path}")


if __name__ == "__main__":
    main()
