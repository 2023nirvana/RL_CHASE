#!/usr/bin/env python3
"""评估 Online PPO (BPPO pretrained) 的 best / final 模型"""

import os, sys, json, math, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from training.offline_ppo_pretrain import HighLevelActorCritic
from training.collect_expert_trajectories import compute_high_reward


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        sd = ckpt["model"]
        obs_dim = ckpt.get("obs_dim", 21)
        act_dim = ckpt.get("act_dim", 2)
        hidden = tuple(ckpt.get("hidden", [256, 128]))
    else:
        raise ValueError(f"Unknown ckpt format: {list(ckpt.keys())}")
    model = HighLevelActorCritic(obs_dim, act_dim, hidden).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def run_episode(env, model, device, record_traj=False):
    """运行一个episode，返回统计信息和可选的轨迹"""
    obs = env.reset()
    eta = env.auv_env.inner.dynamics.eta
    prev_dist = float(np.linalg.norm(np.array([eta[0], eta[1]]) - env.e))

    done = False
    ep_reward = 0.0
    steps = 0
    captured = False

    pursuer_traj = []
    evader_traj = []
    subgoals = []

    while not done:
        if record_traj:
            eta_now = env.auv_env.inner.dynamics.eta
            pursuer_traj.append([eta_now[0], eta_now[1]])
            evader_traj.append([env.e[0], env.e[1]])

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _ = model.get_action(obs_t, deterministic=True)
        a_np = a[0].cpu().numpy()

        next_obs, _, done, info = env.step(a_np)

        if record_traj and "corrected_subgoal" in info:
            sg = info["corrected_subgoal"]
            if sg is not None:
                subgoals.append([sg[0], sg[1]])

        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        curr_dist = float(np.linalg.norm(p - env.e))

        r, _ = compute_high_reward(
            env, prev_dist, curr_dist,
            info.get("captured", False),
            info.get("out_of_bounds", False),
            info.get("collision", False),
            subgoal=info.get("corrected_subgoal", None),
            done=done,
        )
        ep_reward += r
        prev_dist = curr_dist
        obs = next_obs
        steps += 1

        if info.get("captured", False):
            captured = True

    if record_traj:
        eta_now = env.auv_env.inner.dynamics.eta
        pursuer_traj.append([eta_now[0], eta_now[1]])
        evader_traj.append([env.e[0], env.e[1]])

    traj_data = None
    if record_traj:
        traj_data = {
            "pursuer": np.array(pursuer_traj),
            "evader": np.array(evader_traj),
            "subgoals": np.array(subgoals) if subgoals else None,
        }

    return {
        "reward": ep_reward,
        "steps": steps,
        "captured": captured,
        "final_dist": prev_dist,
    }, traj_data


def evaluate_model(env, model, device, n_episodes=30):
    """运行多个episode，收集汇总统计"""
    results = []
    for i in range(n_episodes):
        info, _ = run_episode(env, model, device, record_traj=False)
        results.append(info)
        status = "捕获" if info["captured"] else "失败"
        print(f"  Episode {i+1:3d}/{n_episodes} | "
              f"reward={info['reward']:8.1f} | steps={info['steps']:4d} | "
              f"{status} | dist={info['final_dist']:.2f}")

    rewards = [r["reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    captured_list = [r["captured"] for r in results]
    success_rate = sum(captured_list) / len(captured_list)
    cap_steps = [r["steps"] for r in results if r["captured"]]

    stats = {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_capture_steps": float(np.mean(cap_steps)) if cap_steps else None,
        "median_capture_steps": float(np.median(cap_steps)) if cap_steps else None,
    }
    return stats, results


def plot_trajectory(traj_data, env, ep_idx, out_path, captured=False):
    """绘制单episode追逃轨迹PNG"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    half = env.half

    # 障碍物
    for obs in env.obstacles:
        rect = Rectangle(
            (obs.cx - obs.w / 2, obs.cy - obs.h / 2),
            obs.w, obs.h, fc="#888888", ec="black", alpha=0.5, zorder=2
        )
        ax.add_patch(rect)

    p_traj = traj_data["pursuer"]
    e_traj = traj_data["evader"]

    # 用颜色渐变表示时间
    n = len(p_traj)
    t_norm = np.linspace(0, 1, n)

    # 追击者轨迹 (蓝→红)
    points_p = p_traj.reshape(-1, 1, 2)
    segments_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
    lc_p = LineCollection(segments_p, cmap="coolwarm", linewidths=2, zorder=3)
    lc_p.set_array(t_norm[:-1])
    ax.add_collection(lc_p)

    # 逃逸者轨迹 (绿色虚线)
    n_e = len(e_traj)
    points_e = e_traj.reshape(-1, 1, 2)
    segments_e = np.concatenate([points_e[:-1], points_e[1:]], axis=1)
    lc_e = LineCollection(segments_e, colors="green", linewidths=1.5,
                          linestyles="dashed", alpha=0.7, zorder=3)
    ax.add_collection(lc_e)

    # 起终点
    ax.plot(*p_traj[0], "bs", ms=10, zorder=5, label="Pursuer start")
    ax.plot(*p_traj[-1], "r^", ms=10, zorder=5, label="Pursuer end")
    ax.plot(*e_traj[0], "go", ms=10, zorder=5, label="Evader start")
    ax.plot(*e_traj[-1], "gx", ms=10, mew=3, zorder=5, label="Evader end")

    # 捕获半径
    if captured:
        circle = Circle(e_traj[-1], env.catch_radius,
                        fc="red", alpha=0.2, ec="red", ls="--", zorder=4)
        ax.add_patch(circle)

    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    result_str = "CAPTURED" if captured else "ESCAPED"
    ax.set_title(f"Episode {ep_idx} — {result_str} ({n} steps)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_gif(traj_data, env, ep_idx, out_path, captured=False, fps=10):
    """生成追逃轨迹GIF动画"""
    from matplotlib.animation import FuncAnimation, PillowWriter

    p_traj = traj_data["pursuer"]
    e_traj = traj_data["evader"]
    n = len(p_traj)
    half = env.half

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def init():
        ax.clear()
        for obs in env.obstacles:
            rect = Rectangle(
                (obs.cx - obs.w / 2, obs.cy - obs.h / 2),
                obs.w, obs.h, fc="#888888", ec="black", alpha=0.5, zorder=2
            )
            ax.add_patch(rect)
        ax.set_xlim(-half - 1, half + 1)
        ax.set_ylim(-half - 1, half + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return []

    # 每frame跳若干步, 控制GIF长度
    skip = max(1, n // 200)
    frames = list(range(0, n, skip))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    def update(frame_idx):
        i = frames[frame_idx]
        ax.clear()
        for obs in env.obstacles:
            rect = Rectangle(
                (obs.cx - obs.w / 2, obs.cy - obs.h / 2),
                obs.w, obs.h, fc="#888888", ec="black", alpha=0.5, zorder=2
            )
            ax.add_patch(rect)

        # 历史轨迹
        ax.plot(p_traj[:i+1, 0], p_traj[:i+1, 1], "b-", lw=1.5, alpha=0.6)
        ax.plot(e_traj[:i+1, 0], e_traj[:i+1, 1], "g--", lw=1.2, alpha=0.5)

        # 当前位置
        ax.plot(p_traj[i, 0], p_traj[i, 1], "bo", ms=8, zorder=5)
        ax.plot(e_traj[i, 0], e_traj[i, 1], "go", ms=8, zorder=5)

        # 捕获半径
        circle = Circle(e_traj[i], env.catch_radius,
                        fc="none", ec="red", ls="--", alpha=0.5, zorder=4)
        ax.add_patch(circle)

        ax.set_xlim(-half - 1, half + 1)
        ax.set_ylim(-half - 1, half + 1)
        ax.set_aspect("equal")
        ax.set_title(f"Episode {ep_idx} | Step {i}/{n-1}")
        ax.grid(True, alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frames), blit=False, interval=1000//fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved GIF: {out_path}")


def plot_summary(stats_best, stats_final, out_path):
    """绘制best vs final对比汇总图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    labels = ["Best", "Final"]
    colors = ["#2196F3", "#FF9800"]

    # 1) Success Rate
    ax = axes[0]
    srs = [stats_best["success_rate"] * 100, stats_final["success_rate"] * 100]
    bars = ax.bar(labels, srs, color=colors, width=0.5, edgecolor="black")
    for bar, v in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Capture Success Rate")
    ax.axhline(y=80, color="gray", ls="--", alpha=0.5, label="80% ref")
    ax.legend()

    # 2) Mean Reward
    ax = axes[1]
    rews = [stats_best["mean_reward"], stats_final["mean_reward"]]
    stds = [stats_best["std_reward"], stats_final["std_reward"]]
    bars = ax.bar(labels, rews, yerr=stds, color=colors, width=0.5,
                  edgecolor="black", capsize=5)
    for bar, v in zip(bars, rews):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                y + (20 if y >= 0 else -30),
                f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Episode Reward")
    ax.axhline(y=0, color="gray", ls="-", alpha=0.3)

    # 3) Capture Steps
    ax = axes[2]
    cs = [stats_best.get("mean_capture_steps") or 0,
          stats_final.get("mean_capture_steps") or 0]
    bars = ax.bar(labels, cs, color=colors, width=0.5, edgecolor="black")
    for bar, v in zip(bars, cs):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Mean Steps to Capture")
    ax.set_title("Capture Efficiency")

    fig.suptitle("Online PPO (BPPO pretrain) — Best vs Final", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--n-traj", type=int, default=4,
                        help="Number of trajectory visualizations")
    parser.add_argument("--gif", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low_ckpt = os.path.join(PROJECT_ROOT,
        "low_near/v4_tight_heading_terminal/checkpoints/best.pth")
    nn_policy = load_v4_nn(low_ckpt, device)

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints/online_ppo_bppo_v1")
    out_dir = os.path.join(PROJECT_ROOT, "evaluation/results_online_ppo_bppo_v1")
    os.makedirs(out_dir, exist_ok=True)

    for ckpt_name in ["best", "final"]:
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Skip {ckpt_name}: not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}.pth")
        print(f"{'='*60}")

        model = load_model(ckpt_path, device)

        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy, device=device,
            world_size=30.0, catch_radius=1.0, max_steps=6000,
            subgoal_range=0.7, low_steps=50,
            evader_mode="medium", seed=args.seed,
        )

        # 1) 定量评估
        print(f"\n--- Quantitative Eval ({args.n_episodes} episodes) ---")
        stats, results = evaluate_model(env, model, device, args.n_episodes)
        print(f"\n  Success Rate:    {stats['success_rate']*100:.0f}%")
        print(f"  Mean Reward:     {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        print(f"  Mean Steps:      {stats['mean_steps']:.0f}")
        if stats["mean_capture_steps"]:
            print(f"  Mean Cap Steps:  {stats['mean_capture_steps']:.0f}")

        # 保存统计
        stats_path = os.path.join(out_dir, f"stats_{ckpt_name}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

        # 2) 轨迹可视化 (挑选前n_traj个成功和失败的)
        print(f"\n--- Trajectory Visualization ---")
        env_vis = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy, device=device,
            world_size=30.0, catch_radius=1.0, max_steps=6000,
            subgoal_range=0.7, low_steps=50,
            evader_mode="medium", seed=args.seed + 100,
        )

        success_count = 0
        fail_count = 0
        for ep in range(args.n_traj * 3):  # run extra to get both types
            info, traj = run_episode(env_vis, model, device, record_traj=True)
            if info["captured"] and success_count < args.n_traj:
                success_count += 1
                tag = f"{ckpt_name}_success_{success_count}"
                png_path = os.path.join(out_dir, f"traj_{tag}.png")
                plot_trajectory(traj, env_vis, ep, png_path, captured=True)
                if args.gif:
                    gif_path = os.path.join(out_dir, f"traj_{tag}.gif")
                    make_gif(traj, env_vis, ep, gif_path, captured=True)
            elif not info["captured"] and fail_count < 2:
                fail_count += 1
                tag = f"{ckpt_name}_fail_{fail_count}"
                png_path = os.path.join(out_dir, f"traj_{tag}.png")
                plot_trajectory(traj, env_vis, ep, png_path, captured=False)
                if args.gif:
                    gif_path = os.path.join(out_dir, f"traj_{tag}.gif")
                    make_gif(traj, env_vis, ep, gif_path, captured=False)

            if success_count >= args.n_traj and fail_count >= 2:
                break

    # 3) 汇总对比图
    best_stats_path = os.path.join(out_dir, "stats_best.json")
    final_stats_path = os.path.join(out_dir, "stats_final.json")
    if os.path.exists(best_stats_path) and os.path.exists(final_stats_path):
        with open(best_stats_path) as f:
            sb = json.load(f)
        with open(final_stats_path) as f:
            sf = json.load(f)
        summary_path = os.path.join(out_dir, "summary_best_vs_final.png")
        plot_summary(sb, sf, summary_path)

    print(f"\nAll results in: {out_dir}")


if __name__ == "__main__":
    main()
