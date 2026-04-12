"""
评估 offline/online 训练的高层策略
==================================
在真实 PE 环境中跑 N 个 episode，统计捕获率，
并可视化最佳/最差轨迹。
"""

import os
import sys
import math
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from training.offline_ppo_pretrain import HighLevelActorCritic
from training.collect_expert_trajectories import compute_high_reward


def _model_action(model, obs_t, deterministic=True):
    """统一接口: 支持 HighLevelActorCritic 和 TD3Actor"""
    if hasattr(model, 'get_action'):
        a, _ = model.get_action(obs_t, deterministic=deterministic)
    else:
        # TD3Actor: forward 直接输出 tanh(action)
        a = model(obs_t)
    return a


# ─────────── 单 episode rollout ───────────

def rollout_episode(
    env: PursuitEnvRealAUVFixed,
    model,
    device,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    env.rng = np.random.default_rng(seed)
    obs = env.reset()

    eta = env.auv_env.inner.dynamics.eta
    prev_dist = float(np.linalg.norm(
        np.array([eta[0], eta[1]]) - env.e))

    traj_p, traj_e, subgoals = [], [], []
    rewards, dists = [], []
    vp_list, ve_list = [], []  # 速度记录

    eta = env.auv_env.inner.dynamics.eta
    traj_p.append(np.array([eta[0], eta[1]]).copy())
    traj_e.append(env.e.copy())

    done = False
    info = {}
    ep_reward = 0.0
    step = 0

    while not done:
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            a = _model_action(model, obs_t, deterministic)
        a_np = a[0].cpu().numpy()

        # 记录 subgoal (原始, 修正前)
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        sg = p + a_np * env.subgoal_range
        subgoals.append(sg.copy())

        obs, _env_r, done, info = env.step(a_np)

        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        curr_dist = float(np.linalg.norm(p - env.e))

        r, _ = compute_high_reward(
            env, prev_dist, curr_dist,
            info.get("captured", False),
            info.get("out_of_bounds", False),
            info.get("collision", False),
            subgoal=sg,
            done=done,
        )

        traj_p.append(p.copy())
        traj_e.append(env.e.copy())
        dists.append(curr_dist)
        rewards.append(r)
        ep_reward += r
        prev_dist = curr_dist
        step += 1

        # 记录速度
        vp = env._get_pursuer_vel()
        ve = env.ve.copy()
        vp_list.append(np.linalg.norm(vp))
        ve_list.append(np.linalg.norm(ve))

    # 计算加速度 (速度差分)
    vp_arr = np.array(vp_list)
    ve_arr = np.array(ve_list)
    ap_arr = np.diff(vp_arr, prepend=vp_arr[0]) if len(vp_arr) > 0 else np.array([])
    ae_arr = np.diff(ve_arr, prepend=ve_arr[0]) if len(ve_arr) > 0 else np.array([])

    return {
        "traj_p": np.array(traj_p),
        "traj_e": np.array(traj_e),
        "subgoals": np.array(subgoals) if subgoals else np.zeros((0, 2)),
        "dists": np.array(dists),
        "rewards": np.array(rewards),
        "vp": vp_arr,
        "ve": ve_arr,
        "ap": ap_arr,
        "ae": ae_arr,
        "ep_reward": ep_reward,
        "captured": info.get("captured", False),
        "steps": step,
        "seed": seed,
    }


# ─────────── 绘制单 episode 轨迹 ───────────

def draw_episode(env, trace, save_path, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (0,0) 轨迹图
    ax = axes[0, 0]
    half = env.half
    ax.plot([-half, half, half, -half, -half],
            [-half, -half, half, half, -half],
            "k-", lw=1.5)
    for obs in env.obstacles:
        rect = Rectangle(
            (obs.cx - obs.w / 2, obs.cy - obs.h / 2),
            obs.w, obs.h, fc="gray", ec="black", alpha=0.5)
        ax.add_patch(rect)

    p = trace["traj_p"]
    e = trace["traj_e"]
    sg = trace["subgoals"]

    ax.plot(p[:, 0], p[:, 1], "b-", lw=2, alpha=0.8, label="Pursuer")
    ax.plot(e[:, 0], e[:, 1], "r-", lw=2, alpha=0.8, label="Evader")

    if len(sg) > 0:
        ax.scatter(sg[:, 0], sg[:, 1], c="green", s=15,
                   alpha=0.5, zorder=5, label="Subgoals")

    ax.scatter(*p[0], c="blue", s=100, marker="o", zorder=7)
    ax.scatter(*e[0], c="red", s=100, marker="o", zorder=7)
    ax.scatter(*p[-1], c="blue", s=150, marker="*", zorder=7)
    ax.scatter(*e[-1], c="red", s=150, marker="*", zorder=7)

    catch_c = plt.Circle(
        (p[-1, 0], p[-1, 1]), env.catch_radius,
        fill=False, color="green", lw=2, ls="--")
    ax.add_patch(catch_c)

    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    result = "CAPTURED" if trace["captured"] else "FAILED"
    ax.set_title(f"{title} [{result}] steps={trace['steps']}")

    # (0,1) 距离 + 奖励曲线
    ax2 = axes[0, 1]
    steps = np.arange(len(trace["dists"]))
    ax2.plot(steps, trace["dists"], "b-", lw=1.5, label="Distance")
    ax2.axhline(env.catch_radius, color="green", ls="--",
                lw=1, label=f"catch_r={env.catch_radius}")
    ax2.set_xlabel("High-Level Step")
    ax2.set_ylabel("Distance (m)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    ax3 = ax2.twinx()
    ax3.plot(steps, trace["rewards"], "r-", lw=1, alpha=0.6,
             label="Reward")
    ax3.set_ylabel("Reward", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.legend(loc="upper right", fontsize=9)

    # (1,0) 速度曲线
    ax4 = axes[1, 0]
    if len(trace.get("vp", [])) > 0:
        steps_v = np.arange(len(trace["vp"]))
        ax4.plot(steps_v, trace["vp"], "b-", lw=1.5,
                 label="Pursuer |v|")
        ax4.plot(steps_v, trace["ve"], "r-", lw=1.5,
                 label="Evader |v|")
        ax4.set_xlabel("High-Level Step")
        ax4.set_ylabel("Speed (m/s)")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.2)
        ax4.set_title("Velocity (speed)")
    else:
        ax4.text(0.5, 0.5, "No velocity data", ha="center",
                 va="center", transform=ax4.transAxes)

    # (1,1) 加速度曲线
    ax5 = axes[1, 1]
    if len(trace.get("ap", [])) > 0:
        steps_a = np.arange(len(trace["ap"]))
        ax5.plot(steps_a, trace["ap"], "b-", lw=1, alpha=0.7,
                 label="Pursuer |a|")
        ax5.plot(steps_a, trace["ae"], "r-", lw=1, alpha=0.7,
                 label="Evader |a|")
        ax5.set_xlabel("High-Level Step")
        ax5.set_ylabel("Acceleration (m/s²)")
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.2)
        ax5.set_title("Acceleration (speed diff)")
    else:
        ax5.text(0.5, 0.5, "No accel data", ha="center",
                 va="center", transform=ax5.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────── 汇总统计图 ───────────

def draw_summary(results, save_path, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ep_rewards = [r["ep_reward"] for r in results]
    ep_steps = [r["steps"] for r in results]
    captured = [r["captured"] for r in results]
    min_dists = [r["dists"].min() if len(r["dists"]) > 0
                 else 999 for r in results]

    ax = axes[0, 0]
    ax.bar(range(len(ep_rewards)), ep_rewards,
           color=["green" if c else "red" for c in captured])
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.bar(range(len(ep_steps)), ep_steps,
           color=["green" if c else "red" for c in captured])
    ax.set_title("Episode Length (steps)")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.bar(range(len(min_dists)), min_dists,
           color=["green" if c else "red" for c in captured])
    ax.set_title("Min Distance to Evader")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    sr = sum(captured) / max(1, len(captured))
    ax.text(0.5, 0.5,
            f"Episodes: {len(results)}\n"
            f"Captured: {sum(captured)}\n"
            f"Success Rate: {sr:.1%}\n"
            f"Mean Reward: {np.mean(ep_rewards):.1f}\n"
            f"Mean Steps: {np.mean(ep_steps):.1f}\n"
            f"Mean Min Dist: {np.mean(min_dists):.2f}m",
            transform=ax.transAxes, fontsize=16,
            va="center", ha="center",
            bbox=dict(boxstyle="round", fc="lightyellow"))
    ax.set_axis_off()
    ax.set_title("Summary")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary: {save_path}")


# ─────────── main ───────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--high-ckpt", type=str, required=True)
    pa.add_argument("--low-level-ckpt", type=str, required=True)
    pa.add_argument("--n-episodes", type=int, default=20)
    pa.add_argument("--obs-dim", type=int, default=21)
    pa.add_argument("--act-dim", type=int, default=2)
    pa.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    pa.add_argument("--world-size", type=float, default=30.0)
    pa.add_argument("--catch-radius", type=float, default=1.0)
    pa.add_argument("--subgoal-range", type=float, default=0.7)
    pa.add_argument("--low-steps", type=int, default=50)
    pa.add_argument("--max-steps", type=int, default=6000)
    pa.add_argument("--evader-mode", type=str, default="medium")
    pa.add_argument("--output-dir", type=str,
                    default="evaluation/high_level_eval")
    pa.add_argument("--seed", type=int, default=100)
    pa.add_argument("--deterministic", action="store_true", default=True)
    pa.add_argument("--top-k", type=int, default=3,
                    help="画 top-k 最好和最差的episode轨迹")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载底层
    nn_policy = load_v4_nn(args.low_level_ckpt, device)

    # 加载高层 (自动检测 HighLevelActorCritic 或 TD3Actor)
    ckpt = torch.load(args.high_ckpt, map_location=device,
                      weights_only=False)
    if "model" in ckpt:
        # HighLevelActorCritic (offline PPO / BC)
        model = HighLevelActorCritic(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded HighLevelActorCritic: {args.high_ckpt}")
    elif "actor" in ckpt:
        # TD3Actor (TD3+BC)
        from training.td3bc_pretrain import TD3Actor
        model = TD3Actor(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(device)
        model.load_state_dict(ckpt["actor"])
        print(f"Loaded TD3Actor: {args.high_ckpt}")
    else:
        raise ValueError(f"Unknown ckpt format: {list(ckpt.keys())}")
    model.eval()

    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy, device=device,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        max_steps=args.max_steps,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=args.seed,
        evader_mode=args.evader_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 跑 episodes
    results = []
    for i in range(args.n_episodes):
        seed_i = args.seed + i
        trace = rollout_episode(
            env, model, device, seed_i, args.deterministic)
        results.append(trace)
        tag = "OK" if trace["captured"] else "FAIL"
        print(f"  ep {i+1}/{args.n_episodes}: {tag} "
              f"steps={trace['steps']} "
              f"reward={trace['ep_reward']:.1f} "
              f"min_d={trace['dists'].min():.2f}m")

    # 汇总
    sr = sum(r["captured"] for r in results) / len(results)
    print(f"\n=== Summary: {sr:.1%} success rate "
          f"({sum(r['captured'] for r in results)}/{len(results)}) ===")

    draw_summary(results, os.path.join(args.output_dir, "summary.png"),
                 title=f"High-Level Policy Eval ({args.evader_mode})")

    # 画 top-k best / worst 轨迹
    sorted_r = sorted(results, key=lambda x: x["ep_reward"],
                       reverse=True)
    for k in range(min(args.top_k, len(sorted_r))):
        t = sorted_r[k]
        draw_episode(env, t,
                     os.path.join(args.output_dir, f"best_{k+1}.png"),
                     title=f"Best #{k+1} (seed={t['seed']})")
    for k in range(min(args.top_k, len(sorted_r))):
        t = sorted_r[-(k+1)]
        draw_episode(env, t,
                     os.path.join(args.output_dir, f"worst_{k+1}.png"),
                     title=f"Worst #{k+1} (seed={t['seed']})")

    print("Done.")


if __name__ == "__main__":
    main()
