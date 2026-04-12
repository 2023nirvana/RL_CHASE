#!/usr/bin/env python3
"""
高层策略对比评估
================
在真实AUV动力学追逃环境中测试多种高层策略:
  - 经典策略: PP / Lead / PN / APN / CB / DP
  - 预测启发式 (Expert): 预测逃脱者2秒后位置
  - HRL NN高层 (如有): 训练好的PPO高层策略

对每种策略:
  1. 跑 N 回合, 统计成功率/平均步数/平均距离
  2. 选出 best episode 生成 GIF + PNG (与 pursuit_with_dynamics_expert 同风格)
  3. 最终生成跨策略对比图
"""

import os, sys, math, argparse, json
from typing import Dict, List, Optional, Any

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
from core.agents.classical_pursuit_strategies import make_strategy, BaseStrategy


# ======================== 策略适配器 ========================

class StrategyAdapter:
    """将 classical strategy 适配到 PursuitEnvRealAUVFixed 的 action 空间。
    
    PursuitEnvRealAUVFixed.step(action) 期望:
        action ∈ [-1, 1]^2, 表示 (dx, dy) / subgoal_range
    
    classical strategy 输出:
        subgoal_world (2D position)
    
    转换: action = (subgoal - pursuer_pos) / subgoal_range, clipped to [-1,1]
    """
    
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.name = strategy.name
    
    def reset(self):
        self.strategy.reset()
    
    def get_action(self, env: PursuitEnvRealAUVFixed) -> np.ndarray:
        """从环境状态计算 normalized action ∈ [-1,1]^2"""
        eta = env.auv_env.inner.dynamics.eta
        p_pos = np.array([eta[0], eta[1]])
        p_vel = env._get_pursuer_vel()
        p_yaw = float(eta[5])
        e_pos = env.e.copy()
        e_vel = env.ve.copy()
        
        self.strategy.set_dt(env.dt_evader * env.low_steps)  # high-level dt
        
        subgoal = self.strategy.compute_subgoal(
            p_pos, p_vel, p_yaw, e_pos, e_vel,
            subgoal_radius=env.subgoal_range,
            world_size=env.half,
        )
        
        # 转换为 normalized action
        delta = subgoal - p_pos
        action = delta / max(env.subgoal_range, 1e-6)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class ExpertAdapter:
    """预测启发式策略（与 generate_expert_dynamics_visuals.py 一致）"""
    
    def __init__(self, predict_time: float = 2.0):
        self.name = f"Expert(pred={predict_time}s)"
        self.predict_time = predict_time
    
    def reset(self):
        pass
    
    def get_action(self, env: PursuitEnvRealAUVFixed) -> np.ndarray:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        e_pred = env.e + env.ve * self.predict_time
        e_pred = np.clip(e_pred, -env.half + 1, env.half - 1)
        
        direction = e_pred - p
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            action = direction / max(dist, env.subgoal_range)
        else:
            action = np.zeros(2, dtype=np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


# ======================== Rollout ========================

def rollout(env: PursuitEnvRealAUVFixed, adapter, seed: int = 42):
    """执行一个 episode, 收集轨迹数据 (包含子步密集轨迹用于丝滑渲染)"""
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    adapter.reset()
    
    eta = env.auv_env.inner.dynamics.eta
    init_p = np.array([eta[0], eta[1]], dtype=np.float32)
    init_e = env.e.copy().astype(np.float32)
    init_vp = env._get_pursuer_vel().astype(np.float32)
    init_ve = env.ve.copy().astype(np.float32)

    # 高层步级别轨迹 (用于策略分析)
    traj_p = [init_p.copy()]
    traj_e = [init_e.copy()]
    vp_list = [init_vp.copy()]
    ve_list = [init_ve.copy()]
    distances = [float(np.linalg.norm(init_p - init_e))]
    subgoals = []
    sg_origins = []

    # 子步密集轨迹 (用于丝滑GIF渲染)
    dense_p = [init_p.copy()]
    dense_e = [init_e.copy()]
    dense_vp = [init_vp.copy()]
    dense_ve = [init_ve.copy()]
    
    done = False
    info = {}
    while not done:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]], dtype=np.float32)
        
        action = adapter.get_action(env)
        
        sg_origins.append(p.copy())
        
        obs, reward, done, info = env.step(action)
        
        # 使用环境校正后的 subgoal（经过避障处理的），而非原始 action 计算的
        corrected_sg = info.get('corrected_subgoal', None)
        if corrected_sg is not None:
            subgoals.append(np.asarray(corrected_sg, dtype=np.float32))
        else:
            dx = float(action[0]) * env.subgoal_range
            dy = float(action[1]) * env.subgoal_range
            sg = np.array([p[0] + dx, p[1] + dy], dtype=np.float32)
            sg = np.clip(sg, -env.half + 0.5, env.half - 0.5)
            subgoals.append(sg)
        
        # 高层步记录
        eta = env.auv_env.inner.dynamics.eta
        traj_p.append(np.array([eta[0], eta[1]], dtype=np.float32))
        traj_e.append(env.e.copy().astype(np.float32))
        vp_list.append(env._get_pursuer_vel().astype(np.float32))
        ve_list.append(env.ve.copy().astype(np.float32))
        distances.append(float(np.linalg.norm(traj_p[-1] - traj_e[-1])))
        
        # 子步密集轨迹
        for sp, se, sv, svv in zip(info.get('sub_traj_p', []),
                                    info.get('sub_traj_e', []),
                                    info.get('sub_vp', []),
                                    info.get('sub_ve', [])):
            dense_p.append(np.asarray(sp, dtype=np.float32))
            dense_e.append(np.asarray(se, dtype=np.float32))
            dense_vp.append(np.asarray(sv, dtype=np.float32))
            dense_ve.append(np.asarray(svv, dtype=np.float32))
    
    return {
        "p": np.array(traj_p),
        "e": np.array(traj_e),
        "vp": np.array(vp_list),
        "ve": np.array(ve_list),
        "distance": np.array(distances),
        "subgoals": np.array(subgoals) if subgoals else np.zeros((0, 2)),
        "sg_origins": np.array(sg_origins) if sg_origins else np.zeros((0, 2)),
        # 密集子步轨迹
        "dense_p": np.array(dense_p),
        "dense_e": np.array(dense_e),
        "dense_vp": np.array(dense_vp),
        "dense_ve": np.array(dense_ve),
        "captured": bool(info.get("captured", False)),
        "collision": bool(info.get("collision", False)),
        "steps": int(info.get("steps", len(traj_p) - 1)),
        "seed": seed,
    }


# ======================== 可视化 (同 generate_expert_dynamics_visuals 风格) ========================

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


def save_static_png(trace, env, save_path, strategy_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_world(ax, env)

    # 使用密集轨迹绘制更平滑的路径
    has_dense = "dense_p" in trace and len(trace["dense_p"]) > 1
    p = trace["dense_p"] if has_dense else trace["p"]
    e = trace["dense_e"] if has_dense else trace["e"]
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
        f"[{strategy_name}] {status}, steps={trace['steps']}, seed={trace['seed']}\n"
        f"subgoal_range={env.subgoal_range}, v_e={env.v_max_e:.2f}, a_e={env.a_max_e:.2f}",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {save_path}")


def save_dynamics_gif(trace, env, save_path, strategy_name, fps=8, stride=1):
    # === 完全对齐旧版 generate_expert_dynamics_visuals.py 的渲染方式 ===
    # 用高层步级别数据画所有内容（路径、速度、距离），避免子步高频振荡
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
    ax_world.set_title(f"[{strategy_name}] Pursuit Replay")

    p_line, = ax_world.plot([], [], "b-", linewidth=2, alpha=0.85, label="Pursuer")
    e_line, = ax_world.plot([], [], "r-", linewidth=2, alpha=0.85, label="Evader")
    p_dot = ax_world.scatter([], [], c="blue", s=90, zorder=6)
    e_dot = ax_world.scatter([], [], c="red", s=90, zorder=6)
    sg_hist = ax_world.scatter([], [], c=[], cmap="viridis", s=22, alpha=0.7, zorder=5)
    sg_cur = ax_world.scatter([], [], c="gold", s=120, marker="*", edgecolor="black", zorder=7, label="Subgoal")
    sg_arrow = ax_world.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                               color="green", alpha=0.65)
    txt = ax_world.text(0.02, 0.98, "", transform=ax_world.transAxes, va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax_world.legend(loc="upper right", fontsize=8)

    line_dist, = ax_dist.plot([], [], color="black", linewidth=2.0, label="distance(p,e)")
    cursor_d = ax_dist.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_dist.axhline(env.catch_radius, linestyle=":", color="green", linewidth=1.2,
                    label=f"catch_r={env.catch_radius}")
    ax_dist.set_xlim(0, max(n - 1, 1))
    ax_dist.set_ylim(0, max(float(np.max(dist)), 2.0) * 1.1)
    ax_dist.set_title("Distance to Evader")
    ax_dist.set_xlabel("High-level Step")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="upper right", fontsize=8)

    line_vp, = ax_speed.plot([], [], color="royalblue", linewidth=2.0, label="|v_pursuer|")
    line_ve, = ax_speed.plot([], [], color="crimson", linewidth=2.0, label="|v_evader|")
    cursor_s = ax_speed.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_speed.set_xlim(0, max(n - 1, 1))
    vmax = max(float(np.max(speed_p)), float(np.max(speed_e)), 0.1)
    ax_speed.set_ylim(0, vmax * 1.25)
    ax_speed.set_title("Speed Profile")
    ax_speed.set_xlabel("High-level Step")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(loc="upper right", fontsize=8)

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
            f"[{strategy_name}] step={t:03d}  d={dist[t]:.2f}  "
            f"|v_p|={speed_p[t]:.2f}  |v_e|={speed_e[t]:.2f}  {status}"
        )
        return [p_line, e_line, p_dot, e_dot, sg_hist, sg_cur, sg_arrow, txt,
                line_dist, line_vp, line_ve, cursor_d, cursor_s]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)
    ani.save(save_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print(f"  Saved GIF: {save_path}")


# ======================== 跨策略对比图 ========================

def save_comparison_plot(all_results: Dict[str, Dict], save_path: str):
    """生成跨策略对比柱状图 (成功率 / 平均步数 / 最小距离)"""
    names = list(all_results.keys())
    n = len(names)
    
    catch_rates = []
    avg_steps_caught = []
    avg_min_dist = []
    
    for name in names:
        res = all_results[name]
        episodes = res["episodes"]
        caught_eps = [ep for ep in episodes if ep["captured"]]
        catch_rates.append(len(caught_eps) / max(len(episodes), 1))
        if caught_eps:
            avg_steps_caught.append(np.mean([ep["steps"] for ep in caught_eps]))
        else:
            avg_steps_caught.append(float(res.get("max_steps", 6000)))
        avg_min_dist.append(np.mean([np.min(ep["distance"]) for ep in episodes]))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    
    # 成功率
    bars = axes[0].bar(range(n), [r * 100 for r in catch_rates], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Catch Rate (%)")
    axes[0].set_title("Catch Rate")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, catch_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    
    # 平均步数 (caught episodes)
    bars = axes[1].bar(range(n), avg_steps_caught, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[1].set_ylabel("Avg Steps (caught)")
    axes[1].set_title("Steps to Catch (lower=better)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, avg_steps_caught):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    
    # 最小接近距离
    bars = axes[2].bar(range(n), avg_min_dist, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[2].set_ylabel("Avg Min Distance (m)")
    axes[2].set_title("Closest Approach (lower=better)")
    axes[2].axhline(1.0, color="green", linestyle=":", alpha=0.5, label="catch_radius")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=8)
    for bar, v in zip(bars, avg_min_dist):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    
    fig.suptitle("High-Level Strategy Comparison (Real AUV Dynamics)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison: {save_path}")


# ======================== 主流程 ========================

def find_best_episode(episodes: List[Dict], prefer_caught: bool = True) -> Dict:
    """选出最佳 episode: 优先 captured, 其次步数最长(展示效果好)"""
    caught = [ep for ep in episodes if ep["captured"]]
    if caught and prefer_caught:
        # 选步数适中的 (不太短也不太长)
        caught.sort(key=lambda x: x["steps"])
        idx = len(caught) // 2  # median
        return caught[idx]
    # 没捕获则选最小距离的
    episodes_sorted = sorted(episodes, key=lambda x: np.min(x["distance"]))
    return episodes_sorted[0]


def main():
    parser = argparse.ArgumentParser(description="高层策略对比评估")
    parser.add_argument("--low-level-path", type=str, 
                        default=os.path.join(PROJECT_ROOT, "low_near/v4_tight_vel10_v2/checkpoints/best.pth"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "evaluation/strategy_comparison"))
    parser.add_argument("--episodes", type=int, default=30, help="每策略评估回合数")
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--world-size", type=float, default=30.0)
    parser.add_argument("--catch-radius", type=float, default=1.0)
    parser.add_argument("--subgoal-range", type=float, default=2.0)
    parser.add_argument("--low-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--evader-mode", type=str, default="medium",
                        choices=["simple", "medium", "hard",
                                 "sampling_easy", "sampling_medium", "sampling_hard",
                                 "raycast_easy", "raycast_medium", "raycast_hard",
                                 "unicycle_easy", "unicycle_medium", "unicycle_hard"])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--strategies", type=str, nargs="*", 
                        default=["pp", "lead", "pn", "apn", "cb", "dp", "expert"],
                        help="要评估的策略列表")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载底层模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading low-level model: {args.low_level_path}")
    nn_policy = load_v4_nn(args.low_level_path, device)
    
    # 创建追逃环境
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        max_steps=args.max_steps,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=42,
        evader_mode=args.evader_mode,
    )
    
    # 构建策略列表
    adapters = []
    for s_name in args.strategies:
        if s_name.lower() == "expert":
            adapters.append(ExpertAdapter(predict_time=2.0))
        else:
            strategy = make_strategy(s_name)
            adapters.append(StrategyAdapter(strategy))
    
    all_results = {}
    
    for strat_idx, adapter in enumerate(adapters):
        name = adapter.name
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"{'='*60}")
        
        episodes = []
        caught_count = 0
        
        for i in range(args.episodes):
            # 每个策略使用不同的 seed 偏移，避免所有策略的初始场景完全相同
            seed = args.base_seed + strat_idx * 1000 + i
            trace = rollout(env, adapter, seed=seed)
            episodes.append(trace)
            if trace["captured"]:
                caught_count += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{args.episodes}] caught={caught_count}/{i+1}")
        
        catch_rate = caught_count / args.episodes
        caught_eps = [ep for ep in episodes if ep["captured"]]
        avg_steps = np.mean([ep["steps"] for ep in caught_eps]) if caught_eps else args.max_steps
        avg_min_dist = np.mean([np.min(ep["distance"]) for ep in episodes])
        
        print(f"  Result: catch_rate={catch_rate:.0%}, avg_steps={avg_steps:.0f}, avg_min_dist={avg_min_dist:.2f}m")
        
        all_results[name] = {
            "episodes": episodes,
            "catch_rate": catch_rate,
            "avg_steps": avg_steps,
            "avg_min_dist": avg_min_dist,
            "max_steps": args.max_steps,
        }
        
        # 选择 best episode 生成可视化
        best_ep = find_best_episode(episodes)
        safe_name = name.replace("(", "").replace(")", "").replace("=", "").replace(" ", "_").replace("/", "_")
        
        strat_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(strat_dir, exist_ok=True)
        
        png_path = os.path.join(strat_dir, f"pursuit_{safe_name}.png")
        gif_path = os.path.join(strat_dir, f"pursuit_{safe_name}.gif")
        
        save_static_png(best_ep, env, png_path, name)
        save_dynamics_gif(best_ep, env, gif_path, name, fps=args.fps, stride=args.stride)
    
    # 保存汇总统计
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "catch_rate": res["catch_rate"],
            "avg_steps_caught": float(res["avg_steps"]),
            "avg_min_dist": float(res["avg_min_dist"]),
            "num_episodes": args.episodes,
        }
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")
    
    # 跨策略对比图
    comparison_path = os.path.join(args.output_dir, "strategy_comparison.png")
    save_comparison_plot(all_results, comparison_path)
    
    print("\n" + "="*60)
    print("  ALL DONE")
    print("="*60)
    for name, res in all_results.items():
        print(f"  {name:25s}: catch={res['catch_rate']:.0%}  steps={res['avg_steps']:.0f}  min_d={res['avg_min_dist']:.2f}m")


if __name__ == "__main__":
    main()
