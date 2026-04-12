#!/usr/bin/env python3
"""
可视化专家追击策略的subgoal规划
- 展示每一步的subgoal位置
- 追踪者和逃跑者轨迹
- 带箭头连线：当前位置 → subgoal
"""
import os, sys, math
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed


def expert_predict_action(env):
    """预测追击专家"""
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
        action = np.zeros(2)
    return np.clip(action, -1, 1)


def collect_episode_with_subgoals(env, seed_offset=0):
    """收集一个episode，记录所有subgoal"""
    env.rng = np.random.default_rng(200 + seed_offset)
    obs = env.reset()

    eta = env.auv_env.inner.dynamics.eta
    traj_p = [np.array([eta[0], eta[1]])]
    traj_e = [env.e.copy()]
    subgoals = []
    subgoal_origins = []  # 发出subgoal时追击者的位置

    done = False
    while not done:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        action = expert_predict_action(env)

        # 计算实际subgoal坐标
        dx = action[0] * env.subgoal_range
        dy = action[1] * env.subgoal_range
        sg = np.array([p[0] + dx, p[1] + dy])
        sg = np.clip(sg, -env.half + 0.5, env.half - 0.5)

        subgoal_origins.append(p.copy())
        subgoals.append(sg.copy())

        obs, reward, done, info = env.step(action)

        eta = env.auv_env.inner.dynamics.eta
        traj_p.append(np.array([eta[0], eta[1]]))
        traj_e.append(env.e.copy())

    return {
        'traj_p': np.array(traj_p),
        'traj_e': np.array(traj_e),
        'subgoals': np.array(subgoals),
        'subgoal_origins': np.array(subgoal_origins),
        'info': info,
    }


def draw_world(ax, env):
    """画地图：边界、障碍物、惩罚区"""
    half = env.half
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect('equal')

    # 边界
    for val in [-half, half]:
        ax.axhline(val, color='black', linewidth=2)
        ax.axvline(val, color='black', linewidth=2)

    # 边界惩罚区
    margin = env.boundary_margin
    ax.fill_between([-half, half], half - margin, half, color='red', alpha=0.08)
    ax.fill_between([-half, half], -half, -half + margin, color='red', alpha=0.08)
    ax.fill_betweenx([-half + margin, half - margin], -half, -half + margin, color='red', alpha=0.08)
    ax.fill_betweenx([-half + margin, half - margin], half - margin, half, color='red', alpha=0.08)

    # 障碍物
    for obs in env.obstacles:
        rect = Rectangle(
            (obs.cx - obs.w / 2, obs.cy - obs.h / 2),
            obs.w, obs.h, facecolor='#808080', edgecolor='black', alpha=0.6, linewidth=1.2
        )
        ax.add_patch(rect)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.2)


def plot_episode_subgoals(ax, data, env, title_extra=""):
    """在ax上绘制一个episode的subgoal可视化"""
    draw_world(ax, env)

    traj_p = data['traj_p']
    traj_e = data['traj_e']
    subgoals = data['subgoals']
    origins = data['subgoal_origins']
    info = data['info']
    n_steps = len(subgoals)

    # 颜色映射：时间步
    colors = cm.viridis(np.linspace(0, 1, n_steps))

    # 追击者轨迹（蓝色渐变）
    for i in range(len(traj_p) - 1):
        t = i / max(len(traj_p) - 1, 1)
        ax.plot(traj_p[i:i+2, 0], traj_p[i:i+2, 1],
                color=(0, 0, 1 - 0.5*t), linewidth=1.8, alpha=0.7)

    # 逃跑者轨迹（红色渐变）
    for i in range(len(traj_e) - 1):
        t = i / max(len(traj_e) - 1, 1)
        ax.plot(traj_e[i:i+2, 0], traj_e[i:i+2, 1],
                color=(1, 0.2*t, 0), linewidth=1.8, alpha=0.7)

    # subgoal点 + 箭头（从origin到subgoal）
    for i in range(n_steps):
        c = colors[i]
        # subgoal点
        ax.plot(subgoals[i, 0], subgoals[i, 1], 'o',
                color=c, markersize=6, markeredgecolor='black',
                markeredgewidth=0.5, alpha=0.8, zorder=5)
        # 箭头: origin → subgoal
        dx = subgoals[i, 0] - origins[i, 0]
        dy = subgoals[i, 1] - origins[i, 1]
        ax.annotate('', xy=(subgoals[i, 0], subgoals[i, 1]),
                    xytext=(origins[i, 0], origins[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.0, alpha=0.5))

    # 起点和终点标记
    ax.plot(traj_p[0, 0], traj_p[0, 1], 'b^', markersize=14,
            markeredgecolor='black', markeredgewidth=1.5, label='P start', zorder=10)
    ax.plot(traj_e[0, 0], traj_e[0, 1], 'r^', markersize=14,
            markeredgecolor='black', markeredgewidth=1.5, label='E start', zorder=10)
    ax.plot(traj_p[-1, 0], traj_p[-1, 1], 'bs', markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, label='P end', zorder=10)
    ax.plot(traj_e[-1, 0], traj_e[-1, 1], 'rs', markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, label='E end', zorder=10)

    # 捕获圈
    if info['captured']:
        circle = Circle((traj_p[-1, 0], traj_p[-1, 1]), env.catch_radius,
                        fill=False, color='lime', linewidth=2.5, linestyle='--', zorder=9)
        ax.add_patch(circle)

    # colorbar 表示时间步
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, n_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('High-level step')

    status = "CAPTURED" if info['captured'] else "TIMEOUT"
    ax.set_title(f"{status} | {n_steps} steps{title_extra}", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_policy = load_v4_nn(
        os.path.join(PROJECT_ROOT, 'low_near/v4_tight/checkpoints/best.pth'), device)

    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy, device=device,
        world_size=30.0, catch_radius=1.0, max_steps=6000,
        subgoal_range=2.0, low_steps=50, seed=42)
    # 红线参数: v_e > v_p, a_e < a_p
    env.v_max_e = 0.65
    env.a_max_e = 0.10

    # 收集多个episode，找到捕获和未捕获的
    episodes_cap = []
    episodes_miss = []
    for i in range(40):
        data = collect_episode_with_subgoals(env, seed_offset=i)
        if data['info']['captured']:
            episodes_cap.append(data)
        else:
            episodes_miss.append(data)
        if len(episodes_cap) >= 4 and len(episodes_miss) >= 2:
            break

    print(f"Collected: {len(episodes_cap)} captured, {len(episodes_miss)} missed")

    # === 图1: 4个捕获 + 2个未捕获 ===
    n_show = min(6, len(episodes_cap) + len(episodes_miss))
    show_eps = episodes_cap[:4] + episodes_miss[:2]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, data in enumerate(show_eps[:6]):
        plot_episode_subgoals(axes[idx], data, env,
                              title_extra=f" | v_e={env.v_max_e} a_e={env.a_max_e}")

    # 隐藏多余的axes
    for idx in range(len(show_eps), 6):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'Expert Subgoal Visualization (v_e={env.v_max_e}, a_e={env.a_max_e}, '
        f'range={env.subgoal_range})\n'
        f'AUV effective speed ≈ 0.53 m/s | Circles = subgoals colored by time step',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(PROJECT_ROOT, 'high_tabular', 'expert_subgoal_vis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # === 图2: 单个episode的放大详细图 ===
    if episodes_cap:
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        data = episodes_cap[0]
        plot_episode_subgoals(ax2, data, env, title_extra=" (Detailed)")

        # 额外：在每个subgoal旁标注步数编号
        sg = data['subgoals']
        step_interval = max(1, len(sg) // 20)  # 最多标20个编号
        for i in range(0, len(sg), step_interval):
            ax2.annotate(f'{i}', (sg[i, 0], sg[i, 1]),
                        fontsize=7, color='darkgreen', fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

        fig2.suptitle('Single Episode Detail: Expert Subgoal Planning', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save2 = os.path.join(PROJECT_ROOT, 'high_tabular', 'expert_subgoal_detail.png')
        plt.savefig(save2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save2}")


if __name__ == '__main__':
    main()
