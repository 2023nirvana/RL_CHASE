#!/usr/bin/env python3
"""验证地图构建和碰撞检测 - 用随机策略跑几个episode并可视化"""
import os, sys, math
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from high_tabular.hrl_v3_real_auv_fixed import (
    PursuitEnvRealAUVFixed, load_v4_nn
)

def visualize_map_only(env, save_path):
    """只画地图，展示障碍物和边界区域"""
    fig, ax = plt.subplots(figsize=(10, 10))
    half = env.half
    
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect('equal')
    
    # 边界线
    ax.plot([-half, half, half, -half, -half], 
            [-half, -half, half, half, -half], 'k--', linewidth=1.5, alpha=0.6)
    
    # 边界惩罚区域（红色）
    margin = env.boundary_margin
    ax.fill_between([-half, half], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    ax.fill_between([-half, half], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    
    # 障碍物
    for obs in env.obstacles:
        rect = Rectangle(
            (obs.cx - obs.w/2, obs.cy - obs.h/2),
            obs.w, obs.h, facecolor='gray', edgecolor='black', alpha=0.5
        )
        ax.add_patch(rect)
    
    ax.set_title(f'Map: world={env.world_size}x{env.world_size} (half={half}), '
                 f'{len(env.obstacles)} obstacles, boundary_margin={margin}')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Map saved to: {save_path}")

def visualize_random_episode(env, save_path, seed=42):
    """用随机动作跑一个episode并可视化"""
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    
    eta = env.auv_env.inner.dynamics.eta
    traj_p = [np.array([eta[0], eta[1]])]
    traj_e = [env.e.copy()]
    
    done = False
    step = 0
    info = {}
    while not done and step < 200:
        # 随机动作
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, done, info = env.step(action)
        eta = env.auv_env.inner.dynamics.eta
        traj_p.append(np.array([eta[0], eta[1]]))
        traj_e.append(env.e.copy())
        step += 1
    
    traj_p = np.array(traj_p)
    traj_e = np.array(traj_e)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    half = env.half
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect('equal')
    
    # 边界线
    ax.plot([-half, half, half, -half, -half], 
            [-half, -half, half, half, -half], 'k--', linewidth=1.5, alpha=0.6)
    
    # 边界惩罚区域
    margin = env.boundary_margin
    ax.fill_between([-half, half], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    ax.fill_between([-half, half], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    
    # 障碍物
    for obs_rect in env.obstacles:
        rect = Rectangle(
            (obs_rect.cx - obs_rect.w/2, obs_rect.cy - obs_rect.h/2),
            obs_rect.w, obs_rect.h, facecolor='gray', edgecolor='black', alpha=0.5
        )
        ax.add_patch(rect)
    
    # 轨迹
    ax.plot(traj_p[:, 0], traj_p[:, 1], 'b-', linewidth=1.5, label='Pursuer', alpha=0.7)
    ax.plot(traj_e[:, 0], traj_e[:, 1], 'r-', linewidth=1.5, label='Evader', alpha=0.7)
    
    ax.scatter(traj_p[0, 0], traj_p[0, 1], c='blue', s=100, marker='o', zorder=5, edgecolor='white', label='P Start')
    ax.scatter(traj_e[0, 0], traj_e[0, 1], c='red', s=100, marker='o', zorder=5, edgecolor='white', label='E Start')
    ax.scatter(traj_p[-1, 0], traj_p[-1, 1], c='blue', s=150, marker='*', zorder=5, edgecolor='white')
    ax.scatter(traj_e[-1, 0], traj_e[-1, 1], c='red', s=150, marker='*', zorder=5, edgecolor='white')
    
    status = "CAPTURED" if info.get('captured') else ("COLLISION" if info.get('collision') else "TIMEOUT")
    ax.set_title(f'Random Policy Ep (seed={seed}): {status}, steps={step}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Episode saved to: {save_path}")
    return info

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载底层NN
    nn_path = os.path.join(PROJECT_ROOT, "low_near/v4_tight/checkpoints/best.pth")
    nn_policy = load_v4_nn(nn_path, device)
    
    # 创建环境 (world_size=30.0 对齐tabular的L=15)
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=30.0,
        catch_radius=1.0,
        max_steps=6000,
        subgoal_range=0.5,
        low_steps=50,
        seed=42,
    )
    
    # 输出目录
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints/3d_v3_fixed_run1/verify_map_fix")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 只画地图（对比tabular的地图）
    visualize_map_only(env, os.path.join(save_dir, "map_layout.png"))
    
    # 2. 用随机策略跑3个episode看轨迹
    for i, seed in enumerate([42, 123, 456]):
        save_path = os.path.join(save_dir, f"random_ep_{i}_seed{seed}.png")
        info = visualize_random_episode(env, save_path, seed=seed)
        status = "CAPTURED" if info.get('captured') else ("COLLISION" if info.get('collision') else "TIMEOUT/OOB")
        print(f"  Episode {i} (seed={seed}): {status}")
    
    print(f"\n所有验证结果已保存到: {save_dir}")

if __name__ == '__main__':
    main()
