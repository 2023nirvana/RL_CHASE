"""
高质量训练曲线可视化器
======================

生成平滑、美观的训练曲线图，支持：
- 移动平均平滑
- 置信区间
- 高DPI输出
- 多种配色方案
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

# 设置中文字体支持和高质量输出
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 配色方案
COLORS = {
    'train': '#2196F3',      # 蓝色
    'eval': '#FF5722',       # 橙红色
    'success': '#4CAF50',    # 绿色
    'reward': '#9C27B0',     # 紫色
    'loss': '#F44336',       # 红色
    'entropy': '#00BCD4',    # 青色
    'event': '#8BC34A',      # 亮绿
    'shaping': '#3F51B5',    # 靛蓝
    'sg_obs': '#E91E63',     # 粉红
    'time': '#795548',       # 棕色
}


def smooth_curve(data: np.ndarray, sigma: float = 2.0, mode: str = 'gaussian') -> np.ndarray:
    """平滑曲线
    
    Args:
        data: 原始数据
        sigma: 高斯滤波标准差 (gaussian模式) 或窗口大小 (moving_avg模式)
        mode: 'gaussian' 或 'moving_avg'
    """
    if len(data) < 3:
        return data
    
    if mode == 'gaussian':
        return gaussian_filter1d(data, sigma=sigma)
    elif mode == 'moving_avg':
        window = int(sigma)
        if window < 2:
            return data
        kernel = np.ones(window) / window
        # 边界处理
        smoothed = np.convolve(data, kernel, mode='same')
        # 修正边界
        for i in range(window // 2):
            smoothed[i] = np.mean(data[:i+window//2+1])
            smoothed[-(i+1)] = np.mean(data[-(i+window//2+1):])
        return smoothed
    else:
        return data


def plot_with_confidence(ax, x, y, color, label, sigma=2.0, alpha=0.15, show_raw=True):
    """绘制带置信区间的平滑曲线"""
    y_smooth = smooth_curve(np.array(y), sigma=sigma)
    
    if show_raw:
        ax.plot(x, y, color=color, alpha=0.15, lw=0.5)
    
    ax.plot(x, y_smooth, color=color, lw=2.0, label=label)
    
    # 计算局部标准差作为置信区间
    window = max(5, len(y) // 50)
    y_std = np.zeros_like(y_smooth)
    for i in range(len(y)):
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2 + 1)
        y_std[i] = np.std(y[start:end])
    
    ax.fill_between(x, y_smooth - y_std, y_smooth + y_std, 
                    color=color, alpha=alpha, linewidth=0)


def load_history(checkpoint_dir: str) -> List[Dict]:
    """加载训练历史"""
    history_path = os.path.join(checkpoint_dir, "history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def plot_enhanced_training_curves(
    history: List[Dict],
    output_path: str,
    experiment_name: str = "Online PPO",
    smooth_sigma: float = 3.0,
):
    """绘制增强版训练曲线
    
    布局: 3x3
    - [0,0] Episode Reward
    - [0,1] Success Rate
    - [0,2] Episode Length
    - [1,0] Reward Components (stacked)
    - [1,1] Actor/Critic Loss
    - [1,2] Entropy
    - [2,0] Event Reward
    - [2,1] Shaping Reward
    - [2,2] Subgoal Penalty
    """
    steps = np.array([h["steps"] for h in history]) / 1e6  # 转换为百万
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    # ===== Row 0: 主要指标 =====
    
    # [0,0] Episode Reward
    ax = axes[0, 0]
    train_rew = [h["train_rew"] for h in history]
    plot_with_confidence(ax, steps, train_rew, COLORS['train'], 'Train', sigma=smooth_sigma)
    
    # Eval points
    eval_steps = []
    eval_rews = []
    for h in history:
        if h.get("eval_rew") is not None:
            eval_steps.append(h["steps"] / 1e6)
            eval_rews.append(h["eval_rew"])
    if eval_rews:
        ax.scatter(eval_steps, eval_rews, c=COLORS['eval'], s=50, zorder=5, 
                   marker='o', edgecolors='white', linewidths=1, label='Eval')
        # 连接eval点的平滑线
        if len(eval_rews) > 2:
            eval_smooth = smooth_curve(np.array(eval_rews), sigma=1.5)
            ax.plot(eval_steps, eval_smooth, color=COLORS['eval'], lw=1.5, 
                    linestyle='--', alpha=0.7)
    
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    
    # [0,1] Success Rate
    ax = axes[0, 1]
    train_sr = [h["train_sr"] * 100 for h in history]  # 转换为百分比
    plot_with_confidence(ax, steps, train_sr, COLORS['success'], 'Train', sigma=smooth_sigma)
    
    # Eval success rate
    eval_sr_steps = []
    eval_sr_vals = []
    for h in history:
        if h.get("eval_sr") is not None:
            eval_sr_steps.append(h["steps"] / 1e6)
            eval_sr_vals.append(h["eval_sr"] * 100)
    if eval_sr_vals:
        ax.scatter(eval_sr_steps, eval_sr_vals, c=COLORS['eval'], s=50, zorder=5,
                   marker='o', edgecolors='white', linewidths=1, label='Eval')
        if len(eval_sr_vals) > 2:
            eval_sr_smooth = smooth_curve(np.array(eval_sr_vals), sigma=1.5)
            ax.plot(eval_sr_steps, eval_sr_smooth, color=COLORS['eval'], lw=1.5,
                    linestyle='--', alpha=0.7)
    
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95% target')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    
    # [0,2] Episode Length
    ax = axes[0, 2]
    ep_len = [h["train_ep_len"] for h in history]
    plot_with_confidence(ax, steps, ep_len, COLORS['train'], 'Episode Steps', sigma=smooth_sigma)
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.set_xlim(left=0)
    
    # ===== Row 1: Loss和Policy指标 =====
    
    # [1,0] Reward Components (堆叠区域图)
    ax = axes[1, 0]
    r_event = np.array([h.get("train_r_event", 0) for h in history])
    r_shaping = np.array([h.get("train_r_shaping", 0) for h in history])
    r_sg_obs = np.array([h.get("train_r_sg_obs", 0) for h in history])
    r_time = np.array([h.get("train_r_time", 0) for h in history])
    
    # 平滑处理
    r_event_s = smooth_curve(r_event, sigma=smooth_sigma)
    r_shaping_s = smooth_curve(r_shaping, sigma=smooth_sigma)
    r_sg_obs_s = smooth_curve(r_sg_obs, sigma=smooth_sigma)
    r_time_s = smooth_curve(r_time, sigma=smooth_sigma)
    
    ax.plot(steps, r_event_s, color=COLORS['event'], lw=2, label='Event')
    ax.plot(steps, r_shaping_s, color=COLORS['shaping'], lw=2, label='Shaping')
    ax.plot(steps, r_sg_obs_s, color=COLORS['sg_obs'], lw=2, label='SG Penalty')
    ax.plot(steps, r_time_s, color=COLORS['time'], lw=2, label='Time')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Reward Component')
    ax.set_title('Reward Breakdown')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.set_xlim(left=0)
    
    # [1,1] Actor/Critic Loss
    ax = axes[1, 1]
    actor_loss = [h.get("actor_loss", 0) for h in history]
    critic_loss = [h.get("critic_loss", 0) for h in history]
    
    actor_s = smooth_curve(np.array(actor_loss), sigma=smooth_sigma)
    critic_s = smooth_curve(np.array(critic_loss), sigma=smooth_sigma)
    
    ax.plot(steps, actor_s, color=COLORS['reward'], lw=2, label='Actor Loss')
    ax2 = ax.twinx()
    ax2.plot(steps, critic_s, color=COLORS['loss'], lw=2, label='Critic Loss')
    
    ax.set_ylabel('Actor Loss', color=COLORS['reward'])
    ax2.set_ylabel('Critic Loss', color=COLORS['loss'])
    ax.set_title('Policy Loss')
    ax.tick_params(axis='y', labelcolor=COLORS['reward'])
    ax2.tick_params(axis='y', labelcolor=COLORS['loss'])
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_xlim(left=0)
    
    # [1,2] Entropy
    ax = axes[1, 2]
    entropy = [h.get("entropy", 0) for h in history]
    entropy_s = smooth_curve(np.array(entropy), sigma=smooth_sigma)
    ax.plot(steps, entropy_s, color=COLORS['entropy'], lw=2)
    ax.fill_between(steps, entropy_s * 0.98, entropy_s, color=COLORS['entropy'], alpha=0.3)
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.set_xlim(left=0)
    
    # ===== Row 2: 奖励分项详细 =====
    
    # [2,0] Event Reward
    ax = axes[2, 0]
    plot_with_confidence(ax, steps, r_event, COLORS['event'], 'Train', sigma=smooth_sigma)
    
    eval_event_steps = []
    eval_event_vals = []
    for h in history:
        if h.get("eval_r_event") is not None:
            eval_event_steps.append(h["steps"] / 1e6)
            eval_event_vals.append(h["eval_r_event"])
    if eval_event_vals:
        ax.scatter(eval_event_steps, eval_event_vals, c=COLORS['eval'], s=30, 
                   zorder=5, marker='o', alpha=0.8, label='Eval')
    
    ax.set_xlabel('Steps (M)')
    ax.set_ylabel('Event Reward')
    ax.set_title('Event Reward (Catch/Fail)')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    
    # [2,1] Shaping Reward
    ax = axes[2, 1]
    plot_with_confidence(ax, steps, r_shaping, COLORS['shaping'], 'Train', sigma=smooth_sigma)
    
    eval_shaping_steps = []
    eval_shaping_vals = []
    for h in history:
        if h.get("eval_r_shaping") is not None:
            eval_shaping_steps.append(h["steps"] / 1e6)
            eval_shaping_vals.append(h["eval_r_shaping"])
    if eval_shaping_vals:
        ax.scatter(eval_shaping_steps, eval_shaping_vals, c=COLORS['eval'], s=30,
                   zorder=5, marker='o', alpha=0.8, label='Eval')
    
    ax.set_xlabel('Steps (M)')
    ax.set_ylabel('Shaping Reward')
    ax.set_title('PBRS Distance Shaping')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    
    # [2,2] Subgoal Penalty
    ax = axes[2, 2]
    plot_with_confidence(ax, steps, r_sg_obs, COLORS['sg_obs'], 'Train', sigma=smooth_sigma)
    
    eval_sg_steps = []
    eval_sg_vals = []
    for h in history:
        if h.get("eval_r_sg_obs") is not None:
            eval_sg_steps.append(h["steps"] / 1e6)
            eval_sg_vals.append(h["eval_r_sg_obs"])
    if eval_sg_vals:
        ax.scatter(eval_sg_steps, eval_sg_vals, c=COLORS['eval'], s=30,
                   zorder=5, marker='o', alpha=0.8, label='Eval')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Steps (M)')
    ax.set_ylabel('Subgoal Penalty')
    ax.set_title('Subgoal/Path Quality Penalty')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    
    # 总标题
    fig.suptitle(f'{experiment_name} Training Curves', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved enhanced training curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot enhanced training curves")
    parser.add_argument("--checkpoint-dir", "-c", type=str, required=True,
                        help="Checkpoint directory containing history.json")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: <checkpoint_dir>/training_curves_enhanced.png)")
    parser.add_argument("--name", "-n", type=str, default="Online PPO",
                        help="Experiment name for title")
    parser.add_argument("--sigma", "-s", type=float, default=3.0,
                        help="Smoothing sigma (higher = smoother)")
    args = parser.parse_args()
    
    history = load_history(args.checkpoint_dir)
    
    if args.output is None:
        args.output = os.path.join(args.checkpoint_dir, "training_curves_enhanced.png")
    
    plot_enhanced_training_curves(history, args.output, args.name, args.sigma)


if __name__ == "__main__":
    main()
