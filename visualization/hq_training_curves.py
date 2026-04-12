"""
高质量训练曲线可视化
====================
- 平滑曲线 (Savitzky-Golay / EMA)
- 置信区间阴影
- 无回折线
- 美观配色
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Optional, Tuple

# 设置全局样式
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 美观配色
COLORS = {
    'train': '#2E86AB',       # 深蓝
    'eval': '#F18F01',        # 橙色
    'reward': '#C73E1D',      # 红色
    'success': '#3A7D44',     # 绿色
    'loss': '#7B2CBF',        # 紫色
    'entropy': '#E36414',     # 橙红
    'shading': 0.2,           # 阴影透明度
}


def smooth_curve(y: np.ndarray, method: str = 'ema', window: int = 11, 
                 alpha: float = 0.1) -> np.ndarray:
    """
    平滑曲线
    
    Args:
        y: 原始数据
        method: 'ema'(指数移动平均), 'savgol'(Savitzky-Golay), 'uniform'(均值滤波)
        window: 窗口大小
        alpha: EMA的平滑系数
    """
    if len(y) < 5:
        return y
    
    y = np.array(y, dtype=float)
    
    if method == 'ema':
        smoothed = np.zeros_like(y)
        smoothed[0] = y[0]
        for i in range(1, len(y)):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    elif method == 'savgol':
        window = min(window, len(y))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return y
        return savgol_filter(y, window, polyorder=2)
    
    elif method == 'uniform':
        return uniform_filter1d(y, size=window, mode='nearest')
    
    return y


def compute_running_stats(y: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """计算滑动窗口的均值和标准差"""
    n = len(y)
    mean = np.zeros(n)
    std = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        mean[i] = np.mean(y[start:end])
        std[i] = np.std(y[start:end])
    
    return mean, std


def plot_with_shading(ax, x, y, color, label=None, smooth_method='ema', 
                      alpha_ema=0.15, show_raw=False, shade_std=True):
    """
    绘制带阴影的平滑曲线
    
    Args:
        ax: matplotlib axis
        x: x轴数据
        y: y轴数据
        color: 线条颜色
        label: 图例标签
        smooth_method: 平滑方法
        alpha_ema: EMA平滑系数
        show_raw: 是否显示原始数据点
        shade_std: 是否显示标准差阴影
    """
    y = np.array(y)
    x = np.array(x)
    
    # 平滑曲线 - 主线
    y_smooth = smooth_curve(y, method=smooth_method, alpha=alpha_ema)
    
    # 绘制主线
    line, = ax.plot(x, y_smooth, color=color, label=label, linewidth=2)
    
    # 标准差阴影
    if shade_std and len(y) > 10:
        _, std = compute_running_stats(y, window=max(5, len(y)//20))
        ax.fill_between(x, y_smooth - std, y_smooth + std, 
                       color=color, alpha=COLORS['shading'])
    
    # 原始数据点 (可选)
    if show_raw:
        ax.scatter(x, y, color=color, alpha=0.15, s=8, zorder=1)
    
    return line


def load_history(checkpoint_dir: str) -> Dict:
    """从checkpoint目录加载训练历史"""
    history_path = os.path.join(checkpoint_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return {}


def plot_online_ppo_curves(history: List[Dict], save_path: str, 
                           exp_name: str = 'Online PPO',
                           smooth_alpha: float = 0.12):
    """
    绘制高质量Online PPO训练曲线
    
    Args:
        history: 训练历史列表
        save_path: 保存路径
        exp_name: 实验名称
        smooth_alpha: 平滑系数 (越小越平滑)
    """
    if not history:
        print("Empty history, skipping plot")
        return
    
    steps = [h['steps'] for h in history]
    steps_k = [s / 1000 for s in steps]  # 转换为K
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(f'{exp_name} — Training Curves', fontsize=15, fontweight='bold', y=0.98)
    
    # ===== Row 0: 主要指标 =====
    
    # (0,0) Episode Reward
    ax = axes[0, 0]
    train_rew = [h.get('train_rew', 0) for h in history]
    plot_with_shading(ax, steps_k, train_rew, COLORS['train'], 'Train', 
                     alpha_ema=smooth_alpha)
    # Eval points
    eval_data = [(h['steps']/1000, h['eval_rew']) for h in history if h.get('eval_rew') is not None]
    if eval_data:
        ex, ey = zip(*eval_data)
        ax.scatter(ex, ey, color=COLORS['eval'], s=40, zorder=5, label='Eval', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.legend(loc='lower right')
    
    # (0,1) Success Rate
    ax = axes[0, 1]
    train_sr = [h.get('train_sr', 0) for h in history]
    plot_with_shading(ax, steps_k, train_sr, COLORS['success'], 'Train', 
                     alpha_ema=smooth_alpha)
    eval_sr = [(h['steps']/1000, h['eval_sr']) for h in history if h.get('eval_sr') is not None]
    if eval_sr:
        ex, ey = zip(*eval_sr)
        ax.scatter(ex, ey, color=COLORS['eval'], s=40, zorder=5, label='Eval', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    
    # (0,2) Episode Length
    ax = axes[0, 2]
    ep_len = [h.get('train_ep_len', 0) for h in history]
    plot_with_shading(ax, steps_k, ep_len, COLORS['train'], None, 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    
    # ===== Row 1: 奖励分项 =====
    
    # (1,0) Event Reward
    ax = axes[1, 0]
    r_event = [h.get('train_r_event', 0) for h in history]
    plot_with_shading(ax, steps_k, r_event, COLORS['reward'], 'R_event', 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Reward')
    ax.set_title('Event Reward (Catch/Fail)')
    
    # (1,1) Shaping Reward
    ax = axes[1, 1]
    r_shaping = [h.get('train_r_shaping', 0) for h in history]
    plot_with_shading(ax, steps_k, r_shaping, '#2E86AB', 'R_shaping', 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Reward')
    ax.set_title('Shaping Reward (Distance)')
    
    # (1,2) Subgoal Penalty
    ax = axes[1, 2]
    r_sg = [h.get('train_r_sg_obs', 0) for h in history]
    plot_with_shading(ax, steps_k, r_sg, '#C73E1D', 'R_sg_obs', 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Penalty')
    ax.set_title('Subgoal-Obstacle Penalty')
    
    # ===== Row 2: 训练诊断 =====
    
    # (2,0) Actor Loss
    ax = axes[2, 0]
    actor_loss = [h.get('actor_loss', 0) for h in history]
    plot_with_shading(ax, steps_k, actor_loss, COLORS['loss'], None, 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Loss')
    ax.set_title('Actor (Policy) Loss')
    
    # (2,1) Critic Loss
    ax = axes[2, 1]
    critic_loss = [h.get('critic_loss', 0) for h in history]
    plot_with_shading(ax, steps_k, critic_loss, COLORS['loss'], None, 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Loss')
    ax.set_title('Critic (Value) Loss')
    
    # (2,2) Entropy
    ax = axes[2, 2]
    entropy = [h.get('entropy', 0) for h in history]
    plot_with_shading(ax, steps_k, entropy, COLORS['entropy'], None, 
                     alpha_ema=smooth_alpha)
    ax.set_xlabel('Steps (K)')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_comparison_curves(experiments: Dict[str, List[Dict]], save_path: str,
                          metric: str = 'eval_sr', title: str = 'Success Rate Comparison',
                          ylabel: str = 'Success Rate', smooth_alpha: float = 0.15):
    """
    绘制多实验对比曲线
    
    Args:
        experiments: {name: history_list}
        save_path: 保存路径
        metric: 要对比的指标
        title: 图表标题
        ylabel: Y轴标签
        smooth_alpha: 平滑系数
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 预定义颜色
    colors = ['#2E86AB', '#F18F01', '#3A7D44', '#C73E1D', '#7B2CBF', '#E36414']
    
    for i, (name, history) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        
        steps_k = [h['steps'] / 1000 for h in history]
        values = [h.get(metric, 0) for h in history]
        
        if values:
            plot_with_shading(ax, steps_k, values, color, name, 
                            alpha_ema=smooth_alpha, shade_std=True)
    
    ax.set_xlabel('Steps (K)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    
    if 'sr' in metric.lower() or 'rate' in metric.lower():
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='High-quality training curves')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Checkpoint directory with history.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: checkpoint_dir/hq_curves.png)')
    parser.add_argument('--smooth', type=float, default=0.12,
                       help='Smoothing alpha (smaller=smoother)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name for title')
    args = parser.parse_args()
    
    # 加载历史
    history_path = os.path.join(args.checkpoint_dir, 'history.json')
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # 输出路径
    if args.output is None:
        args.output = os.path.join(args.checkpoint_dir, 'hq_training_curves.png')
    
    # 实验名称
    if args.exp_name is None:
        args.exp_name = os.path.basename(args.checkpoint_dir)
    
    # 绘制
    plot_online_ppo_curves(history, args.output, args.exp_name, args.smooth)


if __name__ == '__main__':
    main()
