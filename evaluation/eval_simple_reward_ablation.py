"""
评估简化奖励消融实验模型
==========================

评估 ablation_sparse_only_20260203_173518 模型
生成评估报告、对比图、轨迹图等
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_simple_reward import LowLevelAUV2DSimpleRewardEnv


def load_actor(checkpoint_path: str, obs_dim: int, num_actions: int, hidden_sizes=[128, 64], device='cuda'):
    """加载Actor网络"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    device = torch.device(device)
    
    layers = []
    prev = obs_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, num_actions))
    
    actor = nn.Sequential(*layers).to(device)
    
    # 尝试加载actor
    if 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'])
    elif 'network' in ckpt:
        # 可能是整个网络，需要提取actor部分
        actor.load_state_dict(ckpt['network'])
    else:
        # 直接是state_dict
        actor.load_state_dict(ckpt)
    
    actor.eval()
    return actor


def evaluate_model(checkpoint_path: str, reward_type: str = 'sparse_only', 
                   n_episodes: int = 20, target_distance: float = 3.0, 
                   device: str = 'cuda'):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {os.path.basename(os.path.dirname(checkpoint_path))}")
    print(f"Reward Type: {reward_type}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")
    
    # 创建环境
    env = LowLevelAUV2DSimpleRewardEnv(
        reward_type=reward_type,
        target_distance=target_distance
    )
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # 加载模型
    actor = load_actor(checkpoint_path, obs_dim, num_actions, device=device)
    if actor is None:
        return None
    
    device = torch.device(device)
    results = {
        'model': os.path.basename(os.path.dirname(checkpoint_path)),
        'reward_type': reward_type,
        'episodes': [],
        'trajectories': []
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        trajectory = {
            'positions': [],
            'headings': [],  # 新增：保存朝向信息
            'target': info.get('target_pos', [0, 0]),
            'distances': [],
            'rewards': []
        }
        
        # 获取初始位置和朝向
        if hasattr(env, 'inner'):
            initial_pos = env.inner.dynamics.eta[:2].copy()
            initial_yaw = env.inner.dynamics.eta[5]
            trajectory['positions'].append(initial_pos.tolist())
            trajectory['headings'].append(initial_yaw)
            initial_dist = np.linalg.norm(initial_pos - np.array(trajectory['target']))
            trajectory['distances'].append(initial_dist)
        
        done, total_r, steps = False, 0, 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
            
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            total_r += r
            steps += 1
            
            if hasattr(env, 'inner'):
                pos = env.inner.dynamics.eta[:2].copy()
                yaw = env.inner.dynamics.eta[5]
                trajectory['positions'].append(pos.tolist())
                trajectory['headings'].append(yaw)
                dist = np.linalg.norm(pos - np.array(trajectory['target']))
                trajectory['distances'].append(dist)
                trajectory['rewards'].append(r)
        
        success = info.get('termination_reason') == 'success'
        results['episodes'].append({
            'success': success,
            'steps': steps,
            'reward': total_r,
            'final_error': info.get('pos_error', trajectory['distances'][-1] if trajectory['distances'] else 0)
        })
        results['trajectories'].append(trajectory)
        
        status = '✓' if success else '✗'
        print(f"  Ep {ep+1:2d}: {status}, steps={steps:4d}, reward={total_r:8.1f}, "
              f"final_error={results['episodes'][-1]['final_error']:.3f}m")
    
    # 统计
    successes = sum(1 for e in results['episodes'] if e['success'])
    results['summary'] = {
        'success_rate': successes / n_episodes,
        'avg_reward': np.mean([e['reward'] for e in results['episodes']]),
        'avg_steps': np.mean([e['steps'] for e in results['episodes']]),
        'avg_final_error': np.mean([e['final_error'] for e in results['episodes']]),
        'std_reward': np.std([e['reward'] for e in results['episodes']]),
        'std_steps': np.std([e['steps'] for e in results['episodes']]),
        'total_episodes': n_episodes
    }
    
    print(f"\n  Summary:")
    print(f"    Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"    Avg Reward: {results['summary']['avg_reward']:.2f} ± {results['summary']['std_reward']:.2f}")
    print(f"    Avg Steps: {results['summary']['avg_steps']:.0f} ± {results['summary']['std_steps']:.0f}")
    print(f"    Avg Final Error: {results['summary']['avg_final_error']:.3f}m")
    
    return results


def plot_trajectory_dual(trajectory: Dict, episode: Dict, model_name: str, 
                         ep_idx: int, save_path: str):
    """绘制dual图：轨迹图 + 距离曲线，带朝向箭头"""
    positions = np.array(trajectory['positions'])
    headings = np.array(trajectory.get('headings', []))
    target = np.array(trajectory['target'])
    distances = np.array(trajectory['distances'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：轨迹
    ax1 = axes[0]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='AUV Trajectory', alpha=0.7)
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
    ax1.plot(target[0], target[1], 'r*', markersize=15, label='Target')
    
    # 添加朝向箭头（黄色）
    if len(headings) > 0:
        # 每隔一定步数绘制一个箭头，避免太密集
        n_arrows = min(15, len(positions) // 20 + 1)
        if n_arrows > 0 and len(positions) > 1:
            indices = np.linspace(0, len(positions)-1, n_arrows, dtype=int)
            arrow_len = 0.3  # 箭头长度
            for i, idx in enumerate(indices):
                if idx < len(headings):
                    x, y = positions[idx]
                    yaw = headings[idx]
                    dx = arrow_len * np.cos(yaw)
                    dy = arrow_len * np.sin(yaw)
                    ax1.arrow(x, y, dx, dy, head_width=0.08, head_length=0.05, 
                             fc='gold', ec='orange', alpha=0.8, zorder=5)
            # 添加图例项
            ax1.plot([], [], color='gold', marker='>', markersize=8, linestyle='', label='Heading')
    
    # 成功圈
    ax1.add_patch(plt.Circle((target[0], target[1]), 0.3, 
                              color='green', fill=False, linestyle='--', lw=2, label='Success (0.3m)'))
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    status = "Success" if episode['success'] else "Fail"
    ax1.set_title(f'{model_name} - Episode {ep_idx+1} ({status})')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：距离曲线
    ax2 = axes[1]
    steps = range(len(distances))
    ax2.plot(steps, distances, 'r-', lw=2, label='Distance to Target')
    ax2.axhline(y=0.3, color='green', linestyle='--', lw=2, label='Success Threshold (0.3m)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title(f'Distance to Target (Final: {distances[-1]:.3f}m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_all_trajectories(results: Dict, save_dir: str, max_individual: int = 5):
    """绘制所有轨迹图，带朝向箭头"""
    os.makedirs(save_dir, exist_ok=True)
    model_name = results['model']
    
    # 保存前几个episode的dual图
    for i in range(min(max_individual, len(results['trajectories']))):
        traj = results['trajectories'][i]
        ep = results['episodes'][i]
        save_path = os.path.join(save_dir, f"{model_name}_ep{i+1}_dual.png")
        plot_trajectory_dual(traj, ep, model_name, i, save_path)
    
    # 汇总图：所有轨迹
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i, (traj, ep) in enumerate(zip(results['trajectories'], results['episodes'])):
        pos = np.array(traj['positions'])
        headings = np.array(traj.get('headings', []))
        target = np.array(traj['target'])
        color = 'green' if ep['success'] else 'red'
        alpha = 0.7 if ep['success'] else 0.3
        ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
        ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6, alpha=alpha)
        
        # 添加朝向箭头（每条轨迹选几个点）
        if len(headings) > 0:
            n_arrows = min(5, len(pos) // 30 + 1)
            if n_arrows > 0 and len(pos) > 1:
                indices = np.linspace(0, len(pos)-1, n_arrows, dtype=int)
                arrow_len = 0.2
                for idx in indices:
                    if idx < len(headings):
                        x, y = pos[idx]
                        yaw = headings[idx]
                        dx = arrow_len * np.cos(yaw)
                        dy = arrow_len * np.sin(yaw)
                        ax.arrow(x, y, dx, dy, head_width=0.06, head_length=0.04, 
                                fc='gold', ec='orange', alpha=0.6, zorder=5)
    
    # 目标点（取第一个episode的）
    if results['trajectories']:
        target = np.array(results['trajectories'][0]['target'])
        ax.plot(target[0], target[1], '*', color='gold', markersize=15, label='Target')
        ax.add_patch(plt.Circle((target[0], target[1]), 0.3, 
                                color='green', fill=False, linestyle='--', lw=2, label='Success (0.3m)'))
    
    successes = sum(1 for e in results['episodes'] if e['success'])
    n_eps = len(results['episodes'])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{model_name}\nAll Trajectories (Success: {successes}/{n_eps} = {successes/n_eps:.1%})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='green', lw=2, label='Success'),
        Line2D([0], [0], color='red', lw=2, alpha=0.3, label='Fail'),
        Line2D([0], [0], marker='*', color='gold', markersize=15, linestyle='', label='Target'),
        Line2D([0], [0], marker='>', color='gold', markersize=8, linestyle='', label='Heading')
    ])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_all_trajectories.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_report(results: Dict, save_path: str):
    """生成评估报告"""
    lines = [
        "=" * 80,
        "MODEL EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        f"Model: {results['model']}",
        f"Reward Type: {results['reward_type']}",
        f"Total Episodes: {results['summary']['total_episodes']}",
        "",
        "=" * 80,
        "SUMMARY STATISTICS",
        "=" * 80,
        "",
        f"Success Rate: {results['summary']['success_rate']:.1%}",
        f"Average Reward: {results['summary']['avg_reward']:.2f} ± {results['summary']['std_reward']:.2f}",
        f"Average Steps: {results['summary']['avg_steps']:.0f} ± {results['summary']['std_steps']:.0f}",
        f"Average Final Error: {results['summary']['avg_final_error']:.3f}m",
        "",
        "=" * 80,
        "EPISODE DETAILS",
        "=" * 80,
        "",
        f"{'Episode':<10} {'Success':<10} {'Steps':<10} {'Reward':<15} {'Final Error (m)':<15}",
        "-" * 80
    ]
    
    for i, ep in enumerate(results['episodes']):
        status = "✓" if ep['success'] else "✗"
        lines.append(f"{i+1:<10} {status:<10} {ep['steps']:<10} {ep['reward']:<15.2f} {ep['final_error']:<15.3f}")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    print("\n" + report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {save_path}")
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Simple Reward Ablation Model")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/ablation_sparse_only_20260203_173518/best_model.pth",
                       help="Path to checkpoint")
    parser.add_argument("--reward-type", type=str, default="sparse_only",
                       choices=['sparse_only', 'distance_linear', 'distance_exp', 
                                'heading_only', 'minimal', 'shaping'],
                       help="Reward type")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--target-distance", type=float, default=3.0, help="Target distance")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    
    # 评估
    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        reward_type=args.reward_type,
        n_episodes=args.episodes,
        target_distance=args.target_distance,
        device=args.device
    )
    
    if results is None:
        print("Evaluation failed!")
        return
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "visualization", 
                                 f"simple_reward_eval_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告
    report_path = os.path.join(output_dir, f"{results['model']}_report.txt")
    generate_report(results, report_path)
    
    # 绘制轨迹图
    plot_all_trajectories(results, output_dir, max_individual=5)
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, f"{results['model']}_results.json")
    json_results = {k: v for k, v in results.items() if k != 'trajectories'}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nJSON results saved to {json_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ All outputs saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
