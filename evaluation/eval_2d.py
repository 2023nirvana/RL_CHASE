"""
2D 模型评估和可视化脚本
========================

使用方法：
    cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
    python evaluation/eval_2d.py --checkpoint checkpoints/2d_test_03/best_model.pth
    
    # 更多 episode
    python evaluation/eval_2d.py --checkpoint checkpoints/2d_test_03/best_model.pth --episodes 20
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d import LowLevelAUV2DEnv


class PolicyEvaluator:
    """策略评估器"""
    
    def __init__(self, checkpoint_path: str, target_distance: float = 5.0, 
                 min_target_distance: float = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_distance = target_distance
        self.min_target_distance = min_target_distance or target_distance * 0.5
        
        # 创建环境
        self.env = LowLevelAUV2DEnv(
            target_distance=target_distance,
            min_target_distance=self.min_target_distance,
            use_simplified_obs=True,
            enable_domain_randomization=False
        )
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        
        # 加载模型
        self._load_model(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Target distance: {self.min_target_distance}m ~ {self.target_distance}m")
    
    def _load_model(self, checkpoint_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 重建网络结构（与训练时一致）
        hidden_sizes = [64, 64]
        
        actor_layers = []
        prev_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(torch.nn.Linear(prev_dim, hidden_size))
            actor_layers.append(torch.nn.ReLU())
            prev_dim = hidden_size
        actor_layers.append(torch.nn.Linear(prev_dim, self.num_actions))
        self.actor = torch.nn.Sequential(*actor_layers).to(self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
    
    def get_action(self, obs: np.ndarray, deterministic: bool = True):
        """获取动作"""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            return action, probs.cpu().numpy()[0]
    
    def run_episode(self, render: bool = False, verbose: bool = True):
        """运行一个 episode"""
        obs, info = self.env.reset()
        
        trajectory = {
            'positions': [],
            'target': info['target_pos'],
            'actions': [],
            'rewards': [],
            'obs': []
        }
        
        init_pos = self.env.inner.dynamics.eta[:2].copy()
        trajectory['positions'].append(init_pos.tolist())
        
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, probs = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # 记录轨迹
            pos = self.env.inner.dynamics.eta[:2].copy()
            trajectory['positions'].append(pos.tolist())
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['obs'].append(obs.tolist())
            
            if verbose and step % 20 == 0:
                print(f"  Step {step}: pos_error={info.get('pos_error', 0):.3f}m, action={self.env.ACTION_NAMES[action]}")
        
        success = info.get('termination_reason') == 'success'
        
        result = {
            'success': success,
            'steps': step,
            'total_reward': total_reward,
            'final_pos_error': info.get('pos_error', 0),
            'termination_reason': info.get('termination_reason', 'unknown'),
            'trajectory': trajectory
        }
        
        if verbose:
            print(f"\nEpisode finished:")
            print(f"  Success: {success}")
            print(f"  Steps: {step}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Final Position Error: {result['final_pos_error']:.3f}m")
            print(f"  Termination: {result['termination_reason']}")
        
        return result
    
    def evaluate(self, n_episodes: int = 10, verbose: bool = True):
        """评估多个 episode"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating {n_episodes} episodes...")
        print(f"{'='*60}\n")
        
        for ep in range(n_episodes):
            print(f"\n--- Episode {ep+1}/{n_episodes} ---")
            result = self.run_episode(verbose=verbose)
            results.append(result)
        
        # 统计
        successes = sum(1 for r in results if r['success'])
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_error = np.mean([r['final_pos_error'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({n_episodes} episodes)")
        print(f"{'='*60}")
        print(f"  Success Rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Average Final Error: {avg_error:.3f}m")
        
        return results
    
    def plot_trajectory(self, result: dict, save_path: str = None, title_prefix: str = ""):
        """绘制单个轨迹"""
        trajectory = result['trajectory']
        positions = np.array(trajectory['positions'])
        target = np.array(trajectory['target'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：2D 轨迹
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
        ax1.plot(target[0], target[1], 'r*', markersize=15, label='Target')
        
        # 绘制成功范围（0.15m）
        circle = plt.Circle((target[0], target[1]), 0.15, color='green', fill=False, 
                           linestyle='--', linewidth=2, label='Success Zone (0.15m)')
        ax1.add_patch(circle)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        title = f'{title_prefix}2D V1 Trajectory ({"Success" if result["success"] else "Fail"})'
        ax1.set_title(title, fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 右图：奖励和位置误差
        ax2 = axes[1]
        steps = range(len(trajectory['rewards']))
        
        # 计算每步的位置误差
        pos_errors = [np.linalg.norm(np.array(p) - target) for p in positions[1:]]
        
        ax2_twin = ax2.twinx()
        
        line1, = ax2.plot(steps, trajectory['rewards'], 'b-', label='Reward')
        line2, = ax2_twin.plot(steps, pos_errors, 'r-', label='Position Error')
        
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Reward', color='blue', fontsize=12)
        ax2_twin.set_ylabel('Position Error (m)', color='red', fontsize=12)
        ax2.set_title('Reward and Position Error over Time', fontsize=14)
        
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # 添加成功线（0.15m）
        ax2_twin.axhline(y=0.15, color='green', linestyle='--', label='Success Threshold')
        
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close(fig)
        return fig
    
    def plot_multiple_trajectories(self, results: list, save_path: str = None, title_prefix: str = ""):
        """绘制多个轨迹"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            trajectory = result['trajectory']
            positions = np.array(trajectory['positions'])
            target = np.array(trajectory['target'])
            
            color = 'green' if result['success'] else 'red'
            alpha = 0.7 if result['success'] else 0.3
            
            ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=alpha, linewidth=1.5)
            ax.plot(positions[0, 0], positions[0, 1], 'o', color=color, markersize=6)
            ax.plot(target[0], target[1], '*', color='gold', markersize=12)
        
        # 统计
        successes = sum(1 for r in results if r['success'])
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = f'{title_prefix}Multiple Trajectories (Success: {successes}/{len(results)} = {successes/len(results):.1%})'
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=2, label='Success'),
            Line2D([0], [0], color='red', linewidth=2, alpha=0.3, label='Fail'),
            Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='', label='Target')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close(fig)
        return fig


def main():
    parser = argparse.ArgumentParser(description="2D V1 Model Evaluation (4 actions)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--target-distance", type=float, default=5.0, help="Max target distance (m)")
    parser.add_argument("--min-target-distance", type=float, default=None, help="Min target distance (m)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save plots to files")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--output-prefix", type=str, default="", help="Output filename prefix")
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = PolicyEvaluator(
        args.checkpoint, 
        target_distance=args.target_distance,
        min_target_distance=args.min_target_distance
    )
    
    # 运行评估
    results = evaluator.evaluate(n_episodes=args.episodes, verbose=args.verbose)
    
    # 绘图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "visualization/eval_plots")
    os.makedirs(save_dir, exist_ok=True)
    
    prefix = args.output_prefix + "_" if args.output_prefix else ""
    
    # 绘制每个轨迹
    for i, result in enumerate(results):
        save_path = os.path.join(save_dir, f"{prefix}trajectory_{timestamp}_{i}.png") if args.save_plots else None
        evaluator.plot_trajectory(result, save_path, title_prefix=f"[{prefix.rstrip('_')}] " if prefix else "")
    
    # 绘制所有轨迹
    save_path = os.path.join(save_dir, f"{prefix}all_trajectories_{timestamp}.png") if args.save_plots else None
    evaluator.plot_multiple_trajectories(results, save_path, title_prefix=f"[{prefix.rstrip('_')}] " if prefix else "")
    
    print(f"\nPlots saved to {save_dir}")


if __name__ == "__main__":
    main()
