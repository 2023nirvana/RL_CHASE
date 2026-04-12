"""
新模型评估和可视化脚本 - 为最近训练的模型生成轨迹图和GIF
================================================================

支持的模型类型：
1. sparse_heading - 改进版稀疏奖励（稀疏+朝向）
2. curriculum - 课程学习
3. obstacle - 避障训练

使用方法：
    # 评估所有新模型
    python evaluation/eval_new_models.py --all
    
    # 评估特定模型
    python evaluation/eval_new_models.py --model sparse_heading_10m
    
    # 生成GIF
    python evaluation/eval_new_models.py --all --save-gif
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_simple_reward import LowLevelAUV2DSimpleRewardEnv
from core.environment.low_level_env_2d_obstacle import LowLevelAUV2DObstacleEnv


# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    'sparse_heading_10m': {
        'checkpoint': 'checkpoints/sparse_heading_10m/best_model.pth',
        'env_type': 'simple_reward',
        'env_kwargs': {
            'reward_type': 'sparse_heading',
            'target_distance': 10.0,
            'goal_reward': 100.0,
            'success_threshold': 0.3,
            'max_steps': 2000
        },
        'description': 'Sparse Heading (10m)'
    },
    'obstacle_5m_from_shaping': {
        'checkpoint': 'checkpoints/obstacle_5m_from_shaping/best_model.pth',
        'env_type': 'obstacle',
        'env_kwargs': {
            'target_distance': 5.0,
            'num_obstacles': 5,
            'goal_reward': 100.0,
            'success_threshold': 0.3,
            'max_steps': 2000
        },
        'description': 'Obstacle (5m, from shaping)'
    },
    'obstacle_5m_5obs': {
        'checkpoint': 'checkpoints/obstacle_5m_5obs/best_model.pth',
        'env_type': 'obstacle',
        'env_kwargs': {
            'target_distance': 5.0,
            'num_obstacles': 5,
            'goal_reward': 100.0,
            'success_threshold': 0.3,
            'max_steps': 2000
        },
        'description': 'Obstacle (5m, 5 obstacles)'
    },
    'curriculum_1to10m': {
        'checkpoint': 'checkpoints/curriculum_1to10m/best_model.pth',
        'env_type': 'simple_reward',
        'env_kwargs': {
            'reward_type': 'sparse_heading',  # 课程学习通常使用sparse_heading
            'target_distance': 10.0,  # 评估时使用最大距离
            'goal_reward': 100.0,
            'success_threshold': 0.3,
            'max_steps': 2000
        },
        'description': 'Curriculum (1m to 10m)'
    },
    'obstacle_5m_curriculum': {
        'checkpoint': 'checkpoints/obstacle_5m_curriculum/best_model.pth',
        'env_type': 'obstacle',
        'env_kwargs': {
            'target_distance': 5.0,
            'num_obstacles': 5,
            'goal_reward': 100.0,
            'success_threshold': 0.3,
            'max_steps': 2000
        },
        'description': 'Obstacle Curriculum (5m)'
    }
}


# ==================== 网络定义 ====================
class MLPActor(nn.Module):
    """MLP Actor网络（与训练脚本一致）"""
    def __init__(self, obs_dim, num_actions, hidden_sizes=[256, 128]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.actor = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.actor(obs)


# ==================== 评估器 ====================
class NewModelEvaluator:
    """新模型评估器"""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Evaluator initialized on {self.device}")
    
    def _create_env(self, env_type: str, env_kwargs: Dict):
        """创建环境"""
        if env_type == 'simple_reward':
            return LowLevelAUV2DSimpleRewardEnv(**env_kwargs)
        elif env_type == 'obstacle':
            return LowLevelAUV2DObstacleEnv(**env_kwargs)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
    
    def _load_model(self, checkpoint_path: str, obs_dim: int, num_actions: int):
        """加载模型"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # 尝试不同的checkpoint格式
        if 'actor' in ckpt:
            actor_state = ckpt['actor']
        elif 'network' in ckpt:
            actor_state = ckpt['network']
        else:
            actor_state = ckpt
        
        actor = MLPActor(obs_dim, num_actions).to(self.device)
        
        # 如果state_dict的键不包含'actor.'前缀，需要适配
        if isinstance(actor_state, dict):
            # 检查键的格式
            first_key = list(actor_state.keys())[0] if actor_state else None
            if first_key and not first_key.startswith('actor.'):
                # 直接使用，因为我们的网络结构是Sequential，键应该是'0.weight', '0.bias'等
                actor.load_state_dict(actor_state, strict=False)
            else:
                # 如果键有'actor.'前缀，需要去掉
                new_state = {k.replace('actor.', ''): v for k, v in actor_state.items()}
                actor.load_state_dict(new_state, strict=False)
        else:
            actor.load_state_dict(actor_state, strict=False)
        
        actor.eval()
        return actor
    
    def evaluate_model(self, model_name: str, config: Dict, n_episodes: int = 10, 
                      verbose: bool = True) -> Dict:
        """评估单个模型"""
        checkpoint_path = os.path.join(PROJECT_ROOT, config['checkpoint'])
        if not os.path.exists(checkpoint_path):
            print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
            return {'model': model_name, 'error': 'checkpoint_not_found'}
        
        # 创建环境
        env = self._create_env(config['env_type'], config['env_kwargs'])
        obs_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        # 加载模型
        actor = self._load_model(checkpoint_path, obs_dim, num_actions)
        
        results = {
            'model': model_name,
            'description': config['description'],
            'episodes': [],
            'trajectories': []
        }
        
        print(f"\n>>> Evaluating: {config['description']}")
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            trajectory = {
                'positions': [],
                'yaw_angles': [],
                'target': info.get('target_pos', None),
                'obstacles': getattr(env, 'obstacles', None),  # 避障环境的障碍物
                'actions': [],
                'rewards': []
            }
            trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            trajectory['yaw_angles'].append(env.inner.dynamics.eta[5])
            
            done, total_reward, steps = False, 0, 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(probs, dim=-1).item()
                
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                total_reward += reward
                steps += 1
                
                trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
                trajectory['yaw_angles'].append(env.inner.dynamics.eta[5])
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                
                if verbose and steps % 200 == 0:
                    pos_error = info.get('pos_error', 0)
                    print(f"    Step {steps}: pos_error={pos_error:.3f}m, reward={reward:.2f}")
            
            success = info.get('termination_reason') == 'success'
            results['episodes'].append({
                'success': success,
                'steps': steps,
                'reward': total_reward,
                'final_error': info.get('pos_error', 0)
            })
            results['trajectories'].append(trajectory)
            
            if verbose:
                status = "✓" if success else "✗"
                print(f"    Ep {ep+1}: {status}, steps={steps}, reward={total_reward:.1f}, error={info.get('pos_error', 0):.3f}m")
        
        # 计算统计
        successes = sum(1 for e in results['episodes'] if e['success'])
        results['summary'] = {
            'success_rate': successes / n_episodes,
            'avg_reward': np.mean([e['reward'] for e in results['episodes']]),
            'avg_steps': np.mean([e['steps'] for e in results['episodes']]),
            'avg_final_error': np.mean([e['final_error'] for e in results['episodes']])
        }
        
        print(f"  Summary: Success {successes}/{n_episodes} = {successes/n_episodes:.1%}")
        
        return results
    
    def plot_trajectory(self, result: Dict, ep_idx: int, save_path: str = None):
        """绘制单条轨迹图"""
        traj = result['trajectories'][ep_idx]
        episode = result['episodes'][ep_idx]
        positions = np.array(traj['positions'])
        yaw = np.array(traj['yaw_angles'])
        target = np.array(traj['target']) if traj['target'] is not None else None
        obstacles = traj.get('obstacles', None)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：轨迹
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='AUV Trajectory', alpha=0.7)
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
        
        # 目标点
        if target is not None:
            ax1.plot(target[0], target[1], 'r*', markersize=15, label='Target')
            ax1.add_patch(plt.Circle((target[0], target[1]), 0.3, 
                                  color='green', fill=False, linestyle='--', lw=2, label='Success (0.3m)'))
        
        # 障碍物
        if obstacles is not None:
            for obs in obstacles:
                circle = plt.Circle((obs.x, obs.y), obs.radius, 
                                  color='gray', alpha=0.5, label='Obstacle' if obstacles.index(obs) == 0 else '')
                ax1.add_patch(circle)
        
        # 航向箭头
        arrow_interval = max(1, len(positions) // 12)
        for i in range(0, len(positions), arrow_interval):
            dx = 0.25 * np.cos(yaw[i])
            dy = 0.25 * np.sin(yaw[i])
            ax1.arrow(positions[i, 0], positions[i, 1], dx, dy,
                    head_width=0.08, head_length=0.04, fc='orange', ec='orange', alpha=0.6)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        status = "Success" if episode['success'] else "Fail"
        ax1.set_title(f'{result["description"]} - Episode {ep_idx+1} ({status})')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 右图：距离和奖励
        ax2 = axes[1]
        steps = range(len(traj['rewards']))
        
        # 计算距离
        if target is not None:
            distances = [np.linalg.norm(np.array(traj['positions'][i+1]) - target[:2]) 
                        for i in range(len(traj['rewards']))]
            ax2.plot(steps, distances, 'r-', lw=2, label='Distance to Target')
            ax2.axhline(y=0.3, color='green', linestyle='--', lw=2, label='Success Threshold (0.3m)')
        
        # 奖励（双y轴）
        ax2_twin = ax2.twinx()
        ax2_twin.plot(steps, traj['rewards'], 'b-', alpha=0.5, label='Reward')
        ax2_twin.set_ylabel('Reward', color='b')
        ax2_twin.tick_params(axis='y', labelcolor='b')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance (m)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Distance and Reward over Time')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)
    
    def plot_multiple_trajectories(self, result: Dict, save_path: str = None):
        """绘制多条轨迹汇总图"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i, (traj, ep) in enumerate(zip(result['trajectories'], result['episodes'])):
            pos = np.array(traj['positions'])
            target = np.array(traj['target']) if traj['target'] is not None else None
            color = 'green' if ep['success'] else 'red'
            alpha = 0.7 if ep['success'] else 0.3
            
            ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6, alpha=alpha)
            
            if target is not None and i == 0:  # 只画一次目标
                ax.plot(target[0], target[1], '*', color='gold', markersize=12, label='Target')
                ax.add_patch(plt.Circle((target[0], target[1]), 0.3, 
                                      color='green', fill=False, linestyle='--', lw=2))
        
        # 障碍物（如果有）
        if result['trajectories'] and result['trajectories'][0].get('obstacles'):
            obstacles = result['trajectories'][0]['obstacles']
            for obs in obstacles:
                circle = plt.Circle((obs.x, obs.y), obs.radius, 
                                  color='gray', alpha=0.5)
                ax.add_patch(circle)
        
        successes = sum(1 for e in result['episodes'] if e['success'])
        n_eps = len(result['episodes'])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{result["description"]} - All Trajectories (Success: {successes}/{n_eps} = {successes/n_eps:.1%})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='green', lw=2, label='Success'),
            Line2D([0], [0], color='red', lw=2, alpha=0.3, label='Fail'),
            Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='', label='Target')
        ])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)
    
    def save_trajectory_gif(self, result: Dict, ep_idx: int, save_path: str, fps: int = 15, every_n: int = 2):
        """生成轨迹GIF动画"""
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("matplotlib.animation not available, skipping GIF")
            return
        
        traj = result['trajectories'][ep_idx]
        positions = np.array(traj['positions'])
        yaw = np.array(traj['yaw_angles'])
        target = np.array(traj['target']) if traj['target'] is not None else None
        obstacles = traj.get('obstacles', None)
        
        n_frames = len(positions)
        indices = list(range(0, n_frames, every_n))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        margin = 1.5
        x_min = positions[:, 0].min() - margin
        x_max = positions[:, 0].max() + margin
        y_min = positions[:, 1].min() - margin
        y_max = positions[:, 1].max() + margin
        if target is not None:
            x_min = min(x_min, target[0] - margin)
            x_max = max(x_max, target[0] + margin)
            y_min = min(y_min, target[1] - margin)
            y_max = max(y_max, target[1] + margin)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 静态元素
        if target is not None:
            ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')
            ax.add_patch(plt.Circle((target[0], target[1]), 0.3, 
                              color='green', fill=False, linestyle='--', lw=2))
        
        if obstacles is not None:
            for obs in obstacles:
                circle = plt.Circle((obs.x, obs.y), obs.radius, 
                                  color='gray', alpha=0.5)
                ax.add_patch(circle)
        
        # 动态元素
        line_auv, = ax.plot([], [], 'b-', lw=2, label='AUV Trajectory')
        point_auv, = ax.plot([], [], 'go', markersize=14, label='AUV')
        arrow_auv = ax.annotate('', xy=(0, 0), xytext=(0, 0), 
                               arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        title_text = ax.set_title('')
        ax.legend(loc='upper right')
        
        def init():
            line_auv.set_data([], [])
            point_auv.set_data([], [])
            arrow_auv.set_position((0, 0))
            title_text.set_text('')
            return line_auv, point_auv, arrow_auv, title_text
        
        def animate(i):
            idx = indices[i] if i < len(indices) else indices[-1]
            line_auv.set_data(positions[:idx+1, 0], positions[:idx+1, 1])
            point_auv.set_data([positions[idx, 0]], [positions[idx, 1]])
            
            # 更新箭头
            dx = 0.3 * np.cos(yaw[idx])
            dy = 0.3 * np.sin(yaw[idx])
            arrow_auv.set_position((positions[idx, 0], positions[idx, 1]))
            arrow_auv.xy = (positions[idx, 0] + dx, positions[idx, 1] + dy)
            
            success = result['episodes'][ep_idx]['success']
            title_text.set_text(f'{result["description"]} | Step {idx}/{n_frames-1} ({"Success" if success else "Fail"})')
            return line_auv, point_auv, arrow_auv, title_text
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                     frames=len(indices), interval=1000//fps, blit=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"  GIF saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate New Models and Generate Visualizations")
    parser.add_argument("--all", action="store_true", help="Evaluate all new models")
    parser.add_argument("--model", type=str, help="Specific model to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per model")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save trajectory plots")
    parser.add_argument("--save-gif", action="store_true", help="Save GIF animations")
    parser.add_argument("--max-plots", type=int, default=3, help="Max individual trajectory plots per model")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF frame rate")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    # 确定要评估的模型
    if args.model:
        if args.model not in MODEL_CONFIGS:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return
        model_names = [args.model]
    elif args.all:
        model_names = list(MODEL_CONFIGS.keys())
    else:
        print("Please specify --all or --model <model_name>")
        return
    
    # 过滤存在的模型
    existing_models = []
    for name in model_names:
        ckpt_path = os.path.join(PROJECT_ROOT, MODEL_CONFIGS[name]['checkpoint'])
        if os.path.exists(ckpt_path):
            existing_models.append(name)
        else:
            print(f"[SKIP] {name}: checkpoint not found")
    
    if not existing_models:
        print("No models to evaluate!")
        return
    
    # 评估
    evaluator = NewModelEvaluator()
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(existing_models)} New Models ({args.episodes} episodes each)")
    print(f"{'='*70}\n")
    
    for name in existing_models:
        config = MODEL_CONFIGS[name]
        all_results[name] = evaluator.evaluate_model(name, config, args.episodes, args.verbose)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "visualization", f"new_models_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成轨迹图
    if args.save_plots:
        plots_dir = os.path.join(output_dir, "trajectory_plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\n>>> Saving trajectory plots to {plots_dir}")
        
        for name, result in all_results.items():
            if 'error' in result:
                continue
            
            model_dir = os.path.join(plots_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            # 单条轨迹图
            n_eps = len(result['trajectories'])
            for i in range(min(args.max_plots, n_eps)):
                path = os.path.join(model_dir, f"trajectory_ep{i+1}.png")
                evaluator.plot_trajectory(result, i, path)
            
            # 多条轨迹汇总图
            path = os.path.join(model_dir, "all_trajectories.png")
            evaluator.plot_multiple_trajectories(result, path)
        
        print(f"  ✓ Trajectory plots saved")
    
    # 生成GIF
    if args.save_gif:
        gif_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        print(f"\n>>> Saving GIFs to {gif_dir}")
        
        for name, result in all_results.items():
            if 'error' in result:
                continue
            
            # 为前3个episode生成GIF
            n_eps = len(result['trajectories'])
            for i in range(min(3, n_eps)):
                gif_path = os.path.join(gif_dir, f"{name}_ep{i+1}.gif")
                evaluator.save_trajectory_gif(result, i, gif_path, fps=args.gif_fps)
        
        print(f"  ✓ GIFs saved")
    
    # 生成汇总报告
    report_path = os.path.join(output_dir, "summary.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NEW MODELS EVALUATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Model':<30} {'Success Rate':>12} {'Avg Reward':>12} {'Avg Steps':>10}\n")
        f.write("-" * 70 + "\n")
        
        for name, result in all_results.items():
            if 'error' in result:
                f.write(f"{name:<30} {'ERROR':>12}\n")
                continue
            summary = result['summary']
            sr = f"{summary['success_rate']:.1%}"
            ar = f"{summary['avg_reward']:.1f}"
            astp = f"{summary['avg_steps']:.0f}"
            f.write(f"{name:<30} {sr:>12} {ar:>12} {astp:>10}\n")
    
    print(f"\n{'='*70}")
    print(f"✓ All outputs saved to {output_dir}")
    print(f"  - summary.txt")
    if args.save_plots:
        print(f"  - trajectory_plots/  (轨迹图)")
    if args.save_gif:
        print(f"  - gifs/  (GIF动画)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
