"""  
底层控制器评估脚本  
==================  
  
评估训练好的底层控制器性能  
  
使用方法:  
    python training/evaluate_low_level.py --model checkpoints/best_model.pth  
      
    # 可视化评估  
    python training/evaluate_low_level.py --model checkpoints/best_model.pth --render  
      
    # 录制视频  
    python training/evaluate_low_level.py --model checkpoints/best_model.pth --record  
"""  
  
import os  
import sys  
import argparse  
import yaml  
import numpy as np  
import torch  
import matplotlib.pyplot as plt  
from typing import Dict, List, Optional  
  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
from core.environment.low_level_env import LowLevelAUVEnv  
from networks.ppo_lstm_network import PPOLSTMNetwork  
from utils.visualizer import TrajectoryVisualizer  
from utils.video_recorder import VideoRecorder  
from utils.metrics import PerformanceMetrics  
  
  
class LowLevelEvaluator:  
    """  
    底层控制器评估器  
    """  
      
    def __init__(self, model_path: str, device: torch.device = None):  
        """  
        初始化评估器  
          
        Args:  
            model_path: 模型路径  
            device: 计算设备  
        """  
        if device is None:  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        else:  
            self.device = device  
              
        # 加载模型
        # 支持两种 checkpoint 格式：
        # 1) PPOLSTMNetwork.save 保存的 { 'state_dict':..., 'config':... }
        # 2) PPO.save 保存的 { 'network_state_dict':..., ... }
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 直接使用网络的 class loader
            self.network = PPOLSTMNetwork.load(model_path, self.device)
            self.network.eval()
            self._loaded_network_state = None
        elif isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
            # PPO.save 格式：延迟构建网络（需要 env dims）
            self.network = None
            self._loaded_network_state = checkpoint['network_state_dict']
            self._loaded_checkpoint = checkpoint
        else:
            raise RuntimeError(f"Unrecognized model checkpoint format: {model_path}")

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
          
    def evaluate(self,  
                 env: LowLevelAUVEnv,  
                 n_episodes: int = 100,  
                 deterministic: bool = True,  
                 render: bool = False,  
                 record: bool = False,  
                 video_dir: str = None) -> Dict:  
        """  
        评估策略  
          
        Args:  
            env: 评估环境  
            n_episodes: 评估episode数  
            deterministic: 是否使用确定性策略  
            render: 是否渲染  
            record: 是否录制视频  
            video_dir: 视频保存目录  
              
        Returns:  
            results: 评估结果字典  
        """  
        # 视频录制器  
        if record:  
            video_recorder = VideoRecorder(save_dir=video_dir or "visualization/videos")  
              
        # 性能指标  
        metrics = PerformanceMetrics()  
          
        # 统计  
        episode_rewards = []  
        episode_lengths = []  
        successes = []  
        position_errors = []  
        orientation_errors = []  
        settling_times = []  
          
        # 轨迹记录（用于可视化）  
        all_trajectories = []  
          
        for ep in range(n_episodes):  
            # 如果 checkpoint 来自 PPO.save（自带 state_dict 但未构建网络），
            # 则在拿到 env 的 obs/action 空间后构建网络并加载参数。
            if getattr(self, '_loaded_network_state', None) is not None and self.network is None:
                obs_dim = env.observation_space.shape[0]
                num_actions = env.action_space.n
                # 构建网络实例并加载 state_dict
                self.network = PPOLSTMNetwork(
                    obs_dim=obs_dim,
                    num_actions=num_actions,
                    feature_hidden_sizes=[256, 256],
                    lstm_hidden_size=256,
                    lstm_num_layers=2,
                )
                try:
                    self.network.load_state_dict(self._loaded_network_state)
                except Exception:
                    # 兼容 key 名称差异
                    self.network.load_state_dict(self._loaded_network_state)
                self.network.to(self.device)
                self.network.eval()

            obs, info = env.reset()
            hidden_state = self.network.lstm.get_initial_state(1, self.device)  
              
            done = False  
            episode_reward = 0  
            episode_length = 0  
              
            trajectory = {  
                'positions': [],  
                'orientations': [],  
                'velocities': [],  
                'actions': [],  
                'rewards': []  
            }  
              
            if record and ep < 5:  # 只录制前5个episode  
                video_recorder.start_recording()  
                  
            metrics.reset()  
              
            while not done:  
                # 获取动作  
                with torch.no_grad():  
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  
                    action_dist, _, hidden_state = self.network(obs_tensor, hidden_state)  
                      
                    if deterministic:  
                        action = action_dist.probs.argmax(dim=-1).item()  
                    else:  
                        action = action_dist.sample().item()  
                          
                # 执行动作  
                obs, reward, terminated, truncated, info = env.step(action)  
                done = terminated or truncated  
                  
                episode_reward += reward  
                episode_length += 1  
                  
                # 记录轨迹  
                state = env.dynamics.get_state()  
                trajectory['positions'].append(state['position'].copy())  
                trajectory['orientations'].append(state['orientation'].copy())  
                trajectory['velocities'].append(np.concatenate([  
                    state['linear_velocity'], state['angular_velocity']  
                ]))  
                trajectory['actions'].append(action)  
                trajectory['rewards'].append(reward)  
                  
                # 性能指标  
                pos_error = info.get('pos_error', 0)  
                ori_error = info.get('ori_error', 0)  
                control_effort = np.abs(env.dynamics.thruster_state).sum()  
                metrics.add_step(pos_error, ori_error,   
                               np.concatenate([state['linear_velocity'], state['angular_velocity']]),  
                               control_effort)  
                  
                # 视频帧  
                if record and ep < 5:  
                    video_recorder.add_frame({  
                        'position': state['position'],  
                        'orientation': state['orientation'],  
                        'target': env.target_state[:3]  
                    })  
                      
                # 渲染  
                if render:  
                    env.render()  
                      
            # Episode结束  
            episode_rewards.append(episode_reward)  
            episode_lengths.append(episode_length)  
              
            # 检查成功  
            success = info.get('termination_reason') == 'success' or (  
                info.get('pos_error', 10) < 0.5 and info.get('ori_error', 10) < 0.2  
            )  
            successes.append(success)  
              
            # 最终误差  
            position_errors.append(info.get('pos_error', 0))  
            orientation_errors.append(info.get('ori_error', 0))  
              
            # 计算指标  
            ep_metrics = metrics.compute_metrics()  
            settling_times.append(ep_metrics.get('settling_time_idx', episode_length))  
              
            # 保存轨迹  
            trajectory['positions'] = np.array(trajectory['positions'])  
            trajectory['orientations'] = np.array(trajectory['orientations'])  
            trajectory['velocities'] = np.array(trajectory['velocities'])  
            all_trajectories.append(trajectory)  
              
            # 保存视频  
            if record and ep < 5:  
                video_recorder.stop_recording()  
                video_recorder.save_video_2d(f"eval_episode_{ep}.mp4")  
                  
            # 打印进度  
            if (ep + 1) % 10 == 0:  
                print(f"Episode {ep+1}/{n_episodes}: "  
                      f"Reward={episode_reward:.2f}, "  
                      f"Success={success}, "  
                      f"Length={episode_length}")  
                  
        # 计算统计  
        results = {  
            'mean_reward': np.mean(episode_rewards),  
            'std_reward': np.std(episode_rewards),  
            'min_reward': np.min(episode_rewards),  
            'max_reward': np.max(episode_rewards),  
            'mean_length': np.mean(episode_lengths),  
            'success_rate': np.mean(successes),  
            'mean_position_error': np.mean(position_errors),  
            'mean_orientation_error': np.mean(orientation_errors),  
            'mean_settling_time': np.mean(settling_times),  
            'episode_rewards': episode_rewards,  
            'trajectories': all_trajectories  
        }  
          
        return results  
      
    def evaluate_skills(self,   
                        env: LowLevelAUVEnv,  
                        n_trials: int = 10) -> Dict:  
        """  
        评估各个技能的表现  
          
        Args:  
            env: 评估环境  
            n_trials: 每个技能的测试次数  
              
        Returns:  
            skill_results: 技能评估结果  
        """  
        skills = {  
            'hover': {'target_delta': np.zeros(6)},  
            'forward': {'target_delta': np.array([5, 0, 0, 0, 0, 0])},  
            'backward': {'target_delta': np.array([-5, 0, 0, 0, 0, 0])},  
            'left': {'target_delta': np.array([0, -5, 0, 0, 0, 0])},  
            'right': {'target_delta': np.array([0, 5, 0, 0, 0, 0])},  
            'up': {'target_delta': np.array([0, 0, 5, 0, 0, 0])},  
            'down': {'target_delta': np.array([0, 0, -5, 0, 0, 0])},  
            'turn_left': {'target_delta': np.array([0, 0, 0, 0, 0, np.pi/2])},  
            'turn_right': {'target_delta': np.array([0, 0, 0, 0, 0, -np.pi/2])},  
        }  
          
        results = {}  
          
        for skill_name, skill_config in skills.items():  
            print(f"\nEvaluating skill: {skill_name}")  
              
            skill_rewards = []  
            skill_errors = []  
            skill_successes = []  
              
            for trial in range(n_trials):  
                obs, info = env.reset()  
                  
                # 设置目标  
                initial_state = env.dynamics.eta.copy()  
                target = initial_state + skill_config['target_delta']  
                env.target_state[:6] = target  
                  
                hidden_state = self.network.lstm.get_initial_state(1, self.device)  
                  
                episode_reward = 0  
                max_steps = 300  
                  
                for step in range(max_steps):  
                    with torch.no_grad():  
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  
                        action_dist, _, hidden_state = self.network(obs_tensor, hidden_state)  
                        action = action_dist.probs.argmax(dim=-1).item()  
                          
                    obs, reward, terminated, truncated, info = env.step(action)  
                    episode_reward += reward  
                      
                    if terminated or truncated:  
                        break  
                          
                # 计算最终误差  
                final_state = env.dynamics.eta  
                pos_error = np.linalg.norm(final_state[:3] - target[:3])  
                ori_error = np.linalg.norm(final_state[3:6] - target[3:6])  
                  
                skill_rewards.append(episode_reward)  
                skill_errors.append({'position': pos_error, 'orientation': ori_error})  
                skill_successes.append(pos_error < 0.5 and ori_error < 0.2)  
                  
            results[skill_name] = {  
                'mean_reward': np.mean(skill_rewards),  
                'success_rate': np.mean(skill_successes),  
                'mean_position_error': np.mean([e['position'] for e in skill_errors]),  
                'mean_orientation_error': np.mean([e['orientation'] for e in skill_errors])  
            }  
              
            print(f"  Success rate: {results[skill_name]['success_rate']:.2%}")  
            print(f"  Mean position error: {results[skill_name]['mean_position_error']:.3f}m")  
              
        return results  
  
  
def print_results(results: Dict):  
    """打印评估结果"""  
    print("\n" + "=" * 60)  
    print("EVALUATION RESULTS")  
    print("=" * 60)  
      
    print(f"\nPerformance Metrics:")  
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")  
    print(f"  Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")  
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")  
    print(f"  Success Rate: {results['success_rate']:.2%}")  
      
    print(f"\nControl Metrics:")  
    print(f"  Mean Position Error: {results['mean_position_error']:.4f} m")  
    print(f"  Mean Orientation Error: {results['mean_orientation_error']:.4f} rad")  
    print(f"  Mean Settling Time: {results['mean_settling_time']:.1f} steps")  
      
    print("=" * 60)  
  
  
def plot_results(results: Dict, save_dir: str = "visualization/plots"):  
    """绘制评估结果图"""  
    os.makedirs(save_dir, exist_ok=True)  
      
    # 奖励分布  
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
      
    # 奖励直方图  
    axes[0, 0].hist(results['episode_rewards'], bins=30, edgecolor='black', alpha=0.7)  
    axes[0, 0].axvline(x=results['mean_reward'], color='r', linestyle='--',   
                       label=f"Mean: {results['mean_reward']:.2f}")  
    axes[0, 0].set_xlabel('Episode Reward')  
    axes[0, 0].set_ylabel('Frequency')  
    axes[0, 0].set_title('Reward Distribution')  
    axes[0, 0].legend()  
      
    # 轨迹示例  
    if results.get('trajectories'):  
        traj = results['trajectories'][0]  
        positions = traj['positions']  
          
        axes[0, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)  
        axes[0, 1].plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')  
        axes[0, 1].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')  
        axes[0, 1].set_xlabel('X (m)')  
        axes[0, 1].set_ylabel('Y (m)')  
        axes[0, 1].set_title('Example Trajectory (XY)')  
        axes[0, 1].legend()  
        axes[0, 1].axis('equal')  
        axes[0, 1].grid(True)  
          
    # 3D轨迹  
    if results.get('trajectories'):  
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')  
        for i, traj in enumerate(results['trajectories'][:5]):  
            positions = traj['positions']  
            ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.7)  
        ax3d.set_xlabel('X (m)')  
        ax3d.set_ylabel('Y (m)')  
        ax3d.set_zlabel('Z (m)')  
        ax3d.set_title('Trajectories (3D)')  
          
    # 动作分布  
    if results.get('trajectories'):  
        all_actions = []  
        for traj in results['trajectories']:  
            all_actions.extend(traj['actions'])  
              
        action_counts = np.bincount(all_actions, minlength=13)  
        axes[1, 1].bar(range(13), action_counts)  
        axes[1, 1].set_xlabel('Action')  
        axes[1, 1].set_ylabel('Count')  
        axes[1, 1].set_title('Action Distribution')  
        axes[1, 1].set_xticks(range(13))  
          
    plt.tight_layout()  
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=150)  
    plt.show()  
  
  
def parse_args():  
    """解析命令行参数"""  
    parser = argparse.ArgumentParser(description="Evaluate low-level AUV controller")  
      
    parser.add_argument('--model', type=str, required=True,  
                       help='Path to model checkpoint')  
    parser.add_argument('--n-episodes', type=int, default=100,  
                       help='Number of evaluation episodes')  
    parser.add_argument('--curriculum-stage', type=int, default=0,  
                       help='Curriculum stage for evaluation')  
    parser.add_argument('--seed', type=int, default=42,  
                       help='Random seed')  
    parser.add_argument('--render', action='store_true',  
                       help='Render evaluation')  
    parser.add_argument('--record', action='store_true',  
                       help='Record video')  
    parser.add_argument('--evaluate-skills', action='store_true',  
                       help='Evaluate individual skills')  
    parser.add_argument('--output-dir', type=str, default='visualization/eval',  
                       help='Output directory')  
      
    return parser.parse_args()  
  
  
def main():  
    args = parse_args()  
      
    # 创建输出目录  
    os.makedirs(args.output_dir, exist_ok=True)  
      
    # 创建评估器  
    evaluator = LowLevelEvaluator(args.model)  
      
    # 创建环境  
    env = LowLevelAUVEnv(  
        curriculum_stage=args.curriculum_stage,  
        enable_domain_randomization=False,  
        render_mode="human" if args.render else None,  
        seed=args.seed  
    )  
      
    # 评估  
    print(f"\nEvaluating model: {args.model}")  
    print(f"Episodes: {args.n_episodes}")  
    print(f"Curriculum stage: {args.curriculum_stage}")  
      
    results = evaluator.evaluate(  
        env,  
        n_episodes=args.n_episodes,  
        deterministic=True,  
        render=args.render,  
        record=args.record,  
        video_dir=os.path.join(args.output_dir, 'videos')  
    )  
      
    # 打印结果  
    print_results(results)  
      
    # 绘图  
    plot_results(results, save_dir=args.output_dir)  
      
    # 技能评估  
    if args.evaluate_skills:  
        print("\n" + "=" * 60)  
        print("SKILL EVALUATION")  
        print("=" * 60)  
          
        skill_results = evaluator.evaluate_skills(env, n_trials=10)  
          
        print("\nSkill Performance Summary:")  
        for skill, metrics in skill_results.items():  
            print(f"  {skill}: "  
                  f"Success={metrics['success_rate']:.2%}, "  
                  f"Pos Error={metrics['mean_position_error']:.3f}m")  
              
    env.close()  
    print(f"\nResults saved to {args.output_dir}")  
  
  
if __name__ == "__main__":  
    main()  
