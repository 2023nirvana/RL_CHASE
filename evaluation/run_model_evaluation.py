"""
综合模型评估脚本 - 批量评估所有训练好的模型
=============================================

功能：
1. 自动扫描并评估所有模型（2D动态MLP/LSTM、帧堆叠、3D固定目标）
2. 生成对比报告
3. 可选生成GIF/MP4动画

使用方法：
    # 评估所有模型
    python evaluation/run_model_evaluation.py --all
    
    # 仅评估动态目标模型
    python evaluation/run_model_evaluation.py --dynamic-only
    
    # 评估并生成GIF
    python evaluation/run_model_evaluation.py --all --save-gif
    
    # 评估特定模型
    python evaluation/run_model_evaluation.py --models dynamic_mlp dynamic_lstm
"""
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_dynamic import LowLevelAUV2DDynamicEnv
from core.environment.low_level_env_3d import LowLevelAUV3DEnv


# ==================== 网络定义 ====================
class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_size=128, lstm_hidden=64):
        super().__init__()
        self.hidden_size, self.lstm_hidden = hidden_size, lstm_hidden
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True)
        self.actor_head = nn.Linear(lstm_hidden, num_actions)
        self.critic_head = nn.Linear(lstm_hidden, 1)

    def forward(self, obs, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch_size, seq_len, _ = obs.shape
        features = self.feature(obs)
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=obs.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=obs.device)
            hidden = (h0, c0)
        lstm_out, new_hidden = self.lstm(features, hidden)
        last_out = lstm_out[:, -1, :]
        logits = self.actor_head(last_out)
        value = self.critic_head(last_out)
        return logits, value.squeeze(-1), new_hidden


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_sizes=[128, 64]):
        super().__init__()
        actor_layers, prev = [], obs_dim
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        actor_layers.append(nn.Linear(prev, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        critic_layers, prev = [], obs_dim
        for h in hidden_sizes:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs, hidden=None):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value, None


# ==================== 帧堆叠包装器 ====================
class FrameStackEnv:
    def __init__(self, env, n_stack=4):
        self.env = env
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.obs_dim = env.obs_dim * n_stack
        
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.obs_dim,)})()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def target_pos(self):
        return self.env.target_pos
    
    @property
    def target_motion(self):
        return self.env.target_motion
    
    @property
    def inner(self):
        return self.env.inner
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, term, trunc, info
    
    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames))


# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    # 动态目标 MLP
    'dynamic_mlp': {
        'type': '2d_dynamic',
        'checkpoint': 'checkpoints/dynamic_mlp/best_model.pth',
        'network': 'mlp',
        'description': '2D Dynamic Target (MLP)'
    },
    # 动态目标 LSTM
    'dynamic_lstm': {
        'type': '2d_dynamic',
        'checkpoint': 'checkpoints/dynamic_lstm/best_model.pth',
        'network': 'lstm',
        'description': '2D Dynamic Target (LSTM)'
    },
    # 帧堆叠系列
    'dynamic_stack4': {
        'type': '2d_framestack',
        'checkpoint': 'checkpoints/dynamic_stack4/best_model.pth',
        'n_stack': 4,
        'description': '2D Dynamic Target (FrameStack-4)'
    },
    'dynamic_stack5': {
        'type': '2d_framestack',
        'checkpoint': 'checkpoints/dynamic_stack5/best_model.pth',
        'n_stack': 5,
        'description': '2D Dynamic Target (FrameStack-5)'
    },
    'dynamic_stack6': {
        'type': '2d_framestack',
        'checkpoint': 'checkpoints/dynamic_stack6/best_model.pth',
        'n_stack': 6,
        'description': '2D Dynamic Target (FrameStack-6)'
    },
    'dynamic_stack8': {
        'type': '2d_framestack',
        'checkpoint': 'checkpoints/dynamic_stack8/best_model.pth',
        'n_stack': 8,
        'description': '2D Dynamic Target (FrameStack-8)'
    },
    # 3D 固定目标
    '3d_dist_3m': {
        'type': '3d_fixed',
        'checkpoint': 'checkpoints/3d_dist_3m/best_model.pth',
        'target_distance': 3.0,
        'description': '3D Fixed Target (3m)'
    },
    '3d_dist_5m': {
        'type': '3d_fixed',
        'checkpoint': 'checkpoints/3d_dist_5m/best_model.pth',
        'target_distance': 5.0,
        'description': '3D Fixed Target (5m)'
    },
}


# ==================== 综合评估器 ====================
class ComprehensiveEvaluator:
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Comprehensive Evaluator initialized on {self.device}")

    def evaluate_model(self, model_name: str, config: Dict, n_episodes: int = 10, 
                       target_motion: str = 'constant', verbose: bool = True) -> Dict:
        """评估单个模型"""
        checkpoint_path = os.path.join(PROJECT_ROOT, config['checkpoint'])
        if not os.path.exists(checkpoint_path):
            print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
            return {'model': model_name, 'error': 'checkpoint_not_found'}
        
        model_type = config['type']
        results = {'model': model_name, 'config': config, 'episodes': []}
        
        if model_type == '2d_dynamic':
            results = self._eval_2d_dynamic(model_name, config, checkpoint_path, n_episodes, target_motion, verbose)
        elif model_type == '2d_framestack':
            results = self._eval_2d_framestack(model_name, config, checkpoint_path, n_episodes, target_motion, verbose)
        elif model_type == '3d_fixed':
            results = self._eval_3d_fixed(model_name, config, checkpoint_path, n_episodes, verbose)
        
        return results

    def _eval_2d_dynamic(self, name: str, config: Dict, ckpt_path: str, n_eps: int, motion: str, verbose: bool) -> Dict:
        """评估 2D 动态目标模型 (MLP/LSTM)"""
        env = LowLevelAUV2DDynamicEnv(target_motion=motion)
        obs_dim, num_actions = env.observation_space.shape[0], env.action_space.n
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get('network', ckpt)
        
        if config['network'] == 'lstm':
            network = LSTMActorCritic(obs_dim, num_actions).to(self.device)
        else:
            network = MLPActorCritic(obs_dim, num_actions).to(self.device)
        network.load_state_dict(state)
        network.eval()
        
        results = {'model': name, 'type': '2d_dynamic', 'motion': motion, 'episodes': [], 'trajectories': []}
        for ep in range(n_eps):
            obs, info = env.reset()
            trajectory = {'positions': [], 'step_target': []}
            trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            trajectory['step_target'].append(env.target_pos.tolist())
            
            hidden, done, total_r, steps = None, False, 0, 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _, hidden = network(obs_t, hidden)
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                total_r += r
                steps += 1
                trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
                trajectory['step_target'].append(info['target_pos'])
            
            success = info.get('termination_reason') == 'success'
            results['episodes'].append({
                'success': success, 'steps': steps, 'reward': total_r, 
                'final_error': info.get('pos_error', 0)
            })
            results['trajectories'].append(trajectory)
            if verbose:
                print(f"    Ep {ep+1}: {'✓' if success else '✗'}, steps={steps}, reward={total_r:.1f}")
        
        results['summary'] = self._compute_summary(results['episodes'])
        return results

    def _eval_2d_framestack(self, name: str, config: Dict, ckpt_path: str, n_eps: int, motion: str, verbose: bool) -> Dict:
        """评估 2D 帧堆叠模型"""
        n_stack = config['n_stack']
        base_env = LowLevelAUV2DDynamicEnv(target_motion=motion)
        env = FrameStackEnv(base_env, n_stack=n_stack)
        obs_dim, num_actions = env.observation_space.shape[0], env.action_space.n
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        hidden1, hidden2 = 256, 64
        actor = nn.Sequential(
            nn.Linear(obs_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, num_actions)
        ).to(self.device)
        actor.load_state_dict(ckpt['actor'])
        actor.eval()
        
        results = {'model': name, 'type': '2d_framestack', 'n_stack': n_stack, 'motion': motion, 
                   'episodes': [], 'trajectories': []}
        for ep in range(n_eps):
            obs, info = env.reset()
            trajectory = {'positions': [], 'step_target': []}
            trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            trajectory['step_target'].append(env.target_pos.tolist())
            
            done, total_r, steps = False, 0, 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                total_r += r
                steps += 1
                trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
                trajectory['step_target'].append(info['target_pos'])
            
            success = info.get('termination_reason') == 'success'
            results['episodes'].append({
                'success': success, 'steps': steps, 'reward': total_r, 
                'final_error': info.get('pos_error', 0)
            })
            results['trajectories'].append(trajectory)
            if verbose:
                print(f"    Ep {ep+1}: {'✓' if success else '✗'}, steps={steps}, reward={total_r:.1f}")
        
        results['summary'] = self._compute_summary(results['episodes'])
        return results

    def _eval_3d_fixed(self, name: str, config: Dict, ckpt_path: str, n_eps: int, verbose: bool) -> Dict:
        """评估 3D 固定目标模型"""
        target_dist = config.get('target_distance', 3.0)
        env = LowLevelAUV3DEnv(target_distance=target_dist)
        obs_dim, num_actions = env.observation_space.shape[0], env.action_space.n
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        hidden_sizes = [128, 64]
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        actor = nn.Sequential(*layers).to(self.device)
        actor.load_state_dict(ckpt['actor'])
        actor.eval()
        
        results = {'model': name, 'type': '3d_fixed', 'target_distance': target_dist, 
                   'episodes': [], 'trajectories': []}
        for ep in range(n_eps):
            obs, info = env.reset()
            trajectory = {'positions': [], 'z': [], 'target': info['target_pos']}
            trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            trajectory['z'].append(env.inner.dynamics.eta[2])
            
            done, total_r, steps = False, 0, 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                total_r += r
                steps += 1
                trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
                trajectory['z'].append(env.inner.dynamics.eta[2])
            
            success = info.get('termination_reason') == 'success'
            results['episodes'].append({
                'success': success, 'steps': steps, 'reward': total_r, 
                'final_error': info.get('pos_error', 0)
            })
            results['trajectories'].append(trajectory)
            if verbose:
                print(f"    Ep {ep+1}: {'✓' if success else '✗'}, steps={steps}, reward={total_r:.1f}")
        
        results['summary'] = self._compute_summary(results['episodes'])
        return results

    def _compute_summary(self, episodes: List[Dict]) -> Dict:
        """计算评估统计"""
        successes = sum(1 for e in episodes if e['success'])
        return {
            'success_rate': successes / len(episodes),
            'avg_reward': np.mean([e['reward'] for e in episodes]),
            'avg_steps': np.mean([e['steps'] for e in episodes]),
            'avg_final_error': np.mean([e['final_error'] for e in episodes]),
            'total_episodes': len(episodes)
        }

    def evaluate_all(self, model_names: List[str] = None, n_episodes: int = 10, 
                     target_motion: str = 'constant', verbose: bool = True) -> Dict[str, Any]:
        """批量评估多个模型"""
        if model_names is None:
            model_names = list(MODEL_CONFIGS.keys())
        
        all_results = {}
        print(f"\n{'='*70}")
        print(f"Comprehensive Model Evaluation ({len(model_names)} models, {n_episodes} eps each)")
        print(f"Target Motion: {target_motion}")
        print(f"{'='*70}\n")
        
        for name in model_names:
            if name not in MODEL_CONFIGS:
                print(f"[WARN] Unknown model: {name}")
                continue
            config = MODEL_CONFIGS[name]
            print(f"\n>>> Evaluating: {config['description']}")
            all_results[name] = self.evaluate_model(name, config, n_episodes, target_motion, verbose)
        
        return all_results

    def generate_comparison_report(self, results: Dict[str, Any], save_path: str = None) -> str:
        """生成对比报告"""
        report_lines = [
            "=" * 80,
            "MODEL COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"{'Model':<25} {'Type':<15} {'Success Rate':>12} {'Avg Reward':>12} {'Avg Steps':>10}",
            "-" * 80
        ]
        
        for name, res in results.items():
            if 'error' in res:
                report_lines.append(f"{name:<25} {'ERROR':<15} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
                continue
            summary = res['summary']
            model_type = res.get('type', 'unknown')
            sr = f"{summary['success_rate']:.1%}"
            ar = f"{summary['avg_reward']:.1f}"
            astp = f"{summary['avg_steps']:.0f}"
            report_lines.append(f"{name:<25} {model_type:<15} {sr:>12} {ar:>12} {astp:>10}")
        
        report_lines.extend(["", "=" * 80])
        report = "\n".join(report_lines)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {save_path}")
        
        return report

    def plot_comparison_bar(self, results: Dict[str, Any], save_path: str = None):
        """绘制对比柱状图"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if not valid_results:
            print("No valid results to plot")
            return
        
        names = list(valid_results.keys())
        success_rates = [valid_results[n]['summary']['success_rate'] for n in names]
        avg_rewards = [valid_results[n]['summary']['avg_reward'] for n in names]
        avg_steps = [valid_results[n]['summary']['avg_steps'] for n in names]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Success Rate
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(names)), [sr * 100 for sr in success_rates], color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 105)
        for bar, sr in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{sr:.0%}', ha='center', fontsize=8)
        
        # Avg Reward
        ax2 = axes[1]
        colors = ['green' if r > 0 else 'red' for r in avg_rewards]
        bars2 = ax2.bar(range(len(names)), avg_rewards, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward Comparison')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Avg Steps
        ax3 = axes[2]
        bars3 = ax3.bar(range(len(names)), avg_steps, color='coral', alpha=0.8)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Average Steps')
        ax3.set_title('Average Steps Comparison')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        plt.close(fig)

    # ==================== 轨迹图绘制 ====================
    def plot_2d_trajectory(self, result: Dict, ep_idx: int, save_path: str = None):
        """绘制2D动态目标单条轨迹图"""
        traj = result['trajectories'][ep_idx]
        episode = result['episodes'][ep_idx]
        positions = np.array(traj['positions'])
        target_path = np.array(traj['step_target'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：轨迹
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='AUV Trajectory', alpha=0.7)
        ax1.plot(target_path[:, 0], target_path[:, 1], 'r--', lw=1.5, label='Target Path', alpha=0.8)
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='AUV Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='AUV End')
        ax1.plot(target_path[0, 0], target_path[0, 1], 'r^', markersize=10, label='Target Start')
        ax1.plot(target_path[-1, 0], target_path[-1, 1], 'r*', markersize=14, label='Target End')
        
        # 成功圈
        ax1.add_patch(plt.Circle((target_path[-1, 0], target_path[-1, 1]), 0.3, 
                                  color='green', fill=False, linestyle='--', lw=2, label='Success (0.3m)'))
        
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
        status = "Success" if episode['success'] else "Fail"
        ax1.set_title(f'{result["model"]} - Episode {ep_idx+1} ({status})')
        ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        # 右图：距离曲线
        ax2 = axes[1]
        steps = range(len(positions) - 1)
        pos_errors = [np.linalg.norm(np.array(positions[i+1]) - np.array(target_path[i+1])) 
                      for i in range(len(positions) - 1)]
        ax2.plot(steps, pos_errors, 'r-', lw=2, label='Distance to Target')
        ax2.axhline(y=0.3, color='green', linestyle='--', lw=2, label='Success Threshold (0.3m)')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance to Moving Target')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)

    def plot_2d_multiple_trajectories(self, result: Dict, save_path: str = None):
        """绘制2D多条轨迹汇总图"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i, (traj, ep) in enumerate(zip(result['trajectories'], result['episodes'])):
            pos = np.array(traj['positions'])
            tgt = np.array(traj['step_target'])
            color = 'green' if ep['success'] else 'red'
            alpha = 0.7 if ep['success'] else 0.3
            ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6, alpha=alpha)
            ax.plot(tgt[:, 0], tgt[:, 1], '--', color='gray', alpha=0.3, lw=1)
        
        successes = sum(1 for e in result['episodes'] if e['success'])
        n_eps = len(result['episodes'])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{result["model"]} - All Trajectories (Success: {successes}/{n_eps} = {successes/n_eps:.1%})')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='green', lw=2, label='Success'),
            Line2D([0], [0], color='red', lw=2, alpha=0.3, label='Fail'),
            Line2D([0], [0], color='gray', lw=1, linestyle='--', alpha=0.3, label='Target Path')
        ])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)

    def plot_3d_trajectory(self, result: Dict, ep_idx: int, save_path: str = None):
        """绘制3D固定目标单条轨迹图（含深度曲线）"""
        traj = result['trajectories'][ep_idx]
        episode = result['episodes'][ep_idx]
        positions = np.array(traj['positions'])
        z_values = np.array(traj['z'])
        target = np.array(traj['target'])
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 左图：XY水平轨迹
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='Trajectory', alpha=0.7)
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
        ax1.plot(target[0], target[1], 'r*', markersize=15, label='Target')
        ax1.add_patch(plt.Circle((target[0], target[1]), 0.3, color='green', fill=False, 
                                  linestyle='--', lw=2, label='Success (0.3m)'))
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
        status = "Success" if episode['success'] else "Fail"
        ax1.set_title(f'{result["model"]} - XY Trajectory ({status})')
        ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        # 中图：深度随时间变化
        ax2 = axes[1]
        ax2.plot(range(len(z_values)), z_values, 'b-', lw=2, label='AUV Depth')
        ax2.axhline(y=target[2], color='r', linestyle='--', lw=2, label=f'Target Depth ({target[2]:.1f}m)')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Z / Depth (m)')
        ax2.set_title('Depth over Time')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        # 右图：3D位置误差
        ax3 = axes[2]
        steps = range(len(positions))
        pos_errors_3d = [np.linalg.norm(np.array([positions[i][0], positions[i][1], z_values[i]]) - target) 
                         for i in range(len(positions))]
        ax3.plot(steps, pos_errors_3d, 'r-', lw=2, label='3D Distance to Target')
        ax3.axhline(y=0.3, color='green', linestyle='--', lw=2, label='Success Threshold (0.3m)')
        ax3.set_xlabel('Step'); ax3.set_ylabel('3D Position Error (m)')
        ax3.set_title('3D Position Error')
        ax3.legend(); ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)

    def plot_3d_multiple_trajectories(self, result: Dict, save_path: str = None):
        """绘制3D多条轨迹汇总图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：XY平面
        ax1 = axes[0]
        for i, (traj, ep) in enumerate(zip(result['trajectories'], result['episodes'])):
            pos = np.array(traj['positions'])
            target = np.array(traj['target'])
            color = 'green' if ep['success'] else 'red'
            alpha = 0.7 if ep['success'] else 0.3
            ax1.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
            ax1.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6, alpha=alpha)
            ax1.plot(target[0], target[1], '*', color='gold', markersize=12)
        
        successes = sum(1 for e in result['episodes'] if e['success'])
        n_eps = len(result['episodes'])
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{result["model"]} - XY Trajectories (Success: {successes}/{n_eps})')
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        from matplotlib.lines import Line2D
        ax1.legend(handles=[
            Line2D([0], [0], color='green', lw=2, label='Success'),
            Line2D([0], [0], color='red', lw=2, alpha=0.3, label='Fail'),
            Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='', label='Target')
        ])
        
        # 右图：深度曲线
        ax2 = axes[1]
        for i, (traj, ep) in enumerate(zip(result['trajectories'], result['episodes'])):
            z_values = np.array(traj['z'])
            target = np.array(traj['target'])
            color = 'green' if ep['success'] else 'red'
            alpha = 0.7 if ep['success'] else 0.3
            ax2.plot(range(len(z_values)), z_values, color=color, alpha=alpha, lw=1.5)
        
        # 画目标深度（取第一个episode的目标）
        if result['trajectories']:
            target_z = result['trajectories'][0]['target'][2]
            ax2.axhline(y=target_z, color='gold', linestyle='--', lw=2, label=f'Target Depth')
        
        ax2.set_xlabel('Step'); ax2.set_ylabel('Z / Depth (m)')
        ax2.set_title('Depth over Time (All Episodes)')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close(fig)

    def save_all_trajectory_plots(self, results: Dict[str, Any], output_dir: str, max_individual: int = 3):
        """保存所有模型的轨迹图"""
        plots_dir = os.path.join(output_dir, "trajectory_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"\n>>> Saving trajectory plots to {plots_dir}")
        
        for name, res in results.items():
            if 'error' in res or 'trajectories' not in res:
                continue
            
            model_dir = os.path.join(plots_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            model_type = res.get('type', '')
            n_eps = len(res['trajectories'])
            
            # 保存单条轨迹图（最多 max_individual 条）
            for i in range(min(max_individual, n_eps)):
                if model_type in ['2d_dynamic', '2d_framestack']:
                    path = os.path.join(model_dir, f"trajectory_ep{i+1}.png")
                    self.plot_2d_trajectory(res, i, path)
                elif model_type == '3d_fixed':
                    path = os.path.join(model_dir, f"trajectory_ep{i+1}.png")
                    self.plot_3d_trajectory(res, i, path)
            
            # 保存多轨迹汇总图
            if model_type in ['2d_dynamic', '2d_framestack']:
                path = os.path.join(model_dir, "all_trajectories.png")
                self.plot_2d_multiple_trajectories(res, path)
            elif model_type == '3d_fixed':
                path = os.path.join(model_dir, "all_trajectories.png")
                self.plot_3d_multiple_trajectories(res, path)
        
        print(f"  ✓ Trajectory plots saved")

    def save_trajectory_gif(self, result: Dict, save_path: str, fps: int = 15, every_n: int = 2):
        """为动态目标结果保存GIF"""
        if 'trajectories' not in result or not result['trajectories']:
            print("No trajectories to save")
            return
        
        traj = result['trajectories'][0]  # 取第一个episode
        positions = np.array(traj['positions'])
        
        if 'step_target' in traj:
            target_path = np.array(traj['step_target'])
            has_dynamic_target = True
        else:
            target_path = np.array([traj['target'][:2]] * len(positions))
            has_dynamic_target = False
        
        n_frames = len(positions)
        indices = list(range(0, n_frames, every_n))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("matplotlib.animation not available")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        margin = 1.5
        x_min = min(positions[:, 0].min(), target_path[:, 0].min()) - margin
        x_max = max(positions[:, 0].max(), target_path[:, 0].max()) + margin
        y_min = min(positions[:, 1].min(), target_path[:, 1].min()) - margin
        y_max = max(positions[:, 1].max(), target_path[:, 1].max()) + margin
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        
        line_auv, = ax.plot([], [], 'b-', lw=2, label='AUV')
        if has_dynamic_target:
            line_tgt, = ax.plot([], [], 'r--', lw=1.5, label='Target Path')
        point_auv, = ax.plot([], [], 'go', markersize=14)
        point_tgt, = ax.plot([], [], 'r*', markersize=18)
        ax.legend(loc='upper right')
        title_text = ax.set_title('')

        def init():
            line_auv.set_data([], [])
            if has_dynamic_target:
                line_tgt.set_data([], [])
            point_auv.set_data([], []); point_tgt.set_data([], [])
            title_text.set_text('')
            return (line_auv, point_auv, point_tgt, title_text) + ((line_tgt,) if has_dynamic_target else ())

        def animate(i):
            idx = indices[i] if i < len(indices) else indices[-1]
            line_auv.set_data(positions[:idx+1, 0], positions[:idx+1, 1])
            if has_dynamic_target:
                line_tgt.set_data(target_path[:idx+1, 0], target_path[:idx+1, 1])
            point_auv.set_data([positions[idx, 0]], [positions[idx, 1]])
            point_tgt.set_data([target_path[idx, 0]], [target_path[idx, 1]])
            success = result['episodes'][0]['success']
            title_text.set_text(f'{result["model"]} | Step {idx}/{n_frames-1} ({"Success" if success else "Fail"})')
            return (line_auv, point_auv, point_tgt, title_text) + ((line_tgt,) if has_dynamic_target else ())

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices), interval=1000//fps, blit=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"  GIF saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--dynamic-only", action="store_true", help="Evaluate dynamic target models only")
    parser.add_argument("--3d-only", action="store_true", help="Evaluate 3D models only")
    parser.add_argument("--framestack-only", action="store_true", help="Evaluate framestack models only")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per model")
    parser.add_argument("--target-motion", type=str, default='constant', 
                        choices=['constant', 'random_walk', 'circular', 'evasive'])
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save trajectory plots (default: True)")
    parser.add_argument("--save-gif", action="store_true", help="Save GIFs for dynamic models")
    parser.add_argument("--max-plots", type=int, default=3, help="Max individual trajectory plots per model")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    # 确定要评估的模型
    if args.models:
        model_names = args.models
    elif args.dynamic_only:
        model_names = ['dynamic_mlp', 'dynamic_lstm']
    elif args.framestack_only:
        model_names = [k for k in MODEL_CONFIGS if 'stack' in k]
    elif args.all or args.all is None:
        model_names = list(MODEL_CONFIGS.keys())
    else:
        model_names = list(MODEL_CONFIGS.keys())
    
    # 过滤存在的模型
    existing_models = []
    for name in model_names:
        if name in MODEL_CONFIGS:
            ckpt_path = os.path.join(PROJECT_ROOT, MODEL_CONFIGS[name]['checkpoint'])
            if os.path.exists(ckpt_path):
                existing_models.append(name)
            else:
                print(f"[SKIP] {name}: checkpoint not found")
    
    if not existing_models:
        print("No models to evaluate!")
        return
    
    # 评估
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_all(existing_models, args.episodes, args.target_motion, args.verbose)
    
    # 输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "visualization/eval_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # 报告
    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")
    evaluator.generate_comparison_report(results, report_path)
    
    # 柱状图
    plot_path = os.path.join(output_dir, f"comparison_chart_{timestamp}.png")
    evaluator.plot_comparison_bar(results, plot_path)
    
    # JSON 结果
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'} for k, v in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {json_path}")
    
    # 轨迹图 (默认开启)
    if args.save_plots:
        evaluator.save_all_trajectory_plots(results, output_dir, max_individual=args.max_plots)
    
    # GIF (可选)
    if args.save_gif:
        gif_dir = os.path.join(output_dir, f"gifs_{timestamp}")
        os.makedirs(gif_dir, exist_ok=True)
        print(f"\n>>> Saving GIFs to {gif_dir}")
        for name, res in results.items():
            if 'error' not in res and res.get('type') in ['2d_dynamic', '2d_framestack', '3d_fixed']:
                gif_path = os.path.join(gif_dir, f"{name}.gif")
                evaluator.save_trajectory_gif(res, gif_path)
        print(f"  ✓ GIFs saved")
    
    print(f"\n{'='*60}")
    print(f"✓ All outputs saved to {output_dir}")
    print(f"  - comparison_report_{timestamp}.txt")
    print(f"  - comparison_chart_{timestamp}.png")
    print(f"  - results_{timestamp}.json")
    if args.save_plots:
        print(f"  - trajectory_plots/  (轨迹图)")
    if args.save_gif:
        print(f"  - gifs_{timestamp}/  (GIF动画)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
