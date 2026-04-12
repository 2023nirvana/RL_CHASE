"""
奖励结构对比评估脚本
====================

对比复杂奖励 vs 简化奖励的训练效果

使用方法：
    python evaluation/eval_reward_comparison.py
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

# 环境
from core.environment.low_level_env_2d import LowLevelAUV2DEnv
from core.environment.low_level_env_2d_v2 import LowLevelAUV2DEnvV2
from core.environment.low_level_env_2d_simple_reward import LowLevelAUV2DSimpleRewardEnv


# 模型配置：复杂奖励 vs 简化奖励
MODEL_CONFIGS = {
    # ========== 复杂奖励（原有环境） ==========
    '2d_v1_far_5m': {
        'type': 'complex',
        'env_class': 'v1',
        'checkpoint': 'checkpoints/2d_v1_far_5m/best_model.pth',
        'target_distance': 5.0,
        'hidden_sizes': [64, 64],
        'description': '2D V1 复杂奖励 (位置+改进+方向+接近+到达+时间)',
        'reward_components': ['position', 'improvement', 'direction', 'proximity', 'goal', 'time']
    },
    '2d_v2_dist_5m': {
        'type': 'complex',
        'env_class': 'v2',
        'checkpoint': 'checkpoints/2d_v2_dist_5m/best_model.pth',
        'target_distance': 5.0,
        'hidden_sizes': [64, 64],
        'description': '2D V2 复杂奖励 (V1+朝向+转圈惩罚)',
        'reward_components': ['position', 'improvement', 'heading', 'direction', 'proximity', 'goal', 'time', 'spin_penalty']
    },
    
    # ========== 简化奖励 ==========
    'sparse_test': {
        'type': 'simple',
        'env_class': 'simple',
        'reward_type': 'sparse_only',
        'checkpoint': 'checkpoints/sparse_test/best_model.pth',
        'target_distance': 3.0,
        'hidden_sizes': [128, 64],
        'description': '纯稀疏奖励 (只有到达给+100)',
        'reward_components': ['goal']
    },
    'linear_test': {
        'type': 'simple',
        'env_class': 'simple',
        'reward_type': 'distance_linear',
        'checkpoint': 'checkpoints/linear_test/best_model.pth',
        'target_distance': 3.0,
        'hidden_sizes': [128, 64],
        'description': '线性距离奖励 (-k*distance + goal)',
        'reward_components': ['distance', 'goal']
    },
    'minimal_test': {
        'type': 'simple',
        'env_class': 'simple',
        'reward_type': 'minimal',
        'checkpoint': 'checkpoints/minimal_test/best_model.pth',
        'target_distance': 3.0,
        'hidden_sizes': [128, 64],
        'description': '最小组合奖励 (距离+朝向+到达)',
        'reward_components': ['distance', 'heading', 'goal', 'time']
    },
    'shaping_test': {
        'type': 'simple',
        'env_class': 'simple',
        'reward_type': 'shaping',
        'checkpoint': 'checkpoints/shaping_test/best_model.pth',
        'target_distance': 3.0,
        'hidden_sizes': [128, 64],
        'description': '势能塑形奖励 (Δdistance+朝向+到达)',
        'reward_components': ['shaping', 'heading', 'goal', 'time']
    },
}


class RewardComparisonEvaluator:
    """奖励结构对比评估器"""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Device: {self.device}")

    def _create_env(self, config: Dict):
        """根据配置创建环境"""
        env_class = config['env_class']
        target_dist = config.get('target_distance', 3.0)
        
        if env_class == 'v1':
            return LowLevelAUV2DEnv(target_distance=target_dist)
        elif env_class == 'v2':
            return LowLevelAUV2DEnvV2(target_distance=target_dist)
        elif env_class == 'simple':
            return LowLevelAUV2DSimpleRewardEnv(
                target_distance=target_dist,
                reward_type=config.get('reward_type', 'minimal')
            )
        else:
            raise ValueError(f"Unknown env_class: {env_class}")

    def _load_actor(self, config: Dict, obs_dim: int, num_actions: int):
        """加载Actor网络"""
        ckpt_path = os.path.join(PROJECT_ROOT, config['checkpoint'])
        if not os.path.exists(ckpt_path):
            return None
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        hidden_sizes = config.get('hidden_sizes', [128, 64])
        
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        
        actor = nn.Sequential(*layers).to(self.device)
        actor.load_state_dict(ckpt['actor'])
        actor.eval()
        return actor

    def evaluate_model(self, name: str, config: Dict, n_episodes: int = 20) -> Dict:
        """评估单个模型"""
        print(f"\n>>> Evaluating: {name}")
        print(f"    {config['description']}")
        
        env = self._create_env(config)
        obs_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        actor = self._load_actor(config, obs_dim, num_actions)
        if actor is None:
            print(f"    [SKIP] Checkpoint not found")
            return {'model': name, 'error': 'checkpoint_not_found'}
        
        results = {
            'model': name,
            'type': config['type'],
            'description': config['description'],
            'reward_components': config['reward_components'],
            'episodes': [],
            'trajectories': []
        }
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            trajectory = {'positions': [], 'target': info.get('target_pos', [0, 0])}
            
            # 获取初始位置
            if hasattr(env, 'inner'):
                trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            
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
                
                if hasattr(env, 'inner'):
                    trajectory['positions'].append(env.inner.dynamics.eta[:2].tolist())
            
            success = info.get('termination_reason') == 'success'
            results['episodes'].append({
                'success': success,
                'steps': steps,
                'reward': total_r,
                'final_error': info.get('pos_error', 0)
            })
            results['trajectories'].append(trajectory)
            
            status = '✓' if success else '✗'
            print(f"    Ep {ep+1:2d}: {status}, steps={steps:4d}, reward={total_r:8.1f}")
        
        # 统计
        successes = sum(1 for e in results['episodes'] if e['success'])
        results['summary'] = {
            'success_rate': successes / n_episodes,
            'avg_reward': np.mean([e['reward'] for e in results['episodes']]),
            'avg_steps': np.mean([e['steps'] for e in results['episodes']]),
            'avg_final_error': np.mean([e['final_error'] for e in results['episodes']])
        }
        
        print(f"    Summary: SR={results['summary']['success_rate']:.1%}, "
              f"Avg Reward={results['summary']['avg_reward']:.1f}")
        
        return results

    def evaluate_all(self, n_episodes: int = 20) -> Dict[str, Any]:
        """评估所有模型"""
        print("=" * 70)
        print("REWARD STRUCTURE COMPARISON EVALUATION")
        print("=" * 70)
        
        all_results = {}
        for name, config in MODEL_CONFIGS.items():
            all_results[name] = self.evaluate_model(name, config, n_episodes)
        
        return all_results

    def generate_comparison_report(self, results: Dict[str, Any], save_path: str = None) -> str:
        """生成对比报告"""
        lines = [
            "=" * 80,
            "REWARD STRUCTURE COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "【复杂奖励 vs 简化奖励】",
            "",
            f"{'Model':<20} {'Type':<10} {'Success Rate':>12} {'Avg Reward':>12} {'Avg Steps':>10}",
            "-" * 70
        ]
        
        # 按类型分组
        complex_models = {k: v for k, v in results.items() if v.get('type') == 'complex'}
        simple_models = {k: v for k, v in results.items() if v.get('type') == 'simple'}
        
        lines.append("\n>>> 复杂奖励模型:")
        for name, res in complex_models.items():
            if 'error' in res:
                lines.append(f"{name:<20} {'ERROR':<10} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            else:
                s = res['summary']
                lines.append(f"{name:<20} {'complex':<10} {s['success_rate']:>11.1%} {s['avg_reward']:>12.1f} {s['avg_steps']:>10.0f}")
        
        lines.append("\n>>> 简化奖励模型:")
        for name, res in simple_models.items():
            if 'error' in res:
                lines.append(f"{name:<20} {'ERROR':<10} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            else:
                s = res['summary']
                lines.append(f"{name:<20} {'simple':<10} {s['success_rate']:>11.1%} {s['avg_reward']:>12.1f} {s['avg_steps']:>10.0f}")
        
        # 对比分析
        lines.extend([
            "",
            "=" * 80,
            "对比分析:",
            "-" * 80
        ])
        
        valid_complex = [v['summary']['success_rate'] for v in complex_models.values() if 'summary' in v]
        valid_simple = [v['summary']['success_rate'] for v in simple_models.values() if 'summary' in v]
        
        if valid_complex and valid_simple:
            avg_complex_sr = np.mean(valid_complex)
            avg_simple_sr = np.mean(valid_simple)
            best_complex = max(complex_models.items(), key=lambda x: x[1].get('summary', {}).get('success_rate', 0))
            best_simple = max(simple_models.items(), key=lambda x: x[1].get('summary', {}).get('success_rate', 0))
            
            lines.extend([
                f"复杂奖励平均成功率: {avg_complex_sr:.1%}",
                f"简化奖励平均成功率: {avg_simple_sr:.1%}",
                "",
                f"最佳复杂奖励: {best_complex[0]} (SR={best_complex[1]['summary']['success_rate']:.1%})",
                f"最佳简化奖励: {best_simple[0]} (SR={best_simple[1]['summary']['success_rate']:.1%})",
            ])
            
            if avg_simple_sr > avg_complex_sr:
                lines.append("\n结论: 简化奖励效果更好！")
            elif avg_complex_sr > avg_simple_sr:
                lines.append("\n结论: 复杂奖励效果更好！")
            else:
                lines.append("\n结论: 两者效果相当")
        
        lines.append("=" * 80)
        report = "\n".join(lines)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {save_path}")
        
        return report

    def plot_comparison_chart(self, results: Dict[str, Any], save_path: str = None):
        """绘制对比柱状图"""
        valid_results = {k: v for k, v in results.items() if 'summary' in v}
        
        # 按类型分组
        complex_models = [(k, v) for k, v in valid_results.items() if v['type'] == 'complex']
        simple_models = [(k, v) for k, v in valid_results.items() if v['type'] == 'simple']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 准备数据
        names = [k for k, _ in complex_models] + [k for k, _ in simple_models]
        colors = ['#2E86AB'] * len(complex_models) + ['#F18F01'] * len(simple_models)
        
        success_rates = [v['summary']['success_rate'] * 100 for _, v in complex_models + simple_models]
        avg_rewards = [v['summary']['avg_reward'] for _, v in complex_models + simple_models]
        avg_steps = [v['summary']['avg_steps'] for _, v in complex_models + simple_models]
        
        # 成功率
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(names)), success_rates, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 105)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        for bar, sr in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{sr:.0f}%', ha='center', fontsize=8)
        
        # 平均奖励
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(names)), avg_rewards, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward Comparison')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 平均步数
        ax3 = axes[2]
        bars3 = ax3.bar(range(len(names)), avg_steps, color=colors, alpha=0.8)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Average Steps')
        ax3.set_title('Average Steps Comparison')
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', alpha=0.8, label='复杂奖励'),
            Patch(facecolor='#F18F01', alpha=0.8, label='简化奖励')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        plt.close(fig)

    def plot_trajectories(self, results: Dict[str, Any], save_dir: str):
        """绘制每个模型的轨迹图"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, res in results.items():
            if 'trajectories' not in res or not res['trajectories']:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for i, (traj, ep) in enumerate(zip(res['trajectories'][:10], res['episodes'][:10])):
                pos = np.array(traj['positions'])
                if len(pos) == 0:
                    continue
                target = np.array(traj['target'])
                color = 'green' if ep['success'] else 'red'
                alpha = 0.7 if ep['success'] else 0.3
                ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
                ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6, alpha=alpha)
                ax.plot(target[0], target[1], '*', color='gold', markersize=12)
            
            successes = sum(1 for e in res['episodes'] if e['success'])
            n_eps = len(res['episodes'])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name}\n{res["description"]}\nSuccess: {successes}/{n_eps} = {successes/n_eps:.1%}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color='green', lw=2, label='Success'),
                Line2D([0], [0], color='red', lw=2, alpha=0.3, label='Fail'),
                Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='', label='Target')
            ])
            
            plt.tight_layout()
            path = os.path.join(save_dir, f'{name}_trajectories.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {path}")


def main():
    evaluator = RewardComparisonEvaluator()
    
    # 评估
    results = evaluator.evaluate_all(n_episodes=20)
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "visualization/eval_reports/reward_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 报告
    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")
    evaluator.generate_comparison_report(results, report_path)
    
    # 柱状图
    chart_path = os.path.join(output_dir, f"comparison_chart_{timestamp}.png")
    evaluator.plot_comparison_chart(results, chart_path)
    
    # 轨迹图
    traj_dir = os.path.join(output_dir, f"trajectories_{timestamp}")
    evaluator.plot_trajectories(results, traj_dir)
    
    # JSON结果
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'} for k, v in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n✓ All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
