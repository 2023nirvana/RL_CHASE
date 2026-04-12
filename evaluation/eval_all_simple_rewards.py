"""
批量评估所有简化奖励模型
==========================

评估所有简化奖励的模型（sparse, linear, minimal, shaping）并生成对比报告
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

from evaluation.eval_simple_reward_ablation import (
    evaluate_model, plot_trajectory_dual, plot_all_trajectories, generate_report
)

# 所有需要评估的模型配置
MODEL_CONFIGS = [
    # Test 系列 (3m)
    {'name': 'sparse_test', 'checkpoint': 'checkpoints/sparse_test/best_model.pth', 
     'reward_type': 'sparse_only', 'target_distance': 3.0, 'description': '稀疏奖励 (3m)'},
    {'name': 'linear_test', 'checkpoint': 'checkpoints/linear_test/best_model.pth',
     'reward_type': 'distance_linear', 'target_distance': 3.0, 'description': '线性距离奖励 (3m)'},
    {'name': 'minimal_test', 'checkpoint': 'checkpoints/minimal_test/best_model.pth',
     'reward_type': 'minimal', 'target_distance': 3.0, 'description': '最小组合奖励 (3m)'},
    {'name': 'shaping_test', 'checkpoint': 'checkpoints/shaping_test/best_model.pth',
     'reward_type': 'shaping', 'target_distance': 3.0, 'description': '势能塑形奖励 (3m)'},
    
    # 5m 系列
    {'name': 'sparse_5m', 'checkpoint': 'checkpoints/sparse_5m/best_model.pth',
     'reward_type': 'sparse_only', 'target_distance': 5.0, 'description': '稀疏奖励 (5m)'},
    {'name': 'linear_5m', 'checkpoint': 'checkpoints/linear_5m/best_model.pth',
     'reward_type': 'distance_linear', 'target_distance': 5.0, 'description': '线性距离奖励 (5m)'},
    {'name': 'minimal_5m', 'checkpoint': 'checkpoints/minimal_5m/best_model.pth',
     'reward_type': 'minimal', 'target_distance': 5.0, 'description': '最小组合奖励 (5m)'},
    {'name': 'shaping_5m', 'checkpoint': 'checkpoints/shaping_5m/best_model.pth',
     'reward_type': 'shaping', 'target_distance': 5.0, 'description': '势能塑形奖励 (5m)'},
    
    # 10m 系列
    {'name': 'sparse_10m', 'checkpoint': 'checkpoints/sparse_10m/best_model.pth',
     'reward_type': 'sparse_only', 'target_distance': 10.0, 'description': '稀疏奖励 (10m)'},
    {'name': 'linear_10m', 'checkpoint': 'checkpoints/linear_10m/best_model.pth',
     'reward_type': 'distance_linear', 'target_distance': 10.0, 'description': '线性距离奖励 (10m)'},
    {'name': 'minimal_10m', 'checkpoint': 'checkpoints/minimal_10m/best_model.pth',
     'reward_type': 'minimal', 'target_distance': 10.0, 'description': '最小组合奖励 (10m)'},
    {'name': 'shaping_10m', 'checkpoint': 'checkpoints/shaping_10m/best_model.pth',
     'reward_type': 'shaping', 'target_distance': 10.0, 'description': '势能塑形奖励 (10m)'},
    
    # Ablation
    {'name': 'ablation_sparse_only_20260203_173518', 
     'checkpoint': 'checkpoints/ablation_sparse_only_20260203_173518/best_model.pth',
     'reward_type': 'sparse_only', 'target_distance': 3.0, 'description': '稀疏奖励 Ablation (3m)'},
]


def evaluate_all_models(n_episodes: int = 20, device: str = 'cuda', 
                       output_dir: str = None, max_individual_plots: int = 3):
    """批量评估所有模型"""
    print("=" * 80)
    print("BATCH EVALUATION: ALL SIMPLE REWARD MODELS")
    print("=" * 80)
    print(f"Total models: {len(MODEL_CONFIGS)}")
    print(f"Episodes per model: {n_episodes}")
    print("=" * 80)
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "visualization", 
                                 f"simple_reward_eval_all_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估每个模型
    for i, config in enumerate(MODEL_CONFIGS):
        print(f"\n[{i+1}/{len(MODEL_CONFIGS)}] Evaluating: {config['name']}")
        checkpoint_path = os.path.join(PROJECT_ROOT, config['checkpoint'])
        
        if not os.path.exists(checkpoint_path):
            print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
            all_results[config['name']] = {'error': 'checkpoint_not_found'}
            continue
        
        try:
            results = evaluate_model(
                checkpoint_path=checkpoint_path,
                reward_type=config['reward_type'],
                n_episodes=n_episodes,
                target_distance=config['target_distance'],
                device=device
            )
            
            if results is None:
                all_results[config['name']] = {'error': 'evaluation_failed'}
                continue
            
            # 添加配置信息
            results['description'] = config['description']
            results['target_distance'] = config['target_distance']
            
            all_results[config['name']] = results
            
            # 为每个模型保存单独的轨迹图（前几个episode）
            model_dir = os.path.join(output_dir, config['name'])
            os.makedirs(model_dir, exist_ok=True)
            
            for j in range(min(max_individual_plots, len(results['trajectories']))):
                traj = results['trajectories'][j]
                ep = results['episodes'][j]
                save_path = os.path.join(model_dir, f"{config['name']}_ep{j+1}_dual.png")
                plot_trajectory_dual(traj, ep, config['name'], j, save_path)
            
            # 保存汇总轨迹图
            plot_all_trajectories(results, model_dir, max_individual=max_individual_plots)
            
            # 保存单独的报告
            report_path = os.path.join(model_dir, f"{config['name']}_report.txt")
            generate_report(results, report_path)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results[config['name']] = {'error': str(e)}
    
    # 生成对比报告
    generate_comparison_report(all_results, output_dir)
    
    # 生成对比图表
    plot_comparison_charts(all_results, output_dir)
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, "all_results.json")
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'} 
                    for k, v in all_results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nJSON results saved to {json_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ All outputs saved to {output_dir}")
    print(f"{'='*80}")
    
    return all_results


def generate_comparison_report(all_results: Dict, output_dir: str):
    """生成对比报告"""
    lines = [
        "=" * 80,
        "SIMPLE REWARD MODELS COMPARISON REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        f"{'Model':<35} {'Distance':<10} {'Success Rate':>12} {'Avg Reward':>12} {'Avg Steps':>10} {'Final Error':>12}",
        "-" * 80
    ]
    
    # 按距离分组
    by_distance = {}
    for name, res in all_results.items():
        if 'error' in res:
            lines.append(f"{name:<35} {'ERROR':<10} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>12}")
            continue
        
        dist = res.get('target_distance', 3.0)
        if dist not in by_distance:
            by_distance[dist] = []
        by_distance[dist].append((name, res))
    
    # 按距离输出
    for dist in sorted(by_distance.keys()):
        lines.append(f"\n>>> Target Distance: {dist}m")
        for name, res in sorted(by_distance[dist], key=lambda x: x[1]['summary']['success_rate'], reverse=True):
            s = res['summary']
            desc = res.get('description', name)
            lines.append(f"{name:<35} {dist:<10.1f} {s['success_rate']:>11.1%} {s['avg_reward']:>12.1f} "
                        f"{s['avg_steps']:>10.0f} {s['avg_final_error']:>12.3f}")
    
    # 最佳模型
    lines.append("\n" + "=" * 80)
    lines.append("BEST MODELS BY METRIC")
    lines.append("=" * 80)
    
    valid_results = {k: v for k, v in all_results.items() if 'summary' in v}
    if valid_results:
        best_sr = max(valid_results.items(), key=lambda x: x[1]['summary']['success_rate'])
        best_reward = max(valid_results.items(), key=lambda x: x[1]['summary']['avg_reward'])
        best_steps = min(valid_results.items(), key=lambda x: x[1]['summary']['avg_steps'])
        best_error = min(valid_results.items(), key=lambda x: x[1]['summary']['avg_final_error'])
        
        lines.append(f"Best Success Rate: {best_sr[0]} ({best_sr[1]['summary']['success_rate']:.1%})")
        lines.append(f"Best Avg Reward: {best_reward[0]} ({best_reward[1]['summary']['avg_reward']:.2f})")
        lines.append(f"Best Avg Steps: {best_steps[0]} ({best_steps[1]['summary']['avg_steps']:.0f})")
        lines.append(f"Best Final Error: {best_error[0]} ({best_error[1]['summary']['avg_final_error']:.3f}m)")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    print("\n" + report)
    
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nComparison report saved to {report_path}")


def plot_comparison_charts(all_results: Dict, output_dir: str):
    """绘制对比图表"""
    valid_results = {k: v for k, v in all_results.items() if 'summary' in v}
    if not valid_results:
        print("No valid results to plot")
        return
    
    # 按距离分组
    by_distance = {}
    for name, res in valid_results.items():
        dist = res.get('target_distance', 3.0)
        if dist not in by_distance:
            by_distance[dist] = []
        by_distance[dist].append((name, res))
    
    # 为每个距离绘制对比图
    for dist in sorted(by_distance.keys()):
        models = by_distance[dist]
        names = [m[0] for m in models]
        success_rates = [m[1]['summary']['success_rate'] * 100 for m in models]
        avg_rewards = [m[1]['summary']['avg_reward'] for m in models]
        avg_steps = [m[1]['summary']['avg_steps'] for m in models]
        avg_errors = [m[1]['summary']['avg_final_error'] for m in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 成功率
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(names)), success_rates, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title(f'Success Rate Comparison ({dist}m)')
        ax1.set_ylim(0, 105)
        for bar, sr in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{sr:.0f}%', ha='center', fontsize=8)
        
        # 平均奖励
        ax2 = axes[0, 1]
        colors2 = ['green' if r > 0 else 'red' for r in avg_rewards]
        bars2 = ax2.bar(range(len(names)), avg_rewards, color=colors2, alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Average Reward')
        ax2.set_title(f'Average Reward Comparison ({dist}m)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 平均步数
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(names)), avg_steps, color='coral', alpha=0.8)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Average Steps')
        ax3.set_title(f'Average Steps Comparison ({dist}m)')
        
        # 最终误差
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(names)), avg_errors, color='purple', alpha=0.8)
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Average Final Error (m)')
        ax4.set_title(f'Final Error Comparison ({dist}m)')
        ax4.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
        ax4.legend()
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_chart_{dist}m.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    # 跨距离对比（按奖励类型）
    reward_types = ['sparse_only', 'distance_linear', 'minimal', 'shaping']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, reward_type in enumerate(reward_types):
        ax = axes[idx // 2, idx % 2]
        type_models = [(k, v) for k, v in valid_results.items() 
                      if v.get('reward_type') == reward_type]
        
        if not type_models:
            ax.set_title(f'{reward_type} (no data)')
            continue
        
        distances = [m[1].get('target_distance', 3.0) for m in type_models]
        success_rates = [m[1]['summary']['success_rate'] * 100 for m in type_models]
        names = [m[0] for m in type_models]
        
        ax.plot(distances, success_rates, 'o-', linewidth=2, markersize=10)
        for i, name in enumerate(names):
            ax.annotate(name, (distances[i], success_rates[i]), 
                       fontsize=8, rotation=45, ha='left')
        ax.set_xlabel('Target Distance (m)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{reward_type} - Success Rate vs Distance')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparison_by_reward_type.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Evaluate All Simple Reward Models")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per model")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-plots", type=int, default=3, help="Max individual plots per model")
    args = parser.parse_args()
    
    evaluate_all_models(
        n_episodes=args.episodes,
        device=args.device,
        output_dir=args.output_dir,
        max_individual_plots=args.max_plots
    )


if __name__ == "__main__":
    main()
