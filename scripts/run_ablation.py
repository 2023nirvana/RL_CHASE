#!/usr/bin/env python3
"""
消融实验运行脚本
================

支持运行单个或批量消融实验，并自动评估和可视化。
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.ablation_config import ABLATION_EXPERIMENTS, get_ablation_config


def build_online_ppo_command(
    experiment_name: str,
    low_level_ckpt: str = "low_near/v4_tight_heading_terminal/checkpoints/best.pth",
    pretrain_ckpt: Optional[str] = "checkpoints/td3bc_v4_2m/best.pth",
    total_timesteps: int = 200000,
    # 默认参数
    hidden: tuple = (256, 128),
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 3,
    lr: float = 1e-5,
    lr_end: float = 1e-6,
    clip_range: float = 0.1,
    beta_kl: float = 0.3,
    kl_anneal_frac: float = 0.3,
    eval_freq: int = 10000,
    eval_episodes: int = 20,
    save_freq: int = 50000,
    evader_mode: str = "medium",
    world_size: float = 30.0,
    catch_radius: float = 1.0,
    subgoal_range: float = 0.7,
    low_steps: int = 50,
    max_steps: int = 6000,
    num_envs: int = 4,
    r_catch: float = 150.0,
    r_fail: float = -50.0,
    alpha_shaping: float = 5.0,
    c_sg: float = 10.0,
    c_path: float = 5.0,
    lam_time: float = 0.02,
    seed: int = 42,
    **kwargs  # 忽略未知参数
) -> List[str]:
    """构建Online PPO训练命令"""
    
    cmd = [
        "python", "-u", "training/online_ppo_finetune.py",
        "--low-level-ckpt", low_level_ckpt,
        "--experiment-name", experiment_name,
        "--hidden", *[str(h) for h in hidden],
        "--total-timesteps", str(total_timesteps),
        "--n-steps", str(n_steps),
        "--batch-size", str(batch_size),
        "--n-epochs", str(n_epochs),
        "--lr", str(lr),
        "--lr-end", str(lr_end),
        "--clip-range", str(clip_range),
        "--beta-kl", str(beta_kl),
        "--kl-anneal-frac", str(kl_anneal_frac),
        "--eval-freq", str(eval_freq),
        "--eval-episodes", str(eval_episodes),
        "--save-freq", str(save_freq),
        "--evader-mode", evader_mode,
        "--world-size", str(world_size),
        "--catch-radius", str(catch_radius),
        "--subgoal-range", str(subgoal_range),
        "--low-steps", str(low_steps),
        "--max-steps", str(max_steps),
        "--num-envs", str(num_envs),
        "--r-catch", str(r_catch),
        "--r-fail", str(r_fail),
        "--alpha-shaping", str(alpha_shaping),
        "--c-sg", str(c_sg),
        "--c-path", str(c_path),
        "--lam-time", str(lam_time),
        "--seed", str(seed),
        "--normalize-reward",
    ]
    
    if pretrain_ckpt is not None:
        cmd.extend(["--pretrain-ckpt", pretrain_ckpt])
    
    return cmd


def run_ablation_experiment(
    experiment_type: str,
    variant_name: str,
    total_timesteps: int = 200000,
    base_config: Dict = None,
    dry_run: bool = False,
    auto_eval: bool = True,
    n_eval_episodes: int = 50,
):
    """运行单个消融实验"""
    
    config = get_ablation_config(experiment_type, variant_name)
    experiment_name = f"ablation_{experiment_type}_{variant_name}"
    
    print(f"\n{'='*60}")
    print(f"Running ablation experiment: {experiment_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    # 合并配置
    run_config = base_config.copy() if base_config else {}
    for key, value in config.items():
        if key not in ['name', 'description']:
            run_config[key] = value
    run_config['total_timesteps'] = total_timesteps
    
    # 构建命令
    cmd = build_online_ppo_command(experiment_name, **run_config)
    
    print("Command:")
    print(" ".join(cmd))
    print()
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return None
    
    # 运行训练
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    print(f"Logging to: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    if process.returncode != 0:
        print(f"\nExperiment {experiment_name} failed with return code {process.returncode}")
        return None
    
    # 自动评估
    if auto_eval:
        checkpoint_path = os.path.join(
            PROJECT_ROOT, "checkpoints", experiment_name, "best.pth"
        )
        if os.path.exists(checkpoint_path):
            print(f"\nRunning evaluation for {experiment_name}...")
            eval_cmd = [
                "python", "evaluation/eval_and_visualize.py",
                "--checkpoint", checkpoint_path,
                "--n-episodes", str(n_eval_episodes),
                "--save-json",
            ]
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT)
    
    return experiment_name


def run_ablation_group(
    experiment_type: str,
    total_timesteps: int = 200000,
    base_config: Dict = None,
    dry_run: bool = False,
    auto_eval: bool = True,
    n_eval_episodes: int = 50,
):
    """运行一组消融实验"""
    
    if experiment_type not in ABLATION_EXPERIMENTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    exp_group = ABLATION_EXPERIMENTS[experiment_type]
    
    print(f"\n{'#'*60}")
    print(f"Running ablation group: {experiment_type}")
    print(f"Description: {exp_group['description']}")
    print(f"Number of experiments: {len(exp_group['experiments'])}")
    print(f"{'#'*60}\n")
    
    results = []
    for exp in exp_group["experiments"]:
        result = run_ablation_experiment(
            experiment_type=experiment_type,
            variant_name=exp["name"],
            total_timesteps=total_timesteps,
            base_config=base_config,
            dry_run=dry_run,
            auto_eval=auto_eval,
            n_eval_episodes=n_eval_episodes,
        )
        results.append((exp["name"], result))
    
    return results


def compare_ablation_results(experiment_type: str):
    """比较消融实验结果"""
    
    if experiment_type not in ABLATION_EXPERIMENTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    exp_group = ABLATION_EXPERIMENTS[experiment_type]
    
    print(f"\n{'='*60}")
    print(f"Ablation Results: {experiment_type}")
    print(f"{'='*60}\n")
    
    results = []
    for exp in exp_group["experiments"]:
        experiment_name = f"ablation_{experiment_type}_{exp['name']}"
        eval_json = os.path.join(
            PROJECT_ROOT, "checkpoints", experiment_name, 
            "eval_visuals", "best_eval_results.json"
        )
        
        if os.path.exists(eval_json):
            with open(eval_json) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            is_baseline = exp["name"] == exp_group["baseline"]
            results.append({
                "name": exp["name"],
                "success_rate": summary.get("success_rate", 0) * 100,
                "avg_steps": summary.get("avg_steps_success", 0),
                "avg_reward": summary.get("avg_reward", 0),
                "baseline": is_baseline,
            })
        else:
            results.append({
                "name": exp["name"],
                "success_rate": None,
                "avg_steps": None,
                "avg_reward": None,
                "baseline": exp["name"] == exp_group["baseline"],
            })
    
    # 打印结果表格
    print(f"{'Variant':<20} {'Success%':>10} {'Avg Steps':>12} {'Avg Reward':>12} {'Note':<10}")
    print("-" * 70)
    
    for r in results:
        if r["success_rate"] is not None:
            note = "(baseline)" if r["baseline"] else ""
            print(f"{r['name']:<20} {r['success_rate']:>9.1f}% {r['avg_steps']:>12.1f} {r['avg_reward']:>12.2f} {note}")
        else:
            print(f"{r['name']:<20} {'N/A':>10} {'N/A':>12} {'N/A':>12} (not run)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--type', type=str, required=True,
                       choices=list(ABLATION_EXPERIMENTS.keys()) + ['all', 'list'],
                       help='Experiment type to run')
    parser.add_argument('--variant', type=str, default=None,
                       help='Specific variant to run (if not specified, run all)')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip automatic evaluation')
    parser.add_argument('--n-eval-episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results of existing experiments')
    parser.add_argument('--pretrain-ckpt', type=str, 
                       default='checkpoints/td3bc_v4_2m/best.pth',
                       help='Pretraining checkpoint')
    parser.add_argument('--low-level-ckpt', type=str,
                       default='low_near/v4_tight_heading_terminal/checkpoints/best.pth',
                       help='Low-level checkpoint')
    args = parser.parse_args()
    
    if args.type == 'list':
        from configs.ablation_config import list_all_experiments
        list_all_experiments()
        return
    
    if args.compare:
        if args.type == 'all':
            for exp_type in ABLATION_EXPERIMENTS.keys():
                compare_ablation_results(exp_type)
        else:
            compare_ablation_results(args.type)
        return
    
    base_config = {
        'pretrain_ckpt': args.pretrain_ckpt,
        'low_level_ckpt': args.low_level_ckpt,
    }
    
    if args.type == 'all':
        for exp_type in ABLATION_EXPERIMENTS.keys():
            run_ablation_group(
                experiment_type=exp_type,
                total_timesteps=args.total_timesteps,
                base_config=base_config,
                dry_run=args.dry_run,
                auto_eval=not args.no_eval,
                n_eval_episodes=args.n_eval_episodes,
            )
    elif args.variant:
        run_ablation_experiment(
            experiment_type=args.type,
            variant_name=args.variant,
            total_timesteps=args.total_timesteps,
            base_config=base_config,
            dry_run=args.dry_run,
            auto_eval=not args.no_eval,
            n_eval_episodes=args.n_eval_episodes,
        )
    else:
        run_ablation_group(
            experiment_type=args.type,
            total_timesteps=args.total_timesteps,
            base_config=base_config,
            dry_run=args.dry_run,
            auto_eval=not args.no_eval,
            n_eval_episodes=args.n_eval_episodes,
        )


if __name__ == '__main__':
    main()
