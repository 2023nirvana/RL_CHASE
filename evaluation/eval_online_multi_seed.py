#!/usr/bin/env python3
"""
多随机种子评估 Online PPO (v3) 模型
=====================================
评估 BPPO/IQL/TD3BC 预训练后的 Online PPO 模型在不同随机种子下的表现
"""

import os, sys, json, argparse
import numpy as np
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed
from training.offline_ppo_pretrain import HighLevelActorCritic


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        sd = ckpt["model"]
        obs_dim = ckpt.get("obs_dim", 21)
        act_dim = ckpt.get("act_dim", 2)
        hidden = tuple(ckpt.get("hidden", [256, 128]))
    else:
        raise ValueError(f"Unknown ckpt format: {list(ckpt.keys())}")
    model = HighLevelActorCritic(obs_dim, act_dim, hidden).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def evaluate_one_seed(env, model, device, n_episodes, seed):
    """用单个种子评估n_episodes个episode"""
    results = []
    env.seed_val = seed
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        captured = False
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _ = model.get_action(obs_t, deterministic=True)
            a_np = a[0].cpu().numpy()
            
            obs, r, done, info = env.step(a_np)
            ep_reward += r
            steps += 1
            
            if info.get("captured", False):
                captured = True
        
        results.append({
            "reward": ep_reward,
            "steps": steps,
            "captured": captured,
        })
    
    return results


def evaluate_model_multi_seed(model_name, ckpt_path, nn_policy, device, 
                               seeds, n_episodes_per_seed):
    """用多个种子评估单个模型"""
    model = load_model(ckpt_path, device)
    
    all_results = {}
    all_success_rates = []
    all_rewards = []
    all_steps = []
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    for seed in seeds:
        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy, device=device,
            world_size=30.0, catch_radius=1.0, max_steps=6000,
            subgoal_range=0.7, low_steps=50,
            evader_mode="medium", seed=seed,
        )
        
        results = evaluate_one_seed(env, model, device, n_episodes_per_seed, seed)
        
        sr = sum(r["captured"] for r in results) / len(results)
        mean_rew = np.mean([r["reward"] for r in results])
        mean_steps = np.mean([r["steps"] for r in results])
        cap_steps = [r["steps"] for r in results if r["captured"]]
        mean_cap_steps = np.mean(cap_steps) if cap_steps else -1
        
        all_success_rates.append(sr)
        all_rewards.append(mean_rew)
        all_steps.append(mean_steps)
        
        all_results[seed] = {
            "success_rate": sr,
            "mean_reward": mean_rew,
            "mean_steps": mean_steps,
            "mean_capture_steps": mean_cap_steps,
        }
        
        print(f"  Seed {seed:3d}: SR={sr*100:5.1f}%  Rew={mean_rew:7.1f}  "
              f"Steps={mean_steps:5.0f}  CapSteps={mean_cap_steps:5.0f}")
    
    # 汇总统计
    summary = {
        "model": model_name,
        "n_seeds": len(seeds),
        "n_episodes_per_seed": n_episodes_per_seed,
        "total_episodes": len(seeds) * n_episodes_per_seed,
        "success_rate_mean": float(np.mean(all_success_rates)),
        "success_rate_std": float(np.std(all_success_rates)),
        "success_rate_min": float(np.min(all_success_rates)),
        "success_rate_max": float(np.max(all_success_rates)),
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "steps_mean": float(np.mean(all_steps)),
        "per_seed_results": all_results,
    }
    
    print(f"\n  ══ Summary ══")
    print(f"  Success Rate: {summary['success_rate_mean']*100:.1f}% ± "
          f"{summary['success_rate_std']*100:.1f}% "
          f"[{summary['success_rate_min']*100:.0f}%-{summary['success_rate_max']*100:.0f}%]")
    print(f"  Mean Reward:  {summary['reward_mean']:.1f} ± {summary['reward_std']:.1f}")
    print(f"  Mean Steps:   {summary['steps_mean']:.0f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for Online PPO models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                        help="Random seeds to use")
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of episodes per seed")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["v3_bppo_A", "v3_bppo_B", "v3_iql_A", "v3_iql_B", 
                                 "v3_td3bc_A", "v3_td3bc_B"],
                        help="Model names to evaluate")
    parser.add_argument("--ckpt-type", type=str, default="best",
                        choices=["best", "latest", "final"],
                        help="Which checkpoint to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.n_episodes}")
    print(f"Models: {args.models}")
    print(f"Checkpoint type: {args.ckpt_type}")
    
    # 加载底层策略
    low_ckpt = os.path.join(PROJECT_ROOT, 
        "low_near/v4_tight_heading_terminal/checkpoints/best.pth")
    nn_policy = load_v4_nn(low_ckpt, device)
    
    # 评估所有模型
    all_summaries = {}
    for model_name in args.models:
        ckpt_path = os.path.join(PROJECT_ROOT, 
            f"checkpoints/{model_name}/{args.ckpt_type}.pth")
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] {model_name}: {ckpt_path} not found")
            continue
        
        summary = evaluate_model_multi_seed(
            model_name, ckpt_path, nn_policy, device,
            args.seeds, args.n_episodes
        )
        all_summaries[model_name] = summary
    
    # 打印总结对比表
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<15} {'SR Mean':>8} {'SR Std':>7} {'SR Range':>12} {'Reward':>10}")
    print("-"*70)
    for name, s in all_summaries.items():
        sr_range = f"[{s['success_rate_min']*100:.0f}-{s['success_rate_max']*100:.0f}]"
        print(f"{name:<15} {s['success_rate_mean']*100:>7.1f}% {s['success_rate_std']*100:>6.1f}% "
              f"{sr_range:>12} {s['reward_mean']:>9.1f}")
    print("="*70)
    
    # 保存结果
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(PROJECT_ROOT, "evaluation/results_multi_seed")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"online_v3_multi_seed_{timestamp}.json")
    
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seeds": args.seeds,
            "n_episodes_per_seed": args.n_episodes,
            "ckpt_type": args.ckpt_type,
            "results": all_summaries,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
