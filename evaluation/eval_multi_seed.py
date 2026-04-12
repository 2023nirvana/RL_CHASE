"""
多随机种子评估底层控制器（V3 连续动作版本）
============================================

用多个随机种子评估底层模型，证明其成功率的稳健性。

使用方法:
    python evaluation/eval_multi_seed.py \
        --model low_near/v4_tight_heading_terminal/checkpoints/best.pth \
        --seeds 10 --episodes 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous


class GaussianActor(nn.Module):
    """V4 高斯策略网络"""
    def __init__(self, obs_dim=8, act_dim=3, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.actor_body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.actor_body(x)
        mu = torch.tanh(self.mu_head(h))
        return mu


def load_v4_nn(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    obs_dim = ckpt.get('obs_dim', 8)
    act_dim = ckpt.get('act_dim', 3)
    config = ckpt.get('config', {})
    network_cfg = config.get('network', {})
    feature_cfg = network_cfg.get('feature_extractor', {})
    hidden = tuple(feature_cfg.get('hidden_sizes', [64, 64]))

    model = GaussianActor(obs_dim, act_dim, hidden)
    state_dict = ckpt['model']
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor_body') or k.startswith('mu_head') or k == 'log_std':
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded V4 NN: obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden}")
    return model, config


def evaluate_one_seed(model, env, device, n_episodes, seed):
    """用给定种子评估一批 episode"""
    successes = []
    rewards = []
    pos_errors = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = model(obs_t).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        success = info.get('termination_reason') == 'success'
        successes.append(success)
        rewards.append(ep_reward)
        pos_errors.append(info.get('pos_error', np.linalg.norm(
            env.inner.dynamics.eta[:2] - env.inner.target_state[:2])))
        lengths.append(ep_len)

        if (ep + 1) % 20 == 0:
            print(f"    Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, "
                  f"success={success}, pos_err={pos_errors[-1]:.4f}")

    return {
        'success_rate': float(np.mean(successes)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_position_error': float(np.mean(pos_errors)),
        'std_position_error': float(np.std(pos_errors)),
        'mean_length': float(np.mean(lengths)),
    }


def run_multi_seed_evaluation(model_path, num_seeds=10, episodes_per_seed=100,
                               base_seed=0, target_distance_max=1.0,
                               target_distance_min=0.3, reach_threshold=0.10,
                               init_velocity_range=1.0, init_yaw_rate_range=0.5):
    """用多个随机种子评估模型（V3 环境）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成种子列表
    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(0, 100000, size=num_seeds).tolist()

    print("=" * 70)
    print(f"Multi-Seed Evaluation (V3): {model_path}")
    print(f"Seeds ({num_seeds}): {seeds}")
    print(f"Episodes per seed: {episodes_per_seed}")
    print(f"Total episodes: {num_seeds * episodes_per_seed}")
    print(f"Target distance: [{target_distance_min}, {target_distance_max}] m")
    print(f"Reach threshold: {reach_threshold} m")
    print(f"Init velocity range: ±{init_velocity_range} m/s")
    print(f"Init yaw rate range: ±{init_yaw_rate_range} rad/s")
    print(f"Device: {device}")
    print("=" * 70)

    # 加载模型
    model, config = load_v4_nn(model_path, device)

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {i+1}/{num_seeds}: {seed} ---")

        env = LowLevelAUV2DEnvV3Continuous(
            seed=seed,
            target_distance_max=target_distance_max,
            target_distance_min=target_distance_min,
            reach_threshold=reach_threshold,
            init_velocity_range=init_velocity_range,
            init_yaw_rate_range=init_yaw_rate_range,
        )

        result = evaluate_one_seed(model, env, device, episodes_per_seed, seed)
        result['seed'] = seed
        all_results.append(result)

        print(f"  Success Rate:    {result['success_rate']:.2%}")
        print(f"  Mean Reward:     {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Mean Pos Error:  {result['mean_position_error']:.4f} m")
        print(f"  Mean Length:     {result['mean_length']:.1f} steps")

        env.close()

    # 汇总统计
    success_rates = [r['success_rate'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    pos_errors = [r['mean_position_error'] for r in all_results]

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (across all seeds)")
    print("=" * 70)
    print(f"\nSuccess Rate:")
    print(f"  Mean:   {np.mean(success_rates):.2%}")
    print(f"  Std:    {np.std(success_rates):.2%}")
    print(f"  Min:    {np.min(success_rates):.2%}")
    print(f"  Max:    {np.max(success_rates):.2%}")
    print(f"  Median: {np.median(success_rates):.2%}")

    print(f"\nMean Reward:")
    print(f"  Mean:   {np.mean(mean_rewards):.2f}")
    print(f"  Std:    {np.std(mean_rewards):.2f}")
    print(f"  Min:    {np.min(mean_rewards):.2f}")
    print(f"  Max:    {np.max(mean_rewards):.2f}")

    print(f"\nPosition Error (m):")
    print(f"  Mean:   {np.mean(pos_errors):.4f}")
    print(f"  Std:    {np.std(pos_errors):.4f}")

    print("\nPer-seed breakdown:")
    print(f"  {'Seed':>8s}  {'Success':>10s}  {'Reward':>10s}  {'PosErr':>10s}  {'Length':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in all_results:
        print(f"  {r['seed']:>8d}  {r['success_rate']:>10.2%}  "
              f"{r['mean_reward']:>10.2f}  {r['mean_position_error']:>10.4f}  "
              f"{r['mean_length']:>10.1f}")

    print("=" * 70)

    # 保存结果
    output = {
        'model_path': model_path,
        'num_seeds': num_seeds,
        'episodes_per_seed': episodes_per_seed,
        'total_episodes': num_seeds * episodes_per_seed,
        'env': 'LowLevelAUV2DEnvV3Continuous',
        'target_distance_range': [target_distance_min, target_distance_max],
        'reach_threshold': reach_threshold,
        'init_velocity_range': init_velocity_range,
        'init_yaw_rate_range': init_yaw_rate_range,
        'timestamp': datetime.now().isoformat(),
        'aggregate': {
            'success_rate_mean': float(np.mean(success_rates)),
            'success_rate_std': float(np.std(success_rates)),
            'success_rate_min': float(np.min(success_rates)),
            'success_rate_max': float(np.max(success_rates)),
            'reward_mean': float(np.mean(mean_rewards)),
            'reward_std': float(np.std(mean_rewards)),
            'pos_error_mean': float(np.mean(pos_errors)),
        },
        'per_seed': all_results,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Multi-seed low-level controller evaluation")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per seed')
    parser.add_argument('--base-seed', type=int, default=0, help='Base seed for generating seed list')
    parser.add_argument('--target-distance-max', type=float, default=1.0, help='Max target distance')
    parser.add_argument('--target-distance-min', type=float, default=0.3, help='Min target distance')
    parser.add_argument('--reach-threshold', type=float, default=0.10, help='Success reach threshold (m)')
    parser.add_argument('--init-velocity-range', type=float, default=1.0, help='Init velocity range (m/s)')
    parser.add_argument('--init-yaw-rate-range', type=float, default=0.5, help='Init yaw rate range (rad/s)')
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                        help='Output directory for results JSON')
    args = parser.parse_args()

    output = run_multi_seed_evaluation(
        model_path=args.model,
        num_seeds=args.seeds,
        episodes_per_seed=args.episodes,
        base_seed=args.base_seed,
        target_distance_max=args.target_distance_max,
        target_distance_min=args.target_distance_min,
        reach_threshold=args.reach_threshold,
        init_velocity_range=args.init_velocity_range,
        init_yaw_rate_range=args.init_yaw_rate_range,
    )

    # 保存 JSON
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model)))
    json_path = os.path.join(args.output_dir, f'{model_name}_multi_seed_eval.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
