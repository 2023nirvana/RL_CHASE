#!/usr/bin/env python3
"""
v5 直接推力底层 多种子评估
==========================
用多个随机种子量化底层控制器的真实成功率。

用法:
    python evaluation/eval_v5_direct_thrust_multiseed.py \
        --checkpoint low_near/v5_direct_thrust/checkpoints/best.pth \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --episodes-per-seed 100
"""

import os, sys, argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_direct_thrust import LowLevelAUV2DEnvV3DirectThrust


def load_direct_thrust_model(ckpt_path, device):
    """加载 v5 直接推力模型。"""
    ckpt = torch.load(ckpt_path, map_location=device)

    obs_dim = ckpt.get("obs_dim", 8)
    act_dim = ckpt.get("act_dim", 8)

    config = ckpt.get("config", {})
    network_cfg = config.get("network", {})
    feature_cfg = network_cfg.get("feature_extractor", {})
    hidden = tuple(feature_cfg.get("hidden_sizes", [256, 128]))

    # 与训练时相同的 GaussianActorCritic
    import torch.nn as nn

    class GaussianActor(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden):
            super().__init__()
            layers = []
            prev = obs_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            self.actor_body = nn.Sequential(*layers)
            self.mu_head = nn.Linear(prev, act_dim)

        def forward(self, x):
            return torch.tanh(self.mu_head(self.actor_body(x)))

    model = GaussianActor(obs_dim, act_dim, hidden)

    state_dict = ckpt["model"]
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor_body") or k.startswith("mu_head"):
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded v5 direct thrust: obs={obs_dim}, act={act_dim}, hidden={hidden}")
    return model


def evaluate_seed(model, device, seed, n_episodes, env_kwargs):
    """在给定种子下跑 n_episodes，返回统计。"""
    env = LowLevelAUV2DEnvV3DirectThrust(seed=seed, **env_kwargs)

    successes = 0
    errors = []
    steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 10000 + ep)
        done = False
        step_count = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_t).cpu().numpy().flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        if info.get("termination_reason") == "success":
            successes += 1
            steps_list.append(step_count)
            errors.append(info.get("pos_error", 0.0))
        else:
            errors.append(info.get("pos_error", float("inf")))

    env.close()

    success_rate = successes / n_episodes
    avg_error = np.mean([e for e in errors if e < float("inf")]) if successes > 0 else float("inf")
    avg_steps = np.mean(steps_list) if steps_list else float("inf")
    median_steps = np.median(steps_list) if steps_list else float("inf")

    return {
        "seed": seed,
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_error": avg_error,
        "avg_steps": avg_steps,
        "median_steps": median_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="low_near/v5_direct_thrust/checkpoints/best.pth")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    # 环境参数（与训练一致）
    parser.add_argument("--target-distance-max", type=float, default=1.0)
    parser.add_argument("--target-distance-min", type=float, default=0.3)
    parser.add_argument("--reach-threshold", type=float, default=0.10)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    parser.add_argument("--init-velocity-range", type=float, default=1.0)
    parser.add_argument("--thrust-scale", type=float, default=40.0)
    parser.add_argument("--reward-heading-terminal", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_direct_thrust_model(args.checkpoint, device)

    env_kwargs = dict(
        target_distance_max=args.target_distance_max,
        target_distance_min=args.target_distance_min,
        reach_threshold=args.reach_threshold,
        max_episode_seconds=args.max_episode_seconds,
        init_velocity_range=args.init_velocity_range,
        thrust_scale=args.thrust_scale,
        reward_heading_terminal=args.reward_heading_terminal,
    )

    print(f"\n{'='*60}")
    print(f" v5 Direct Thrust Multi-Seed Evaluation")
    print(f" Seeds: {args.seeds}")
    print(f" Episodes per seed: {args.episodes_per_seed}")
    print(f" init_velocity_range: {args.init_velocity_range}")
    print(f"{'='*60}\n")

    all_results = []
    total_success = 0
    total_episodes = 0

    for seed in args.seeds:
        result = evaluate_seed(model, device, seed, args.episodes_per_seed, env_kwargs)
        all_results.append(result)
        total_success += result["successes"]
        total_episodes += result["episodes"]

        print(f"  Seed {seed:3d}: {result['success_rate']*100:6.1f}% "
              f"({result['successes']}/{result['episodes']}) "
              f"avg_err={result['avg_error']:.4f}m "
              f"median_steps={result['median_steps']:.0f}")

    overall_rate = total_success / total_episodes
    rates = [r["success_rate"] for r in all_results]
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    print(f"\n{'='*60}")
    print(f" Overall: {overall_rate*100:.1f}% ({total_success}/{total_episodes})")
    print(f" Mean ± Std: {mean_rate*100:.1f}% ± {std_rate*100:.1f}%")
    print(f" Min: {min(rates)*100:.1f}%  Max: {max(rates)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
