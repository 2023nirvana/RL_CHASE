"""
直接推力版 - 专家数据采集
============================

用 PID 版本已训练好的模型（v4_tight_heading_terminal）跑 episode，
将每步实际下达给动力学的 8 维推力 (last_thrust_command) 作为专家动作保存。

保存格式与高层 pipeline 的 expert_dataset.npz 一致：
  states (N, 8), actions (N, 8), rewards (N,),
  next_states (N, 8), dones (N,), meta (JSON)

直接推力环境的观测空间 = PID 版本的观测空间（8 维），
只是动作从 3 维 PID 命令变成了对应的 8 维推力。

用法：
    python training/collect_expert_direct_thrust.py \\
        --pid-model low_near/v4_tight_heading_terminal/checkpoints/best.pth \\
        --episodes 2000 \\
        --output data/expert_direct_thrust/expert_dataset.npz
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous
from training.train_2d_v3_continuous import GaussianActorCritic


def load_pid_model(ckpt_path, device):
    """加载 PID 版本（3 维动作）的训练好模型"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden = ckpt.get("config", {}).get("network", {}).get(
        "feature_extractor", {}
    ).get("hidden_sizes", [128, 64])
    model = GaussianActorCritic(obs_dim, act_dim, hidden=tuple(hidden)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def collect_episodes(model, device, args):
    """用 PID 模型跑 episode，记录推力 as expert action"""
    env = LowLevelAUV2DEnvV3Continuous(
        reach_threshold=args.reach_threshold,
        target_distance_min=args.target_distance_min,
        target_distance_max=args.target_distance_max,
        max_episode_seconds=args.max_episode_seconds,
        reward_heading_terminal=args.reward_heading_terminal,
        reward_time_penalty=args.reward_time_penalty,
        init_velocity_range=args.init_velocity_range,
    )

    thrust_scale = args.thrust_scale
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    success_count = 0
    total_transitions = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed is not None else None)
        done = False
        ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []

        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_3d, _, _ = model.get_action(x, deterministic=True)
            a_np = action_3d[0].cpu().numpy()

            # 执行 PID 版本的 step（内部会计算推力并赋给 last_thrust_command）
            next_obs, reward, term, trunc, info = env.step(a_np)

            # 读取 PID 控制器实际下达的 8 维推力
            thrust_8d = env.inner.last_thrust_command.copy()
            # 归一化到 [-1, 1] 用于训练直接推力网络
            thrust_normalized = np.clip(thrust_8d / thrust_scale, -1.0, 1.0)

            ep_states.append(obs.copy())
            ep_actions.append(thrust_normalized.astype(np.float32))
            ep_rewards.append(reward)
            ep_next_states.append(next_obs.copy())
            ep_dones.append(float(term or trunc))

            obs = next_obs
            done = term or trunc

        is_success = info.get("termination_reason") == "success"

        # 可选：只保留成功 episode
        if args.only_success and not is_success:
            continue

        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_next_states.extend(ep_next_states)
        all_dones.extend(ep_dones)
        total_transitions += len(ep_states)

        if is_success:
            success_count += 1

        if (ep + 1) % 100 == 0:
            print(f"  [{ep+1}/{args.episodes}] transitions={total_transitions} "
                  f"succ={success_count}/{ep+1}={success_count/(ep+1):.0%}",
                  flush=True)

    env.close()

    return {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "next_states": np.array(all_next_states, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
        "meta": {
            "total_transitions": total_transitions,
            "total_episodes": args.episodes,
            "success_count": success_count,
            "success_rate": success_count / max(1, args.episodes),
            "only_success": args.only_success,
            "thrust_scale": args.thrust_scale,
            "reach_threshold": args.reach_threshold,
            "init_velocity_range": args.init_velocity_range,
            "pid_model": args.pid_model,
            "obs_dim": 8,
            "act_dim": 8,
            "date": datetime.now().isoformat(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert data: PID model → 8-dim thrust")
    parser.add_argument("--pid-model", type=str, required=True,
                        help="PID 版本模型 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default="data/expert_direct_thrust/expert_dataset.npz")
    parser.add_argument("--only-success", action="store_true",
                        help="只保留成功 episode")
    parser.add_argument("--seed", type=int, default=0)
    # 环境参数（与 v4_tight_heading_terminal 一致）
    parser.add_argument("--reach-threshold", type=float, default=0.10)
    parser.add_argument("--target-distance-min", type=float, default=0.3)
    parser.add_argument("--target-distance-max", type=float, default=1.0)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    parser.add_argument("--reward-heading-terminal", type=float, default=0.5)
    parser.add_argument("--reward-time-penalty", type=float, default=-0.2)
    parser.add_argument("--init-velocity-range", type=float, default=1.0)
    parser.add_argument("--thrust-scale", type=float, default=40.0,
                        help="推力归一化尺度 (N)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading PID model: {args.pid_model}")

    model = load_pid_model(args.pid_model, device)
    print(f"Collecting {args.episodes} episodes...")

    data = collect_episodes(model, device, args)

    # 保存
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
        dones=data["dones"],
        meta=json.dumps(data["meta"]),
    )
    print(f"\nSaved {data['states'].shape[0]} transitions → {args.output}")
    print(f"  obs_dim={data['states'].shape[1]}, act_dim={data['actions'].shape[1]}")
    print(f"  success_rate={data['meta']['success_rate']:.0%}")


if __name__ == "__main__":
    main()
