#!/usr/bin/env python3
"""Render multiple successful GIF cases for xx_continuous PPO checkpoints."""

import argparse
import os
from typing import List

import torch

from xx_continuous import (
    DEVICE,
    ContinuousHighLevelPursuitEvasion,
    GaussianActorCritic,
    greedy_policy_from_model,
    plot_episode,
    run_episode,
    save_gif,
)


def load_model(ckpt_path: str) -> tuple[GaussianActorCritic, dict]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    cfg = ckpt["cfg"]
    model = GaussianActorCritic(obs_dim, act_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def collect_success_cases(
    ckpt_path: str,
    num_cases: int,
    max_trials: int,
    seed_start: int,
    output_prefix: str,
) -> List[str]:
    model, cfg = load_model(ckpt_path)
    policy = greedy_policy_from_model(model)

    saved = []
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    for offset in range(max_trials):
        seed = seed_start + offset
        env = ContinuousHighLevelPursuitEvasion(cfg)
        frames, rewards = run_episode(env, policy, seed=seed)
        info = frames[-1][2] if frames else {}
        if not info.get("captured", False):
            continue

        case_idx = len(saved) + 1
        gif_path = f"{output_prefix}_success_{case_idx}.gif"
        png_path = f"{output_prefix}_success_{case_idx}.png"
        save_gif(frames, env, f"PPO Success #{case_idx} | seed={seed}", gif_path)
        plot_episode(frames, env, f"PPO Success #{case_idx} | seed={seed}", png_path)
        total_reward = sum(rewards)
        print(
            f"saved case={case_idx} seed={seed} reward={total_reward:+.1f} "
            f"steps={info.get('step_n', len(frames) - 1)} gif={gif_path}"
        )
        saved.append(gif_path)
        if len(saved) >= num_cases:
            break

    return saved


def main():
    parser = argparse.ArgumentParser(description="Render successful xx_continuous PPO episodes")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="high_tabular/xx_continuous_u500_final.pth",
        help="Checkpoint path produced by xx_continuous.py",
    )
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="high_tabular/xx_continuous_u500",
        help="Output prefix for generated gif/png cases",
    )
    args = parser.parse_args()

    saved = collect_success_cases(
        ckpt_path=args.ckpt,
        num_cases=args.num_cases,
        max_trials=args.max_trials,
        seed_start=args.seed_start,
        output_prefix=args.output_prefix,
    )

    if not saved:
        print("No successful cases found in searched seeds.")
    else:
        print(f"Saved {len(saved)} successful case(s).")


if __name__ == "__main__":
    main()