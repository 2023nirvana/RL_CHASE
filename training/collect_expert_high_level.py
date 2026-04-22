"""
高层专家数据采集 - 追逃任务
================================

用经典策略（Expert + Lead + PN 混合）作为"专家"在
PursuitEnvDirectThrust 环境中跑 episode，
记录每步 (high_obs_21d, subgoal_action_2d, reward, next_obs, done) 作为高层专家数据。

保存格式:
  states      (N, 21)
  actions     (N, 2)       高层动作（subgoal 相对位移，-1~1）
  rewards     (N,)
  next_states (N, 21)
  dones       (N,)
  meta        JSON (包含 episodes / strategies / constraints 等)

用法:
  python training/collect_expert_high_level.py \
    --episodes 2000 \
    --strategies expert lead pn \
    --output data/expert_high_level/expert_dataset.npz
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

from high_tabular.hrl_v3_real_auv_direct_thrust import (
    load_v5_nn, PursuitEnvDirectThrust,
)
from evaluation.evaluate_direct_thrust_strategies import (
    StrategyAdapter, ExpertAdapter,
)
from core.agents.classical_pursuit_strategies import make_strategy


def make_adapter(name: str):
    if name.lower() == "expert":
        return ExpertAdapter(predict_time=2.0)
    strat = make_strategy(name)
    return StrategyAdapter(strat)


def collect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[collect] device={device}")
    nn_policy = load_v5_nn(args.nn_ckpt, device)

    adapters = [make_adapter(s) for s in args.strategies]
    print(f"[collect] adapters: {[a.name for a in adapters]}")

    env = PursuitEnvDirectThrust(
        nn_policy=nn_policy, device=device,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        max_steps=args.max_steps,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=args.seed,
        evader_mode=args.evader_mode,
        v_max_pursuer=args.v_max_pursuer,
        a_max_pursuer=args.a_max_pursuer,
        v_max_evader=args.v_max_evader,
        a_max_evader=args.a_max_evader,
    )

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    success_count = 0
    total_transitions = 0
    t0 = time.time()

    for ep in range(args.episodes):
        adapter = adapters[ep % len(adapters)]
        env.rng = np.random.default_rng(args.seed + ep)
        obs = env.reset()
        adapter.reset()

        ep_s, ep_a, ep_r, ep_ns, ep_d = [], [], [], [], []
        done = False
        info = {}

        while not done:
            action = adapter.get_action(env)
            next_obs, reward, done, info = env.step(action)

            ep_s.append(obs.astype(np.float32))
            ep_a.append(action.astype(np.float32))
            ep_r.append(float(reward))
            ep_ns.append(next_obs.astype(np.float32))
            ep_d.append(bool(done))

            obs = next_obs

        captured = bool(info.get("captured", False))
        if captured:
            success_count += 1

        # --only-success 过滤
        if args.only_success and not captured:
            continue

        all_s.extend(ep_s)
        all_a.extend(ep_a)
        all_r.extend(ep_r)
        all_ns.extend(ep_ns)
        all_d.extend(ep_d)
        total_transitions += len(ep_s)

        if (ep + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (ep + 1) / max(dt, 1e-6)
            eta = (args.episodes - ep - 1) / max(rate, 1e-6)
            print(f"  [{ep+1}/{args.episodes}] "
                  f"success={success_count}/{ep+1} ({success_count/(ep+1):.1%}) "
                  f"transitions={total_transitions} "
                  f"rate={rate:.2f} ep/s ETA={eta/60:.1f}min",
                  flush=True)

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    states = np.stack(all_s).astype(np.float32)
    actions = np.stack(all_a).astype(np.float32)
    rewards = np.array(all_r, dtype=np.float32)
    next_states = np.stack(all_ns).astype(np.float32)
    dones = np.array(all_d, dtype=bool)

    meta = {
        "episodes_run": args.episodes,
        "episodes_kept": int(success_count if args.only_success else args.episodes),
        "success_count": int(success_count),
        "success_rate": float(success_count / args.episodes),
        "transitions": int(states.shape[0]),
        "obs_dim": int(states.shape[1]),
        "act_dim": int(actions.shape[1]),
        "strategies": args.strategies,
        "evader_mode": args.evader_mode,
        "v_max_pursuer": args.v_max_pursuer,
        "a_max_pursuer": args.a_max_pursuer,
        "v_max_evader": args.v_max_evader,
        "a_max_evader": args.a_max_evader,
        "subgoal_range": args.subgoal_range,
        "low_steps": args.low_steps,
        "only_success": args.only_success,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    np.savez(
        args.output,
        states=states, actions=actions, rewards=rewards,
        next_states=next_states, dones=dones,
        meta=json.dumps(meta),
    )

    print(f"\n[collect] saved: {args.output}")
    print(f"[collect] transitions: {states.shape[0]}  obs_dim: {states.shape[1]}  act_dim: {actions.shape[1]}")
    print(f"[collect] success: {success_count}/{args.episodes} ({success_count/args.episodes:.1%})")
    print(f"[collect] elapsed: {(time.time()-t0)/60:.1f} min")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--nn-ckpt", type=str,
                   default="low_near/v5_direct_thrust/checkpoints/best.pth")
    p.add_argument("--output", type=str,
                   default="data/expert_high_level/expert_dataset.npz")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--strategies", nargs="+", default=["expert", "lead", "pn"])
    p.add_argument("--only-success", action="store_true", default=True)
    p.add_argument("--no-only-success", dest="only_success", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    # env config
    p.add_argument("--world-size", type=float, default=30.0)
    p.add_argument("--catch-radius", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--subgoal-range", type=float, default=2.0)
    p.add_argument("--low-steps", type=int, default=50)
    p.add_argument("--evader-mode", type=str, default="medium")
    p.add_argument("--v-max-pursuer", type=float, default=0.65)
    p.add_argument("--a-max-pursuer", type=float, default=0.5)
    p.add_argument("--v-max-evader", type=float, default=0.85)
    p.add_argument("--a-max-evader", type=float, default=0.25)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    collect(args)
