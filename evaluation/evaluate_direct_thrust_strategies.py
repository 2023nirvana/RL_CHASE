#!/usr/bin/env python3
"""
直接推力版 高层策略对比评估
===========================
用 v5 直接推力底层 + 传统高层策略跑追逃评估。
基于 evaluate_high_level_strategies.py，替换底层为直接推力版。
"""
import os, sys, math, argparse, json
from typing import Dict, List, Optional, Any
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from high_tabular.hrl_v3_real_auv_direct_thrust import (
    load_v5_nn, PursuitEnvDirectThrust,
)
from core.agents.classical_pursuit_strategies import make_strategy, BaseStrategy


class StrategyAdapter:
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.name = strategy.name

    def reset(self):
        self.strategy.reset()

    def get_action(self, env) -> np.ndarray:
        eta = env.auv_env.inner.dynamics.eta
        p_pos = np.array([eta[0], eta[1]])
        p_vel = env._get_pursuer_vel()
        p_yaw = float(eta[5])
        e_pos = env.e.copy()
        e_vel = env.ve.copy()
        self.strategy.set_dt(env.dt_evader * env.low_steps)
        subgoal = self.strategy.compute_subgoal(
            p_pos, p_vel, p_yaw, e_pos, e_vel,
            subgoal_radius=env.subgoal_range,
            world_size=env.half,
        )
        delta = subgoal - p_pos
        action = delta / max(env.subgoal_range, 1e-6)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class ExpertAdapter:
    def __init__(self, predict_time: float = 2.0):
        self.name = f"Expert(pred={predict_time}s)"
        self.predict_time = predict_time

    def reset(self):
        pass

    def get_action(self, env) -> np.ndarray:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        e_pred = env.e + env.ve * self.predict_time
        e_pred = np.clip(e_pred, -env.half + 1, env.half - 1)
        direction = e_pred - p
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            action = direction / max(dist, env.subgoal_range)
        else:
            action = np.zeros(2, dtype=np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def rollout(env, adapter, seed: int = 42):
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    adapter.reset()

    eta = env.auv_env.inner.dynamics.eta
    init_p = np.array([eta[0], eta[1]], dtype=np.float32)
    init_e = env.e.copy().astype(np.float32)

    traj_p = [init_p.copy()]
    traj_e = [init_e.copy()]
    vp_list = [env._get_pursuer_vel().copy()]
    ve_list = [env.ve.copy()]
    distances = [float(np.linalg.norm(init_p - init_e))]
    subgoals, sg_origins = [], []
    dense_p = [init_p.copy()]
    dense_e = [init_e.copy()]

    done, info = False, {}
    while not done:
        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]], dtype=np.float32)
        action = adapter.get_action(env)
        sg_origins.append(p.copy())

        obs, reward, done, info = env.step(action)

        corrected = info.get('corrected_subgoal', None)
        if corrected is not None:
            subgoals.append(np.asarray(corrected, dtype=np.float32))
        else:
            dx = float(action[0]) * env.subgoal_range
            dy = float(action[1]) * env.subgoal_range
            sg = np.clip(np.array([p[0]+dx, p[1]+dy], dtype=np.float32),
                         -env.half+0.5, env.half-0.5)
            subgoals.append(sg)

        eta = env.auv_env.inner.dynamics.eta
        traj_p.append(np.array([eta[0], eta[1]], dtype=np.float32))
        traj_e.append(env.e.copy().astype(np.float32))
        vp_list.append(env._get_pursuer_vel().copy())
        ve_list.append(env.ve.copy())
        distances.append(float(np.linalg.norm(traj_p[-1] - traj_e[-1])))

        for sp, se in zip(info.get('sub_traj_p', []), info.get('sub_traj_e', [])):
            dense_p.append(np.asarray(sp, dtype=np.float32))
            dense_e.append(np.asarray(se, dtype=np.float32))

    return {
        "p": np.array(traj_p), "e": np.array(traj_e),
        "vp": np.array(vp_list), "ve": np.array(ve_list),
        "distance": np.array(distances),
        "subgoals": np.array(subgoals) if subgoals else np.zeros((0, 2)),
        "sg_origins": np.array(sg_origins) if sg_origins else np.zeros((0, 2)),
        "dense_p": np.array(dense_p), "dense_e": np.array(dense_e),
        "captured": bool(info.get("captured", False)),
        "steps": int(info.get("steps", len(traj_p) - 1)),
        "seed": seed,
    }


# ---- 可视化函数单独导入自原版 ----
from evaluation.evaluate_high_level_strategies import (
    draw_world, save_static_png, save_dynamics_gif,
    save_comparison_plot, find_best_episode,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-path", type=str,
                        default="low_near/v5_direct_thrust/checkpoints/best.pth")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation/strategy_comparison_direct_thrust")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--world-size", type=float, default=30.0)
    parser.add_argument("--catch-radius", type=float, default=1.0)
    parser.add_argument("--subgoal-range", type=float, default=2.0)
    parser.add_argument("--low-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--thrust-scale", type=float, default=40.0)
    parser.add_argument("--evader-mode", type=str, default="medium")
    parser.add_argument("--v-max-pursuer", type=float, default=0.65)
    parser.add_argument("--a-max-pursuer", type=float, default=0.5)
    parser.add_argument("--v-max-evader", type=float, default=None)
    parser.add_argument("--a-max-evader", type=float, default=None)
    parser.add_argument("--strategies", type=str, nargs="*",
                        default=["pp", "lead", "pn", "apn", "cb", "dp", "expert"])
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading v5 direct thrust model: {args.low_level_path}")
    nn_policy = load_v5_nn(args.low_level_path, device)

    env = PursuitEnvDirectThrust(
        nn_policy=nn_policy, device=device,
        world_size=args.world_size, catch_radius=args.catch_radius,
        max_steps=args.max_steps, subgoal_range=args.subgoal_range,
        low_steps=args.low_steps, seed=42,
        evader_mode=args.evader_mode, thrust_scale=args.thrust_scale,
        v_max_pursuer=args.v_max_pursuer,
        a_max_pursuer=args.a_max_pursuer,
        v_max_evader=args.v_max_evader,
        a_max_evader=args.a_max_evader,
    )

    adapters = []
    for s in args.strategies:
        if s.lower() == "expert":
            adapters.append(ExpertAdapter(predict_time=2.0))
        else:
            adapters.append(StrategyAdapter(make_strategy(s)))

    all_results = {}
    for si, adapter in enumerate(adapters):
        name = adapter.name
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        episodes, caught = [], 0
        for i in range(args.episodes):
            seed = args.base_seed + si * 1000 + i
            trace = rollout(env, adapter, seed=seed)
            episodes.append(trace)
            if trace["captured"]:
                caught += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{args.episodes}] caught={caught}/{i+1}")

        rate = caught / args.episodes
        caught_eps = [ep for ep in episodes if ep["captured"]]
        avg_steps = np.mean([ep["steps"] for ep in caught_eps]) if caught_eps else 0
        avg_min_d = np.mean([np.min(ep["distance"]) for ep in episodes])
        print(f"  => catch={rate:.0%}  steps={avg_steps:.0f}  min_d={avg_min_d:.2f}")

        all_results[name] = {
            "episodes": episodes, "catch_rate": rate,
            "avg_steps": avg_steps, "avg_min_dist": avg_min_d,
            "max_steps": args.max_steps,
        }

        best = find_best_episode(episodes)
        safe = name.replace("(","").replace(")","").replace("=","").replace(" ","_")
        sdir = os.path.join(args.output_dir, safe)
        os.makedirs(sdir, exist_ok=True)
        save_static_png(best, env, os.path.join(sdir, f"{safe}.png"), name)
        if not args.no_gif:
            save_dynamics_gif(best, env, os.path.join(sdir, f"{safe}.gif"), name)

    summary = {n: {"catch_rate": r["catch_rate"], "avg_steps": float(r["avg_steps"]),
                    "avg_min_dist": float(r["avg_min_dist"])}
               for n, r in all_results.items()}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_comparison_plot(all_results, os.path.join(args.output_dir, "comparison.png"))

    print(f"\n{'='*50}\n  DONE\n{'='*50}")
    for n, r in all_results.items():
        print(f"  {n:25s}: catch={r['catch_rate']:.0%}  steps={r['avg_steps']:.0f}")


if __name__ == "__main__":
    main()
