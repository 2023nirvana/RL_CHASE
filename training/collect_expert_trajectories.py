"""
收集经典策略的专家轨迹用于 Offline PPO 预训练
==============================================
混合多种经典策略 (PP, Lead, PN, APN, CB, DP, Expert) 的成功 episode，
记录 (state, action, reward, next_state, done) 五元组。
"""

import os
import sys
import math
import json
import argparse
import time
import multiprocessing as mp
from functools import partial
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from core.agents.classical_pursuit_strategies import make_strategy, BaseStrategy


# ─────────── 策略适配器 ───────────

class StrategyAdapter:
    """将经典策略输出转换为 [-1,1]^2 归一化动作"""

    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.name = strategy.name

    def reset(self):
        self.strategy.reset()

    def get_action(self, env: PursuitEnvRealAUVFixed) -> np.ndarray:
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
    """预测启发式策略"""

    def __init__(self, predict_time: float = 2.0):
        self.name = f"Expert_pred{predict_time:.1f}s"
        self.predict_time = predict_time

    def reset(self):
        pass

    def get_action(self, env: PursuitEnvRealAUVFixed) -> np.ndarray:
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


# ─────────── 奖励函数 (HIGH.pdf 设计) ───────────

def compute_high_reward(
    env: PursuitEnvRealAUVFixed,
    prev_dist: float,
    curr_dist: float,
    captured: bool,
    out_of_bounds: bool,
    collision: bool,
    subgoal: np.ndarray = None,
    done: bool = False,
    gamma: float = 0.99,
    R_catch: float = 100.0,
    R_fail: float = -50.0,
    alpha: float = 5.0,
    c_sg: float = 15.0,
    d_sg_safe: float = 1.5,
    c_path: float = 8.0,
    lam_time: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """四项奖励: event + shaping + subgoal_path_quality + time

    设计逻辑:
    - 高层本质是路径规划(输出subgoal)，只应惩罚规划质量，不惩罚执行位置
    - 移除 r_obs (追踪者位置惩罚)：追踪者位置是底层控制结果，高层无法直接控制
    - r_sg_obs 统一为路径质量惩罚，包含:
      (a) 子目标是否靠近障碍物 (d_sg_safe 从 0.5→1.5m，覆盖原 r_obs 的安全距离)
      (b) 追踪者→子目标路径是否穿越障碍物 (新增路径检测)
    """
    r_event = 0.0
    r_shaping = 0.0
    r_sg_obs = 0.0
    r_time = -lam_time

    # 1) 事件奖励 —— 只在 episode 终止时给
    if done:
        if captured:
            r_event = R_catch
        elif out_of_bounds or collision:
            r_event = R_fail

    # 2) PBRS 距离整形
    phi_prev = -prev_dist
    phi_curr = -curr_dist
    r_shaping = alpha * (gamma * phi_curr - phi_prev)

    # 3) 路径质量惩罚 (统一的子目标+路径检测)
    eta = env.auv_env.inner.dynamics.eta
    p = np.array([eta[0], eta[1]])
    d_sg_nearest = 999.0

    if subgoal is not None:
        # (a) 子目标靠近障碍物
        d_sg_nearest = min(
            (ob.dist(subgoal) for ob in env.obstacles), default=999.0)
        if d_sg_nearest < d_sg_safe:
            sg_ratio = max(0.0, 1.0 - d_sg_nearest / d_sg_safe)
            r_sg_obs = -c_sg * sg_ratio * sg_ratio  # 二次惩罚，越近越重

        # (b) 路径穿越障碍物：检查 pursuer→subgoal 是否穿过任何障碍物
        for ob in env.obstacles:
            hit, t, _ = ob.line_intersects(p, subgoal)
            if hit and 0.01 < t < 0.99:
                r_sg_obs -= c_path  # 路径穿越障碍物，附加惩罚
                break  # 只惩罚一次

    total = r_event + r_shaping + r_sg_obs + r_time
    info = {
        "r_event": r_event, "r_shaping": r_shaping,
        "r_sg_obs": r_sg_obs, "r_time": r_time,
        "d_sg_nearest": d_sg_nearest,
        "dist": curr_dist,
    }
    return total, info


# ─────────── 单 episode 收集 ───────────

def collect_one_episode(
    env: PursuitEnvRealAUVFixed,
    adapter,
    seed: int,
) -> Dict[str, Any]:
    """跑一个 episode，记录 (s, a, r, s', done)"""
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    adapter.reset()

    eta = env.auv_env.inner.dynamics.eta
    prev_dist = float(np.linalg.norm(
        np.array([eta[0], eta[1]]) - env.e
    ))

    states, actions, rewards, next_states, dones = [], [], [], [], []
    reward_infos: List[Dict] = []

    done = False
    info = {}
    while not done:
        s = obs.copy()
        a = adapter.get_action(env)

        # 计算原始子目标（用于奖励中的子目标避障惩罚）
        eta_pre = env.auv_env.inner.dynamics.eta
        p_pre = np.array([eta_pre[0], eta_pre[1]])
        sg = p_pre + a * env.subgoal_range

        obs, _env_r, done, info = env.step(a)

        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        curr_dist = float(np.linalg.norm(p - env.e))

        captured = info.get("captured", False)
        oob = info.get("out_of_bounds", False)
        coll = info.get("collision", False)

        r, r_info = compute_high_reward(
            env, prev_dist, curr_dist, captured, oob, coll,
            subgoal=sg,
            done=done,
        )

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(obs.copy())
        dones.append(done)
        reward_infos.append(r_info)

        prev_dist = curr_dist

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "reward_infos": reward_infos,
        "captured": info.get("captured", False),
        "steps": len(states),
        "strategy": adapter.name,
        "seed": seed,
    }


# ─────────── 单策略 worker (用于多进程并行) ───────────

def _worker_collect_strategy(
    strategy_name: str,
    episodes: int,
    only_success: bool,
    base_seed: int,
    low_ckpt: str,
    world_size: float,
    catch_radius: float,
    max_steps: int,
    subgoal_range: float,
    low_steps: int,
    evader_mode: str,
    worker_device: str = "cpu",
):
    """在独立进程中收集一个策略的全部 episode"""
    device = torch.device(worker_device)
    nn_policy = load_v4_nn(low_ckpt, device)
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=world_size,
        catch_radius=catch_radius,
        max_steps=max_steps,
        subgoal_range=subgoal_range,
        low_steps=low_steps,
        seed=base_seed,
        evader_mode=evader_mode,
    )

    if strategy_name.startswith("expert"):
        parts = strategy_name.split("_")
        pt = float(parts[1]) if len(parts) > 1 else 2.0
        adapter = ExpertAdapter(predict_time=pt)
    else:
        adapter = StrategyAdapter(make_strategy(strategy_name))

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    ok, fail = 0, 0
    strat_transitions = 0
    t0 = time.time()

    for ep_idx in range(episodes):
        seed_i = base_seed + ep_idx
        ep = collect_one_episode(env, adapter, seed_i)
        if only_success and not ep["captured"]:
            fail += 1
        else:
            ok += 1
            strat_transitions += ep["steps"]
            all_s.append(ep["states"])
            all_a.append(ep["actions"])
            all_r.append(ep["rewards"])
            all_ns.append(ep["next_states"])
            all_d.append(ep["dones"])

        if (ep_idx + 1) % 20 == 0 or (ep_idx + 1) == episodes:
            elapsed = time.time() - t0
            print(f"  [{adapter.name}] ep {ep_idx+1:3d}/{episodes}"
                  f" | ok={ok} fail={fail} trans={strat_transitions}"
                  f" ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    stats = {
        "success": ok, "fail": fail,
        "success_rate": ok / max(1, ok + fail),
        "mean_steps": strat_transitions / max(1, ok),
    }
    print(f"  {adapter.name}: {ok} ok, {fail} fail, "
          f"{strat_transitions} transitions ({elapsed:.0f}s)")

    if len(all_s) == 0:
        return strategy_name, stats, None

    result = {
        "states": np.concatenate(all_s),
        "actions": np.concatenate(all_a),
        "rewards": np.concatenate(all_r),
        "next_states": np.concatenate(all_ns),
        "dones": np.concatenate(all_d),
    }
    return strategy_name, stats, result


# ─────────── 可视化 ───────────

def visualize_dataset(dataset_path: str, save_dir: str):
    """对收集到的数据集画统计图"""
    data = np.load(dataset_path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) 每策略成功率
    ax = axes[0, 0]
    strats = list(meta["per_strategy"].keys())
    rates = [meta["per_strategy"][s]["success_rate"] for s in strats]
    ax.bar(strats, rates, color="steelblue")
    ax.set_ylabel("Success Rate")
    ax.set_title("Per-Strategy Success Rate")
    ax.set_ylim(0, 1)

    # 2) 奖励分布
    ax = axes[0, 1]
    rewards = data["rewards"]
    ax.hist(rewards, bins=80, color="salmon", edgecolor="black", alpha=0.7)
    ax.set_title(f"Reward Distribution (n={len(rewards)})")
    ax.set_xlabel("Reward")

    # 3) episode 长度分布
    ax = axes[1, 0]
    ep_lens = [meta["per_strategy"][s]["mean_steps"]
               for s in strats]
    ax.bar(strats, ep_lens, color="seagreen")
    ax.set_ylabel("Mean Episode Steps")
    ax.set_title("Mean Episode Length")

    # 4) 动作分布
    ax = axes[1, 1]
    actions = data["actions"]
    ax.scatter(actions[:, 0], actions[:, 1], s=1, alpha=0.15)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("action[0]")
    ax.set_ylabel("action[1]")
    ax.set_title("Action Distribution")
    ax.set_aspect("equal")

    plt.suptitle(
        f"Expert Dataset: {meta['total_transitions']} transitions, "
        f"{meta['total_episodes']} episodes",
        fontsize=13,
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "dataset_stats.png")
    plt.savefig(out, dpi=150) 
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────── 主函数 ───────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-ckpt", type=str, required=True)
    parser.add_argument("--episodes-per-strategy", type=int, default=100)
    parser.add_argument("--only-success", action="store_true", default=True,
                        help="只保留成功(captured) episode")
    parser.add_argument("--evader-mode", type=str, default="medium")
    parser.add_argument("--world-size", type=float, default=30.0)
    parser.add_argument("--catch-radius", type=float, default=1.0)
    parser.add_argument("--subgoal-range", type=float, default=0.7)
    parser.add_argument("--low-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output-dir", type=str,
                        default="data/expert_trajectories")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["lead", "pn", "apn", "cb", "expert"])
    parser.add_argument("--merge", action="store_true", default=False,
                        help="合并到已有数据集而非覆盖")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="并行 worker 数 (每策略一个进程, 0=sequential)")
    args = parser.parse_args()

    print(f"Strategies: {args.strategies}")
    print(f"Episodes per strategy: {args.episodes_per_strategy}")
    print(f"Only success: {args.only_success}")
    print(f"Num workers: {args.num_workers}")

    # ── 收集循环 ──
    os.makedirs(args.output_dir, exist_ok=True)
    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    per_strategy: Dict[str, Dict] = {}
    t0 = time.time()

    num_workers = min(args.num_workers, len(args.strategies))
    use_parallel = num_workers > 1

    if use_parallel:
        # ── 多进程并行: 每策略一个进程 ──
        mp.set_start_method("spawn", force=True)
        worker_args = []
        for i, strat_name in enumerate(args.strategies):
            base_seed = args.seed + i * args.episodes_per_strategy
            worker_args.append((
                strat_name,
                args.episodes_per_strategy,
                args.only_success,
                base_seed,
                args.low_level_ckpt,
                args.world_size,
                args.catch_radius,
                args.max_steps,
                args.subgoal_range,
                args.low_steps,
                args.evader_mode,
                "cpu",  # worker_device: 并行时用 CPU 避免 CUDA 冲突
            ))

        print(f"\n启动 {num_workers} 个并行 worker...")
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(_worker_collect_strategy, worker_args)

        for strat_name, stats, data in results:
            per_strategy[strat_name] = stats
            if data is not None:
                all_s.append(data["states"])
                all_a.append(data["actions"])
                all_r.append(data["rewards"])
                all_ns.append(data["next_states"])
                all_d.append(data["dones"])

    else:
        # ── 单进程串行 ──
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_policy = load_v4_nn(args.low_level_ckpt, device)
        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy, device=device,
            world_size=args.world_size, catch_radius=args.catch_radius,
            max_steps=args.max_steps, subgoal_range=args.subgoal_range,
            low_steps=args.low_steps, seed=args.seed,
            evader_mode=args.evader_mode,
        )

        adapters = []
        for name in args.strategies:
            if name.startswith("expert"):
                parts = name.split("_")
                pt = float(parts[1]) if len(parts) > 1 else 2.0
                adapters.append(ExpertAdapter(predict_time=pt))
            else:
                adapters.append(StrategyAdapter(make_strategy(name)))

        total_ep = 0
        for adapter in adapters:
            ok, fail = 0, 0
            strat_transitions = 0
            for ep_idx in range(args.episodes_per_strategy):
                seed_i = args.seed + total_ep + ep_idx
                ep = collect_one_episode(env, adapter, seed_i)
                if args.only_success and not ep["captured"]:
                    fail += 1
                else:
                    ok += 1
                    strat_transitions += ep["steps"]
                    all_s.append(ep["states"])
                    all_a.append(ep["actions"])
                    all_r.append(ep["rewards"])
                    all_ns.append(ep["next_states"])
                    all_d.append(ep["dones"])
                if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == args.episodes_per_strategy:
                    elapsed = time.time() - t0
                    print(f"\r  [{adapter.name}] ep {ep_idx+1:3d}/{args.episodes_per_strategy}"
                          f" | ok={ok} fail={fail} trans={strat_transitions}"
                          f" ({elapsed:.0f}s)", end="", flush=True)
            print()
            per_strategy[adapter.name] = {
                "success": ok, "fail": fail,
                "success_rate": ok / max(1, ok + fail),
                "mean_steps": strat_transitions / max(1, ok),
            }
            total_ep += args.episodes_per_strategy
            elapsed = time.time() - t0
            print(f"  {adapter.name}: {ok} ok, {fail} fail, "
                  f"{strat_transitions} transitions ({elapsed:.0f}s)")

    states = np.concatenate(all_s)
    actions = np.concatenate(all_a)
    rewards = np.concatenate(all_r)
    next_states = np.concatenate(all_ns)
    dones = np.concatenate(all_d)

    # 如果指定了 merge 且旧数据存在，则合并
    out_path = os.path.join(args.output_dir, "expert_dataset.npz")
    if args.merge and os.path.exists(out_path):
        print(f"\n合并已有数据: {out_path}")
        old = np.load(out_path, allow_pickle=True)
        old_meta = json.loads(str(old["meta"]))  
        states = np.concatenate([old["states"], states])
        actions = np.concatenate([old["actions"], actions])
        rewards = np.concatenate([old["rewards"], rewards])
        next_states = np.concatenate([old["next_states"], next_states])
        dones = np.concatenate([old["dones"], dones])
        # 合并 per_strategy 统计
        for k, v in old_meta.get("per_strategy", {}).items():
            if k in per_strategy:
                per_strategy[k]["success"] += v["success"]
                per_strategy[k]["fail"] += v["fail"]
                tot = per_strategy[k]["success"] + per_strategy[k]["fail"]
                per_strategy[k]["success_rate"] = per_strategy[k]["success"] / max(1, tot)
            else:
                per_strategy[k] = v
        print(f"  旧数据: {old_meta['total_transitions']} transitions")
        print(f"  合并后: {len(states)} transitions")

    meta = {
        "total_transitions": int(len(states)),
        "total_episodes": int(sum(
            v["success"] for v in per_strategy.values())),
        "per_strategy": per_strategy,
        "args": vars(args),
    }

    out_path = os.path.join(args.output_dir, "expert_dataset.npz")
    np.savez_compressed(
        out_path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        meta=json.dumps(meta),
    )
    print(f"\nSaved {len(states)} transitions → {out_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # 可视化
    visualize_dataset(out_path, args.output_dir)


if __name__ == "__main__":
    main()
