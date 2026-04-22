"""
DAgger 数据收集 (v5 直接推力版, 高层策略)
==========================================
适配 PursuitEnvDirectThrust + v5 底层，替代旧的 dagger_collect.py (v4)。

流程:
  Round 0 : 纯经典策略收集 -> D_0  (只保留成功 episode)
  Round k : 用 D_{k-1} 训练 BC -> π_k
             以 beta_k 概率执行专家，(1-beta_k) 执行 π_k
             记录「专家在该状态的标签动作」聚合进数据集

用法:
  python training/dagger_collect_high_level.py \\
      --nn-ckpt low_near/v5_direct_thrust/checkpoints/best.pth \\
      --n-rounds 4 --episodes-per-round 500 \\
      --output data/expert_high_level/expert_dataset.npz
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_direct_thrust import (
    load_v5_nn, PursuitEnvDirectThrust,
)
from evaluation.evaluate_direct_thrust_strategies import (
    StrategyAdapter, ExpertAdapter,
)
from core.agents.classical_pursuit_strategies import make_strategy
from training.offline_ppo_high_level import HighLevelActorCritic


# ──────────────────────────────────────────────
#  单 episode DAgger rollout
# ──────────────────────────────────────────────

def dagger_one_episode(
    env: PursuitEnvDirectThrust,
    learner,          # HighLevelActorCritic or None
    expert_adapter,
    beta: float,
    device: torch.device,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    混合策略 rollout:
      - 以概率 beta  执行专家动作
      - 以概率 1-beta 执行学习策略动作
      - 无论谁执行，都记录「专家标签动作」用于训练

    返回 dict: states, actions(专家标签), rewards, next_states, dones,
               captured, steps, seed
    """
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    expert_adapter.reset()

    states, expert_actions, rewards, next_states, dones = [], [], [], [], []
    done = False
    info = {}

    while not done:
        s = obs.copy()

        # 专家标签（始终计算）
        a_expert = expert_adapter.get_action(env)

        # 学习策略动作
        if learner is not None and beta < 1.0:
            with torch.no_grad():
                s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
                a_out, _ = learner.get_action(s_t, deterministic=deterministic)
                a_learner = a_out[0].cpu().numpy().astype(np.float32)
                a_learner = np.clip(a_learner, -1.0, 1.0)
        else:
            a_learner = a_expert

        # 混合执行
        a_exec = a_expert if (np.random.random() < beta) else a_learner

        obs, reward, done, info = env.step(a_exec)

        states.append(s)
        expert_actions.append(a_expert)   # 标签始终是专家动作
        rewards.append(float(reward))
        next_states.append(obs.copy())
        dones.append(bool(done))

    return {
        "states":   np.array(states,        dtype=np.float32),
        "actions":  np.array(expert_actions, dtype=np.float32),
        "rewards":  np.array(rewards,        dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones":    np.array(dones,          dtype=np.float32),
        "captured": bool(info.get("captured", False)),
        "steps":    len(states),
    }


# ──────────────────────────────────────────────
#  轮间 BC 快速训练 (在 tanh 空间做 MSE，与 offline 一致)
# ──────────────────────────────────────────────

def quick_bc_train(
    dataset_path: str,
    obs_dim: int = 21,
    act_dim: int = 2,
    hidden: tuple = (256, 128),
    n_epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: torch.device = None,
    seed: int = 42,
) -> HighLevelActorCritic:
    """快速 BC，返回已训练模型，不保存中间 checkpoint。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(dataset_path, allow_pickle=True)
    states  = torch.from_numpy(data["states"]).float().to(device)
    actions = torch.from_numpy(data["actions"]).float().to(device)
    n = len(states)

    model = HighLevelActorCritic(
        obs_dim=obs_dim, act_dim=act_dim, hidden=hidden
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    torch.manual_seed(seed)
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s, a = states[idx], actions[idx]
            a_pred, _ = model.get_action(s, deterministic=True)
            loss = F.mse_loss(a_pred, a)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total += loss.item(); nb += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  BC epoch {epoch+1}/{n_epochs}: mse={total/nb:.5f}", flush=True)

    model.eval()
    return model


# ──────────────────────────────────────────────
#  辅助：构建 adapter 列表
# ──────────────────────────────────────────────

def _make_adapters(strategy_names: List[str]):
    adapters = []
    for name in strategy_names:
        if name.startswith("expert"):
            parts = name.split("_")
            pt = float(parts[1]) if len(parts) > 1 else 2.0
            adapters.append(ExpertAdapter(predict_time=pt))
        else:
            adapters.append(StrategyAdapter(make_strategy(name)))
    return adapters


# ──────────────────────────────────────────────
#  辅助：保存/追加数据集
# ──────────────────────────────────────────────

def _save_dataset(path: str, new_s, new_a, new_r, new_ns, new_d,
                  round_idx: int, stats: list):
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        old_n = len(old["states"])
        new_s  = np.concatenate([old["states"],      new_s])
        new_a  = np.concatenate([old["actions"],     new_a])
        new_r  = np.concatenate([old["rewards"],     new_r])
        new_ns = np.concatenate([old["next_states"], new_ns])
        new_d  = np.concatenate([old["dones"],       new_d])
        print(f"  Aggregated: {old_n} + {len(new_s)-old_n} = {len(new_s)}", flush=True)

    meta = {
        "total_transitions": int(len(new_s)),
        "dagger_round": round_idx,
        "dagger_stats": stats,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        states=new_s, actions=new_a, rewards=new_r,
        next_states=new_ns, dones=new_d,
        meta=json.dumps(meta),
    )
    print(f"  Saved {len(new_s)} transitions -> {path}", flush=True)


# ──────────────────────────────────────────────
#  主 DAgger 循环
# ──────────────────────────────────────────────

def run_dagger(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAgger] device={device}")

    nn_policy = load_v5_nn(args.nn_ckpt, device)
    adapters  = _make_adapters(args.strategies)
    print(f"[DAgger] strategies: {[a.name for a in adapters]}")

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

    learner: HighLevelActorCritic = None
    stats_per_round: List[dict] = []

    for round_idx in range(args.n_rounds):
        t0 = time.time()
        beta = max(args.beta_min, 1.0 - round_idx * args.beta_decay)
        eps  = args.episodes_per_round

        print(f"\n{'='*55}", flush=True)
        print(f"DAgger Round {round_idx}/{args.n_rounds-1} | "
              f"beta={beta:.2f} | episodes={eps}", flush=True)
        print(f"{'='*55}", flush=True)

        all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
        n_success, n_fail, n_trans = 0, 0, 0

        for ep_i in range(eps):
            adapter = adapters[ep_i % len(adapters)]
            seed_i  = args.seed + round_idx * 10000 + ep_i

            ep = dagger_one_episode(
                env, learner, adapter, beta, device, seed_i,
                deterministic=not args.stochastic_learner,
            )

            if ep["captured"]:
                n_success += 1
            else:
                n_fail += 1

            # Round 0 只保留成功 episode，后续全部保留
            if round_idx == 0 and args.only_success_round0 and not ep["captured"]:
                continue

            all_s.append(ep["states"])
            all_a.append(ep["actions"])
            all_r.append(ep["rewards"])
            all_ns.append(ep["next_states"])
            all_d.append(ep["dones"])
            n_trans += ep["steps"]

            if (ep_i + 1) % 50 == 0 or ep_i == eps - 1:
                elapsed = time.time() - t0
                print(f"  [R{round_idx}] ep {ep_i+1}/{eps} | "
                      f"ok={n_success} fail={n_fail} trans={n_trans} "
                      f"({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        rs = {
            "round": round_idx,
            "beta": beta,
            "success": n_success,
            "fail": n_fail,
            "success_rate": n_success / max(1, n_success + n_fail),
            "transitions": n_trans,
            "time_sec": elapsed,
        }
        stats_per_round.append(rs)
        print(f"  Round {round_idx} done: ok={n_success} fail={n_fail} "
              f"trans={n_trans} ({elapsed:.0f}s)", flush=True)

        if n_trans == 0:
            print(f"  WARNING: Round {round_idx} 没有收集到数据，跳过。", flush=True)
            continue

        # 聚合并保存
        _save_dataset(
            args.output,
            np.concatenate(all_s),
            np.concatenate(all_a),
            np.concatenate(all_r),
            np.concatenate(all_ns),
            np.concatenate(all_d),
            round_idx, stats_per_round,
        )

        # 训练下一轮用的 BC 策略（最后一轮不需要）
        if round_idx < args.n_rounds - 1:
            print(f"\n  Training BC for round {round_idx+1}...", flush=True)
            learner = quick_bc_train(
                args.output,
                obs_dim=args.obs_dim,
                act_dim=args.act_dim,
                hidden=tuple(args.hidden),
                n_epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
                device=device,
                seed=args.seed + round_idx,
            )

    # ── 汇总 ──
    print(f"\n{'='*55}")
    print(f"DAgger 完成: {args.n_rounds} 轮")
    for rs in stats_per_round:
        print(f"  Round {rs['round']}: beta={rs['beta']:.2f} "
              f"ok={rs['success']} fail={rs['fail']} "
              f"rate={rs['success_rate']:.1%} trans={rs['transitions']}")
    data = np.load(args.output, allow_pickle=True)
    print(f"最终数据集: {len(data['states'])} transitions -> {args.output}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="DAgger 高层数据采集 (v5 直接推力)")
    # 底层 & 环境
    p.add_argument("--nn-ckpt", type=str,
                   default="low_near/v5_direct_thrust/checkpoints/best.pth")
    p.add_argument("--output", type=str,
                   default="data/expert_high_level/expert_dataset.npz")
    p.add_argument("--evader-mode", type=str, default="medium")
    p.add_argument("--world-size",     type=float, default=30.0)
    p.add_argument("--catch-radius",   type=float, default=1.0)
    p.add_argument("--max-steps",      type=int,   default=6000)
    p.add_argument("--subgoal-range",  type=float, default=2.0)
    p.add_argument("--low-steps",      type=int,   default=50)
    p.add_argument("--v-max-pursuer",  type=float, default=0.65)
    p.add_argument("--a-max-pursuer",  type=float, default=0.5)
    p.add_argument("--v-max-evader",   type=float, default=0.85)
    p.add_argument("--a-max-evader",   type=float, default=0.25)
    # DAgger
    p.add_argument("--n-rounds",            type=int,   default=4)
    p.add_argument("--episodes-per-round",  type=int,   default=500)
    p.add_argument("--beta-decay",          type=float, default=0.3)
    p.add_argument("--beta-min",            type=float, default=0.1)
    p.add_argument("--only-success-round0", action="store_true", default=True)
    p.add_argument("--no-only-success-round0",
                   dest="only_success_round0", action="store_false")
    p.add_argument("--stochastic-learner",  action="store_true", default=False)
    p.add_argument("--strategies", nargs="+",
                   default=["expert", "expert_1.0", "expert_3.0",
                            "lead", "pn"])
    # BC 中间训练
    p.add_argument("--bc-epochs",      type=int,   default=60)
    p.add_argument("--bc-batch-size",  type=int,   default=256)
    p.add_argument("--bc-lr",          type=float, default=1e-4)
    # 网络
    p.add_argument("--obs-dim",  type=int,        default=21)
    p.add_argument("--act-dim",  type=int,        default=2)
    p.add_argument("--hidden",   type=int, nargs="+", default=[256, 128])
    # 其他
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_dagger(args)


if __name__ == "__main__":
    main()
