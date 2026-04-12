"""
DAgger (Dataset Aggregation) 数据收集
=====================================
核心思路:
  Round 0: 纯经典策略收集 -> 得到 D_0
  Round k (k>=1):
    1. 用 D_{k-1} 训练 BC 策略 π_k
    2. Roll out π_k 与环境交互，在 π_k 访问的状态上用经典策略标注动作
    3. 混合策略: β_k π_expert + (1-β_k) π_k , β 逐轮衰减
    4. 聚合: D_k = D_{k-1} ∪ new_data

为什么有效:
  - 纯专家数据只覆盖"专家会到达的状态"
  - 学到的策略部署时会有 compounding error，进入未见过的状态
  - DAgger 让专家在这些"策略犯错后的状态"上给标签，减小分布偏移
"""

import os
import sys
import json
import argparse
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from core.agents.classical_pursuit_strategies import make_strategy
from training.collect_expert_trajectories import (
    StrategyAdapter, ExpertAdapter, compute_high_reward,
)
from training.offline_ppo_pretrain import HighLevelActorCritic


# ────────── DAgger 单 episode 收集 ──────────

def dagger_one_episode(
    env: PursuitEnvRealAUVFixed,
    learner: HighLevelActorCritic,
    expert_adapter,
    beta: float,
    device: torch.device,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    混合策略 rollout:
      以概率 beta 执行专家动作，以概率 (1-beta) 执行学习策略动作
      但无论谁执行，都记录「专家在当前状态会给出的动作」作为标签

    Args:
        beta: 专家介入概率。round 0 = 1.0 (纯专家)，逐轮衰减
        deterministic: 学习策略是否用 mean (True) 还是采样 (False)
    """
    env.rng = np.random.default_rng(seed)
    obs = env.reset()
    expert_adapter.reset()

    eta = env.auv_env.inner.dynamics.eta
    prev_dist = float(np.linalg.norm(np.array([eta[0], eta[1]]) - env.e))

    states, expert_actions, rewards, next_states, dones = [], [], [], [], []
    executed_actions = []
    done = False
    info = {}

    while not done:
        s = obs.copy()

        # 获取专家动作 (永远计算，作为标签)
        a_expert = expert_adapter.get_action(env)

        # 获取学习策略动作
        if learner is not None and beta < 1.0:
            with torch.no_grad():
                s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
                if deterministic:
                    a_out = learner.get_action(s_t, deterministic=True)
                else:
                    a_out = learner.get_action(s_t, deterministic=False)
                # get_action returns (action, logprob), take action only
                if isinstance(a_out, tuple):
                    a_out = a_out[0]
                a_learner = a_out.cpu().numpy().flatten()
                a_learner = np.clip(a_learner, -1.0, 1.0).astype(np.float32)
        else:
            a_learner = a_expert  # round 0 或 beta=1 时无需学习策略

        # 混合: 以概率 beta 执行专家，否则执行学习策略
        if np.random.random() < beta:
            a_exec = a_expert
        else:
            a_exec = a_learner

        # 计算原始子目标（用于奖励计算）
        eta_pre = env.auv_env.inner.dynamics.eta
        p_pre = np.array([eta_pre[0], eta_pre[1]])
        sg = p_pre + a_exec * env.subgoal_range

        obs, _env_r, done, info = env.step(a_exec)

        eta = env.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        curr_dist = float(np.linalg.norm(p - env.e))

        captured = info.get("captured", False)
        oob = info.get("out_of_bounds", False)
        coll = info.get("collision", False)

        r, r_info = compute_high_reward(
            env, prev_dist, curr_dist, captured, oob, coll,
            subgoal=sg, done=done,
        )

        # 记录: 状态 & 专家动作 (标签)
        states.append(s)
        expert_actions.append(a_expert)   # 标签始终是专家动作
        executed_actions.append(a_exec)    # 实际执行的动作
        rewards.append(r)
        next_states.append(obs.copy())
        dones.append(done)

        prev_dist = curr_dist

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(expert_actions, dtype=np.float32),   # DAgger 标签
        "executed_actions": np.array(executed_actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "captured": info.get("captured", False),
        "steps": len(states),
        "seed": seed,
    }


# ────────── BC 快速训练 (用于 DAgger 中间轮) ──────────

def quick_bc_train(
    dataset_path: str,
    obs_dim: int = 21,
    act_dim: int = 2,
    hidden: tuple = (256, 128),
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: torch.device = None,
    seed: int = 42,
) -> HighLevelActorCritic:
    """快速训练 BC 策略，返回模型（不保存中间 checkpoint）"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(dataset_path, allow_pickle=True)
    states = torch.from_numpy(data["states"]).float().to(device)
    actions = torch.from_numpy(data["actions"]).float().to(device)
    n = len(states)

    model = HighLevelActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                                  hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    torch.manual_seed(seed)
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s, a = states[idx], actions[idx]
            dist = model.get_dist(s)
            mu = torch.tanh(dist.mean)
            loss = F.mse_loss(mu, a)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  BC epoch {epoch+1}/{n_epochs}: "
                  f"mse={total_loss / n_batch:.5f}", flush=True)

    model.eval()
    return model


# ────────── DAgger 多进程 worker ──────────

def _dagger_worker(
    worker_id: int,
    episode_indices: List[int],
    learner_state: dict,       # None or state_dict
    beta: float,
    round_idx: int,
    args_dict: dict,
    return_dict: dict,         # mp.Manager dict for results
):
    """单个 worker：创建独立的 env，跑分配到的 episodes"""
    import torch
    device = torch.device("cpu")  # worker 强制 CPU 避免 CUDA fork 问题

    nn_policy = load_v4_nn(args_dict["low_level_ckpt"], device)
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy, device=device,
        world_size=args_dict["world_size"],
        catch_radius=args_dict["catch_radius"],
        max_steps=args_dict["max_steps"],
        subgoal_range=args_dict["subgoal_range"],
        low_steps=args_dict["low_steps"],
        seed=args_dict["seed"] + worker_id * 777,
        evader_mode=args_dict["evader_mode"],
    )

    # 构建专家适配器
    expert_adapters = []
    for name in args_dict["expert_strategies"]:
        if name.startswith("expert"):
            parts = name.split("_")
            pt = float(parts[1]) if len(parts) > 1 else 2.0
            expert_adapters.append(ExpertAdapter(predict_time=pt))
        else:
            expert_adapters.append(StrategyAdapter(make_strategy(name)))

    # 加载 learner (如果有)
    learner = None
    if learner_state is not None:
        learner = HighLevelActorCritic(
            obs_dim=args_dict["obs_dim"],
            act_dim=args_dict["act_dim"],
            hidden=tuple(args_dict["hidden"]),
        ).to(device)
        learner.load_state_dict(learner_state)
        learner.eval()

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    n_success, n_fail = 0, 0
    n_transitions = 0

    for i, ep_i in enumerate(episode_indices):
        adapter = expert_adapters[ep_i % len(expert_adapters)]
        seed_i = args_dict["seed"] + round_idx * 10000 + ep_i

        ep = dagger_one_episode(
            env, learner, adapter, beta, device, seed_i,
            deterministic=not args_dict.get("stochastic_learner", False),
        )

        if ep["captured"]:
            n_success += 1
        else:
            n_fail += 1

        skip = (round_idx == 0 and args_dict.get("only_success_round0", True)
                and not ep["captured"])
        if not skip:
            all_s.append(ep["states"])
            all_a.append(ep["actions"])
            all_r.append(ep["rewards"])
            all_ns.append(ep["next_states"])
            all_d.append(ep["dones"])
            n_transitions += ep["steps"]

        if (i + 1) % 10 == 0 or (i + 1) == len(episode_indices):
            print(f"  [R{round_idx} W{worker_id}] {i+1}/{len(episode_indices)} "
                  f"ok={n_success} fail={n_fail} trans={n_transitions}",
                  flush=True)

    result = {
        "n_success": n_success,
        "n_fail": n_fail,
        "n_transitions": n_transitions,
    }
    if n_transitions > 0:
        result["states"] = np.concatenate(all_s)
        result["actions"] = np.concatenate(all_a)
        result["rewards"] = np.concatenate(all_r)
        result["next_states"] = np.concatenate(all_ns)
        result["dones"] = np.concatenate(all_d)

    return_dict[worker_id] = result


# ────────── DAgger 主循环 ──────────

def run_dagger(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = getattr(args, "num_workers", 8)

    # 构建 args_dict 供 worker 使用
    args_dict = vars(args)

    # 若 num_workers=1, 需要加载 env 到主进程（BC 训练仍在主进程）
    nn_policy = load_v4_nn(args.low_level_ckpt, device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── DAgger 轮次 ──
    learner = None
    stats_per_round = []
    dataset_path = os.path.join(output_dir, "expert_dataset.npz")

    for round_idx in range(args.n_rounds):
        t0 = time.time()
        beta = max(args.beta_min, 1.0 - round_idx * args.beta_decay)
        eps_this_round = args.episodes_per_round

        print(f"\n{'='*60}", flush=True)
        print(f"DAgger Round {round_idx}/{args.n_rounds - 1} | "
              f"beta={beta:.2f} | episodes={eps_this_round} | "
              f"workers={num_workers}", flush=True)
        print(f"{'='*60}", flush=True)

        # 准备 learner state_dict (picklable)
        learner_state = None
        if learner is not None:
            learner_state = {k: v.cpu() for k, v in learner.state_dict().items()}

        if num_workers > 1:
            # 并行收集
            mp_ctx = mp.get_context("spawn")
            manager = mp_ctx.Manager()
            return_dict = manager.dict()

            # 分配 episodes 到 workers
            all_eps = list(range(eps_this_round))
            chunks = [[] for _ in range(num_workers)]
            for i, ep_i in enumerate(all_eps):
                chunks[i % num_workers].append(ep_i)

            procs = []
            for w_id in range(num_workers):
                if len(chunks[w_id]) == 0:
                    continue
                p = mp_ctx.Process(
                    target=_dagger_worker,
                    args=(w_id, chunks[w_id], learner_state, beta,
                          round_idx, args_dict, return_dict),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            # 合并 worker 结果
            all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
            n_success, n_fail, n_transitions = 0, 0, 0
            for w_id in sorted(return_dict.keys()):
                res = return_dict[w_id]
                n_success += res["n_success"]
                n_fail += res["n_fail"]
                n_transitions += res["n_transitions"]
                if res["n_transitions"] > 0:
                    all_s.append(res["states"])
                    all_a.append(res["actions"])
                    all_r.append(res["rewards"])
                    all_ns.append(res["next_states"])
                    all_d.append(res["dones"])
        else:
            # 单进程 fallback
            env = PursuitEnvRealAUVFixed(
                nn_policy=nn_policy, device=device,
                world_size=args.world_size,
                catch_radius=args.catch_radius,
                max_steps=args.max_steps,
                subgoal_range=args.subgoal_range,
                low_steps=args.low_steps,
                seed=args.seed,
                evader_mode=args.evader_mode,
            )
            expert_adapters = []
            for name in args.expert_strategies:
                if name.startswith("expert"):
                    parts = name.split("_")
                    pt = float(parts[1]) if len(parts) > 1 else 2.0
                    expert_adapters.append(ExpertAdapter(predict_time=pt))
                else:
                    expert_adapters.append(StrategyAdapter(make_strategy(name)))

            all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
            n_success, n_fail, n_transitions = 0, 0, 0

            for ep_i in range(eps_this_round):
                adapter = expert_adapters[ep_i % len(expert_adapters)]
                seed_i = args.seed + round_idx * 10000 + ep_i

                ep = dagger_one_episode(
                    env, learner, adapter, beta, device, seed_i,
                    deterministic=not args.stochastic_learner,
                )

                if ep["captured"]:
                    n_success += 1
                else:
                    n_fail += 1

                skip = (round_idx == 0 and args.only_success_round0
                        and not ep["captured"])
                if not skip:
                    all_s.append(ep["states"])
                    all_a.append(ep["actions"])
                    all_r.append(ep["rewards"])
                    all_ns.append(ep["next_states"])
                    all_d.append(ep["dones"])
                    n_transitions += ep["steps"]

                if (ep_i + 1) % 20 == 0 or ep_i == eps_this_round - 1:
                    elapsed = time.time() - t0
                    print(f"  [{round_idx}] ep {ep_i+1}/{eps_this_round} | "
                          f"ok={n_success} fail={n_fail} "
                          f"trans={n_transitions} ({elapsed:.0f}s)")

        elapsed = time.time() - t0
        round_stats = {
            "round": round_idx,
            "beta": beta,
            "success": n_success,
            "fail": n_fail,
            "success_rate": n_success / max(1, n_success + n_fail),
            "transitions": n_transitions,
            "time_sec": elapsed,
        }
        stats_per_round.append(round_stats)
        print(f"  Round {round_idx} done: ok={n_success} fail={n_fail} "
              f"trans={n_transitions} ({elapsed:.0f}s)", flush=True)

        if n_transitions == 0:
            print(f"  WARNING: Round {round_idx} 没有收集到数据!")
            continue

        # 聚合
        new_s = np.concatenate(all_s)
        new_a = np.concatenate(all_a)
        new_r = np.concatenate(all_r)
        new_ns = np.concatenate(all_ns)
        new_d = np.concatenate(all_d)

        if os.path.exists(dataset_path):
            old = np.load(dataset_path, allow_pickle=True)
            old_n = len(old['states'])
            new_s = np.concatenate([old["states"], new_s])
            new_a = np.concatenate([old["actions"], new_a])
            new_r = np.concatenate([old["rewards"], new_r])
            new_ns = np.concatenate([old["next_states"], new_ns])
            new_d = np.concatenate([old["dones"], new_d])
            print(f"  Aggregated: {old_n} + {n_transitions} "
                  f"= {len(new_s)} transitions", flush=True)

        meta = {
            "total_transitions": int(len(new_s)),
            "dagger_round": round_idx,
            "dagger_stats": stats_per_round,
            "args": {k: v for k, v in vars(args).items()
                     if isinstance(v, (int, float, str, bool, list))},
        }
        np.savez_compressed(
            dataset_path,
            states=new_s, actions=new_a, rewards=new_r,
            next_states=new_ns, dones=new_d,
            meta=json.dumps(meta),
        )
        print(f"  Saved {len(new_s)} transitions -> {dataset_path}", flush=True)

        # 训练新的 BC 策略用于下一轮 rollout (最后一轮不用训)
        if round_idx < args.n_rounds - 1:
            print(f"\n  Training BC for round {round_idx + 1}...", flush=True)
            learner = quick_bc_train(
                dataset_path,
                obs_dim=args.obs_dim,
                act_dim=args.act_dim,
                hidden=tuple(args.hidden),
                n_epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
                device=device,
                seed=args.seed + round_idx,
            )

    # ── 最终统计 ──
    print(f"\n{'='*60}")
    print(f"DAgger 完成: {args.n_rounds} 轮")
    print(f"{'='*60}")
    for rs in stats_per_round:
        print(f"  Round {rs['round']}: beta={rs['beta']:.2f}, "
              f"ok={rs['success']}, fail={rs['fail']}, "
              f"rate={rs['success_rate']:.2%}, "
              f"trans={rs['transitions']}, {rs['time_sec']:.0f}s")

    data = np.load(dataset_path, allow_pickle=True)
    print(f"\n最终数据集: {len(data['states'])} transitions")

    # 可视化 DAgger 统计
    _plot_dagger_stats(stats_per_round, output_dir)


def _plot_dagger_stats(stats: List[Dict], save_dir: str):
    """画 DAgger 各轮统计图"""
    rounds = [s["round"] for s in stats]
    betas = [s["beta"] for s in stats]
    rates = [s["success_rate"] for s in stats]
    trans = [s["transitions"] for s in stats]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(rounds, betas, "o-", color="steelblue")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Beta (expert ratio)")
    axes[0].set_title("Expert Intervention Rate")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(rounds, rates, "s-", color="seagreen")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Per-Round Capture Rate")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(rounds, trans, color="salmon")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Transitions")
    axes[2].set_title("Transitions per Round")

    # 累计转移数折线叠加
    cum = np.cumsum(trans)
    ax2 = axes[2].twinx()
    ax2.plot(rounds, cum, "D-", color="navy", alpha=0.7)
    ax2.set_ylabel("Cumulative", color="navy")

    plt.tight_layout()
    out = os.path.join(save_dir, "dagger_stats.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ────────── CLI ──────────

def main():
    parser = argparse.ArgumentParser(
        description="DAgger 数据收集: 迭代式聚合专家-学习者混合轨迹")

    # 环境
    parser.add_argument("--low-level-ckpt", type=str, required=True)
    parser.add_argument("--evader-mode", type=str, default="medium")
    parser.add_argument("--world-size", type=float, default=30.0)
    parser.add_argument("--catch-radius", type=float, default=1.0)
    parser.add_argument("--subgoal-range", type=float, default=0.7)
    parser.add_argument("--low-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6000)

    # DAgger 参数
    parser.add_argument("--n-rounds", type=int, default=4,
                        help="DAgger 总轮数 (round 0 = 纯专家)")
    parser.add_argument("--episodes-per-round", type=int, default=500,
                        help="每轮收集的 episode 数")
    parser.add_argument("--beta-decay", type=float, default=0.3,
                        help="beta 每轮衰减量: beta_k = max(beta_min, 1.0 - k * decay)")
    parser.add_argument("--beta-min", type=float, default=0.1,
                        help="beta 最小值 (保持一定比例的专家介入)")
    parser.add_argument("--only-success-round0", action="store_true", default=True,
                        help="Round 0 只保留成功 episode (高质量专家数据); Round 1+ 始终保留全部")
    parser.add_argument("--stochastic-learner", action="store_true", default=False,
                        help="学习策略 rollout 时是否采样而非用 mean")

    # 专家策略
    parser.add_argument("--expert-strategies", type=str, nargs="+",
                        default=["lead", "pn", "apn", "cb",
                                 "expert_1.0", "expert_2.0", "expert_3.0"])

    # BC 中间训练
    parser.add_argument("--bc-epochs", type=int, default=60,
                        help="每轮间 BC 训练 epoch 数")
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=3e-4)

    # 网络
    parser.add_argument("--obs-dim", type=int, default=21)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])

    # 输出
    parser.add_argument("--output-dir", type=str,
                        default="data/expert_trajectories")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="并行 worker 数 (每个 worker 独立 env)")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    run_dagger(args)


if __name__ == "__main__":
    main()
