#!/usr/bin/env python
"""
统一评估多个 TD3+BC checkpoint，输出 reward 分解对比
用法: python scripts/eval_compare_td3bc.py
"""
import os, sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed
from training.collect_expert_trajectories import compute_high_reward
from training.td3bc_pretrain import TD3Actor
from training.bppo_pretrain import GaussianPolicy


def _load_actor(ckpt, device):
    """根据 checkpoint config 判断使用 TD3Actor 还是 GaussianPolicy"""
    od = ckpt.get("obs_dim", 21)
    ad = ckpt.get("act_dim", 2)
    hid = tuple(ckpt.get("hidden", [256, 128]))
    cfg = ckpt.get("config", {})
    # BPPO checkpoints 含 old_policy key
    if "old_policy" in ckpt:
        actor = GaussianPolicy(od, ad, hid).to(device)
        actor.load_state_dict(ckpt["actor"])
        # 包装成统一接口：__call__ 返回 (action,)
        class _Wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            def forward(self, s):
                return self.policy.get_action(s, deterministic=True)
        return _Wrapper(actor).to(device)
    else:
        actor = TD3Actor(od, ad, hid).to(device)
        actor.load_state_dict(ckpt["actor"])
        return actor


def evaluate_checkpoint(ckpt_path, env, device, n_episodes=30):
    """评估单个 checkpoint，返回详细统计"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = _load_actor(ckpt, device)
    actor.eval()

    results = {
        "ep_rewards": [], "captures": [], "ep_steps": [],
        "total_event": 0, "total_shaping": 0,
        "total_sg_obs": 0, "total_time": 0, "total_steps": 0,
    }

    for i in range(n_episodes):
        env.rng = np.random.default_rng(9999 + i)
        obs = env.reset()
        eta = env.auv_env.inner.dynamics.eta
        prev_dist = float(np.linalg.norm(np.array([eta[0], eta[1]]) - env.e))
        done, ep_r, steps = False, 0.0, 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = actor(obs_t)[0].cpu().numpy()
            eta = env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            sg = p + a * env.subgoal_range
            obs, _, done, info = env.step(a)
            eta = env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            curr_dist = float(np.linalg.norm(p - env.e))
            r, r_info = compute_high_reward(
                env, prev_dist, curr_dist,
                info.get("captured", False),
                info.get("out_of_bounds", False),
                info.get("collision", False),
                subgoal=sg,
                done=done,
            )
            ep_r += r
            prev_dist = curr_dist
            steps += 1
            results["total_event"] += r_info["r_event"]
            results["total_shaping"] += r_info["r_shaping"]
            results["total_sg_obs"] += r_info["r_sg_obs"]
            results["total_time"] += r_info["r_time"]

        results["ep_rewards"].append(ep_r)
        results["captures"].append(info.get("captured", False))
        results["ep_steps"].append(steps)
        results["total_steps"] += steps

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_policy = load_v4_nn(
        os.path.join(PROJECT_ROOT,
                     "low_near/v4_tight_heading_terminal/checkpoints/best.pth"),
        device)

    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy, device=device,
        world_size=30.0, catch_radius=1.0,
        max_steps=120 * 50,
        subgoal_range=0.7, low_steps=50,
        seed=9999, evader_mode="medium",
    )

    # 要评估的 checkpoints
    checkpoints = {
        "TD3+BC v1 (100k, old data)": "checkpoints/td3bc_phase1/best.pth",
        "TD3+BC v3 (150k, new data)": "checkpoints/td3bc_phase1_v3/best.pth",
    }
    # 也评估新算法（如果存在）
    for name, path in [
        ("IQL v1", "checkpoints/iql_phase1_v1/best.pth"),
        ("AWAC v1", "checkpoints/awac_phase1_v1/best.pth"),
        ("REBRAC v1", "checkpoints/rebrac_phase1_v1/best.pth"),
        ("BPPO v1", "checkpoints/bppo_phase1_v1/best.pth"),
    ]:
        full = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full):
            checkpoints[name] = path

    n_eval = 30
    all_results = {}

    for name, path in checkpoints.items():
        full_path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  Path: {path}")
        res = evaluate_checkpoint(full_path, env, device, n_eval)
        all_results[name] = res

        success = np.mean(res["captures"]) * 100
        avg_r = np.mean(res["ep_rewards"])
        std_r = np.std(res["ep_rewards"])
        avg_steps = np.mean(res["ep_steps"])
        ts = res["total_steps"]

        print(f"  Success: {success:.0f}%")
        print(f"  Reward: {avg_r:.1f} ± {std_r:.1f}")
        print(f"  Avg steps: {avg_steps:.0f}")
        print(f"  Reward breakdown (total / per-step):")
        print(f"    event:   {res['total_event']:8.1f}  ({res['total_event']/ts:+.4f}/step)")
        print(f"    shaping: {res['total_shaping']:8.1f}  ({res['total_shaping']/ts:+.4f}/step)")
        print(f"    sg_obs:  {res['total_sg_obs']:8.1f}  ({res['total_sg_obs']/ts:+.4f}/step)")
        print(f"    time:    {res['total_time']:8.1f}  ({res['total_time']/ts:+.4f}/step)")

    # ─── Visualization ───
    if len(all_results) < 2:
        print("Not enough results to compare")
        return

    names = list(all_results.keys())
    n = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Success Rate
    ax = axes[0, 0]
    successes = [np.mean(all_results[nm]["captures"]) * 100 for nm in names]
    bars = ax.bar(range(n), successes, color=["tab:blue", "tab:orange", "tab:green",
                                               "tab:red", "tab:purple"][:n])
    for bar, val in zip(bars, successes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Success %"); ax.set_title("Success Rate")
    ax.set_ylim(0, 110); ax.axhline(80, color="gray", ls="--", lw=0.8)
    ax.grid(axis="y", alpha=0.25)

    # 2. Mean Episode Reward
    ax = axes[0, 1]
    rewards = [np.mean(all_results[nm]["ep_rewards"]) for nm in names]
    stds = [np.std(all_results[nm]["ep_rewards"]) for nm in names]
    bars = ax.bar(range(n), rewards, yerr=stds,
                  color=["tab:blue", "tab:orange", "tab:green",
                         "tab:red", "tab:purple"][:n], capsize=5)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Return"); ax.set_title("Mean Episode Return")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.grid(axis="y", alpha=0.25)

    # 3. Avg Steps per Episode
    ax = axes[0, 2]
    avg_steps = [np.mean(all_results[nm]["ep_steps"]) for nm in names]
    bars = ax.bar(range(n), avg_steps,
                  color=["tab:blue", "tab:orange", "tab:green",
                         "tab:red", "tab:purple"][:n])
    for bar, val in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=10)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Steps"); ax.set_title("Avg Steps/Episode")
    ax.grid(axis="y", alpha=0.25)

    # 4. Reward Breakdown (stacked bar, per episode)
    ax = axes[1, 0]
    components = ["event", "shaping", "obs", "sg_obs", "time"]
    colors_comp = ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:gray"]
    x = np.arange(n)
    width = 0.6
    bottoms_pos = np.zeros(n)
    bottoms_neg = np.zeros(n)
    for comp, cc in zip(components, colors_comp):
        vals = np.array([all_results[nm][f"total_{comp}"] / n_eval for nm in names])
        pos = np.maximum(vals, 0)
        neg = np.minimum(vals, 0)
        ax.bar(x, pos, width, bottom=bottoms_pos, label=comp, color=cc, alpha=0.8)
        ax.bar(x, neg, width, bottom=bottoms_neg, color=cc, alpha=0.8)
        bottoms_pos += pos
        bottoms_neg += neg
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Reward/Episode"); ax.set_title("Reward Breakdown (per ep)")
    ax.legend(fontsize=8, loc="lower left")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.25)

    # 5. Per-step reward breakdown
    ax = axes[1, 1]
    bar_w = 0.15
    for i, (comp, cc) in enumerate(zip(components, colors_comp)):
        vals = [all_results[nm][f"total_{comp}"] / all_results[nm]["total_steps"]
                for nm in names]
        ax.bar(x + i * bar_w - 0.3, vals, bar_w, label=comp, color=cc, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Reward/Step"); ax.set_title("Per-Step Reward Breakdown")
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.25)

    # 6. Episode Reward Distribution (box plot)
    ax = axes[1, 2]
    data = [all_results[nm]["ep_rewards"] for nm in names]
    bp = ax.boxplot(data, labels=[nm[:12] for nm in names], patch_artist=True)
    box_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][:n]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Return"); ax.set_title("Reward Distribution")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.grid(axis="y", alpha=0.25)

    plt.suptitle(f"Offline RL Algorithm Comparison ({n_eval} eval episodes)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(PROJECT_ROOT, "visualization/logs/offline_rl_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
