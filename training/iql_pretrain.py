"""
IQL (Implicit Q-Learning) 离线强化学习预训练
=============================================
Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022

核心思想:
- 用 expectile regression 训练 V(s)，隐式估计 max_a Q(s,a)
- 用 V(s') 做 Bellman backup 训练 Q(s,a)，避免查询 OOD 动作
- 用 advantage-weighted regression (AWR) 提取策略
"""

import os, sys, json, math, argparse, time, copy
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from training.collect_expert_trajectories import compute_high_reward
from training.td3bc_pretrain import OfflineReplayBuffer, TD3Actor, TD3Critic


# ─────────── Value Network V(s) ───────────

class ValueNet(nn.Module):
    def __init__(self, obs_dim=21, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, s):
        return self.net(s).squeeze(-1)


# ─────────── IQL Trainer ───────────

class IQLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        od, ad = args.obs_dim, args.act_dim
        hid = tuple(args.hidden)

        self.actor = TD3Actor(od, ad, hid).to(self.device)
        self.critic = TD3Critic(od, ad, hid).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.value = ValueNet(od, hid).to(self.device)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=args.value_lr)

        self.buffer = OfflineReplayBuffer(
            args.dataset, self.device,
            normalize_reward=args.normalize_reward,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # Target Q clipping
        r_abs_max = max(abs(self.buffer.r.min().item()),
                        abs(self.buffer.r.max().item()), 10.0)
        self.q_max = r_abs_max / (1 - args.gamma) * 1.5
        print(f"  Target Q clip: [-{self.q_max:.1f}, {self.q_max:.1f}]")

        self.exp_name = (args.experiment_name
                         or f"iql_{datetime.now():%Y%m%d_%H%M}")
        self.log_dir = os.path.join(
            PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(
            PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.history: List[Dict[str, float]] = []
        self.eval_history: List[Dict] = []

        n_params = (sum(p.numel() for p in self.actor.parameters())
                    + sum(p.numel() for p in self.critic.parameters())
                    + sum(p.numel() for p in self.value.parameters()))
        print(f"Total params: {n_params:,}")

    @torch.no_grad()
    def _soft_update(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

    def _expectile_loss(self, diff, expectile):
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, step: int):
        args = self.args
        s, a, r, ns, d = self.buffer.sample(args.batch_size)

        # ── 1. Value update: expectile regression ──
        with torch.no_grad():
            q1, q2 = self.critic_target(s, a)
            q = torch.min(q1, q2)
        v = self.value(s)
        value_loss = self._expectile_loss(q - v, args.expectile)

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.value.parameters(), args.max_grad_norm)
        self.value_opt.step()

        # ── 2. Critic update: use V(s') for backup ──
        with torch.no_grad():
            next_v = self.value(ns)
            target_q = r + args.gamma * (1 - d) * next_v
            target_q = target_q.clamp(-self.q_max, self.q_max)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), args.max_grad_norm)
        self.critic_opt.step()

        self._soft_update(self.critic_target, self.critic, args.tau)

        # ── 3. Actor update: advantage-weighted regression ──
        with torch.no_grad():
            q1, q2 = self.critic_target(s, a)
            q = torch.min(q1, q2)
            v = self.value(s)
            adv = q - v
            # Clamp for numerical stability
            weights = torch.exp(adv / args.beta).clamp(max=100.0)

        pi = self.actor(s)
        # AWR: weighted MSE to dataset actions
        actor_loss = (weights * (pi - a).pow(2).sum(dim=-1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), args.max_grad_norm)
        self.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": ((q1.mean() + q2.mean()) / 2).item(),
            "v_mean": self.value(s).mean().item(),
            "target_q_mean": target_q.mean().item(),
            "reward_batch_mean": r.mean().item(),
            "adv_mean": adv.mean().item(),
            "weight_mean": weights.mean().item(),
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value": self.value.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "obs_dim": self.args.obs_dim,
            "act_dim": self.args.act_dim,
            "hidden": list(self.args.hidden),
            "config": vars(self.args),
        }, path)
        print(f"  Saved: {path}")

    def evaluate_in_env(self, n_episodes: int = 10) -> Dict[str, float]:
        if not hasattr(self, '_eval_env'):
            if not self.args.low_level_ckpt:
                return {"eval_reward": float("nan"),
                        "eval_success": float("nan")}
            nn_policy = load_v4_nn(
                self.args.low_level_ckpt, self.device)
            self._eval_env = PursuitEnvRealAUVFixed(
                nn_policy=nn_policy, device=self.device,
                world_size=self.args.world_size,
                catch_radius=self.args.catch_radius,
                max_steps=120 * self.args.low_steps,
                subgoal_range=self.args.subgoal_range,
                low_steps=self.args.low_steps,
                seed=9999,
                evader_mode=self.args.evader_mode,
            )

        env = self._eval_env
        self.actor.eval()
        ep_rewards, captures, ep_steps_list = [], [], []
        # 诊断：累计各 reward 分量
        total_r_event, total_r_shaping = 0.0, 0.0
        total_r_sg_obs, total_r_time = 0.0, 0.0

        for i in range(n_episodes):
            env.rng = np.random.default_rng(9999 + i)
            obs = env.reset()
            eta = env.auv_env.inner.dynamics.eta
            prev_dist = float(np.linalg.norm(
                np.array([eta[0], eta[1]]) - env.e))
            done, ep_r, ep_steps = False, 0.0, 0
            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a = self.actor(obs_t)[0].cpu().numpy()
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
                    subgoal=sg,                    done=done,                )
                ep_r += r
                prev_dist = curr_dist
                ep_steps += 1
                total_r_event += r_info["r_event"]
                total_r_shaping += r_info["r_shaping"]
                total_r_sg_obs += r_info["r_sg_obs"]
                total_r_time += r_info["r_time"]
            ep_rewards.append(ep_r)
            captures.append(info.get("captured", False))
            ep_steps_list.append(ep_steps)

        self.actor.train()
        # 诊断：打印首个 eval episode 的 reward 分解
        if ep_steps_list:
            total_steps = sum(ep_steps_list)
            print(f"    [Eval diag] avg_steps={np.mean(ep_steps_list):.0f} "
                  f"max_steps={max(ep_steps_list)}")
            print(f"    [Reward breakdown per step] "
                  f"event={total_r_event/total_steps:.3f} "
                  f"shaping={total_r_shaping/total_steps:.3f} "
                  f"sg_obs={total_r_sg_obs/total_steps:.3f} "
                  f"time={total_r_time/total_steps:.3f}")
            print(f"    [Reward total] "
                  f"event={total_r_event:.1f} "
                  f"shaping={total_r_shaping:.1f} "
                  f"sg_obs={total_r_sg_obs:.1f} "
                  f"time={total_r_time:.1f}", flush=True)
        return {
            "eval_reward": float(np.mean(ep_rewards)),
            "eval_success": float(np.mean(captures)),
            "eval_reward_std": float(np.std(ep_rewards)),
            "eval_steps": float(np.mean(ep_steps_list)),
        }

    def train(self):
        args = self.args
        print(f"\nIQL: {args.n_steps} steps, "
              f"expectile={args.expectile}, beta={args.beta}, "
              f"gamma={args.gamma}")
        t0 = time.time()
        best_success = -1.0
        best_eval = -float("inf")

        for step in range(1, args.n_steps + 1):
            info = self.train_step(step)

            if step % args.log_every == 0:
                self.history.append({"step": step, **info})

            if step % args.print_every == 0:
                elapsed = time.time() - t0
                cl = info["critic_loss"]
                vl = info["value_loss"]
                al = info["actor_loss"]
                qm = info["q_mean"]
                vm = info["v_mean"]
                print(f"  Step {step:6d}/{args.n_steps} | "
                      f"c={cl:.4f} v={vl:.4f} a={al:.4f} "
                      f"Q={qm:.2f} V={vm:.2f} ({elapsed:.0f}s)",
                      flush=True)

            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_info = self.evaluate_in_env(
                    n_episodes=args.eval_episodes)
                eval_info["step"] = step
                self.eval_history.append(eval_info)
                er = eval_info["eval_reward"]
                es = eval_info["eval_success"]
                print(f"    [Eval] step={step} reward={er:.1f} "
                      f"success={es*100:.0f}%", flush=True)
                if es > best_success or (es == best_success and er > best_eval):
                    best_success = es
                    best_eval = er
                    self.save_checkpoint("best")

            if step % args.save_every == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        self.plot_curves()
        print(f"\nIQL done. Best eval success: {best_success*100:.0f}%, "
              f"Best eval reward: {best_eval:.1f}")

    def plot_curves(self):
        if not self.history:
            return
        steps = [h["step"] for h in self.history]
        has_eval = len(self.eval_history) > 0

        fig, axes = plt.subplots(4, 2, figsize=(14, 20))

        def _smooth(vals, w=None):
            if w is None:
                w = min(50, len(vals) // 4 + 1)
            if w > 1 and len(vals) >= w:
                return np.convolve(vals, np.ones(w)/w, mode="valid")
            return vals

        w_offset = lambda vals: len(steps) - len(_smooth(vals))

        # Row 0: Critic Loss + Value Loss
        ax = axes[0, 0]
        vals = [h["critic_loss"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:blue")
        sm = _smooth(vals); w = w_offset(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:blue")
        ax.set_title("Critic Loss"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        vals = [h["value_loss"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:cyan")
        sm = _smooth(vals); w = w_offset(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:cyan")
        ax.set_title("Value Loss (expectile)"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Row 1: Q & V + Actor Loss
        ax = axes[1, 0]
        qvals = [h["q_mean"] for h in self.history]
        vvals = [h["v_mean"] for h in self.history]
        ax.plot(steps, qvals, lw=0.8, alpha=0.3, color="tab:orange")
        ax.plot(steps[w_offset(qvals):], _smooth(qvals), lw=2,
                color="tab:orange", label="Q")
        ax.plot(steps, vvals, lw=0.8, alpha=0.3, color="tab:green")
        ax.plot(steps[w_offset(vvals):], _smooth(vvals), lw=2,
                color="tab:green", label="V")
        ax.legend(fontsize=8)
        ax.set_title("Mean Q & V"); ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 1]
        vals = [h["actor_loss"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:red")
        sm = _smooth(vals); w = w_offset(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:red")
        ax.set_title("Actor Loss (AWR)"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Row 2: Advantage + AWR Weights
        ax = axes[2, 0]
        vals = [h["adv_mean"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:purple")
        sm = _smooth(vals); w = w_offset(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:purple")
        ax.set_title("Mean Advantage (Q-V)"); ax.set_ylabel("Adv")
        ax.grid(True, alpha=0.25)

        ax = axes[2, 1]
        vals = [h["weight_mean"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:brown")
        sm = _smooth(vals); w = w_offset(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:brown")
        ax.set_title("AWR Weight Mean"); ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.25)

        # Row 3: Eval Return + Success
        ax = axes[3, 0]
        if has_eval:
            e_s = [h["step"] for h in self.eval_history]
            e_r = [h["eval_reward"] for h in self.eval_history]
            e_std = [h.get("eval_reward_std", 0) for h in self.eval_history]
            ax.plot(e_s, e_r, "o-", lw=2, color="tab:purple", ms=5)
            ax.fill_between(e_s,
                            [r-s for r,s in zip(e_r, e_std)],
                            [r+s for r,s in zip(e_r, e_std)],
                            alpha=0.2, color="tab:purple")
            ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title("Eval Episode Return"); ax.set_ylabel("Return")
        ax.grid(True, alpha=0.25)

        ax = axes[3, 1]
        if has_eval:
            e_sc = [h["eval_success"] for h in self.eval_history]
            ax.plot(e_s, [s*100 for s in e_sc], "o-", lw=2,
                    color="tab:green", ms=5)
            ax.set_ylim(-5, 105)
            ax.axhline(80, color="gray", ls="--", lw=0.8,
                       label="Expert baseline 80%")
            ax.legend(fontsize=8)
        ax.set_title("Eval Success Rate (%)"); ax.set_ylabel("Success %")
        ax.grid(True, alpha=0.25)

        for ax in axes.flat:
            ax.set_xlabel("Step")

        plt.suptitle(f"IQL Offline — {self.exp_name}", fontsize=14, y=1.0)
        plt.tight_layout()
        out = os.path.join(self.log_dir, "iql_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser("IQL offline pretrain")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--experiment-name", type=str, default=None)

    p.add_argument("--obs-dim", type=int, default=21)
    p.add_argument("--act-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])

    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=150000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # IQL 核心参数
    p.add_argument("--expectile", type=float, default=0.7,
                   help="Expectile τ (0.5=median, 0.7~0.9=upper)")
    p.add_argument("--beta", type=float, default=3.0,
                   help="AWR 温度 (越大越接近 greedy)")
    p.add_argument("--normalize-reward", action="store_true",
                   help="对数据集 reward 做 z-score 归一化")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="验证集比例")

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=15)
    p.add_argument("--low-level-ckpt", type=str, default="")
    p.add_argument("--world-size", type=float, default=30.0)
    p.add_argument("--catch-radius", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--subgoal-range", type=float, default=0.7)
    p.add_argument("--low-steps", type=int, default=50)
    p.add_argument("--evader-mode", type=str, default="medium")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = IQLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
