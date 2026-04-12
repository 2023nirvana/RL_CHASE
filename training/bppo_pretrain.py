"""
BPPO (Behavior Proximal Policy Optimization) 离线强化学习预训练
===============================================================
Zhuang et al., "Behavior Proximal Policy Optimization", ICLR 2023

核心思想 (三阶段):
1. 在离线数据上训练 V(s) 和 Q(s,a)
2. 用行为克隆 (BC) 估计行为策略 π_β
3. 用 PPO 的 clipped surrogate 目标改进 π_β
   - advantage = Q(s,a) - V(s)  (来自离线评估器)
   - ratio = π_new(a|s) / π_old(a|s)
   - 与 PPO 唯一区别：advantage 用 Q-V 而非 GAE
"""

import os, sys, json, math, argparse, time, copy
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from training.collect_expert_trajectories import compute_high_reward
from training.td3bc_pretrain import OfflineReplayBuffer, TD3Critic

CONST_EPS = 1e-10


# ─────────── Gaussian Policy (带 log_prob) ───────────

class GaussianPolicy(nn.Module):
    """输出高斯分布的策略网络，支持 log_prob 计算 (BPPO 需要)"""
    def __init__(self, obs_dim=21, act_dim=2, hidden=(256, 128),
                 log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, act_dim)
        self.log_std_head = nn.Linear(prev, act_dim)
        self._init()

    def _init(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, s):
        """返回 (mean_action, dist)"""
        h = self.backbone(s)
        mean = torch.tanh(self.mean_head(h))
        log_std = self.log_std_head(h)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        dist = Normal(mean, std)
        return mean, dist

    def log_prob(self, s, a):
        """计算 log π(a|s)，对各维度求和"""
        _, dist = self.forward(s)
        lp = dist.log_prob(a)
        if lp.dim() > 1:
            lp = lp.sum(-1, keepdim=True)
        return lp

    def get_action(self, s, deterministic=False):
        """用于 eval：返回 numpy action"""
        mean, dist = self.forward(s)
        if deterministic:
            a = mean
        else:
            a = dist.sample()
        return a.clamp(-1.0, 1.0)


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


# ─────────── BPPO Trainer ───────────

class BPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        od, ad = args.obs_dim, args.act_dim
        hid = tuple(args.hidden)

        # 策略网络 (Gaussian)
        self.policy = GaussianPolicy(od, ad, hid).to(self.device)
        self.old_policy = copy.deepcopy(self.policy)

        # Value 和 Q 网络 (用于离线训练 advantage)
        self.value = ValueNet(od, hid).to(self.device)
        self.critic = TD3Critic(od, ad, hid).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=args.bc_lr)
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=args.value_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # 数据集
        self.buffer = OfflineReplayBuffer(
            args.dataset, self.device,
            normalize_reward=args.normalize_reward,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # 目录
        self.exp_name = (args.experiment_name
                         or f"bppo_{datetime.now():%Y%m%d_%H%M}")
        self.log_dir = os.path.join(
            PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(
            PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.history: List[Dict] = []
        self.eval_history: List[Dict] = []

        n_params = (sum(p.numel() for p in self.policy.parameters())
                    + sum(p.numel() for p in self.critic.parameters())
                    + sum(p.numel() for p in self.value.parameters()))
        print(f"Total params: {n_params:,}")

    # ─── Phase 1: 训练 V(s) ───
    def train_value_step(self):
        args = self.args
        s, a, r, ns, d = self.buffer.sample(args.batch_size)

        with torch.no_grad():
            next_v = self.value(ns)
            target_v = r + args.gamma * (1 - d) * next_v
        v = self.value(s)
        loss = F.mse_loss(v, target_v)

        self.value_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), args.max_grad_norm)
        self.value_opt.step()
        return loss.item(), v.mean().item()

    # ─── Phase 1: 训练 Q(s,a) (SARSA-style) ───
    def train_critic_step(self):
        args = self.args
        s, a, r, ns, d = self.buffer.sample(args.batch_size)

        with torch.no_grad():
            next_v = self.value(ns)
            target_q = r + args.gamma * (1 - d) * next_v

        q1, q2 = self.critic(s, a)
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
        self.critic_opt.step()

        # soft update target
        with torch.no_grad():
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.mul_(1 - args.tau).add_(sp.data, alpha=args.tau)

        return loss.item(), ((q1.mean() + q2.mean()) / 2).item()

    # ─── Phase 2: 行为克隆 (BC) ───
    def train_bc_step(self):
        args = self.args
        s, a, r, ns, d = self.buffer.sample(args.batch_size)
        log_prob = self.policy.log_prob(s, a)
        loss = -log_prob.mean()

        self.policy_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        self.policy_opt.step()
        return loss.item()

    # ─── Phase 3: BPPO 策略改进 ───
    def train_bppo_step(self, clip_ratio: float):
        args = self.args
        s, a, r, ns, d = self.buffer.sample(args.batch_size)

        # Advantage = Q(s,a) - V(s)  ← BPPO 与 PPO 的核心区别
        with torch.no_grad():
            q1, q2 = self.critic(s, a)
            q = torch.min(q1, q2)
            v = self.value(s)
            advantage = q - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)

            # Weighted advantage (omega)
            if args.omega != 0.5:
                weight = torch.where(advantage > 0, args.omega,
                                     1.0 - args.omega)
                advantage = weight * advantage

            # old policy log prob
            old_log_prob = self.old_policy.log_prob(s, a)

        # new policy log prob
        new_log_prob = self.policy.log_prob(s, a)
        ratio = (new_log_prob - old_log_prob).exp()

        # PPO clipped surrogate
        loss1 = ratio * advantage
        loss2 = torch.clamp(ratio, 1 - clip_ratio,
                            1 + clip_ratio) * advantage

        # entropy bonus
        _, dist = self.policy(s)
        entropy_loss = dist.entropy().sum(-1, keepdim=True) * args.entropy_weight

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        self.policy_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        self.policy_opt.step()

        return {
            "bppo_loss": loss.item(),
            "ratio_mean": ratio.mean().item(),
            "adv_mean": advantage.mean().item(),
            "entropy": dist.entropy().mean().item(),
        }

    @torch.no_grad()
    def set_old_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "actor": self.policy.state_dict(),
            "old_policy": self.old_policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value": self.value.state_dict(),
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
        self.policy.eval()
        ep_rewards, captures, ep_steps_list = [], [], []
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
                    a = self.policy.get_action(
                        obs_t, deterministic=True)[0].cpu().numpy()
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
                ep_steps += 1
                total_r_event += r_info["r_event"]
                total_r_shaping += r_info["r_shaping"]
                total_r_sg_obs += r_info["r_sg_obs"]
                total_r_time += r_info["r_time"]
            ep_rewards.append(ep_r)
            captures.append(info.get("captured", False))
            ep_steps_list.append(ep_steps)

        self.policy.train()
        if ep_steps_list:
            ts = sum(ep_steps_list)
            print(f"    [Eval diag] avg_steps={np.mean(ep_steps_list):.0f} "
                  f"max_steps={max(ep_steps_list)}")
            print(f"    [Reward breakdown per step] "
                  f"event={total_r_event/ts:.3f} "
                  f"shaping={total_r_shaping/ts:.3f} "
                  f"sg_obs={total_r_sg_obs/ts:.3f} "
                  f"time={total_r_time/ts:.3f}")
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
        print(f"\n{'='*60}")
        print(f"BPPO Training Pipeline")
        print(f"  Phase 1: V steps={args.v_steps}, Q steps={args.q_steps}")
        print(f"  Phase 2: BC steps={args.bc_steps}")
        print(f"  Phase 3: BPPO steps={args.bppo_steps}, "
              f"clip={args.clip_ratio}, omega={args.omega}")
        print(f"{'='*60}\n")
        t0 = time.time()

        # ──────── Phase 1: 训练 V(s) ────────
        print("Phase 1a: Training Value function V(s) ...")
        for step in range(1, args.v_steps + 1):
            v_loss, v_mean = self.train_value_step()
            if step % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"  [V] step {step:6d}/{args.v_steps} | "
                      f"loss={v_loss:.4f} V={v_mean:.2f} ({elapsed:.0f}s)",
                      flush=True)
            if step % args.log_every == 0:
                self.history.append({
                    "step": step, "phase": "V",
                    "value_loss": v_loss, "v_mean": v_mean,
                })

        # ──────── Phase 1b: 训练 Q(s,a) ────────
        print("\nPhase 1b: Training Q function Q(s,a) ...")
        q_offset = args.v_steps
        for step in range(1, args.q_steps + 1):
            q_loss, q_mean = self.train_critic_step()
            gstep = q_offset + step
            if step % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"  [Q] step {step:6d}/{args.q_steps} | "
                      f"loss={q_loss:.4f} Q={q_mean:.2f} ({elapsed:.0f}s)",
                      flush=True)
            if step % args.log_every == 0:
                self.history.append({
                    "step": gstep, "phase": "Q",
                    "critic_loss": q_loss, "q_mean": q_mean,
                })

        # ──────── Phase 2: 行为克隆 BC ────────
        print("\nPhase 2: Behavior Cloning ...")
        bc_offset = q_offset + args.q_steps
        best_bc_success = -1.0
        # 设置 BC 学习率
        for pg in self.policy_opt.param_groups:
            pg['lr'] = args.bc_lr

        for step in range(1, args.bc_steps + 1):
            bc_loss = self.train_bc_step()
            gstep = bc_offset + step
            if step % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"  [BC] step {step:6d}/{args.bc_steps} | "
                      f"loss={bc_loss:.4f} ({elapsed:.0f}s)", flush=True)
            if step % args.log_every == 0:
                self.history.append({
                    "step": gstep, "phase": "BC",
                    "bc_loss": bc_loss,
                })
            # BC 阶段也做 eval
            if (args.eval_every > 0
                    and step % args.eval_every == 0):
                eval_info = self.evaluate_in_env(
                    n_episodes=args.eval_episodes)
                eval_info["step"] = gstep
                eval_info["phase"] = "BC"
                self.eval_history.append(eval_info)
                es = eval_info["eval_success"]
                er = eval_info["eval_reward"]
                print(f"    [BC Eval] step={step} reward={er:.1f} "
                      f"success={es*100:.0f}%", flush=True)
                if es > best_bc_success:
                    best_bc_success = es
                    self.save_checkpoint("bc_best")

        self.save_checkpoint("bc_final")
        print(f"  BC done. Best BC success: {best_bc_success*100:.0f}%")

        # ──────── Phase 3: BPPO 策略改进 ────────
        print("\nPhase 3: BPPO policy improvement ...")
        bppo_offset = bc_offset + args.bc_steps
        # 重置优化器为 BPPO 学习率
        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=args.bppo_lr)
        self.bppo_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_opt, step_size=2, gamma=0.98)

        # 初始化 old_policy
        self.set_old_policy()
        clip_ratio = args.clip_ratio
        best_success = -1.0
        best_eval = -float("inf")

        for step in range(1, args.bppo_steps + 1):
            gstep = bppo_offset + step

            # clip decay: 前 200 步衰减，之后停止
            if args.is_clip_decay and step <= 200:
                clip_ratio = clip_ratio * args.decay
            is_lr_decay = args.is_lr_decay and step <= 200

            info = self.train_bppo_step(clip_ratio)

            if is_lr_decay:
                self.bppo_scheduler.step()

            if step % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"  [BPPO] step {step:6d}/{args.bppo_steps} | "
                      f"loss={info['bppo_loss']:.4f} "
                      f"ratio={info['ratio_mean']:.3f} "
                      f"ent={info['entropy']:.3f} "
                      f"clip={clip_ratio:.4f} ({elapsed:.0f}s)",
                      flush=True)
            if step % args.log_every == 0:
                self.history.append({
                    "step": gstep, "phase": "BPPO", **info,
                })

            bppo_eval = args.bppo_eval_every or args.eval_every
            if (bppo_eval > 0
                    and step % bppo_eval == 0):
                eval_info = self.evaluate_in_env(
                    n_episodes=args.eval_episodes)
                eval_info["step"] = gstep
                eval_info["phase"] = "BPPO"
                self.eval_history.append(eval_info)
                es = eval_info["eval_success"]
                er = eval_info["eval_reward"]
                print(f"    [BPPO Eval] step={step} reward={er:.1f} "
                      f"success={es*100:.0f}%", flush=True)
                if es > best_success or (es == best_success and er > best_eval):
                    best_success = es
                    best_eval = er
                    self.save_checkpoint("best")
                    if args.is_update_old_policy:
                        self.set_old_policy()
                        print("    -> Updated old policy")

            if step % args.save_every == 0:
                self.save_checkpoint("latest")

        # BPPO 阶段结束后强制 eval 一次
        print("\n  [BPPO] Final evaluation ...")
        eval_info = self.evaluate_in_env(n_episodes=args.eval_episodes)
        eval_info["step"] = bppo_offset + args.bppo_steps
        eval_info["phase"] = "BPPO"
        self.eval_history.append(eval_info)
        es = eval_info["eval_success"]
        er = eval_info["eval_reward"]
        print(f"    [BPPO Final Eval] reward={er:.1f} "
              f"success={es*100:.0f}%", flush=True)
        if es > best_success or (es == best_success and er > best_eval):
            best_success = es
            best_eval = er
            self.save_checkpoint("best")

        self.save_checkpoint("final")
        self.plot_curves()
        print(f"\nBPPO done. Best success: {best_success*100:.0f}%, "
              f"Best eval reward: {best_eval:.1f}")
        print(f"Total time: {time.time()-t0:.0f}s")

    def plot_curves(self):
        if not self.history:
            return

        fig, axes = plt.subplots(4, 2, figsize=(14, 20))

        def _smooth(vals, w=None):
            if w is None:
                w = min(50, len(vals) // 4 + 1)
            if w > 1 and len(vals) >= w:
                return np.convolve(vals, np.ones(w)/w, mode="valid")
            return vals

        # Phase 1a: V loss
        ax = axes[0, 0]
        v_h = [h for h in self.history if h.get("phase") == "V"]
        if v_h:
            xs = [h["step"] for h in v_h]
            ys = [h["value_loss"] for h in v_h]
            ax.plot(xs, ys, lw=0.8, alpha=0.3, color="tab:blue")
            sm = _smooth(ys)
            off = len(xs) - len(sm)
            ax.plot(xs[off:], sm, lw=2, color="tab:blue")
        ax.set_title("Phase 1a: Value Loss"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Phase 1b: Q loss
        ax = axes[0, 1]
        q_h = [h for h in self.history if h.get("phase") == "Q"]
        if q_h:
            xs = [h["step"] for h in q_h]
            ys = [h["critic_loss"] for h in q_h]
            ax.plot(xs, ys, lw=0.8, alpha=0.3, color="tab:orange")
            sm = _smooth(ys)
            off = len(xs) - len(sm)
            ax.plot(xs[off:], sm, lw=2, color="tab:orange")
        ax.set_title("Phase 1b: Critic Loss"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Phase 2: BC loss
        ax = axes[1, 0]
        bc_h = [h for h in self.history if h.get("phase") == "BC"]
        if bc_h:
            xs = [h["step"] for h in bc_h]
            ys = [h["bc_loss"] for h in bc_h]
            ax.plot(xs, ys, lw=0.8, alpha=0.3, color="tab:green")
            sm = _smooth(ys)
            off = len(xs) - len(sm)
            ax.plot(xs[off:], sm, lw=2, color="tab:green")
        ax.set_title("Phase 2: BC Loss"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Phase 3: BPPO loss
        ax = axes[1, 1]
        bp_h = [h for h in self.history if h.get("phase") == "BPPO"]
        if bp_h:
            xs = [h["step"] for h in bp_h]
            ys = [h["bppo_loss"] for h in bp_h]
            ax.plot(xs, ys, lw=0.8, alpha=0.3, color="tab:red")
            sm = _smooth(ys)
            off = len(xs) - len(sm)
            ax.plot(xs[off:], sm, lw=2, color="tab:red")
        ax.set_title("Phase 3: BPPO Loss"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        # Ratio + Entropy (BPPO)
        ax = axes[2, 0]
        if bp_h:
            xs = [h["step"] for h in bp_h]
            ys = [h["ratio_mean"] for h in bp_h]
            ax.plot(xs, ys, lw=1.5, color="tab:purple", label="ratio")
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
            ax.legend(fontsize=8)
        ax.set_title("BPPO Importance Ratio"); ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.25)

        ax = axes[2, 1]
        if bp_h:
            xs = [h["step"] for h in bp_h]
            ys = [h["entropy"] for h in bp_h]
            ax.plot(xs, ys, lw=1.5, color="tab:brown")
        ax.set_title("Policy Entropy"); ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.25)

        # Eval reward + success
        has_eval = len(self.eval_history) > 0
        ax = axes[3, 0]
        if has_eval:
            e_s = [h["step"] for h in self.eval_history]
            e_r = [h["eval_reward"] for h in self.eval_history]
            e_std = [h.get("eval_reward_std", 0) for h in self.eval_history]
            # 标注 BC vs BPPO phase
            bc_mask = [h.get("phase") == "BC" for h in self.eval_history]
            bp_mask = [h.get("phase") == "BPPO" for h in self.eval_history]
            if any(bc_mask):
                ax.plot([s for s, m in zip(e_s, bc_mask) if m],
                        [r for r, m in zip(e_r, bc_mask) if m],
                        "s-", lw=2, color="tab:green", ms=5, label="BC")
            if any(bp_mask):
                ax.plot([s for s, m in zip(e_s, bp_mask) if m],
                        [r for r, m in zip(e_r, bp_mask) if m],
                        "o-", lw=2, color="tab:red", ms=5, label="BPPO")
            ax.legend(fontsize=8)
            ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title("Eval Episode Return"); ax.set_ylabel("Return")
        ax.grid(True, alpha=0.25)

        ax = axes[3, 1]
        if has_eval:
            e_sc = [h["eval_success"] for h in self.eval_history]
            if any(bc_mask):
                ax.plot([s for s, m in zip(e_s, bc_mask) if m],
                        [v*100 for v, m in zip(e_sc, bc_mask) if m],
                        "s-", lw=2, color="tab:green", ms=5, label="BC")
            if any(bp_mask):
                ax.plot([s for s, m in zip(e_s, bp_mask) if m],
                        [v*100 for v, m in zip(e_sc, bp_mask) if m],
                        "o-", lw=2, color="tab:red", ms=5, label="BPPO")
            ax.set_ylim(-5, 105)
            ax.axhline(80, color="gray", ls="--", lw=0.8,
                       label="Expert 80%")
            ax.legend(fontsize=8)
        ax.set_title("Eval Success Rate (%)"); ax.set_ylabel("Success %")
        ax.grid(True, alpha=0.25)

        for ax in axes.flat:
            ax.set_xlabel("Step")

        plt.suptitle(f"BPPO Offline — {self.exp_name}", fontsize=14, y=1.0)
        plt.tight_layout()
        out = os.path.join(self.log_dir, "bppo_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser("BPPO offline pretrain")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--experiment-name", type=str, default=None)

    p.add_argument("--obs-dim", type=int, default=21)
    p.add_argument("--act-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])

    # Phase 1: V/Q 训练
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--v-steps", type=int, default=100000,
                   help="Value function 训练步数")
    p.add_argument("--q-steps", type=int, default=100000,
                   help="Q function 训练步数")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)

    # Phase 2: BC 训练
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--bc-steps", type=int, default=100000,
                   help="行为克隆训练步数")

    # Phase 3: BPPO 训练
    p.add_argument("--bppo-lr", type=float, default=1e-4)
    p.add_argument("--bppo-steps", type=int, default=1000,
                   help="BPPO 策略改进步数 (通常 <2000)")
    p.add_argument("--clip-ratio", type=float, default=0.25)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--decay", type=float, default=0.96,
                   help="Clip ratio decay factor")
    p.add_argument("--omega", type=float, default=0.9,
                   help="Weighted advantage: 0.5=均匀, >0.5=偏好正advantage")
    p.add_argument("--is-clip-decay", action="store_true", default=True)
    p.add_argument("--is-lr-decay", action="store_true", default=True)
    p.add_argument("--is-update-old-policy", action="store_true", default=True)
    p.add_argument("--bppo-eval-every", type=int, default=0,
                   help="BPPO\u9636\u6bb5\u7684eval\u95f4\u9694 (0=\u7528eval-every)")

    # 通用
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--normalize-reward", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="验证集比例")

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    # 环境评估
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

    trainer = BPPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
