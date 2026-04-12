"""
TD3+BC 离线强化学习预训练高层策略
=================================
Phase 1: 在专家数据集上用 TD3+BC 离线训练。
- Twin Delayed DDPG + Behavioral Cloning 正则
- Actor: max Q(s,a) + alpha * BC_loss
- Critic: Twin Q-networks, Bellman backup
- 用奖励函数训练，可验证奖励设计
"""

import os, sys, json, math, argparse, time, copy
from datetime import datetime
from typing import Dict, List, Tuple

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


# ─────────── Actor ───────────

class TD3Actor(nn.Module):
    def __init__(self, obs_dim=21, act_dim=2, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # 输出层小初始化
        last = list(self.net.children())[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, s):
        return torch.tanh(self.net(s))


# ─────────── Twin Critic ───────────

class TD3Critic(nn.Module):
    def __init__(self, obs_dim=21, act_dim=2, hidden=(256, 128)):
        super().__init__()
        self.q1 = self._build(obs_dim + act_dim, hidden)
        self.q2 = self._build(obs_dim + act_dim, hidden)

    def _build(self, in_dim, hidden):
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        return net

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1)


# ─────────── Replay Buffer ───────────

class OfflineReplayBuffer:
    def __init__(self, path: str, device: str = "cpu",
                 normalize_reward: bool = False,
                 val_ratio: float = 0.1, seed: int = 42):
        raw = np.load(path, allow_pickle=True)
        all_s = torch.tensor(raw["states"], dtype=torch.float32)
        all_a = torch.tensor(raw["actions"], dtype=torch.float32)
        all_r = torch.tensor(raw["rewards"], dtype=torch.float32)
        all_ns = torch.tensor(raw["next_states"], dtype=torch.float32)
        all_d = torch.tensor(raw["dones"], dtype=torch.float32)

        meta = json.loads(str(raw["meta"]))
        n_total = len(all_s)
        print(f"Loaded {n_total} transitions")
        print(f"  reward: [{all_r.min():.1f}, {all_r.max():.1f}], "
              f"mean={all_r.mean():.2f}, std={all_r.std():.2f}")

        # Reward normalization (对整个数据集做)
        self.r_mean = all_r.mean().item()
        self.r_std = all_r.std().item() + 1e-6
        if normalize_reward:
            all_r = (all_r - self.r_mean) / self.r_std
            print(f"  Reward normalized: mean={self.r_mean:.2f}, "
                  f"std={self.r_std:.2f}")
            print(f"  After: [{all_r.min():.2f}, {all_r.max():.2f}], "
                  f"mean={all_r.mean():.4f}")

        # Train / validation split
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total)
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val

        idx_t = perm[:n_train]
        idx_v = perm[n_train:]

        self.s = all_s[idx_t]
        self.a = all_a[idx_t]
        self.r = all_r[idx_t]
        self.ns = all_ns[idx_t]
        self.d = all_d[idx_t]
        self.n = n_train

        self.val_s = all_s[idx_v]
        self.val_a = all_a[idx_v]
        self.val_r = all_r[idx_v]
        self.val_ns = all_ns[idx_v]
        self.val_d = all_d[idx_v]
        self.n_val = n_val

        self.device = device
        print(f"  Split: train={n_train}, val={n_val}")

    def sample(self, batch_size):
        idx = torch.randint(0, self.n, (batch_size,))
        return (self.s[idx].to(self.device),
                self.a[idx].to(self.device),
                self.r[idx].to(self.device),
                self.ns[idx].to(self.device),
                self.d[idx].to(self.device))

    def sample_val(self, batch_size):
        """采样验证集 batch"""
        idx = torch.randint(0, self.n_val, (batch_size,))
        return (self.val_s[idx].to(self.device),
                self.val_a[idx].to(self.device),
                self.val_r[idx].to(self.device),
                self.val_ns[idx].to(self.device),
                self.val_d[idx].to(self.device))


# ─────────── TD3+BC Trainer ───────────

class TD3BCTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        od, ad = args.obs_dim, args.act_dim
        hid = tuple(args.hidden)

        self.actor = TD3Actor(od, ad, hid).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TD3Critic(od, ad, hid).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        self.buffer = OfflineReplayBuffer(
            args.dataset, self.device,
            normalize_reward=args.normalize_reward,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # TD3+BC 的 alpha 自动计算
        # alpha = alpha_ratio / mean(|Q|)  (论文推荐)
        self.alpha_ratio = args.alpha_ratio

        # Target Q clipping: 防止 critic 爆炸
        # 估算理论最大累积回报上界
        r_abs_max = max(abs(self.buffer.r.min().item()),
                        abs(self.buffer.r.max().item()), 10.0)
        self.q_max = r_abs_max / (1 - args.gamma) * 1.5  # 留余量
        print(f"  Target Q clip: [-{self.q_max:.1f}, {self.q_max:.1f}]")

        self.exp_name = (args.experiment_name
                         or f"td3bc_{datetime.now():%Y%m%d_%H%M}")
        self.log_dir = os.path.join(
            PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(
            PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.history: List[Dict[str, float]] = []

        n_params = (sum(p.numel() for p in self.actor.parameters())
                    + sum(p.numel() for p in self.critic.parameters()))
        print(f"Total params: {n_params:,}")

    @torch.no_grad()
    def _soft_update(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

    def train_step(self, step: int):
        args = self.args
        bs = args.batch_size
        s, a, r, ns, d = self.buffer.sample(bs)

        # ── Critic update ──
        with torch.no_grad():
            # Target action + clipped noise
            noise = (torch.randn_like(a) * args.policy_noise
                     ).clamp(-args.noise_clip, args.noise_clip)
            na = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(ns, na)
            tq = torch.min(tq1, tq2)
            target_q = r + args.gamma * (1 - d) * tq
            # Target Q clipping 防止 critic 发散
            target_q = target_q.clamp(-self.q_max, self.q_max)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        c_grad = nn.utils.clip_grad_norm_(
            self.critic.parameters(), args.max_grad_norm)
        self.critic_opt.step()

        info = {
            "critic_loss": critic_loss.item(),
            "q_mean": ((q1.mean() + q2.mean()) / 2).item(),
            "q_max": max(q1.max().item(), q2.max().item()),
            "q_min": min(q1.min().item(), q2.min().item()),
            "q_std": ((q1.std() + q2.std()) / 2).item(),
            "target_q_mean": target_q.mean().item(),
            "reward_batch_mean": r.mean().item(),
            "critic_grad_norm": c_grad.item(),
        }

        # ── Actor update (delayed) ──
        if step % args.policy_delay == 0:
            pi = self.actor(s)
            q_val = self.critic.q1_forward(s, pi)

            # TD3+BC: alpha = alpha_ratio / mean(|Q|)
            lam = self.alpha_ratio / (q_val.abs().mean().detach() + 1e-6)
            bc_loss = F.mse_loss(pi, a)
            actor_loss = -lam * q_val.mean() + bc_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            a_grad = nn.utils.clip_grad_norm_(
                self.actor.parameters(), args.max_grad_norm)
            self.actor_opt.step()

            self._soft_update(
                self.actor_target, self.actor, args.tau)
            self._soft_update(
                self.critic_target, self.critic, args.tau)

            info["actor_loss"] = actor_loss.item()
            info["bc_loss"] = bc_loss.item()
            info["q_pi"] = q_val.mean().item()
            info["lambda"] = lam.item()
            info["actor_grad_norm"] = a_grad.item()

        return info

    @torch.no_grad()
    def compute_val_loss(self):
        """在验证集上计算 critic / actor loss（不更新参数）"""
        bs = min(self.args.batch_size, self.buffer.n_val)
        s, a, r, ns, d = self.buffer.sample_val(bs)

        # Critic validation
        noise = (torch.randn_like(a) * self.args.policy_noise
                 ).clamp(-self.args.noise_clip, self.args.noise_clip)
        na = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
        tq1, tq2 = self.critic_target(ns, na)
        tq = torch.min(tq1, tq2)
        target_q = (r + self.args.gamma * (1 - d) * tq
                     ).clamp(-self.q_max, self.q_max)
        q1, q2 = self.critic(s, a)
        val_critic = (F.mse_loss(q1, target_q)
                      + F.mse_loss(q2, target_q)).item()

        # Actor validation (BC loss on val set)
        pi = self.actor(s)
        val_bc = F.mse_loss(pi, a).item()

        return {"val_critic_loss": val_critic, "val_bc_loss": val_bc}

    def save_checkpoint(self, name: str):
        """保存格式兼容 HighLevelActorCritic 加载"""
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "obs_dim": self.args.obs_dim,
            "act_dim": self.args.act_dim,
            "hidden": list(self.args.hidden),
            "config": vars(self.args),
        }, path)
        print(f"  Saved: {path}")

    # ─────────── 环境评估 ───────────

    def evaluate_in_env(self, n_episodes: int = 10) -> Dict[str, float]:
        """在环境中评估当前策略，返回统计指标"""
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
        total_r_event, total_r_shaping = 0.0, 0.0
        total_r_sg_obs, total_r_time = 0.0, 0.0

        for i in range(n_episodes):
            env.rng = np.random.default_rng(9999 + i)
            obs = env.reset()
            eta = env.auv_env.inner.dynamics.eta
            prev_dist = float(np.linalg.norm(
                np.array([eta[0], eta[1]]) - env.e))
            done = False
            ep_r = 0.0
            ep_steps = 0
            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a = self.actor(obs_t)[0].cpu().numpy()
                # 原始子目标
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

    # ─────────── 训练主循环 ───────────

    def train(self):
        args = self.args
        print(f"\nTD3+BC: {args.n_steps} steps, "
              f"alpha_ratio={args.alpha_ratio}, "
              f"gamma={args.gamma}")
        t0 = time.time()
        best_q = -float("inf")
        best_success = -1.0
        best_eval = -float("inf")
        self.eval_history: List[Dict] = []

        for step in range(1, args.n_steps + 1):
            info = self.train_step(step)

            if step % args.log_every == 0:
                self.history.append({"step": step, **info})

            if step % args.print_every == 0:
                elapsed = time.time() - t0
                cl = info["critic_loss"]
                qm = info["q_mean"]
                qmx = info["q_max"]
                qmn = info["q_min"]
                al = info.get("actor_loss", float("nan"))
                bc = info.get("bc_loss", float("nan"))
                lm = info.get("lambda", float("nan"))
                cg = info["critic_grad_norm"]
                ag = info.get("actor_grad_norm", float("nan"))
                print(f"  Step {step:7d}/{args.n_steps} | "
                      f"c_loss={cl:.4f} q=[{qmn:.1f},{qm:.1f},{qmx:.1f}] "
                      f"a_loss={al:.4f} bc={bc:.5f} "
                      f"lam={lm:.3f} grad_c={cg:.3f} grad_a={ag:.3f} "
                      f"({elapsed:.0f}s)", flush=True)

            # 验证集 loss
            if step % (args.print_every * 5) == 0:
                val_info = self.compute_val_loss()
                print(f"    [Val] critic={val_info['val_critic_loss']:.4f} "
                      f"bc={val_info['val_bc_loss']:.5f}", flush=True)
                self.history[-1].update(val_info) if self.history else None

            # 周期性环境评估
            if (args.eval_every > 0
                    and step % args.eval_every == 0):
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

            # 保存 latest
            q_now = info["q_mean"]
            if q_now > best_q:
                best_q = q_now

            if step % args.save_every == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        self.plot_curves()
        print(f"\nTD3+BC done. Best Q: {best_q:.2f}, "
              f"Best eval success: {best_success*100:.0f}%, "
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

        # ===== Row 0: Critic Loss + Mean Q =====
        ax = axes[0, 0]
        vals = [h["critic_loss"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:blue")
        sm = _smooth(vals)
        w = len(steps) - len(sm)
        ax.plot(steps[w:], sm, lw=2, color="tab:blue")
        ax.set_title("Critic Loss")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        vals = [h["q_mean"] for h in self.history]
        ax.plot(steps, vals, lw=0.8, alpha=0.3, color="tab:orange")
        sm = _smooth(vals)
        ax.plot(steps[w:], sm, lw=2, color="tab:orange")
        # 叠加 target Q
        tq_vals = [h.get("target_q_mean", float("nan"))
                    for h in self.history]
        sm_t = _smooth(tq_vals)
        ax.plot(steps[w:], sm_t, lw=1.5, color="tab:cyan",
                ls="--", label="Target Q")
        ax.legend(fontsize=8)
        ax.set_title("Mean Q Value")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)

        # ===== Row 1: Actor Loss + BC Loss =====
        ax = axes[1, 0]
        actor_h = [h for h in self.history if "actor_loss" in h]
        if actor_h:
            ss = [h["step"] for h in actor_h]
            al = [h["actor_loss"] for h in actor_h]
            ax.plot(ss, al, lw=0.8, alpha=0.3, color="tab:green")
            sm_a = _smooth(al)
            wa = len(ss) - len(sm_a)
            ax.plot(ss[wa:], sm_a, lw=2, color="tab:green")
        ax.set_title("Actor Loss")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 1]
        if actor_h:
            bc = [h["bc_loss"] for h in actor_h]
            ax.plot(ss, bc, lw=0.8, alpha=0.3, color="tab:red")
            sm_b = _smooth(bc)
            ax.plot(ss[wa:], sm_b, lw=2, color="tab:red")
        ax.set_title("BC Loss (MSE to expert)")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.25)

        # ===== Row 2: Lambda + Batch Reward =====
        ax = axes[2, 0]
        if actor_h:
            lm = [h.get("lambda", float("nan")) for h in actor_h]
            ax.plot(ss, lm, lw=0.8, alpha=0.3, color="tab:purple")
            sm_l = _smooth(lm)
            ax.plot(ss[wa:], sm_l, lw=2, color="tab:purple")
        ax.set_title("Lambda (Q-weight)")
        ax.set_ylabel("λ")
        ax.grid(True, alpha=0.25)

        ax = axes[2, 1]
        rvals = [h.get("reward_batch_mean", float("nan"))
                 for h in self.history]
        ax.plot(steps, rvals, lw=0.8, alpha=0.3, color="tab:brown")
        sm_r = _smooth(rvals)
        ax.plot(steps[w:], sm_r, lw=2, color="tab:brown")
        ax.set_title("Batch Reward Mean (dataset)")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.25)

        # ===== Row 3: Eval Return + Eval Success Rate =====
        ax = axes[3, 0]
        if has_eval:
            e_steps = [h["step"] for h in self.eval_history]
            e_rewards = [h["eval_reward"] for h in self.eval_history]
            e_std = [h.get("eval_reward_std", 0)
                     for h in self.eval_history]
            ax.plot(e_steps, e_rewards, "o-", lw=2,
                    color="tab:purple", markersize=5)
            ax.fill_between(
                e_steps,
                [r - s for r, s in zip(e_rewards, e_std)],
                [r + s for r, s in zip(e_rewards, e_std)],
                alpha=0.2, color="tab:purple")
            ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title("Eval Episode Return")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.25)

        ax = axes[3, 1]
        if has_eval:
            e_success = [h["eval_success"] for h in self.eval_history]
            ax.plot(e_steps, [s * 100 for s in e_success],
                    "o-", lw=2, color="tab:green", markersize=5)
            ax.set_ylim(-5, 105)
            ax.axhline(80, color="gray", ls="--", lw=0.8,
                       label="Expert baseline 80%")
            ax.legend(fontsize=8)
        ax.set_title("Eval Success Rate (%)")
        ax.set_ylabel("Success %")
        ax.grid(True, alpha=0.25)

        for ax in axes.flat:
            ax.set_xlabel("Step")

        plt.suptitle(f"TD3+BC Offline — {self.exp_name}",
                     fontsize=14, y=1.0)
        plt.tight_layout()
        out = os.path.join(self.log_dir, "td3bc_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ─────────── main ───────────

def main():
    p = argparse.ArgumentParser("TD3+BC offline pretrain")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--experiment-name", type=str, default=None)

    p.add_argument("--obs-dim", type=int, default=21)
    p.add_argument("--act-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])

    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=100000)

    # TD3 参数
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # BC 正则
    p.add_argument("--alpha-ratio", type=float, default=2.5,
                   help="BC 正则强度 (论文推荐 2.5)")

    # 稳定性
    p.add_argument("--normalize-reward", action="store_true",
                   help="对奖励做标准化 (防止 Critic 发散)")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="验证集比例")

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    # 环境评估参数
    p.add_argument("--eval-every", type=int, default=5000,
                   help="每多少步在环境中评估 (0=不评估)")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--low-level-ckpt", type=str, default="",
                   help="底层控制策略路径 (用于eval)")
    p.add_argument("--world-size", type=float, default=30.0)
    p.add_argument("--catch-radius", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--subgoal-range", type=float, default=0.7)
    p.add_argument("--low-steps", type=int, default=50)
    p.add_argument("--evader-mode", type=str, default="medium")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = TD3BCTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
