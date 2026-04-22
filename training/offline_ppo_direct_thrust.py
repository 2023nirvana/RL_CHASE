"""
直接推力版 - Offline PPO 预训练
================================

在 PID 专家采集的 8 维推力数据上做离线 PPO 预训练。
数据来自 collect_expert_direct_thrust.py 生成的 expert_dataset.npz。

网络：8D obs → [256, 128] → 8D action (thrust normalized to [-1,1])

与 offline_ppo_pretrain.py 结构相同，仅维度和默认参数适配底层直接推力。

用法：
    python training/offline_ppo_direct_thrust.py \\
        --dataset data/expert_direct_thrust/expert_dataset.npz \\
        --experiment-name offline_direct_thrust_v1
"""

import os
import sys
import json
import math
import argparse
import time
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


# ─────────── 直接推力 Actor-Critic ───────────

class DirectThrustActorCritic(nn.Module):
    """底层直接推力策略：8D obs → 8D thrust action"""
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 0.5

    def __init__(self, obs_dim: int = 8, act_dim: int = 8,
                 hidden: Tuple[int, ...] = (256, 128)):
        super().__init__()
        layers_a: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_a += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        self.actor_body = nn.Sequential(*layers_a)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        layers_c: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers_c += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers_c.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers_c)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, x):
        mu = torch.tanh(self.mu_head(self.actor_body(x)))
        v = self.critic(x).squeeze(-1)
        return mu, v

    def get_dist(self, x):
        h = self.actor_body(x)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_action(self, x, deterministic=False):
        dist = self.get_dist(x)
        z = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp

    def evaluate_actions(self, x, a):
        dist = self.get_dist(x)
        z = torch.clamp(a, -0.999, 0.999)
        z = 0.5 * (torch.log1p(z) - torch.log1p(-z))  # atanh
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent

    def get_value(self, x):
        return self.critic(x).squeeze(-1)


# ─────────── 离线数据集 ───────────

class OfflineDataset:
    def __init__(self, path: str, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = "cpu"):
        raw = np.load(path, allow_pickle=True)
        self.states = torch.tensor(raw["states"], dtype=torch.float32)
        self.actions = torch.tensor(raw["actions"], dtype=torch.float32)
        self.rewards = torch.tensor(raw["rewards"], dtype=torch.float32)
        self.next_states = torch.tensor(raw["next_states"], dtype=torch.float32)
        self.dones = torch.tensor(raw["dones"], dtype=torch.float32)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.n = len(self.states)
        meta = json.loads(str(raw["meta"]))
        print(f"Loaded {self.n} transitions from {path}")
        print(f"  obs_dim={self.states.shape[1]}, act_dim={self.actions.shape[1]}")
        print(f"  success_rate={meta.get('success_rate', '?')}")

        self.advantages = torch.zeros(self.n)
        self.returns = torch.zeros(self.n)

    def compute_gae(self, model: DirectThrustActorCritic):
        model.eval()
        with torch.no_grad():
            values = model.get_value(self.states.to(self.device)).cpu()
            next_values = model.get_value(self.next_states.to(self.device)).cpu()

        adv = torch.zeros(self.n)
        last_gae = 0.0
        for t in reversed(range(self.n)):
            non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + self.gamma * next_values[t] * non_terminal
                     - values[t])
            if self.dones[t] > 0.5:
                last_gae = 0.0
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            adv[t] = last_gae

        self.advantages = adv
        self.returns = adv + values
        model.train()


# ─────────── Offline PPO 训练器 ───────────

class OfflinePPODirectThrustTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = DirectThrustActorCritic(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.dataset = OfflineDataset(
            args.dataset, gamma=args.gamma,
            gae_lambda=args.gae_lambda, device=self.device,
        )

        self.exp_name = (args.experiment_name
                         or f"offline_direct_thrust_{datetime.now():%Y%m%d_%H%M}")
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.exp_name)
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history: List[Dict[str, float]] = []

    def train_step(self) -> Dict[str, float]:
        args = self.args
        self.dataset.compute_gae(self.model)
        self.model.train()

        with torch.no_grad():
            all_s = self.dataset.states.to(self.device)
            all_a = self.dataset.actions.to(self.device)
            old_logp, _ = self.model.evaluate_actions(all_s, all_a)
            old_logp = old_logp.cpu()

        total_loss, total_actor, total_critic, total_ent = 0., 0., 0., 0.
        n_updates = 0

        for _epoch in range(args.n_epochs):
            perm = torch.randperm(self.dataset.n)
            for start in range(0, self.dataset.n, args.batch_size):
                idx = perm[start:start + args.batch_size]
                b_s = self.dataset.states[idx].to(self.device)
                b_a = self.dataset.actions[idx].to(self.device)
                b_adv = self.dataset.advantages[idx].to(self.device)
                b_ret = self.dataset.returns[idx].to(self.device)
                b_old_logp = old_logp[idx].to(self.device)

                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                new_logp, ent = self.model.evaluate_actions(b_s, b_a)
                ratio = torch.exp(new_logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - args.clip_range,
                                    1 + args.clip_range) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.model.get_value(b_s)
                critic_loss = F.mse_loss(values, b_ret)

                entropy = ent.mean()
                loss = actor_loss + args.vf_coef * critic_loss - args.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_actor += actor_loss.item()
                total_critic += critic_loss.item()
                total_ent += entropy.item()
                n_updates += 1

            # KL early stopping
            if args.target_kl is not None:
                with torch.no_grad():
                    all_new_logp, _ = self.model.evaluate_actions(all_s, all_a)
                    kl = (old_logp.to(self.device) - all_new_logp).mean().item()
                if kl > args.target_kl:
                    break

        return {
            "loss": total_loss / max(1, n_updates),
            "actor_loss": total_actor / max(1, n_updates),
            "critic_loss": total_critic / max(1, n_updates),
            "entropy": total_ent / max(1, n_updates),
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.args.obs_dim,
            "act_dim": self.args.act_dim,
            "hidden": list(self.args.hidden),
            "config": vars(self.args),
        }, path)
        print(f"Saved: {path}")

    def train(self):
        n_iters = self.args.n_iters
        print(f"Offline PPO (direct thrust): {n_iters} iters, "
              f"{self.dataset.n} transitions")
        t0 = time.time()
        best_loss = float("inf")

        for it in range(1, n_iters + 1):
            info = self.train_step()
            self.history.append(info)

            if it % 10 == 0 or it == 1:
                elapsed = time.time() - t0
                print(f"  Iter {it}/{n_iters} | "
                      f"loss={info['loss']:.4f} "
                      f"actor={info['actor_loss']:.4f} "
                      f"critic={info['critic_loss']:.4f} "
                      f"ent={info['entropy']:.3f} "
                      f"({elapsed:.0f}s)", flush=True)

            if info["loss"] < best_loss:
                best_loss = info["loss"]
                self.save_checkpoint("best")

            if it % 50 == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        self.plot_curves()
        print("Offline PPO direct thrust pretraining done.")

    def plot_curves(self):
        if not self.history:
            return
        iters = np.arange(1, len(self.history) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        for ax, key, title in [
            (axes[0, 0], "loss", "Total Loss"),
            (axes[0, 1], "actor_loss", "Actor Loss"),
            (axes[1, 0], "critic_loss", "Critic Loss"),
            (axes[1, 1], "entropy", "Entropy"),
        ]:
            vals = [h[key] for h in self.history]
            ax.plot(iters, vals, lw=1.5)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.25)

        plt.suptitle(f"Offline PPO Direct Thrust — {self.exp_name}")
        plt.tight_layout()
        out = os.path.join(self.log_dir, "offline_ppo_curves.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline PPO for direct thrust (8D action)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--act-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.01)
    args = parser.parse_args()

    trainer = OfflinePPODirectThrustTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
