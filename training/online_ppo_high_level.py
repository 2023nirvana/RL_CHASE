"""
高层 Online PPO 微调（追逃环境）
==================================

加载 offline 预训练的高层策略，在 PursuitEnvDirectThrust 中在线
微调，底层固定为 v5 直接推力控制器。

用法：
  python training/online_ppo_high_level.py \\
    --pretrain-ckpt checkpoints/offline_high_level_v1/best.pth \\
    --experiment-name online_high_level_v1 \\
    --total-timesteps 500000
"""
import os
import sys
import argparse
import csv
import json
import copy
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_direct_thrust import (
    load_v5_nn, PursuitEnvDirectThrust,
)
from training.offline_ppo_high_level import HighLevelActorCritic


def make_env(args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_policy = load_v5_nn(args.nn_ckpt, device)
    return PursuitEnvDirectThrust(
        nn_policy=nn_policy, device=device,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        max_steps=args.max_steps,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=seed,
        evader_mode=args.evader_mode,
        v_max_pursuer=args.v_max_pursuer,
        a_max_pursuer=args.a_max_pursuer,
        v_max_evader=args.v_max_evader,
        a_max_evader=args.a_max_evader,
    )


class OnlineHighLevelTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.exp_name = args.experiment_name or f"online_high_level_{datetime.now():%Y%m%d_%H%M}"
        self.out_dir = os.path.join(PROJECT_ROOT, "low_near", self.exp_name)
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 单环境（高层 step 已经包含 low_steps 个底层步）
        self.env = make_env(args, seed=args.seed)
        self.eval_env = make_env(args, seed=args.seed + 10000)

        self.obs_dim = 21
        self.act_dim = 2

        self._init_network()

        self.episode_rewards: List[float] = []
        self.episode_successes: List[int] = []
        self.history: List[Dict[str, float]] = []

    def _init_network(self):
        hidden = tuple(self.args.hidden)
        self.model = HighLevelActorCritic(
            obs_dim=self.obs_dim, act_dim=self.act_dim, hidden=hidden
        ).to(self.device)

        if self.args.pretrain_ckpt:
            ckpt = torch.load(self.args.pretrain_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrain: {self.args.pretrain_ckpt}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.bc_model = None
        if self.args.beta_kl > 0 and self.args.pretrain_ckpt:
            self.bc_model = copy.deepcopy(self.model)
            self.bc_model.eval()
            for p in self.bc_model.parameters():
                p.requires_grad = False
            print(f"KL constraint: beta={self.args.beta_kl}")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network params: {total_params:,}")

    def _get_lr(self, progress):
        return self.args.lr_end + (self.args.lr - self.args.lr_end) * (1.0 - progress)

    def _get_beta_kl(self, progress):
        if self.args.kl_anneal_frac <= 0:
            return self.args.beta_kl
        if progress > self.args.kl_anneal_frac:
            return 0.0
        return self.args.beta_kl * (1.0 - progress / self.args.kl_anneal_frac)

    def collect_rollouts(self, n_steps: int, cur_obs=None, cur_ep_rew=0.0):
        """单环境 rollout，累积 n_steps 个高层 transition。"""
        obs_buf, act_buf, rew_buf = [], [], []
        done_buf, val_buf, logp_buf = [], [], []

        if cur_obs is None:
            self.env.rng = np.random.default_rng(
                self.args.seed + len(self.episode_rewards) * 7919)
            cur_obs = self.env.reset()
            cur_ep_rew = 0.0

        for _ in range(n_steps):
            obs_t = torch.tensor(cur_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, logp = self.model.get_action(obs_t)
                v = self.model.get_value(obs_t)

            action_np = a[0].cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_np)

            obs_buf.append(cur_obs.astype(np.float32))
            act_buf.append(action_np.astype(np.float32))
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(float(v.item()))
            logp_buf.append(float(logp.item()))

            cur_ep_rew += float(reward)

            if done:
                self.episode_rewards.append(cur_ep_rew)
                self.episode_successes.append(int(bool(info.get("captured", False))))
                self.env.rng = np.random.default_rng(
                    self.args.seed + len(self.episode_rewards) * 7919)
                cur_obs = self.env.reset()
                cur_ep_rew = 0.0
            else:
                cur_obs = next_obs

        with torch.no_grad():
            last_v = self.model.get_value(torch.tensor(
                cur_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)).item()

        rews = np.array(rew_buf, dtype=np.float32)
        vals = np.array(val_buf, dtype=np.float32)
        dns = np.array(done_buf, dtype=np.float32)

        advs = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            nv = last_v if t == n_steps - 1 else vals[t + 1]
            nt = 1.0 - dns[t]
            delta = rews[t] + self.args.gamma * nv * nt - vals[t]
            last_gae = delta + self.args.gamma * self.args.gae_lambda * nt * last_gae
            advs[t] = last_gae
        returns = advs + vals

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advs,
            "returns": returns,
            "mean_reward": float(rews.mean()),
        }, cur_obs, cur_ep_rew

    def train_step(self, rollout, progress):
        obs = torch.tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(rollout["logprobs"], dtype=torch.float32, device=self.device)
        advs = torch.tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        rets = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        lr = self._get_lr(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        beta_kl = self._get_beta_kl(progress)

        total_loss = 0.0
        n_updates = 0
        for _ in range(self.args.n_epochs):
            idx = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.args.batch_size):
                bi = idx[start:start + self.args.batch_size]
                b_obs, b_act, b_old_logp = obs[bi], actions[bi], old_logp[bi]
                b_adv, b_ret = advs[bi], rets[bi]

                new_logp, ent_each = self.model.evaluate_actions(b_obs, b_act)
                entropy = ent_each.mean()
                values = self.model.get_value(b_obs)

                ratio = torch.exp(new_logp - b_old_logp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * b_adv
                actor_loss = -torch.min(s1, s2).mean()
                value_loss = nn.functional.mse_loss(values, b_ret)

                loss = actor_loss + self.args.vf_coef * value_loss - self.args.ent_coef * entropy

                if self.bc_model is not None and beta_kl > 0:
                    with torch.no_grad():
                        bc_logp, _ = self.bc_model.evaluate_actions(b_obs, b_act)
                    kl = (bc_logp - new_logp).mean()
                    loss = loss + beta_kl * kl

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1
        return total_loss / max(1, n_updates)

    def evaluate(self, num_episodes=10):
        success = 0
        rewards = []
        for i in range(num_episodes):
            self.eval_env.rng = np.random.default_rng(self.args.seed + 99999 + i)
            obs = self.eval_env.reset()
            done = False
            ep_rew = 0.0
            info = {}
            while not done:
                x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a, _ = self.model.get_action(x, deterministic=True)
                obs, r, done, info = self.eval_env.step(a[0].cpu().numpy())
                ep_rew += r
            rewards.append(ep_rew)
            if info.get("captured", False):
                success += 1
        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(success / max(1, num_episodes)),
        }

    def save_checkpoint(self, name):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": list(self.args.hidden),
            "experiment_name": self.exp_name,
            "config": vars(self.args),
        }, path)
        print(f"Saved: {path}")

    def _save_history(self):
        if not self.history:
            return
        csv_path = os.path.join(self.out_dir, "training_metrics.csv")
        fields = list(self.history[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.history)

    def _plot_curves(self):
        if not self.history:
            return
        steps = [h["steps"] for h in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, key, title in [
            (axes[0, 0], "train_mean_reward", "Train Reward"),
            (axes[0, 1], "train_loss", "Loss"),
            (axes[1, 0], "train_success_rate", "Train Success Rate"),
            (axes[1, 1], "eval_success_rate", "Eval Success Rate"),
        ]:
            vals = [h[key] for h in self.history]
            ax.plot(steps, vals, lw=1.5)
            ax.set_title(title); ax.set_xlabel("high-level steps"); ax.grid(True, alpha=0.25)
        plt.suptitle(f"Online High-Level Fine-tune — {self.exp_name}")
        plt.tight_layout()
        out = os.path.join(self.out_dir, "training_curves.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    def train(self):
        n_steps = self.args.n_steps
        total_ts = self.args.total_timesteps
        num_updates = total_ts // n_steps

        print(f"Online PPO (high-level): {num_updates} updates, n_steps={n_steps}, total={total_ts}")

        cur_obs = None
        cur_ep_rew = 0.0
        best_eval_sr = -1.0
        last_eval_rew = float("nan")
        last_eval_sr = float("nan")
        t0 = time.time()

        for up in range(1, num_updates + 1):
            progress = up / num_updates
            rollout, cur_obs, cur_ep_rew = self.collect_rollouts(n_steps, cur_obs, cur_ep_rew)
            loss = self.train_step(rollout, progress)

            cur_steps = up * n_steps
            recent = self.episode_successes[-50:] if self.episode_successes else [0]
            train_sr = float(np.mean(recent)) if recent else 0.0

            if up % self.args.log_every == 0 or up == 1:
                elapsed = time.time() - t0
                print(f"Update {up}/{num_updates} | Steps {cur_steps} | "
                      f"Rew {rollout['mean_reward']:.3f} | Loss {loss:.4f} | "
                      f"TrainSucc {train_sr:.2%} | Eps {len(self.episode_rewards)} | "
                      f"({elapsed:.0f}s)", flush=True)

            if up % self.args.eval_every == 0:
                ev = self.evaluate(num_episodes=self.args.eval_episodes)
                last_eval_rew = ev["mean_reward"]
                last_eval_sr = ev["success_rate"]
                print(f"  [Eval@{cur_steps}] rew={ev['mean_reward']:.3f} succ={ev['success_rate']:.2%}", flush=True)
                if ev["success_rate"] > best_eval_sr:
                    best_eval_sr = ev["success_rate"]
                    self.save_checkpoint("best")

            if up % self.args.save_every == 0:
                self.save_checkpoint("latest")

            self.history.append({
                "update": int(up),
                "steps": int(cur_steps),
                "train_mean_reward": float(rollout["mean_reward"]),
                "train_loss": float(loss),
                "train_success_rate": float(train_sr),
                "eval_mean_reward": float(last_eval_rew),
                "eval_success_rate": float(last_eval_sr),
            })

            if up % 20 == 0:
                self._save_history()

        self.save_checkpoint("final")
        self._save_history()
        self._plot_curves()
        print("Online high-level fine-tuning finished.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nn-ckpt", type=str,
                   default="low_near/v5_direct_thrust/checkpoints/best.pth")
    p.add_argument("--pretrain-ckpt", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--total-timesteps", type=int, default=500000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--beta-kl", type=float, default=0.3)
    p.add_argument("--kl-anneal-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)
    # env
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
    args = p.parse_args()

    trainer = OnlineHighLevelTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
