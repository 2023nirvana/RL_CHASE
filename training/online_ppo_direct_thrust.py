"""
直接推力版 - Online PPO 微调
==============================

加载 offline 预训练的直接推力策略权重，
在底层小目标到达环境中在线交互做 PPO 微调。

与 train_2d_v3_direct_thrust.py 相同的训练循环，
额外功能：
- 加载预训练 checkpoint
- KL 约束（防止偏离预训练策略太远）
- 学习率线性衰减

用法：
    python training/online_ppo_direct_thrust.py \
        --pretrain-ckpt checkpoints/offline_direct_thrust_v1/best.pth \
        --experiment-name online_direct_thrust_v1 \
        --total-timesteps 2000000
"""

import os
import sys
import argparse
import yaml
import csv
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_direct_thrust import (
    LowLevelAUV2DEnvV3DirectThrust,
    VecLowLevelAUV2DEnvV3DirectThrust,
)
from training.offline_ppo_direct_thrust import DirectThrustActorCritic


class OnlineDirectThrustTrainer:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.experiment_name = (
            args.experiment_name
            or f"online_direct_thrust_{datetime.now():%Y%m%d_%H%M}")
        self.out_dir = os.path.join(
            PROJECT_ROOT, "low_near", self.experiment_name)
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._init_environments()
        self._init_network()

        self.success_count = 0
        self.episode_count = 0
        self.history = []

    def _init_environments(self):
        tc = self.config["training"]
        self.num_envs = int(tc.get("n_envs", 4))
        env_kwargs = dict(
            target_distance_max=self.args.target_distance_max,
            target_distance_min=self.args.target_distance_min,
            reach_threshold=self.args.reach_threshold,
            max_episode_seconds=self.args.max_episode_seconds,
            reward_reach=self.args.reward_reach,
            reward_speed_bonus=self.args.reward_speed_bonus,
            reward_improve_scale=self.args.reward_improve_scale,
            reward_heading_scale=self.args.reward_heading_scale,
            reward_heading_terminal=self.args.reward_heading_terminal,
            reward_vel_align_scale=self.args.reward_vel_align_scale,
            reward_time_penalty=self.args.reward_time_penalty,
            init_velocity_range=self.args.init_velocity_range,
            init_yaw_rate_range=self.args.init_yaw_rate_range,
            thrust_scale=self.args.thrust_scale,
        )
        self.train_env = VecLowLevelAUV2DEnvV3DirectThrust(
            num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV2DEnvV3DirectThrust(**env_kwargs)

        self.obs_dim = self.train_env.observation_space.shape[0]
        self.act_dim = self.train_env.action_space.shape[0]
        print(f"Env: obs_dim={self.obs_dim}, act_dim={self.act_dim}")

    def _init_network(self):
        tc = self.config["training"]
        hidden = tuple(self.args.hidden)

        self.model = DirectThrustActorCritic(
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            hidden=hidden,
        ).to(self.device)

        # 加载预训练权重
        if self.args.pretrain_ckpt:
            ckpt = torch.load(
                self.args.pretrain_ckpt, map_location=self.device,
                weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrain: {self.args.pretrain_ckpt}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(tc.get("learning_rate", 3e-4)))

        # KL 参考策略（冻结）
        self.bc_model = None
        if self.args.beta_kl > 0 and self.args.pretrain_ckpt:
            self.bc_model = copy.deepcopy(self.model)
            self.bc_model.eval()
            for p in self.bc_model.parameters():
                p.requires_grad = False
            print(f"  KL constraint: beta={self.args.beta_kl}")

        self.gamma = float(tc.get("gamma", 0.99))
        self.gae_lambda = float(tc.get("gae_lambda", 0.95))
        self.clip_range = float(tc.get("clip_range", 0.2))
        self.ent_coef = float(tc.get("ent_coef", 0.05))
        self.vf_coef = float(tc.get("vf_coef", 0.5))
        self.max_grad_norm = float(tc.get("max_grad_norm", 0.5))
        self.n_steps = int(tc.get("n_steps", 1024))
        self.batch_size = int(tc.get("batch_size", 256))
        self.n_epochs = int(tc.get("n_epochs", 4))

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network params: {total_params:,}")

    def _get_lr(self, progress: float) -> float:
        """线性衰减学习率"""
        lr_start = self.args.lr
        lr_end = self.args.lr_end
        return lr_end + (lr_start - lr_end) * (1.0 - progress)

    def _get_beta_kl(self, progress: float) -> float:
        """KL 系数线性退火"""
        if self.args.kl_anneal_frac <= 0:
            return self.args.beta_kl
        if progress > self.args.kl_anneal_frac:
            return 0.0
        return self.args.beta_kl * (
            1.0 - progress / self.args.kl_anneal_frac)

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf = [], [], []
        done_buf, val_buf, logp_buf = [], [], []
        obs, _ = self.train_env.reset()

        for _ in range(self.n_steps):
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp = self.model.get_action(obs_t)
                value = self.model.get_value(obs_t)

            action_np = action.cpu().numpy()
            next_obs, rewards, terms, truncs, infos = \
                self.train_env.step(action_np)
            dones = np.logical_or(terms, truncs)

            obs_buf.append(obs)
            act_buf.append(action_np)
            rew_buf.append(rewards)
            done_buf.append(dones.astype(np.float32))
            val_buf.append(value.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())

            for i, info in enumerate(infos):
                if info.get("termination_reason") == "success":
                    self.success_count += 1
                if dones[i]:
                    self.episode_count += 1

            obs = next_obs

        with torch.no_grad():
            last_vals = self.model.get_value(torch.tensor(
                obs, dtype=torch.float32, device=self.device
            )).cpu().numpy()

        rews = np.array(rew_buf)
        vals = np.array(val_buf)
        dns = np.array(done_buf)

        advs = np.zeros_like(rews, dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            nv = last_vals if t == self.n_steps - 1 else vals[t + 1]
            nt = 1.0 - dns[t]
            delta = rews[t] + self.gamma * nv * nt - vals[t]
            last_gae = delta + self.gamma * self.gae_lambda * nt * last_gae
            advs[t] = last_gae
        returns = advs + vals

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advs,
            "returns": returns,
            "mean_reward": float(np.mean(rews)),
        }

    def train_step(self, rollout, progress):
        obs = torch.tensor(
            rollout["obs"].reshape(-1, self.obs_dim),
            dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            rollout["actions"].reshape(-1, self.act_dim),
            dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(
            rollout["logprobs"].reshape(-1),
            dtype=torch.float32, device=self.device)
        advs = torch.tensor(
            rollout["advantages"].reshape(-1),
            dtype=torch.float32, device=self.device)
        rets = torch.tensor(
            rollout["returns"].reshape(-1),
            dtype=torch.float32, device=self.device)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # 更新学习率
        lr = self._get_lr(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        beta_kl = self._get_beta_kl(progress)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.batch_size):
                bi = idx[start:start + self.batch_size]
                b_obs = obs[bi]
                b_act = actions[bi]
                b_old_logp = old_logp[bi]
                b_adv = advs[bi]
                b_ret = rets[bi]

                new_logp, ent_each = self.model.evaluate_actions(
                    b_obs, b_act)
                entropy = ent_each.mean()
                values = self.model.get_value(b_obs)

                ratio = torch.exp(new_logp - b_old_logp)
                s1 = ratio * b_adv
                s2 = torch.clamp(
                    ratio, 1 - self.clip_range,
                    1 + self.clip_range) * b_adv
                actor_loss = -torch.min(s1, s2).mean()
                value_loss = nn.functional.mse_loss(values, b_ret)

                loss = (actor_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy)

                # KL 约束
                if self.bc_model is not None and beta_kl > 0:
                    with torch.no_grad():
                        bc_logp, _ = self.bc_model.evaluate_actions(
                            b_obs, b_act)
                    kl = (bc_logp - new_logp).mean()
                    loss = loss + beta_kl * kl

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / max(1, self.n_epochs)

    def evaluate(self, num_episodes=10):
        rewards, final_errors = [], []
        success = 0
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done, ep_rew = False, 0.0
            while not done:
                x = torch.tensor(
                    obs, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.model.get_action(x, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(
                    action[0].cpu().numpy())
                ep_rew += r
                done = term or trunc
                if done:
                    final_errors.append(
                        info.get("pos_error", float("nan")))
                    if info.get("termination_reason") == "success":
                        success += 1
            rewards.append(ep_rew)
        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(success / max(1, num_episodes)),
            "mean_final_error": float(np.nanmean(final_errors)),
        }

    def train(self):
        total_ts = self.args.total_timesteps
        ts_per_up = self.n_steps * self.num_envs
        num_updates = total_ts // ts_per_up

        print(f"Online PPO: {num_updates} updates "
              f"({total_ts} timesteps)")

        best_eval = -1e9
        eval_freq = int(self.config["training"].get("eval_freq", 5000))
        save_freq = int(self.config["training"].get("save_freq", 20000))
        last_eval_rew = float("nan")
        last_eval_sr = float("nan")

        for up in range(1, num_updates + 1):
            progress = up / num_updates
            rollout = self.collect_rollouts()
            loss = self.train_step(rollout, progress)

            cur_steps = up * ts_per_up
            sr = self.success_count / max(1, self.episode_count)

            if up % 10 == 0:
                print(
                    f"Update {up}/{num_updates} | "
                    f"Steps {cur_steps} | "
                    f"Rew {rollout['mean_reward']:.3f} | "
                    f"Loss {loss:.4f} | "
                    f"Succ {sr:.2%}",
                    flush=True)

            if cur_steps % eval_freq < ts_per_up:
                ev = self.evaluate()
                last_eval_rew = ev["mean_reward"]
                last_eval_sr = ev["success_rate"]
                print(
                    f"  [Eval] steps={cur_steps} "
                    f"rew={ev['mean_reward']:.3f} "
                    f"succ={ev['success_rate']:.2%} "
                    f"err={ev['mean_final_error']:.4f}",
                    flush=True)
                if ev["mean_reward"] > best_eval:
                    best_eval = ev["mean_reward"]
                    self.save_checkpoint("best")

            if cur_steps % save_freq < ts_per_up:
                self.save_checkpoint("latest")

            self.history.append({
                "update": int(up),
                "steps": int(cur_steps),
                "train_mean_reward": float(rollout["mean_reward"]),
                "train_loss": float(loss),
                "train_success_rate": float(sr),
                "eval_mean_reward": float(last_eval_rew),
                "eval_success_rate": float(last_eval_sr),
            })

            if up % 20 == 0:
                self._save_history()

        self.save_checkpoint("final")
        self._save_history()
        self._plot_curves()
        print("Online fine-tuning finished.")

    def save_checkpoint(self, name):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": list(self.args.hidden),
            "config": self.config,
            "experiment_name": self.experiment_name,
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
            ax.set_title(title)
            ax.set_xlabel("timesteps")
            ax.grid(True, alpha=0.25)
        plt.suptitle(f"Online Fine-tune — {self.experiment_name}")
        plt.tight_layout()
        out = os.path.join(self.out_dir, "training_curves.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(
        description="Online PPO fine-tune for direct thrust")
    p.add_argument("--config", type=str,
                   default="configs/low_level_config_2d.yaml")
    p.add_argument("--pretrain-ckpt", type=str, default=None,
                   help="Offline pretrained checkpoint")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--total-timesteps", type=int, default=2000000)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    # 学习率
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=1e-5)
    # KL 约束
    p.add_argument("--beta-kl", type=float, default=0.3)
    p.add_argument("--kl-anneal-frac", type=float, default=0.3)
    # 环境参数
    p.add_argument("--target-distance-max", type=float, default=1.0)
    p.add_argument("--target-distance-min", type=float, default=0.3)
    p.add_argument("--reach-threshold", type=float, default=0.10)
    p.add_argument("--max-episode-seconds", type=float, default=30.0)
    p.add_argument("--thrust-scale", type=float, default=40.0)
    # 奖励
    p.add_argument("--reward-reach", type=float, default=100.0)
    p.add_argument("--reward-speed-bonus", type=float, default=50.0)
    p.add_argument("--reward-improve-scale", type=float, default=50.0)
    p.add_argument("--reward-heading-scale", type=float, default=0.0)
    p.add_argument("--reward-heading-terminal", type=float, default=0.5)
    p.add_argument("--reward-vel-align-scale", type=float, default=0.0)
    p.add_argument("--reward-time-penalty", type=float, default=-0.2)
    p.add_argument("--init-velocity-range", type=float, default=1.0)
    p.add_argument("--init-yaw-rate-range", type=float, default=0.0)
    args = p.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        config["training"]["n_envs"] = args.n_envs

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = OnlineDirectThrustTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
