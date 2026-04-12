"""
2D V2 连续动作环境训练脚本（公平对比）
========================================

目标：与 train_low_level_2d_v2.py 使用相同任务设定，
仅替换为连续动作环境 LowLevelAUV2DEnvV2Continuous。

使用方法：
    python training/train_low_level_2d_v2_continuous.py --target-distance 2.0 --experiment-name 2d_v2_cont_dist_2m
"""

import os
import sys
import argparse
import yaml
import csv
import json
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

from core.environment.low_level_env_2d_v2_continuous import (
    LowLevelAUV2DEnvV2Continuous,
    VecLowLevelAUV2DEnvV2Continuous,
)


class GaussianActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 64)):
        super().__init__()

        actor_layers = []
        prev = obs_dim
        for h in hidden:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*actor_layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        critic_layers = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.critic = nn.Sequential(*critic_layers, nn.Linear(prev, 1))

    def forward(self, x):
        mu = self.mu_head(self.actor_body(x))
        v = self.critic(x).squeeze(-1)
        return mu, v

    def get_dist(self, x):
        mu, _ = self.forward(x)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_action(self, x, deterministic=False):
        dist = self.get_dist(x)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        a = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return a, logp, ent

    def evaluate_actions(self, x, a):
        dist = self.get_dist(x)
        z = self._atanh(a)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent

    def get_value(self, x):
        return self.critic(x).squeeze(-1)


class Continuous2DTrainerV2:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.experiment_name = args.experiment_name or f"2d_v2_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_environments()
        self._init_network()

        self.success_count = 0
        self.episode_count = 0
        self.history = []

    def _init_environments(self):
        training_config = self.config["training"]
        self.num_envs = int(training_config.get("n_envs", 4))

        self.train_env = VecLowLevelAUV2DEnvV2Continuous(
            num_envs=self.num_envs,
            target_distance=self.args.target_distance,
            enable_domain_randomization=False,
        )
        self.eval_env = LowLevelAUV2DEnvV2Continuous(
            target_distance=self.args.target_distance,
            enable_domain_randomization=False,
        )

        self.obs_dim = self.train_env.observation_space.shape[0]
        self.act_dim = self.train_env.action_space.shape[0]

        print("2D V2 Continuous Environment initialized:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.act_dim} (continuous)")
        print(f"  Target distance: {self.args.target_distance}m")
        print(f"  Num envs: {self.num_envs}")

    def _init_network(self):
        training_config = self.config["training"]
        hidden_sizes = self.config.get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64])

        self.model = GaussianActorCritic(self.obs_dim, self.act_dim, hidden=tuple(hidden_sizes)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(training_config.get("learning_rate", 3e-4)))

        self.gamma = float(training_config.get("gamma", 0.99))
        self.gae_lambda = float(training_config.get("gae_lambda", 0.95))
        self.clip_range = float(training_config.get("clip_range", 0.2))
        self.ent_coef = float(training_config.get("ent_coef", 0.05))
        self.vf_coef = float(training_config.get("vf_coef", 0.5))
        self.max_grad_norm = float(training_config.get("max_grad_norm", 0.5))
        self.n_steps = int(training_config.get("n_steps", 1024))
        self.batch_size = int(training_config.get("batch_size", 256))
        self.n_epochs = int(training_config.get("n_epochs", 4))

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network parameters: {total_params:,}")

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = self.train_env.reset()

        for _ in range(self.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp, _ = self.model.get_action(obs_tensor, deterministic=False)
                value = self.model.get_value(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.train_env.step(action_np)
            dones = np.logical_or(terminateds, truncateds)

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
            last_values = self.model.get_value(torch.tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()

        rewards = np.array(rew_buf)
        values = np.array(val_buf)
        dones = np.array(done_buf)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advantages,
            "returns": returns,
            "mean_reward": float(np.mean(rewards)),
        }

    def train_step(self, rollout_data: dict):
        obs = torch.tensor(rollout_data["obs"].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"].reshape(-1, self.act_dim), dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(rollout_data["logprobs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data["advantages"].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data["returns"].reshape(-1), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                new_logprobs, ent_each = self.model.evaluate_actions(batch_obs, batch_actions)
                entropy = ent_each.mean()
                values = self.model.get_value(batch_obs)

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.item())

        return total_loss / max(1, self.n_epochs)

    def evaluate(self, num_episodes: int = 10):
        rewards = []
        success = 0

        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                with torch.no_grad():
                    x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _ = self.model.get_action(x, deterministic=True)
                    action_np = action[0].cpu().numpy()

                obs, reward, term, trunc, info = self.eval_env.step(action_np)
                ep_reward += reward
                done = term or trunc
                if done and info.get("termination_reason") == "success":
                    success += 1

            rewards.append(ep_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(success / max(1, num_episodes)),
            "std_reward": float(np.std(rewards)),
        }

    def train(self):
        total_timesteps = int(self.config["training"].get("total_timesteps", 2_000_000))
        timesteps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // timesteps_per_update

        print(f"Start training for {num_updates} updates ({total_timesteps} timesteps)")

        best_eval_reward = -1e9
        eval_freq = int(self.config["training"].get("eval_freq", 5000))
        save_freq = int(self.config["training"].get("save_freq", 20000))
        last_eval_mean_reward = np.nan
        last_eval_success_rate = np.nan

        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            train_loss = self.train_step(rollout_data)

            current_steps = update * timesteps_per_update
            success_rate = self.success_count / max(1, self.episode_count)

            if update % 10 == 0:
                print(
                    f"Update {update}/{num_updates} | "
                    f"Steps {current_steps} | "
                    f"TrainRew {rollout_data['mean_reward']:.3f} | "
                    f"Loss {train_loss:.4f} | "
                    f"Succ {success_rate:.2%}",
                    flush=True,
                )

            if current_steps % eval_freq < timesteps_per_update:
                eval_metrics = self.evaluate(num_episodes=int(self.config["training"].get("eval_episodes", 10)))
                last_eval_mean_reward = float(eval_metrics["mean_reward"])
                last_eval_success_rate = float(eval_metrics["success_rate"])
                print(
                    f"  [Eval] steps={current_steps} mean_reward={eval_metrics['mean_reward']:.3f} "
                    f"success_rate={eval_metrics['success_rate']:.2%}",
                    flush=True,
                )

                if eval_metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["mean_reward"]
                    self.save_checkpoint("best")

            if current_steps % save_freq < timesteps_per_update:
                self.save_checkpoint("latest")

            self.history.append(
                {
                    "update": int(update),
                    "steps": int(current_steps),
                    "train_mean_reward": float(rollout_data["mean_reward"]),
                    "train_loss": float(train_loss),
                    "train_success_rate": float(success_rate),
                    "eval_mean_reward": float(last_eval_mean_reward),
                    "eval_success_rate": float(last_eval_success_rate),
                }
            )

            if update % 20 == 0:
                self.save_training_history()

        self.save_checkpoint("final")
        self.save_training_history()
        self.plot_training_curves()
        print("Training finished.")

    def save_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "config": self.config,
                "experiment_name": self.experiment_name,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def save_training_history(self):
        if len(self.history) == 0:
            return

        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        json_path = os.path.join(self.log_dir, "training_metrics.json")
        fields = [
            "update",
            "steps",
            "train_mean_reward",
            "train_loss",
            "train_success_rate",
            "eval_mean_reward",
            "eval_success_rate",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def plot_training_curves(self):
        if len(self.history) == 0:
            return

        steps = np.array([h["steps"] for h in self.history], dtype=np.float64)
        train_rew = np.array([h["train_mean_reward"] for h in self.history], dtype=np.float64)
        train_loss = np.array([h["train_loss"] for h in self.history], dtype=np.float64)
        train_sr = np.array([h["train_success_rate"] for h in self.history], dtype=np.float64)
        eval_rew = np.array([h["eval_mean_reward"] for h in self.history], dtype=np.float64)
        eval_sr = np.array([h["eval_success_rate"] for h in self.history], dtype=np.float64)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        ax = axes[0, 0]
        ax.plot(steps, train_rew, lw=1.8, label="train_mean_reward")
        valid = ~np.isnan(eval_rew)
        if np.any(valid):
            ax.plot(steps[valid], eval_rew[valid], lw=1.8, label="eval_mean_reward")
        ax.set_title("Reward Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(steps, train_loss, lw=1.8, color="tab:orange")
        ax.set_title("Loss Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 0]
        ax.plot(steps, train_sr, lw=1.8, label="train_success_rate")
        valid = ~np.isnan(eval_sr)
        if np.any(valid):
            ax.plot(steps[valid], eval_sr[valid], lw=1.8, label="eval_success_rate")
        ax.set_ylim(0, 1)
        ax.set_title("Success Rate Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(np.arange(1, len(train_rew) + 1), train_rew, lw=1.8, label="reward by update")
        ax.set_title("Reward vs Update")
        ax.set_xlabel("update")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 2D V2 continuous low-level policy (fair comparison)")
    parser.add_argument("--config", type=str, default="configs/low_level_config_2d.yaml")
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = int(args.total_timesteps)
    if args.n_envs is not None:
        config["training"]["n_envs"] = int(args.n_envs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = Continuous2DTrainerV2(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
