"""
2D 追逃训练 - 复用完整 LowLevelAUVEnv（与「到达固定目标」同套动力学+PID）
==========================================================================

与 train_low_level_2d_v2 同构：相同 total_timesteps（默认 2M）、n_steps、n_envs、网络结构，
仅环境换为 LowLevelAUV2DPursuitEvasionEnv（目标=每步更新的逃脱者），训练时长应与到达固定目标相当（约 2h+）。

使用：
    python training/train_pe_low_level_2d.py --total-timesteps 2000000
    python training/train_pe_low_level_2d.py --experiment-name pe2d_ll_2m
"""
import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional, List, Any
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_pursuit_evasion import (
    LowLevelAUV2DPursuitEvasionEnv,
    VecLowLevelAUV2DPursuitEvasionEnv,
)


class Simple2DPETrainer:
    """2D 追逃训练器（与 Simple2DTrainerV2 同构，环境为追逃）"""

    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.experiment_name = args.experiment_name or f"pe2d_ll_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_environments()
        self._init_network()

        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        # 训练曲线记录（用于可视化）
        self.history: Dict[str, List[Any]] = {
            "timesteps": [],
            "mean_reward": [],
            "eval_reward": [],
            "eval_success_rate": [],
            "train_success_rollout": [],   # 本 rollout 成功率
            "train_success_cumulative": [],  # 累计成功率
            "loss": [],
            "entropy": [],
        }

    def _init_environments(self):
        training_config = self.config.get("training", {})
        self.num_envs = getattr(self.args, "n_envs", None) or training_config.get("n_envs", 4)
        if getattr(self.args, "curriculum_stage", None) is not None:
            stage = int(self.args.curriculum_stage)
        else:
            stage = 0 if getattr(self.args, "curriculum", False) else 2
        init_dist = getattr(self.args, "init_distance_range", (2.0, 6.0))
        if isinstance(init_dist, (list, tuple)) and len(init_dist) == 2:
            init_distance_range = tuple(float(x) for x in init_dist)
        else:
            init_distance_range = (2.0, 6.0)
        world_size = getattr(self.args, "world_size", 10.0)
        catch_radius = getattr(self.args, "catch_radius", 0.5)
        max_episode_seconds = getattr(self.args, "max_episode_seconds", 90.0)

        env_kwargs = dict(
            init_distance_range=init_distance_range,
            world_size=world_size,
            catch_radius=catch_radius,
            max_episode_seconds=max_episode_seconds,
            enable_domain_randomization=False,
            curriculum_stage=stage,
        )

        self.train_env = VecLowLevelAUV2DPursuitEvasionEnv(
            num_envs=self.num_envs,
            **env_kwargs,
        )

        self.eval_env = LowLevelAUV2DPursuitEvasionEnv(**env_kwargs)
        self.curriculum_stage = stage

        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n

        print("2D Pursuit-Evasion (same low_level pipeline as reach-target):")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.num_actions}")
        print(f"  Actions: {LowLevelAUV2DPursuitEvasionEnv.ACTION_NAMES}")
        print(f"  Init distance range: {init_distance_range}")
        print(f"  World size: {world_size}, catch_radius: {catch_radius}, max_episode_seconds: {max_episode_seconds}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Curriculum stage: {self.curriculum_stage}")

    def _init_network(self):
        network_config = self.config.get("network", {})
        hidden_sizes = network_config.get("feature_extractor", {}).get("hidden_sizes", [128, 64])

        actor_layers = []
        prev_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(torch.nn.Linear(prev_dim, hidden_size))
            actor_layers.append(torch.nn.ReLU())
            prev_dim = hidden_size
        actor_layers.append(torch.nn.Linear(prev_dim, self.num_actions))
        self.actor = torch.nn.Sequential(*actor_layers).to(self.device)

        critic_layers = []
        prev_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            critic_layers.append(torch.nn.Linear(prev_dim, hidden_size))
            critic_layers.append(torch.nn.ReLU())
            prev_dim = hidden_size
        critic_layers.append(torch.nn.Linear(prev_dim, 1))
        self.critic = torch.nn.Sequential(*critic_layers).to(self.device)

        training_config = self.config.get("training", {})
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=training_config.get("learning_rate", 3e-4),
        )
        self.gamma = training_config.get("gamma", 0.99)
        self.gae_lambda = training_config.get("gae_lambda", 0.95)
        self.clip_range = training_config.get("clip_range", 0.2)
        self.ent_coef = training_config.get("ent_coef", 0.05)
        self.vf_coef = training_config.get("vf_coef", 0.5)
        self.max_grad_norm = training_config.get("max_grad_norm", 0.5)
        self.n_steps = training_config.get("n_steps", 1024)
        self.batch_size = training_config.get("batch_size", 256)
        self.n_epochs = training_config.get("n_epochs", 4)

        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Network parameters: {total_params:,}")

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits = self.actor(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                actions = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
            return actions.cpu().numpy()

    def collect_rollouts(self):
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        logprob_buffer = []

        obs, _ = self.train_env.reset()
        rollout_successes = 0
        rollout_episodes = 0
        rollout_term_counts = {"success": 0, "timeout": 0, "out_of_bounds": 0, "other": 0}

        for step in range(self.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.actor(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = self.critic(obs_tensor).squeeze(-1)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.train_env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)
            obs_buffer.append(obs)
            action_buffer.append(actions_np)
            reward_buffer.append(rewards)
            done_buffer.append(dones)
            value_buffer.append(values.cpu().numpy())
            logprob_buffer.append(logprobs.cpu().numpy())
            for i, info in enumerate(infos):
                if dones[i]:
                    rollout_episodes += 1
                    self.episode_count += 1
                term_reason = info.get("termination_reason")
                if term_reason == "success":
                    rollout_successes += 1
                    self.success_count += 1
                    rollout_term_counts["success"] += 1
                elif term_reason == "timeout":
                    rollout_term_counts["timeout"] += 1
                elif term_reason == "out_of_bounds":
                    rollout_term_counts["out_of_bounds"] += 1
                elif term_reason is not None:
                    rollout_term_counts["other"] += 1
            obs = next_obs

        with torch.no_grad():
            last_values = self.critic(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1).cpu().numpy()
        advantages, returns = self._compute_gae(
            np.array(reward_buffer),
            np.array(value_buffer),
            np.array(done_buffer),
            last_values,
        )
        return {
            "obs": np.array(obs_buffer),
            "actions": np.array(action_buffer),
            "logprobs": np.array(logprob_buffer),
            "advantages": advantages,
            "returns": returns,
            "mean_reward": np.mean(reward_buffer),
            "rollout_successes": rollout_successes,
            "rollout_episodes": rollout_episodes,
            "rollout_term_counts": rollout_term_counts,
        }

    def _compute_gae(self, rewards, values, dones, last_values):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_values = last_values if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        returns = advantages + values
        return advantages, returns

    def train_step(self, rollout_data: Dict):
        obs = torch.tensor(rollout_data["obs"].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"].reshape(-1), dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(rollout_data["logprobs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data["advantages"].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data["returns"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0
        for epoch in range(self.n_epochs):
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                logits = self.actor(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values = self.critic(batch_obs).squeeze(-1)
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                total_loss += loss.item()
        return {
            "loss": total_loss / (self.n_epochs * (len(obs) // self.batch_size + 1)),
            "actor_loss": actor_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def evaluate(self, n_episodes: int = 10):
        rewards = []
        successes = 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.get_action(obs[np.newaxis, :], deterministic=True)[0]
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                if info.get("termination_reason") == "success":
                    successes += 1
            rewards.append(episode_reward)
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": successes / n_episodes,
        }

    def _save_history_and_plots(self):
        """将 history 写入 JSON 并绘制 loss / reward / success 曲线到 log_dir。"""
        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        ts = self.history["timesteps"]
        if not ts:
            return
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Reward
        ax = axes[0, 0]
        ax.plot(ts, self.history["mean_reward"], label="Train (rollout mean)", alpha=0.8)
        ax.plot(ts, self.history["eval_reward"], label="Eval mean", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Success rate
        ax = axes[0, 1]
        ax.plot(ts, np.array(self.history["eval_success_rate"]) * 100, label="Eval Success %", alpha=0.8)
        ax.plot(ts, np.array(self.history["train_success_rollout"]) * 100, label="Train (rollout) %", alpha=0.8)
        ax.plot(ts, np.array(self.history["train_success_cumulative"]) * 100, label="Train (cumulative) %", alpha=0.6)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Loss
        ax = axes[1, 0]
        ax.plot(ts, self.history["loss"], color="C2", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        # Entropy
        ax = axes[1, 1]
        ax.plot(ts, self.history["entropy"], color="C3", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Training curves saved to {plot_path}", flush=True)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self, total_timesteps: Optional[int] = None):
        if total_timesteps is None:
            total_timesteps = self.config.get("training", {}).get("total_timesteps", 2_000_000)
        training_config = self.config.get("training", {})
        eval_freq = training_config.get("eval_freq", 5000)
        save_freq = training_config.get("save_freq", 20000)
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update

        print(f"\n{'='*60}")
        print("2D Pursuit-Evasion (same low_level as reach-target, 2M steps)")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per update: {steps_per_update:,}")
        print(f"Number of updates: {num_updates:,}")
        print(f"{'='*60}\n")

        start_time = time.time()
        timesteps = 0
        best_success_rate = 0.0

        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(rollout_data)

            if timesteps % eval_freq < steps_per_update:
                eval_info = self.evaluate(n_episodes=10)
                elapsed = time.time() - start_time
                fps = timesteps / elapsed
                # 本 rollout 成功率（本次 update 内结束的回合中有多少是 success）
                ro_succ = rollout_data.get("rollout_successes", 0)
                ro_eps = rollout_data.get("rollout_episodes", 1)
                ro_terms = rollout_data.get("rollout_term_counts", {})
                train_success_rollout = ro_succ / max(1, ro_eps)
                train_success_cumulative = self.success_count / max(1, self.episode_count)
                print(f"\nUpdate {update}/{num_updates} | Timesteps: {timesteps:,}", flush=True)
                print(f"  FPS: {fps:.1f}", flush=True)
                print(f"  Mean Reward: {rollout_data['mean_reward']:.3f}", flush=True)
                print(f"  Eval Reward: {eval_info['mean_reward']:.3f}", flush=True)
                print(f"  Eval Success Rate: {eval_info['success_rate']:.1%}", flush=True)
                print(f"  Train Success (rollout): {train_success_rollout:.1%} ({ro_succ}/{ro_eps})", flush=True)
                print(f"  Train Success (cumulative): {train_success_cumulative:.1%}", flush=True)
                print(
                    "  Rollout terminations: "
                    f"success={ro_terms.get('success', 0)}, "
                    f"timeout={ro_terms.get('timeout', 0)}, "
                    f"out_of_bounds={ro_terms.get('out_of_bounds', 0)}, "
                    f"other={ro_terms.get('other', 0)}",
                    flush=True,
                )
                print(f"  Loss: {train_info['loss']:.4f}", flush=True)
                print(f"  Entropy: {train_info['entropy']:.4f}", flush=True)
                if getattr(self.args, "curriculum", False):
                    if eval_info["success_rate"] >= self.args.curriculum_threshold:
                        if self.curriculum_stage < self.args.curriculum_max_stage:
                            self.curriculum_stage += 1
                            self.train_env.set_curriculum_stage(self.curriculum_stage)
                            self.eval_env.set_curriculum_stage(self.curriculum_stage)
                            print(
                                f"  -> Curriculum stage increased to {self.curriculum_stage}",
                                flush=True,
                            )
                if eval_info["success_rate"] > best_success_rate:
                    best_success_rate = eval_info["success_rate"]
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best model! Success rate: {best_success_rate:.1%}", flush=True)
                # 记录历史并绘图
                self.history["timesteps"].append(timesteps)
                self.history["mean_reward"].append(float(rollout_data["mean_reward"]))
                self.history["eval_reward"].append(float(eval_info["mean_reward"]))
                self.history["eval_success_rate"].append(float(eval_info["success_rate"]))
                self.history["train_success_rollout"].append(float(train_success_rollout))
                self.history["train_success_cumulative"].append(float(train_success_cumulative))
                self.history["loss"].append(float(train_info["loss"]))
                self.history["entropy"].append(float(train_info["entropy"]))
                self._save_history_and_plots()
            if timesteps % save_freq < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))

        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining completed!")
        print(f"Best success rate: {best_success_rate:.1%}")
        print(f"Final model saved to {self.checkpoint_dir}")


def _parse_init_distance(s: str):
    if not s:
        return (2.0, 6.0)
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        return (2.0, 6.0)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (2.0, 6.0)


def main():
    parser = argparse.ArgumentParser(description="2D Pursuit-Evasion (same low_level pipeline as reach-target)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None, help="Default 2M, same as train_low_level_2d_v2")
    parser.add_argument("--seed", type=int, default=42)
    # 场景/难度参数，便于多组对比
    parser.add_argument("--init-distance-range", type=str, default="2,6", help="e.g. 2,6 or 1,3 for closer start")
    parser.add_argument("--world-size", type=float, default=10.0, help="Half extent of 2D world (default 10)")
    parser.add_argument("--catch-radius", type=float, default=0.5, help="Success when distance < this")
    parser.add_argument("--max-episode-seconds", type=float, default=90.0, help="Max episode length in seconds")
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel envs (default from config, usually 4)")
    parser.add_argument("--curriculum", action="store_true", help="Enable success-based curriculum")
    parser.add_argument("--curriculum-threshold", type=float, default=0.6, help="Success rate to advance stage")
    parser.add_argument("--curriculum-max-stage", type=int, default=2, help="Max curriculum stage")
    parser.add_argument("--curriculum-stage", type=int, default=None, help="Force a curriculum stage (overrides --curriculum)")
    args = parser.parse_args()
    args.init_distance_range = _parse_init_distance(args.init_distance_range)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "network" not in config:
        config["network"] = {}
    config["network"]["feature_extractor"] = {"hidden_sizes": [256, 128, 64]}
    if args.total_timesteps is not None:
        config.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    if "training" not in config:
        config["training"] = {}
    if "total_timesteps" not in config["training"]:
        config["training"]["total_timesteps"] = 2_000_000

    trainer = Simple2DPETrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
