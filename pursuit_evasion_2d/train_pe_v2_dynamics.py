"""
非对称追逃训练脚本 V2 - 动力学版（基于 low_level_env 动力学）
================================================================

与 train_pe_v2.py 的区别：
- 使用 env_asymmetric_pe_v2_dynamics 环境（M*a = τ - D*v - D*|v|*v，含加速度）
- 观测 22 维（比简化版多 2 维加速度）
- 动作为推力方向，由动力学模块计算实际加速度
- 不修改原 train_pe_v2 / env_asymmetric_pe_v2
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from collections import deque
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_asymmetric_pe_v2_dynamics import (
    PursuitEvasion2DAsymmetricEnvV2Dynamics,
    make_easy_env,
    make_medium_env,
    make_hard_env,
    make_expert_env,
    NUM_ACCEL_ACTIONS,
)
from pursuit_evasion_2d.env_asymmetric_pe_v2_pid import PursuitEvasion2DAsymmetricEnvV2PID

# 观测维度（动力学版与 PID 版均为 22 维）
OBS_DIM_DYNAMICS = 22


class ActorCriticMLP(nn.Module):
    """MLP Actor-Critic 网络（与 train_pe_v2 相同结构，输入维度 22）"""
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 128, 64)):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            prev_dim = h

        self.feature_net = nn.Sequential(*layers)

        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, n_actions),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_net(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action(self, x, deterministic=False):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, dist.entropy()

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy


class CurriculumManager:
    """课程学习管理器"""
    def __init__(
        self,
        initial_difficulty: float = 0.2,
        target_success_rate: float = 0.6,
        difficulty_increment: float = 0.05,
        window_size: int = 100,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
    ):
        self.difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.difficulty_increment = difficulty_increment
        self.window_size = window_size
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.success_history = deque(maxlen=window_size)

    def record_episode(self, success: bool):
        self.success_history.append(1.0 if success else 0.0)

    def get_success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def update_difficulty(self) -> bool:
        if len(self.success_history) < self.window_size // 2:
            return False

        success_rate = self.get_success_rate()
        old_difficulty = self.difficulty

        if success_rate > self.target_success_rate + 0.1:
            self.difficulty = min(
                self.max_difficulty,
                self.difficulty + self.difficulty_increment
            )
        elif success_rate < self.target_success_rate - 0.15:
            self.difficulty = max(
                self.min_difficulty,
                self.difficulty - self.difficulty_increment * 0.5
            )

        return abs(self.difficulty - old_difficulty) > 1e-6

    def get_evader_mode(self) -> str:
        if self.difficulty < 0.3:
            return "simple"
        elif self.difficulty < 0.7:
            return "smart"
        else:
            return "smart"


class PPOTrainer:
    """PPO 训练器（与 train_pe_v2 相同逻辑，环境为动力学版）"""
    def __init__(
        self,
        env_fn,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        n_steps: int = 256,
        n_envs: int = 8,
    ):
        self.env_fn = env_fn
        self.model = model
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        self.envs = [env_fn() for _ in range(n_envs)]
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.n_actions = self.envs[0].action_space.n

    def collect_rollout(self, curriculum: CurriculumManager):
        obs_buf = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        rew_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        done_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        logp_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        val_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        obs_list = []
        for i, env in enumerate(self.envs):
            difficulty = curriculum.difficulty
            evader_mode = curriculum.get_evader_mode()
            obs, _ = env.reset(options={"difficulty": difficulty, "evader_mode": evader_mode})
            obs_list.append(obs)
        obs = np.stack(obs_list)

        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        for step in range(self.n_steps):
            obs_buf[step] = obs

            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=self.device)
                logits, val = self.model.forward(x)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample()
                logp = dist.log_prob(act)

            act_np = act.cpu().numpy()
            logp_buf[step] = logp.cpu().numpy()
            val_buf[step] = val.cpu().numpy()
            act_buf[step] = act_np

            next_obs_list = []
            for i, env in enumerate(self.envs):
                next_obs, reward, terminated, truncated, info = env.step(int(act_np[i]))
                done = terminated or truncated

                if done:
                    success = info.get("success", False)
                    curriculum.record_episode(success)
                    episode_successes.append(success)
                    episode_rewards.append(info.get("episode_reward", 0))
                    episode_lengths.append(info.get("episode_length", 0))

                    difficulty = curriculum.difficulty
                    evader_mode = curriculum.get_evader_mode()
                    next_obs, _ = env.reset(options={"difficulty": difficulty, "evader_mode": evader_mode})

                next_obs_list.append(next_obs)
                rew_buf[step, i] = reward
                done_buf[step, i] = 1.0 if done else 0.0

            obs = np.stack(next_obs_list)

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)
            _, last_val = self.model.forward(x)
            last_val = last_val.cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_val if t == self.n_steps - 1 else val_buf[t + 1]
            next_non_terminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_val * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        ret_buf = adv_buf + val_buf

        rollout_data = {
            "obs": obs_buf.reshape(-1, self.obs_dim),
            "act": act_buf.reshape(-1),
            "logp": logp_buf.reshape(-1),
            "adv": adv_buf.reshape(-1),
            "ret": ret_buf.reshape(-1),
        }

        stats = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_successes": episode_successes,
            "mean_reward": np.mean(rew_buf),
        }

        return rollout_data, stats

    def train_step(self, rollout_data: dict) -> dict:
        obs = torch.tensor(rollout_data["obs"], dtype=torch.float32, device=self.device)
        act = torch.tensor(rollout_data["act"], dtype=torch.long, device=self.device)
        logp_old = torch.tensor(rollout_data["logp"], dtype=torch.float32, device=self.device)
        adv = torch.tensor(rollout_data["adv"], dtype=torch.float32, device=self.device)
        ret = torch.tensor(rollout_data["ret"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_samples = len(obs)
        all_pg_loss, all_vf_loss, all_ent_loss, all_loss = [], [], [], []

        for _ in range(self.n_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]

                logits, val = self.model.forward(obs[idx])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp_new = dist.log_prob(act[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp_new - logp_old[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(val, ret[idx])
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_pg_loss.append(pg_loss.item())
                all_vf_loss.append(vf_loss.item())
                all_ent_loss.append(entropy.item())
                all_loss.append(loss.item())

        return {
            "pg_loss": np.mean(all_pg_loss),
            "vf_loss": np.mean(all_vf_loss),
            "entropy": np.mean(all_ent_loss),
            "total_loss": np.mean(all_loss),
        }


def evaluate_policy(model, env_fn, n_episodes=20, device="cpu", deterministic=True):
    model.eval()
    env = env_fn()
    successes, rewards, lengths = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model.forward(x)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    probs = F.softmax(logits, dim=-1)
                    action = torch.distributions.Categorical(probs).sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        successes.append(info.get("success", False))
        rewards.append(total_reward)
        lengths.append(steps)

    model.train()
    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
    }


def train(
    experiment_name: str = None,
    total_timesteps: int = 2_000_000,
    n_envs: int = 16,
    n_steps: int = 256,
    batch_size: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.02,
    use_curriculum: bool = True,
    initial_difficulty: float = 0.2,
    target_success_rate: float = 0.55,
    seed: int = 42,
    log_interval: int = 10,
    eval_interval: int = 50,
    save_interval: int = 100,
    env_type: str = "dynamics",
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_pid = env_type.lower() == "pid"
    EnvClass = PursuitEvasion2DAsymmetricEnvV2PID if use_pid else PursuitEvasion2DAsymmetricEnvV2Dynamics
    ckpt_subdir = "pe2d_v2_pid" if use_pid else "pe2d_v2_dynamics"
    env_label = "PE2D V2 PID (RL target + 2D PID)" if use_pid else "PE2D V2 Dynamics (low_level-style)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Environment: {env_label}, obs_dim=22")

    exp_name = experiment_name or f"{ckpt_subdir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", ckpt_subdir, exp_name)
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "experiment_name": exp_name,
        "env": "PursuitEvasion2DAsymmetricEnvV2PID" if use_pid else "PursuitEvasion2DAsymmetricEnvV2Dynamics",
        "obs_dim": OBS_DIM_DYNAMICS,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "use_curriculum": use_curriculum,
        "initial_difficulty": initial_difficulty,
        "target_success_rate": target_success_rate,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def make_env():
        return EnvClass(
            difficulty=initial_difficulty,
            evader_mode="simple" if initial_difficulty < 0.3 else "smart",
            seed=seed,
        )

    model = ActorCriticMLP(OBS_DIM_DYNAMICS, NUM_ACCEL_ACTIONS, hidden_sizes=(256, 128, 64)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = PPOTrainer(
        env_fn=make_env,
        model=model,
        device=device,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_steps=n_steps,
        n_envs=n_envs,
        batch_size=batch_size,
    )

    curriculum = CurriculumManager(
        initial_difficulty=initial_difficulty,
        target_success_rate=target_success_rate,
        difficulty_increment=0.05,
        window_size=200,
    )

    n_updates = total_timesteps // (n_steps * n_envs)
    timesteps = 0
    best_success_rate = 0.0
    training_log = []

    print(f"\nStarting training for {total_timesteps:,} timesteps ({n_updates} updates)")
    print(f"Checkpoint dir: {ckpt_dir}")
    print("-" * 60)

    start_time = time.time()

    for update in range(1, n_updates + 1):
        rollout_data, rollout_stats = trainer.collect_rollout(curriculum)
        timesteps += n_steps * n_envs
        train_stats = trainer.train_step(rollout_data)

        if use_curriculum and update % 5 == 0:
            curriculum.update_difficulty()

        log_entry = {
            "update": int(update),
            "timesteps": int(timesteps),
            "difficulty": float(curriculum.difficulty),
            "success_rate": float(curriculum.get_success_rate()),
            "mean_reward": float(rollout_stats["mean_reward"]),
            "pg_loss": float(train_stats["pg_loss"]),
            "vf_loss": float(train_stats["vf_loss"]),
            "entropy": float(train_stats["entropy"]),
        }
        training_log.append(log_entry)

        if update % log_interval == 0:
            elapsed = time.time() - start_time
            fps = timesteps / elapsed
            ep_rewards = rollout_stats["episode_rewards"]
            print(f"Update {update:5d} | Steps {timesteps:8,d} | "
                  f"Difficulty {curriculum.difficulty:.2f} | "
                  f"Success {curriculum.get_success_rate():.1%} | "
                  f"Reward {np.mean(ep_rewards) if ep_rewards else 0:.1f} | "
                  f"Loss {train_stats['total_loss']:.4f} | "
                  f"FPS {fps:.0f}")

        if update % eval_interval == 0:
            for difficulty, mode in [(0.3, "simple"), (0.6, "smart"), (1.0, "smart")]:
                eval_env_fn = lambda d=difficulty, m=mode, E=EnvClass: E(
                    difficulty=d, evader_mode=m, seed=seed + 1000
                )
                eval_stats = evaluate_policy(model, eval_env_fn, n_episodes=20, device=device)
                print(f"  Eval (d={difficulty:.1f}, {mode}): "
                      f"Success {eval_stats['success_rate']:.1%}, "
                      f"Reward {eval_stats['mean_reward']:.1f}")

            if curriculum.get_success_rate() > best_success_rate:
                best_success_rate = curriculum.get_success_rate()
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "update": update,
                    "success_rate": best_success_rate,
                    "difficulty": curriculum.difficulty,
                    "obs_dim": OBS_DIM_DYNAMICS,
                }, os.path.join(ckpt_dir, "best.pth"))

        if update % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "update": update,
                "timesteps": timesteps,
                "difficulty": curriculum.difficulty,
                "obs_dim": OBS_DIM_DYNAMICS,
            }, os.path.join(ckpt_dir, f"checkpoint_{update}.pth"))

            with open(os.path.join(log_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "update": n_updates,
        "timesteps": timesteps,
        "final_difficulty": curriculum.difficulty,
        "final_success_rate": curriculum.get_success_rate(),
        "obs_dim": OBS_DIM_DYNAMICS,
    }, os.path.join(ckpt_dir, "final.pth"))

    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Final difficulty: {curriculum.difficulty:.2f}")
    print(f"Final success rate: {curriculum.get_success_rate():.1%}")
    print(f"Best success rate: {best_success_rate:.1%}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    return model, ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Train PE2D V2 (dynamics or PID env)")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--env", type=str, default="dynamics", choices=["dynamics", "pid"],
                        help="dynamics=推力方向+动力学; pid=RL目标点+2D双环PID算tau")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--use-curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", action="store_false", dest="use_curriculum")
    parser.add_argument("--initial-difficulty", type=float, default=0.2)
    parser.add_argument("--target-success-rate", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        experiment_name=args.experiment_name,
        env_type=args.env,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ent_coef=args.ent_coef,
        use_curriculum=args.use_curriculum,
        initial_difficulty=args.initial_difficulty,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
