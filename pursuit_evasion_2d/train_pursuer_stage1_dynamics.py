"""
阶段一：训练 2D 追击者（动力学 + 加速度选项）
============================================

在 env_pursuer_low_level_dynamics 上训练追击者策略：
- 完整动力学（pe_2d_dynamics），不简化
- 25 动作（8 方向 × 3 档推力 + 无推力）
- 目标：移动目标（constant / evasive）

训练完成后保存到 checkpoints/pursuer_stage1_dynamics/，供阶段二加载。
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_pursuer_low_level_dynamics import (
    PursuerLowLevelDynamicsEnv,
    NUM_PURSUER_ACCEL_ACTIONS,
    make_pursuer_env,
)

OBS_DIM_STAGE1 = 18


class ActorCriticMLP(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_net(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value


class PPOTrainerStage1:
    def __init__(
        self,
        env_fn,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,
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

    def collect_rollout(self):
        obs_buf = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        rew_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        done_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        logp_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        val_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        obs_list = [e.reset()[0] for e in self.envs]
        obs = np.stack(obs_list)
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
                    episode_successes.append(info.get("success", False))
                    next_obs, _ = env.reset()
                next_obs_list.append(next_obs)
                rew_buf[step, i] = reward
                done_buf[step, i] = 1.0 if done else 0.0
            obs = np.stack(next_obs_list)

        with torch.no_grad():
            _, last_val = self.model.forward(torch.tensor(obs, dtype=torch.float32, device=self.device))
            last_val = last_val.cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_val if t == self.n_steps - 1 else val_buf[t + 1]
            next_non_terminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_val * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        ret_buf = adv_buf + val_buf

        return {
            "obs": obs_buf.reshape(-1, self.obs_dim),
            "act": act_buf.reshape(-1),
            "logp": logp_buf.reshape(-1),
            "adv": adv_buf.reshape(-1),
            "ret": ret_buf.reshape(-1),
        }, {"episode_successes": episode_successes, "mean_reward": np.mean(rew_buf)}

    def train_step(self, rollout_data: dict):
        obs = torch.tensor(rollout_data["obs"], dtype=torch.float32, device=self.device)
        act = torch.tensor(rollout_data["act"], dtype=torch.long, device=self.device)
        logp_old = torch.tensor(rollout_data["logp"], dtype=torch.float32, device=self.device)
        adv = torch.tensor(rollout_data["adv"], dtype=torch.float32, device=self.device)
        ret = torch.tensor(rollout_data["ret"], dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_samples = len(obs)
        losses = []
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
                losses.append(loss.item())
        return {"loss": np.mean(losses)}


def evaluate_policy(model, env_fn, n_episodes=20, device="cpu"):
    model.eval()
    env = env_fn()
    successes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model.forward(x)
                action = torch.argmax(logits, dim=-1).item()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(info.get("success", False))
    model.train()
    return np.mean(successes)


def train(
    experiment_name: str = None,
    total_timesteps: int = 2_000_000,
    n_envs: int = 16,
    n_steps: int = 256,
    batch_size: int = 128,
    target_motion: str = "constant",
    target_speed: float = 0.5,
    lr: float = 3e-4,
    seed: int = 42,
    log_interval: int = 10,
    eval_interval: int = 50,
    save_interval: int = 100,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = experiment_name or f"pursuer_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pursuer_stage1_dynamics", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {
        "experiment_name": exp_name,
        "obs_dim": OBS_DIM_STAGE1,
        "n_actions": NUM_PURSUER_ACCEL_ACTIONS,
        "total_timesteps": total_timesteps,
        "target_motion": target_motion,
        "target_speed": target_speed,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def make_env():
        return make_pursuer_env(
            target_motion=target_motion,
            target_speed=target_speed,
            seed=seed,
        )

    model = ActorCriticMLP(OBS_DIM_STAGE1, NUM_PURSUER_ACCEL_ACTIONS).to(device)
    trainer = PPOTrainerStage1(
        env_fn=make_env,
        model=model,
        device=device,
        lr=lr,
        n_steps=n_steps,
        n_envs=n_envs,
        batch_size=batch_size,
    )

    n_updates = total_timesteps // (n_steps * n_envs)
    steps_per_update = n_steps * n_envs
    timesteps = 0
    best_sr = 0.0
    start_time = time.time()

    print(f"Stage1: Training pursuer (dynamics + 9 actions, same as before)")
    print(f"  target_motion={target_motion}, target_speed={target_speed}")
    print(f"  Total timesteps: {total_timesteps:,}  (={n_updates} updates x {steps_per_update} steps/update)")
    print(f"  Checkpoints: {ckpt_dir}")
    print("-" * 60)

    for update in range(1, n_updates + 1):
        rollout_data, stats = trainer.collect_rollout()
        timesteps += n_steps * n_envs
        train_stats = trainer.train_step(rollout_data)

        if update % log_interval == 0:
            sr = np.mean(stats["episode_successes"]) if stats["episode_successes"] else 0.0
            print(f"Update {update:5d} | Steps {timesteps:8,d} | "
                  f"Success {sr:.1%} | Reward {stats['mean_reward']:.2f} | Loss {train_stats['loss']:.4f}")

        if update % eval_interval == 0:
            eval_sr = evaluate_policy(model, make_env, n_episodes=20, device=device)
            print(f"  Eval success rate: {eval_sr:.1%}")
            if eval_sr > best_sr:
                best_sr = eval_sr
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "update": update,
                    "success_rate": best_sr,
                    "obs_dim": OBS_DIM_STAGE1,
                    "n_actions": NUM_PURSUER_ACCEL_ACTIONS,
                }, os.path.join(ckpt_dir, "best.pth"))
                print(f"  -> New best saved: {best_sr:.1%}")

        if update % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "update": update,
                "timesteps": timesteps,
                "obs_dim": OBS_DIM_STAGE1,
                "n_actions": NUM_PURSUER_ACCEL_ACTIONS,
            }, os.path.join(ckpt_dir, f"checkpoint_{update}.pth"))

    torch.save({
        "model": model.state_dict(),
        "update": n_updates,
        "timesteps": timesteps,
        "obs_dim": OBS_DIM_STAGE1,
        "n_actions": NUM_PURSUER_ACCEL_ACTIONS,
    }, os.path.join(ckpt_dir, "final.pth"))

    print("\n" + "=" * 60)
    print("Stage1 completed. Best success rate: {:.1%}".format(best_sr))
    print("Checkpoints: {}".format(ckpt_dir))
    return model, ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Stage1: Train 2D pursuer (dynamics + accel options)")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="Total env steps (default 2M, same as previous 2D training)")
    parser.add_argument("--target-motion", type=str, default="constant", choices=["constant", "evasive"])
    parser.add_argument("--target-speed", type=float, default=0.5)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        target_motion=args.target_motion,
        target_speed=args.target_speed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
