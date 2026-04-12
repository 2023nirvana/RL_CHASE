"""
2D 追逃训练 - AUV 配置 + 双环 PID 环境（PursuitEvasion2DAUVPIDEnv）
====================================================================

在 PursuitEvasion2DAUVPIDEnv 上训练追击者：双环 PID 来自 low_level_config_2d，
动力学来自 robot_config，硬约束 v_max/a_max，默认 5x5 世界。
观测 22 维，动作 9，PPO 与 train_pe_stage2_dynamics 同构。
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
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_pe_2d_auv_pid import PursuitEvasion2DAUVPIDEnv

OBS_DIM = 22
N_ACTIONS = 9


class ActorCriticMLP(nn.Module):
    """MLP Actor-Critic，22 维观测，9 动作。"""
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_sizes=(256, 128, 64)):
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


class CurriculumManager:
    def __init__(self, initial_difficulty=0.2, target_success_rate=0.55, difficulty_increment=0.05, window_size=200):
        self.difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.difficulty_increment = difficulty_increment
        self.window_size = window_size
        self.success_history = deque(maxlen=window_size)

    def record_episode(self, success: bool):
        self.success_history.append(1.0 if success else 0.0)

    def get_success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def get_evader_mode(self) -> str:
        return "simple" if self.difficulty < 0.3 else "smart"

    def update_difficulty(self):
        if len(self.success_history) < self.window_size // 2:
            return
        sr = self.get_success_rate()
        if sr >= self.target_success_rate:
            self.difficulty = min(1.0, self.difficulty + self.difficulty_increment)


class PPOTrainer:
    def __init__(self, env_fn, model, device, lr=3e-4, n_steps=256, n_envs=16, batch_size=128,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.env_fn = env_fn
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            obs, _ = env.reset(seed=None, options={"difficulty": curriculum.difficulty, "evader_mode": curriculum.get_evader_mode()})
            obs_list.append(obs)
        obs = np.stack(obs_list)
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
                    curriculum.record_episode(info.get("success", False))
                    next_obs, _ = env.reset(seed=None, options={"difficulty": curriculum.difficulty, "evader_mode": curriculum.get_evader_mode()})
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
        }, {"mean_reward": np.mean(rew_buf)}

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


def evaluate_policy(model, env_fn, n_episodes=20, device="cpu", difficulty=0.6, evader_mode="smart"):
    model.eval()
    env = env_fn()
    successes = []
    for _ in range(n_episodes):
        obs, _ = env.reset(options={"difficulty": difficulty, "evader_mode": evader_mode})
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
    experiment_name=None,
    total_timesteps=1_500_000,
    world_size=5.0,
    n_envs=16,
    n_steps=256,
    batch_size=128,
    lr=3e-4,
    use_curriculum=True,
    initial_difficulty=0.2,
    target_success_rate=0.5,
    difficulty_increment=0.04,
    seed=42,
    log_interval=10,
    eval_interval=50,
    save_interval=100,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = experiment_name or f"pe2d_auv_pid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d_auv_pid", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "experiment_name": exp_name,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "total_timesteps": total_timesteps,
        "world_size": world_size,
        "use_curriculum": use_curriculum,
        "initial_difficulty": initial_difficulty,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def make_env():
        return PursuitEvasion2DAUVPIDEnv(
            world_size=world_size,
            difficulty=initial_difficulty,
            evader_mode="simple" if initial_difficulty < 0.3 else "smart",
            seed=seed,
        )

    model = ActorCriticMLP(OBS_DIM, N_ACTIONS).to(device)
    trainer = PPOTrainer(env_fn=make_env, model=model, device=device, lr=lr, n_steps=n_steps, n_envs=n_envs, batch_size=batch_size)
    curriculum = CurriculumManager(
        initial_difficulty=initial_difficulty,
        target_success_rate=target_success_rate,
        difficulty_increment=difficulty_increment,
        window_size=200,
    )

    n_updates = total_timesteps // (n_steps * n_envs)
    steps_per_update = n_steps * n_envs
    timesteps = 0
    best_sr = 0.0
    training_log = []

    print("Training PE2D AUV+PID (dual-loop from low_level_config_2d, world_size={})".format(world_size))
    print("  22-dim obs, 9 actions, curriculum={}".format(use_curriculum))
    print("  Total timesteps: {:,}  Checkpoints: {}".format(total_timesteps, ckpt_dir))
    print("-" * 60)

    for update in range(1, n_updates + 1):
        rollout_data, stats = trainer.collect_rollout(curriculum)
        timesteps += n_steps * n_envs
        train_stats = trainer.train_step(rollout_data)
        if use_curriculum and update % 5 == 0:
            curriculum.update_difficulty()

        log_entry = {
            "update": int(update),
            "timesteps": int(timesteps),
            "difficulty": float(curriculum.difficulty),
            "success_rate": float(curriculum.get_success_rate()),
            "mean_reward": float(stats["mean_reward"]),
            "loss": float(train_stats["loss"]),
        }
        training_log.append(log_entry)

        if update % log_interval == 0:
            print("Update {:5d} | Steps {:8,d} | Difficulty {:.2f} | Success {:.1%} | Reward {:.2f} | Loss {:.4f}".format(
                update, timesteps, curriculum.difficulty, curriculum.get_success_rate(), stats["mean_reward"], train_stats["loss"]))

        if update % eval_interval == 0:
            for d, m in [(0.3, "simple"), (0.6, "smart"), (1.0, "smart")]:
                eval_env_fn = lambda diff=d, mode=m: PursuitEvasion2DAUVPIDEnv(world_size=world_size, difficulty=diff, evader_mode=mode, seed=seed + 1000)
                sr = evaluate_policy(model, eval_env_fn, n_episodes=20, device=device, difficulty=d, evader_mode=m)
                print("  Eval (d={}, {}): success rate {:.1%}".format(d, m, sr))
            if curriculum.get_success_rate() > best_sr:
                best_sr = curriculum.get_success_rate()
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "update": update,
                    "success_rate": best_sr,
                    "difficulty": curriculum.difficulty,
                    "obs_dim": OBS_DIM,
                    "n_actions": N_ACTIONS,
                }, os.path.join(ckpt_dir, "best.pth"))
                print("  -> New best saved: {:.1%}".format(best_sr))

        if update % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "update": update,
                "timesteps": timesteps,
                "difficulty": curriculum.difficulty,
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
            }, os.path.join(ckpt_dir, "checkpoint_{}.pth".format(update)))
            with open(os.path.join(log_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "update": n_updates,
        "timesteps": timesteps,
        "final_difficulty": curriculum.difficulty,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
    }, os.path.join(ckpt_dir, "final.pth"))
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 60)
    print("Training completed. Best success rate: {:.1%}".format(best_sr))
    print("Checkpoints: {}".format(ckpt_dir))
    return model, ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Train PE2D AUV+PID (dual-loop, 5x5 default)")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--world-size", type=float, default=5.0, help="World size (default 5 for small test)")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-curriculum", action="store_false", dest="use_curriculum")
    parser.add_argument("--initial-difficulty", type=float, default=0.2)
    parser.add_argument("--target-success-rate", type=float, default=0.5)
    parser.add_argument("--difficulty-increment", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        world_size=args.world_size,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_curriculum=args.use_curriculum,
        initial_difficulty=args.initial_difficulty,
        target_success_rate=args.target_success_rate,
        difficulty_increment=args.difficulty_increment,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
