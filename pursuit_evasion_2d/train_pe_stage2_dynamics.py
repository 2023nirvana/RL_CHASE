"""
阶段二：训练追逃策略（动力学 + 速度劣势/加速度优势）
==================================================

在 env_asymmetric_pe_v2_dynamics_accel 上训练追击者追逃策略：
- 物理配置：追击者速度劣势、加速度优势（与之前一致）
- 追击者 25 动作（与阶段一一致）
- 可选：从阶段一 checkpoint 加载并初始化网络（前 18 维观测对应权重 + 其余层）

不修改原 train_pe_v2_dynamics。
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

from pursuit_evasion_2d.env_asymmetric_pe_v2_dynamics_accel import (
    PursuitEvasion2DAsymmetricEnvV2DynamicsAccel,
    NUM_PURSUER_ACCEL_ACTIONS,
)

OBS_DIM_STAGE2 = 22
N_ACTIONS_STAGE2 = NUM_PURSUER_ACCEL_ACTIONS  # 9，与之前动作空间一致


class ActorCriticMLP(nn.Module):
    """Stage2 网络：输入 22 维，输出 9 动作。与 stage1 同结构 (256,128,64) 便于迁移。"""
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


def init_from_stage1(stage2_model: nn.Module, stage1_ckpt_path: str, device: torch.device):
    """
    用阶段一 checkpoint 部分初始化阶段二模型。
    阶段一：obs_dim=18, n_actions=9；阶段二：obs_dim=22, n_actions=9。
    第一层只复制前 18 维输入对应的权重，其余层按 state_dict 键名复制。
    """
    ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model") or ckpt
    stage2_state = stage2_model.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            if k not in stage2_state:
                continue
            t = stage2_state[k]
            if v.shape == t.shape:
                t.copy_(v)
            elif k == "feature_net.0.weight":
                # stage1 weight (256, 18) -> stage2 (256, 22)，只复制前 18 列
                t[:, :18].copy_(v)
            # feature_net.0.bias 形状相同 (256,)，上面已复制
    print(f"Initialized stage2 model from stage1: {stage1_ckpt_path}")


class CurriculumManager:
    def __init__(
        self,
        initial_difficulty: float = 0.2,
        target_success_rate: float = 0.55,
        difficulty_increment: float = 0.05,
        window_size: int = 200,
    ):
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

    def update_difficulty(self):
        if len(self.success_history) < self.window_size // 2:
            return
        sr = self.get_success_rate()
        if sr > self.target_success_rate + 0.1:
            self.difficulty = min(1.0, self.difficulty + self.difficulty_increment)
        elif sr < self.target_success_rate - 0.15:
            self.difficulty = max(0.1, self.difficulty - self.difficulty_increment * 0.5)

    def get_evader_mode(self) -> str:
        return "simple" if self.difficulty < 0.3 else "smart"


class PPOTrainerStage2:
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
        batch_size: int = 128,
        n_steps: int = 256,
        n_envs: int = 16,
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
            obs, _ = env.reset(options={"difficulty": curriculum.difficulty, "evader_mode": curriculum.get_evader_mode()})
            obs_list.append(obs)
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
                    curriculum.record_episode(info.get("success", False))
                    episode_successes.append(info.get("success", False))
                    next_obs, _ = env.reset(options={"difficulty": curriculum.difficulty, "evader_mode": curriculum.get_evader_mode()})
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
    experiment_name: str = None,
    total_timesteps: int = 2_000_000,
    n_envs: int = 16,
    n_steps: int = 256,
    batch_size: int = 128,
    lr: float = 3e-4,
    use_curriculum: bool = True,
    initial_difficulty: float = 0.15,
    target_success_rate: float = 0.5,
    difficulty_increment: float = 0.03,
    seed: int = 42,
    reward_catch: float = 150.0,
    reward_distance_improve: float = 8.0,
    reward_closing_speed: float = 2.0,
    penalty_time: float = 0.03,
    log_interval: int = 10,
    eval_interval: int = 50,
    save_interval: int = 100,
    stage1_checkpoint: str = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = experiment_name or f"pe_stage2_accel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d_stage2_dynamics_accel", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "experiment_name": exp_name,
        "obs_dim": OBS_DIM_STAGE2,
        "n_actions": N_ACTIONS_STAGE2,
        "total_timesteps": total_timesteps,
        "use_curriculum": use_curriculum,
        "initial_difficulty": initial_difficulty,
        "target_success_rate": target_success_rate,
        "stage1_checkpoint": stage1_checkpoint,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def make_env():
        return PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(
            difficulty=initial_difficulty,
            evader_mode="simple" if initial_difficulty < 0.3 else "smart",
            seed=seed,
            reward_catch=reward_catch,
            reward_distance_improve=reward_distance_improve,
            reward_closing_speed=reward_closing_speed,
            penalty_time=penalty_time,
        )

    model = ActorCriticMLP(OBS_DIM_STAGE2, N_ACTIONS_STAGE2).to(device)
    if stage1_checkpoint and os.path.isfile(stage1_checkpoint):
        init_from_stage1(model, stage1_checkpoint, device)
    else:
        if stage1_checkpoint:
            print(f"Stage1 checkpoint not found: {stage1_checkpoint}, training from scratch.")

    trainer = PPOTrainerStage2(
        env_fn=make_env,
        model=model,
        device=device,
        lr=lr,
        n_steps=n_steps,
        n_envs=n_envs,
        batch_size=batch_size,
    )
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
    start_time = time.time()
    training_log = []

    print("Stage2: Training pursuit-evasion strategy (speed disadvantage, accel advantage)")
    print(f"  9 actions (same as before), 22-dim obs, curriculum={use_curriculum}")
    print(f"  Total timesteps: {total_timesteps:,}  (={n_updates} updates x {steps_per_update} steps/update)")
    print(f"  Checkpoints: {ckpt_dir}")
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
            print(f"Update {update:5d} | Steps {timesteps:8,d} | Difficulty {curriculum.difficulty:.2f} | "
                  f"Success {curriculum.get_success_rate():.1%} | Reward {stats['mean_reward']:.2f} | Loss {train_stats['loss']:.4f}")

        if update % eval_interval == 0:
            for d, m in [(0.3, "simple"), (0.6, "smart"), (1.0, "smart")]:
                eval_env_fn = lambda diff=d, mode=m: PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(
                    difficulty=diff, evader_mode=mode, seed=seed + 1000
                )
                sr = evaluate_policy(model, eval_env_fn, n_episodes=20, device=device, difficulty=d, evader_mode=m)
                print(f"  Eval (d={d}, {m}): success rate {sr:.1%}")
            if curriculum.get_success_rate() > best_sr:
                best_sr = curriculum.get_success_rate()
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "update": update,
                    "success_rate": best_sr,
                    "difficulty": curriculum.difficulty,
                    "obs_dim": OBS_DIM_STAGE2,
                    "n_actions": N_ACTIONS_STAGE2,
                }, os.path.join(ckpt_dir, "best.pth"))
                print(f"  -> New best saved: {best_sr:.1%}")

        if update % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "update": update,
                "timesteps": timesteps,
                "difficulty": curriculum.difficulty,
                "obs_dim": OBS_DIM_STAGE2,
                "n_actions": N_ACTIONS_STAGE2,
            }, os.path.join(ckpt_dir, f"checkpoint_{update}.pth"))
            with open(os.path.join(log_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "update": n_updates,
        "timesteps": timesteps,
        "final_difficulty": curriculum.difficulty,
        "obs_dim": OBS_DIM_STAGE2,
        "n_actions": N_ACTIONS_STAGE2,
    }, os.path.join(ckpt_dir, "final.pth"))
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 60)
    print("Stage2 completed. Best success rate: {:.1%}".format(best_sr))
    print("Checkpoints: {}".format(ckpt_dir))
    return model, ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Stage2: Train PE strategy (dynamics + accel options)")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="Total env steps (default 2M, same as train_pe_v2_dynamics)")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Path to stage1 best.pth or final.pth to initialize")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-curriculum", action="store_false", dest="use_curriculum")
    parser.add_argument("--initial-difficulty", type=float, default=0.15)
    parser.add_argument("--target-success-rate", type=float, default=0.5)
    parser.add_argument("--difficulty-increment", type=float, default=0.03)
    parser.add_argument("--reward-catch", type=float, default=150.0)
    parser.add_argument("--reward-distance-improve", type=float, default=8.0)
    parser.add_argument("--reward-closing-speed", type=float, default=2.0)
    parser.add_argument("--penalty-time", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        stage1_checkpoint=args.stage1_checkpoint,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_curriculum=args.use_curriculum,
        initial_difficulty=args.initial_difficulty,
        target_success_rate=args.target_success_rate,
        difficulty_increment=args.difficulty_increment,
        reward_catch=args.reward_catch,
        reward_distance_improve=args.reward_distance_improve,
        reward_closing_speed=args.reward_closing_speed,
        penalty_time=args.penalty_time,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
