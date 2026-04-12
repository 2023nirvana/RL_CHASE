"""
低层策略训练：目标到达任务，观测含加速度
==========================================

在 LowLevelTarget2DEnv 上训练 PPO，得到可复用于 HRL 的底层策略。
观测包含 pos_error, vel, acc，使策略学会利用机动性（加速度）。
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_low_level_target import LowLevelTarget2DEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple = (128, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*layers)
        self.actor_head = nn.Linear(prev, n_actions)
        self.critic_body = nn.Sequential(*layers)
        self.critic_head = nn.Linear(prev, 1)

    def forward(self, x):
        a_feat = self.actor_body(x)
        logits = self.actor_head(a_feat)
        v = self.critic_head(self.critic_body(x)).squeeze(-1)
        return logits, v

    def get_action(self, x, deterministic=False):
        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample(), dist.log_prob(dist.sample()), dist.entropy()

    def get_value(self, x):
        _, v = self.forward(x)
        return v


def train(
    experiment_name: str = None,
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    n_steps: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    v_max: float = 1.0,
    a_max: float = 2.0,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = experiment_name or f"pe2d_lowlevel_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join(PROJECT_ROOT, "visualization", "logs", "pe2d", exp_name)
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    def make_env():
        return LowLevelTarget2DEnv(
            v_max=v_max,
            a_max=a_max,
            max_steps=400,
            target_distance_range=(1.0, 5.0),
            seed=seed,
        )

    envs = [make_env() for _ in range(n_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
    rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

    obs, _ = np.array([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]), None
    if obs.ndim == 1:
        obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    n_updates = total_timesteps // (n_steps * n_envs)
    success_count = 0
    ep_count = 0

    for update in range(1, n_updates + 1):
        for step in range(n_steps):
            obs_buf[step] = obs
            x = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, val = model(x)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample()
                logp = dist.log_prob(act)
            act_buf[step] = act.cpu().numpy()
            logp_buf[step] = logp.cpu().numpy()
            val_buf[step] = val.cpu().numpy()

            next_obs_list = []
            for i, e in enumerate(envs):
                no, r, term, trunc, info = e.step(int(act_buf[step, i]))
                if term or trunc:
                    if info.get("success"):
                        success_count += 1
                    ep_count += 1
                    no, _ = e.reset()
                next_obs_list.append(no)
                rew_buf[step, i] = r
                done_buf[step, i] = 1.0 if (term or trunc) else 0.0
            obs = np.array(next_obs_list)

        last_val = model.get_value(torch.tensor(obs, dtype=torch.float32, device=device)).detach().cpu().numpy()
        adv = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            next_non_term = 1.0 - done_buf[t]
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_non_term - val_buf[t]
            adv[t] = last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            last_val = val_buf[t]
        ret = adv + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        b_act = torch.tensor(act_buf.reshape(-1), dtype=torch.long, device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(4):
            perm = torch.randperm(len(b_obs), device=device)
            for start in range(0, len(b_obs), batch_size):
                idx = perm[start : start + batch_size]
                logits, v = model(b_obs[idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp_new = dist.log_prob(b_act[idx])
                ent = dist.entropy().mean()
                vf_loss = nn.functional.mse_loss(v, b_ret[idx])
                ratio = torch.exp(logp_new - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                loss = pg_loss + 0.5 * vf_loss - ent_coef * ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if update % 50 == 0:
            eval_rew = []
            eval_env = LowLevelTarget2DEnv(v_max=v_max, a_max=a_max, seed=seed + 999)
            for _ in range(10):
                o, _ = eval_env.reset()
                r_sum = 0.0
                for _ in range(400):
                    with torch.no_grad():
                        a, _, _ = model.get_action(torch.tensor(o[np.newaxis], dtype=torch.float32, device=device))
                        a = a.item()
                    o, r, term, trunc, info = eval_env.step(a)
                    r_sum += r
                    if term or trunc:
                        break
                eval_rew.append(r_sum)
            mean_rew = np.mean(eval_rew)
            sr = success_count / max(1, ep_count)
            print(f"update {update} | eval reward {mean_rew:.2f} | train success rate {sr:.2%}", flush=True)
            torch.save({
                "model": model.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "v_max": v_max,
                "a_max": a_max,
            }, os.path.join(ckpt_dir, "latest.pth"))

    torch.save({
        "model": model.state_dict(),
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "v_max": v_max,
        "a_max": a_max,
    }, os.path.join(ckpt_dir, "final.pth"))
    print(f"Low-level policy saved to {ckpt_dir}", flush=True)
    return model, ckpt_dir


def main():
    p = argparse.ArgumentParser(description="Train low-level policy with acceleration in obs")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--v-max", type=float, default=1.0, help="Max speed (for compatibility with pursuer)")
    p.add_argument("--a-max", type=float, default=2.0, help="Max acceleration")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        v_max=args.v_max,
        a_max=args.a_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
