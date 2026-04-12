"""
Option A 训练：高层离散策略 + 底层连续 PID 执行
==============================================

- 高层每 K 步输出一个宏动作（9 离散）
- 底层将宏动作映射为连续 waypoint，并由 PID 连续执行
- 环境为 PursuitEvasion2DAsymmetricEnvV2PIDOptionA
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_asymmetric_pe_v2_pid_option_a import (
    PursuitEvasion2DAsymmetricEnvV2PIDOptionA,
    NUM_HIGH_ACTIONS,
)


class HighLevelActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*layers)
        self.actor_head = nn.Linear(prev, n_actions)

        critic_layers = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.critic_body = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(prev, 1)

    def forward(self, x):
        logits = self.actor_head(self.actor_body(x))
        value = self.critic_head(self.critic_body(x)).squeeze(-1)
        return logits, value


def train_option_a_pid(
    experiment_name: str = None,
    total_timesteps: int = 600_000,
    n_envs: int = 8,
    n_steps: int = 64,
    macro_steps: int = 5,
    batch_size: int = 128,
    n_epochs: int = 4,
    lr: float = 2e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    difficulty: float = 0.6,
    evader_mode: str = "smart",
    include_acc_in_high_obs: bool = True,
    high_waypoint_dist: float = 2.0,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = experiment_name or f"pe2d_optionA_pid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d_optionA_pid", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    def make_env(env_seed: int):
        return PursuitEvasion2DAsymmetricEnvV2PIDOptionA(
            difficulty=difficulty,
            evader_mode=evader_mode,
            include_acc_in_high_obs=include_acc_in_high_obs,
            high_waypoint_dist=high_waypoint_dist,
            seed=env_seed,
        )

    envs = [make_env(seed + i) for i in range(n_envs)]

    obs_dim = 8 if include_acc_in_high_obs else 6
    model = HighLevelActorCritic(obs_dim, NUM_HIGH_ACTIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def get_high_obs(env):
        return env.get_high_level_obs()

    obs_high = []
    for i, env in enumerate(envs):
        env.reset(seed=seed + i)
        obs_high.append(get_high_obs(env))
    obs_high = np.stack(obs_high)

    n_updates = total_timesteps // (n_steps * n_envs * macro_steps)
    catch_count = 0
    ep_count = 0

    obs_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
    rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

    for update in range(1, n_updates + 1):
        for step in range(n_steps):
            obs_buf[step] = obs_high
            x = torch.tensor(obs_high, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, val = model(x)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample()
                logp = dist.log_prob(act)

            act_np = act.cpu().numpy()
            act_buf[step] = act_np
            logp_buf[step] = logp.cpu().numpy()
            val_buf[step] = val.cpu().numpy()

            next_obs = []
            for i, env in enumerate(envs):
                _, macro_reward, term, trunc, info = env.step_macro(
                    high_action=int(act_np[i]),
                    macro_steps=macro_steps,
                )
                rew_buf[step, i] = macro_reward
                done_buf[step, i] = 1.0 if (term or trunc) else 0.0

                if term or trunc:
                    if info.get("caught", False):
                        catch_count += 1
                    ep_count += 1
                    env.reset(seed=seed + i + update * n_envs)

                next_obs.append(get_high_obs(env))

            obs_high = np.stack(next_obs)

        with torch.no_grad():
            _, last_val = model(torch.tensor(obs_high, dtype=torch.float32, device=device))
            last_val = last_val.cpu().numpy()

        adv = np.zeros_like(rew_buf)
        last_gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_non_term = 1.0 - done_buf[t]
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_non_term - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            adv[t] = last_gae
        ret = adv + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        b_act = torch.tensor(act_buf.reshape(-1), dtype=torch.long, device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(n_epochs):
            perm = torch.randperm(len(b_obs), device=device)
            for start in range(0, len(b_obs), batch_size):
                idx = perm[start : start + batch_size]
                logits, v = model(b_obs[idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp_new = dist.log_prob(b_act[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp_new - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = nn.functional.mse_loss(v, b_ret[idx])
                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if update % 20 == 0:
            catch_rate = catch_count / max(1, ep_count)
            print(
                f"[OptionA] update={update} mean_macro_rew={rew_buf.mean():.3f} catch_rate={catch_rate:.2%}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "high_obs_dim": obs_dim,
                    "n_actions": NUM_HIGH_ACTIONS,
                    "macro_steps": macro_steps,
                    "include_acc_in_high_obs": include_acc_in_high_obs,
                    "high_waypoint_dist": high_waypoint_dist,
                },
                os.path.join(ckpt_dir, "high_latest.pth"),
            )

    torch.save(
        {
            "model": model.state_dict(),
            "high_obs_dim": obs_dim,
            "n_actions": NUM_HIGH_ACTIONS,
            "macro_steps": macro_steps,
            "include_acc_in_high_obs": include_acc_in_high_obs,
            "high_waypoint_dist": high_waypoint_dist,
        },
        os.path.join(ckpt_dir, "high_final.pth"),
    )
    print(f"Option A high-level model saved to {ckpt_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=600000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--macro-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--difficulty", type=float, default=0.6)
    parser.add_argument("--evader-mode", type=str, default="smart")
    parser.add_argument("--high-waypoint-dist", type=float, default=2.0)
    parser.add_argument("--disable-acc-high-obs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_option_a_pid(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        macro_steps=args.macro_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        difficulty=args.difficulty,
        evader_mode=args.evader_mode,
        include_acc_in_high_obs=not args.disable_acc_high_obs,
        high_waypoint_dist=args.high_waypoint_dist,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
