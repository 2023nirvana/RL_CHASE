"""
分层 HRL 追逃：高层输出子目标，底层策略执行
============================================

先加载预训练底层策略（含加速度观测），高层在追逃环境中每 K 步输出一个子目标位置，
底层在该子目标下执行 K 步，再由高层根据新状态决策。
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

from pursuit_evasion_2d.env_asymmetric_pe import (
    PursuitEvasion2DAsymmetricEnv,
    DEFAULT_PURSUER_V_MAX,
    DEFAULT_PURSUER_A_MAX,
    DEFAULT_EVADER_V_MAX,
    DEFAULT_EVADER_A_MAX,
)
from pursuit_evasion_2d.env_asymmetric_pe import _action_to_acceleration
from pursuit_evasion_2d.evader_policy import scripted_evader_run_away

# 高层子目标：8 方向 + 当前逃脱者位置，共 9 个
NUM_HIGH_ACTIONS = 9
WAYPOINT_DIST = 2.0
LOW_LEVEL_STEPS = 5
OBS_SCALE_POS = 1.0 / 10.0
OBS_SCALE_VEL = 1.0 / 2.0


def build_low_level_obs(goal, p_pos, p_vel, p_acc):
    pos_err = goal - p_pos
    obs = np.concatenate([
        pos_err * OBS_SCALE_POS,
        p_vel * OBS_SCALE_VEL,
        p_acc * (1.0 / 3.0),
    ]).astype(np.float32)
    return np.clip(obs, -10.0, 10.0)


def high_action_to_waypoint(high_action, p_pos, e_pos):
    """高层动作 -> 追击者子目标位置"""
    if high_action == 0:
        return e_pos.copy()
    angle = (high_action - 1) * (2 * np.pi / 8)
    offset = WAYPOINT_DIST * np.array([np.cos(angle), np.sin(angle)])
    return p_pos + offset


class LowLevelWrapper:
    def __init__(self, model, device, obs_dim=6, n_actions=9):
        self.model = model
        self.device = device
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    def get_action(self, obs, deterministic=True):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            if deterministic:
                return torch.argmax(logits, dim=-1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1)
            return torch.distributions.Categorical(probs).sample().cpu().numpy()


class HighLevelActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(64, 32)):
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


def train_hrl(
    low_level_ckpt: str,
    experiment_name=None,
    total_timesteps=500_000,
    n_envs=4,
    n_steps=32,
    batch_size=32,
    lr=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    pursuer_v_max=DEFAULT_PURSUER_V_MAX,
    pursuer_a_max=DEFAULT_PURSUER_A_MAX,
    evader_v_max=DEFAULT_EVADER_V_MAX,
    evader_a_max=DEFAULT_EVADER_A_MAX,
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(low_level_ckpt, map_location=device)
    from pursuit_evasion_2d.train_low_level_accel import ActorCritic as LowLevelNet
    low_net = LowLevelNet(ckpt["obs_dim"], ckpt["n_actions"]).to(device)
    low_net.load_state_dict(ckpt["model"])
    low_level = LowLevelWrapper(low_net, device, ckpt["obs_dim"], ckpt["n_actions"])

    exp_name = experiment_name or f"pe2d_hrl_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    def make_env():
        return PursuitEvasion2DAsymmetricEnv(
            pursuer_v_max=pursuer_v_max,
            pursuer_a_max=pursuer_a_max,
            evader_v_max=evader_v_max,
            evader_a_max=evader_a_max,
            seed=seed,
        )
    envs = [make_env() for _ in range(n_envs)]
    high_obs_dim = 6
    high_model = HighLevelActorCritic(high_obs_dim, NUM_HIGH_ACTIONS).to(device)
    optimizer = torch.optim.Adam(high_model.parameters(), lr=lr)

    obs_buf = np.zeros((n_steps, n_envs, high_obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
    rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

    def get_high_obs(env):
        s = env.get_state()
        p_pos, p_vel = s["pursuer_pos"], s["pursuer_vel"]
        e_pos, e_vel = s["evader_pos"], s["evader_vel"]
        rel = (e_pos - p_pos) * OBS_SCALE_POS
        return np.concatenate([rel, p_vel * OBS_SCALE_VEL, e_vel * OBS_SCALE_VEL]).astype(np.float32)

    obs_high = np.stack([get_high_obs(e) for e in envs])
    n_updates = total_timesteps // (n_steps * n_envs * LOW_LEVEL_STEPS)
    catch_count, ep_count = 0, 0

    for update in range(1, n_updates + 1):
        for step in range(n_steps):
            obs_buf[step] = obs_high
            x = torch.tensor(obs_high, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, val = high_model(x)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample()
                logp = dist.log_prob(act)
            act_buf[step] = act.cpu().numpy()
            logp_buf[step] = logp.cpu().numpy()
            val_buf[step] = val.cpu().numpy()

            total_rew = np.zeros(n_envs)
            for i, e in enumerate(envs):
                high_act = int(act_buf[step, i])
                s = e.get_state()
                waypoint = high_action_to_waypoint(high_act, s["pursuer_pos"], s["evader_pos"])
                p_pos, p_vel = s["pursuer_pos"].copy(), s["pursuer_vel"].copy()
                p_acc = np.zeros(2)
                for _ in range(LOW_LEVEL_STEPS):
                    obs_lo = build_low_level_obs(waypoint, p_pos, p_vel, p_acc)
                    a_lo = low_level.get_action(obs_lo[np.newaxis])[0]
                    ev_act = scripted_evader_run_away(s["evader_pos"], p_pos, evader_a_max, e.np_random)
                    no, r, term, trunc, info = e.step(a_lo, evader_action=ev_act)
                    total_rew[i] += r
                    s = e.get_state()
                    p_pos, p_vel = s["pursuer_pos"].copy(), s["pursuer_vel"].copy()
                    p_acc = _action_to_acceleration(a_lo, pursuer_a_max)
                    if term or trunc:
                        if info.get("caught"):
                            catch_count += 1
                        ep_count += 1
                        no, _ = e.reset(seed=seed + i + update * n_envs)
                        break
                rew_buf[step, i] = total_rew[i]
                done_buf[step, i] = 1.0 if (term or trunc) else 0.0
            obs_high = np.stack([get_high_obs(e) for e in envs])

        last_val = high_model.get_value(torch.tensor(obs_high, dtype=torch.float32, device=device)).detach().cpu().numpy()
        adv = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            next_non_term = 1.0 - done_buf[t]
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_non_term - val_buf[t]
            adv[t] = last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            last_val = val_buf[t]
        ret = adv + val_buf
        b_obs = torch.tensor(obs_buf.reshape(-1, high_obs_dim), dtype=torch.float32, device=device)
        b_act = torch.tensor(act_buf.reshape(-1), dtype=torch.long, device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        for _ in range(4):
            perm = torch.randperm(len(b_obs), device=device)
            for start in range(0, len(b_obs), batch_size):
                idx = perm[start : start + batch_size]
                logits, v = high_model(b_obs[idx])
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
                nn.utils.clip_grad_norm_(high_model.parameters(), 0.5)
                optimizer.step()
        if update % 50 == 0:
            sr = catch_count / max(1, ep_count)
            print(f"hrl update {update} | mean_rew {rew_buf.mean():.2f} | catch_rate {sr:.2%}", flush=True)
            torch.save({"model": high_model.state_dict(), "high_obs_dim": high_obs_dim}, os.path.join(ckpt_dir, "high_latest.pth"))
    torch.save({"model": high_model.state_dict(), "high_obs_dim": high_obs_dim}, os.path.join(ckpt_dir, "high_final.pth"))
    print(f"HRL high-level saved to {ckpt_dir}", flush=True)
    return high_model, ckpt_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low-level-ckpt", type=str, required=True, help="Path to low-level checkpoint (e.g. checkpoints/pe2d/pe2d_lowlevel_xxx/final.pth)")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_hrl(low_level_ckpt=args.low_level_ckpt, experiment_name=args.experiment_name,
              total_timesteps=args.total_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
