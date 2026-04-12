"""
阶段一：连续动作低层控制训练（PPO-Gaussian）
=========================================

目标：先验证连续动作底层在目标到达任务上的效果，再对接高层追逃任务。
环境：LowLevelTarget2DContinuousEnv
动作：连续 2D（归一化到 [-1,1]，映射到加速度命令）
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_low_level_target_continuous import LowLevelTarget2DContinuousEnv
from pursuit_evasion_2d.continuous_policy_nets import build_continuous_actor_critic


def _init_obs_history(obs: np.ndarray, seq_len: int) -> List[deque]:
    return [deque([obs_i.copy() for _ in range(seq_len)], maxlen=seq_len) for obs_i in obs]


def _stack_obs_history(histories: List[deque]) -> np.ndarray:
    return np.stack([np.stack(list(history), axis=0) for history in histories], axis=0).astype(np.float32)


def _update_history(history: deque, obs: np.ndarray, seq_len: int, reset: bool = False):
    if reset:
        history.clear()
        for _ in range(seq_len):
            history.append(obs.copy())
    else:
        history.append(obs.copy())


def train(
    experiment_name: str = None,
    total_timesteps: int = 800_000,
    n_envs: int = 8,
    control_mode: str = "dual_pid",
    low_level_config_path: str = None,
    robot_config_path: str = None,
    n_steps: int = 128,
    batch_size: int = 256,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    v_max: float = 1.2,
    a_max: float = 2.5,
    seed: int = 42,
    network_type: str = "mlp",
    seq_len: int = 8,
    hidden_sizes: tuple[int, ...] = (128, 64),
    rnn_hidden: int = 128,
    rnn_layers: int = 1,
    enable_current_disturbance: bool = False,
    current_scale: float = 0.25,
    current_ou_theta: float = 0.12,
    current_ou_sigma: float = 0.08,
    current_drift_scale: float = 0.10,
    obs_mode: str = "full",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = experiment_name or f"pe2d_lowlevel_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pe2d_lowlevel_cont", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    def make_env(local_seed: int):
        return LowLevelTarget2DContinuousEnv(
            v_max=v_max,
            a_max=a_max,
            control_mode=control_mode,
            low_level_config_path=low_level_config_path,
            robot_config_path=robot_config_path,
            enable_current_disturbance=enable_current_disturbance,
            current_scale=current_scale,
            current_ou_theta=current_ou_theta,
            current_ou_sigma=current_ou_sigma,
            current_drift_scale=current_drift_scale,
            obs_mode=obs_mode,
            seed=local_seed,
        )

    envs = [make_env(seed + i) for i in range(n_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]

    model = build_continuous_actor_critic(
        obs_dim,
        act_dim,
        network_type=network_type,
        hidden=hidden_sizes,
        seq_len=seq_len,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    histories = _init_obs_history(obs, seq_len)
    n_updates = total_timesteps // (n_steps * n_envs)

    success_count = 0
    ep_count = 0
    recent_success = deque(maxlen=200)
    speed_sum = 0.0
    thrust_norm_sum = 0.0
    info_count = 0

    for update in range(1, n_updates + 1):
        if network_type == "mlp":
            obs_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        else:
            obs_buf = np.zeros((n_steps, n_envs, seq_len, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for t in range(n_steps):
            obs_input = obs if network_type == "mlp" else _stack_obs_history(histories)
            obs_buf[t] = obs_input
            x = torch.tensor(obs_input, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, _ = model.get_action(x, deterministic=False)
                v = model.get_value(x)

            act_np = a.cpu().numpy()
            act_buf[t] = act_np
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = v.cpu().numpy()

            next_obs = []
            for i, env in enumerate(envs):
                no, r, term, trunc, info = env.step(act_np[i])
                speed_sum += float(info.get("speed", 0.0))
                thrust_norm_sum += float(info.get("thrust_norm", 0.0))
                info_count += 1
                if term or trunc:
                    if info.get("success", False):
                        success_count += 1
                    recent_success.append(1.0 if info.get("success", False) else 0.0)
                    ep_count += 1
                    no, _ = env.reset()
                    _update_history(histories[i], no, seq_len, reset=True)
                else:
                    _update_history(histories[i], no, seq_len, reset=False)
                next_obs.append(no)
                rew_buf[t, i] = r
                done_buf[t, i] = 1.0 if (term or trunc) else 0.0
            obs = np.array(next_obs)

        with torch.no_grad():
            obs_input = obs if network_type == "mlp" else _stack_obs_history(histories)
            x_last = torch.tensor(obs_input, dtype=torch.float32, device=device)
            last_val = model.get_value(x_last).cpu().numpy()

        adv = np.zeros_like(rew_buf)
        last_gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_non_term = 1.0 - done_buf[t]
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_non_term - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            adv[t] = last_gae
        ret = adv + val_buf

        if network_type == "mlp":
            b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        else:
            b_obs = torch.tensor(obs_buf.reshape(-1, seq_len, obs_dim), dtype=torch.float32, device=device)
        b_act = torch.tensor(act_buf.reshape(-1, act_dim), dtype=torch.float32, device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(n_epochs):
            perm = torch.randperm(len(b_obs), device=device)
            for start in range(0, len(b_obs), batch_size):
                idx = perm[start:start + batch_size]
                logp_new, ent_each = model.evaluate_actions(b_obs[idx], b_act[idx])
                ent = ent_each.mean()
                v = model.get_value(b_obs[idx])

                ratio = torch.exp(logp_new - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = nn.functional.mse_loss(v, b_ret[idx])
                loss = pg_loss + vf_coef * vf_loss - ent_coef * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if update % 25 == 0:
            sr = success_count / max(1, ep_count)
            sr_recent = float(np.mean(recent_success)) if len(recent_success) > 0 else 0.0
            mean_speed = speed_sum / max(1, info_count)
            mean_thrust_norm = thrust_norm_sum / max(1, info_count)
            print(
                f"[cont-ll] update={update} mode={control_mode} "
                f"net={network_type} "
                f"mean_rew={rew_buf.mean():.3f} success_rate={sr:.2%} recent_success_rate={sr_recent:.2%} "
                f"mean_speed={mean_speed:.3f} mean_thrust_norm={mean_thrust_norm:.3f}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "v_max": v_max,
                    "a_max": a_max,
                    "network_type": network_type,
                    "seq_len": seq_len,
                    "hidden_sizes": list(hidden_sizes),
                    "rnn_hidden": rnn_hidden,
                    "rnn_layers": rnn_layers,
                    "enable_current_disturbance": enable_current_disturbance,
                    "current_scale": current_scale,
                    "current_ou_theta": current_ou_theta,
                    "current_ou_sigma": current_ou_sigma,
                    "current_drift_scale": current_drift_scale,
                    "obs_mode": obs_mode,
                },
                os.path.join(ckpt_dir, "latest.pth"),
            )

    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "v_max": v_max,
            "a_max": a_max,
            "network_type": network_type,
            "seq_len": seq_len,
            "hidden_sizes": list(hidden_sizes),
            "rnn_hidden": rnn_hidden,
            "rnn_layers": rnn_layers,
            "enable_current_disturbance": enable_current_disturbance,
            "current_scale": current_scale,
            "current_ou_theta": current_ou_theta,
            "current_ou_sigma": current_ou_sigma,
            "current_drift_scale": current_drift_scale,
            "obs_mode": obs_mode,
        },
        os.path.join(ckpt_dir, "final.pth"),
    )
    print(f"Continuous low-level model saved to {ckpt_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=800000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--control-mode",
        type=str,
        default="dual_pid",
        choices=["dual_pid", "direct"],
        help="dual_pid=RL连续动作->双环PID->动力学; direct=RL动作直接映射加速度",
    )
    parser.add_argument("--low-level-config-path", type=str, default=None)
    parser.add_argument("--robot-config-path", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--v-max", type=float, default=1.2)
    parser.add_argument("--a-max", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-type", type=str, default="mlp", choices=["mlp", "lstm", "gru"])
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--rnn-hidden", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--enable-current-disturbance", action="store_true")
    parser.add_argument("--current-scale", type=float, default=0.25)
    parser.add_argument("--current-ou-theta", type=float, default=0.12)
    parser.add_argument("--current-ou-sigma", type=float, default=0.08)
    parser.add_argument("--current-drift-scale", type=float, default=0.10)
    parser.add_argument("--obs-mode", type=str, default="full", choices=["full", "no_acc", "pos_only"])
    args = parser.parse_args()

    train(
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        control_mode=args.control_mode,
        low_level_config_path=args.low_level_config_path,
        robot_config_path=args.robot_config_path,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ent_coef=args.ent_coef,
        v_max=args.v_max,
        a_max=args.a_max,
        seed=args.seed,
        network_type=args.network_type,
        seq_len=args.seq_len,
        hidden_sizes=tuple(args.hidden_sizes),
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        enable_current_disturbance=args.enable_current_disturbance,
        current_scale=args.current_scale,
        current_ou_theta=args.current_ou_theta,
        current_ou_sigma=args.current_ou_sigma,
        current_drift_scale=args.current_drift_scale,
        obs_mode=args.obs_mode,
    )


if __name__ == "__main__":
    main()
