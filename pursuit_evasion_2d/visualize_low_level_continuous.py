"""
连续低层目标到达策略可视化
=========================

加载 train_low_level_continuous.py 训练得到的 checkpoint，
在 LowLevelTarget2DContinuousEnv 中执行若干 episode，输出：
- 轨迹/速度/加速度/距离 图（PNG）
- 轨迹动画（GIF）

示例：
  cd /root/autodl-tmp/pursue_20260302
  python pursuit_evasion_2d/visualize_low_level_continuous.py \
    --ckpt checkpoints/pe2d_lowlevel_cont/smoke_log_fmt/final.pth \
    --episodes 3 --control-mode dual_pid
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_low_level_target_continuous import LowLevelTarget2DContinuousEnv
from pursuit_evasion_2d.continuous_policy_nets import build_continuous_actor_critic


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", 6))
    act_dim = int(ckpt.get("act_dim", 2))
    network_type = str(ckpt.get("network_type", "mlp"))
    hidden_sizes = tuple(ckpt.get("hidden_sizes", [128, 64]))
    seq_len = int(ckpt.get("seq_len", 8))
    rnn_hidden = int(ckpt.get("rnn_hidden", 128))
    rnn_layers = int(ckpt.get("rnn_layers", 1))
    model = build_continuous_actor_critic(
        obs_dim,
        act_dim,
        network_type=network_type,
        hidden=hidden_sizes,
        seq_len=seq_len,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_episode(model, env, device, deterministic=True):
    obs, info = env.reset()
    network_type = getattr(model, "rnn_type", None)
    uses_sequence = network_type in {"lstm", "gru"}
    seq_len = int(getattr(model, "seq_len", 8)) if hasattr(model, "seq_len") else None
    if seq_len is None:
        seq_len = 1 if not uses_sequence else 8
    obs_history = deque([obs.copy() for _ in range(seq_len)], maxlen=seq_len)

    traj = {
        "pos": [env._pos.copy()],
        "goal": [info["goal"].copy()],
        "vel": [env._vel.copy()],
        "acc": [env._acc.copy()],
        "distance": [float(info["distance"])],
        "reward": [],
        "thrust_norm": [0.0],
        "speed": [0.0],
    }

    done = False
    while not done:
        with torch.no_grad():
            model_in = np.stack(list(obs_history), axis=0) if uses_sequence else obs
            x = torch.tensor(model_in, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = model.act_deterministic(x)[0].cpu().numpy()
            else:
                action = model.act_deterministic(x)[0].cpu().numpy()

        obs, rew, term, trunc, step_info = env.step(action)
        obs_history.append(obs.copy())

        traj["pos"].append(env._pos.copy())
        traj["goal"].append(env._goal.copy())
        traj["vel"].append(env._vel.copy())
        traj["acc"].append(env._acc.copy())
        traj["distance"].append(float(step_info.get("distance", 0.0)))
        traj["reward"].append(float(rew))
        traj["thrust_norm"].append(float(step_info.get("thrust_norm", 0.0)))
        traj["speed"].append(float(step_info.get("speed", np.linalg.norm(env._vel))))

        done = term or trunc

    traj["success"] = bool(step_info.get("success", False))
    traj["steps"] = len(traj["pos"]) - 1
    traj["ep_return"] = float(np.sum(traj["reward"]))
    return traj


def plot_episode(traj, save_dir: str, episode_id: int, world_size: float, goal_tol: float, dt: float):
    pos = np.asarray(traj["pos"]) 
    goal = np.asarray(traj["goal"]) 
    vel = np.asarray(traj["vel"]) 
    acc = np.asarray(traj["acc"]) 
    dist = np.asarray(traj["distance"]) 
    rew = np.asarray(traj["reward"]) if len(traj["reward"]) > 0 else np.array([0.0])
    speed = np.asarray(traj["speed"]) 
    thrust_norm = np.asarray(traj["thrust_norm"]) 

    steps = np.arange(len(pos))
    steps_r = np.arange(len(rew))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], "b-", lw=2, alpha=0.85, label="agent")
    ax.scatter(pos[0, 0], pos[0, 1], c="blue", s=70, marker="o", label="start")
    ax.scatter(pos[-1, 0], pos[-1, 1], c="blue", s=70, marker="x", label="end")
    ax.scatter(goal[0, 0], goal[0, 1], c="red", s=80, marker="*", label="goal")
    ax.add_patch(Circle((goal[0, 0], goal[0, 1]), goal_tol, fill=False, color="red", ls="--", lw=1.3, alpha=0.8))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, edgecolor="black", lw=1.5))
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title(f"Trajectory (Ep{episode_id}) - {'Success' if traj['success'] else 'Timeout'}")

    ax = axes[0, 1]
    ax.plot(steps, speed, "g-", lw=2, label="|v|")
    ax.plot(steps, np.linalg.norm(acc, axis=1), "m-", lw=2, label="|a|")
    ax.set_xlabel("step")
    ax.set_ylabel("norm")
    ax.set_title("Speed / Acceleration")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, dist, "c-", lw=2, label="distance")
    ax.axhline(y=goal_tol, color="r", ls="--", lw=1.2, label="goal tolerance")
    ax.set_xlabel("step")
    ax.set_ylabel("distance")
    ax.set_title("Distance to Goal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps_r, rew, "b-", lw=1.6, label="reward")
    ax.plot(steps, thrust_norm, "orange", lw=1.6, label="thrust norm")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("Reward / Thrust")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(f"Return={traj['ep_return']:.2f}, Steps={traj['steps']}, dt={dt}")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"ll_cont_ep{episode_id}_summary.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def save_gif(
    traj,
    save_path: str,
    world_size: float,
    goal_tol: float,
    fps: int = 20,
    frame_step: int = 3,
):
    pos = np.asarray(traj["pos"]) 
    goal = np.asarray(traj["goal"]) 
    dist = np.asarray(traj["distance"]) 
    frame_step = max(1, int(frame_step))
    frame_indices = list(range(0, len(pos), frame_step))
    if frame_indices[-1] != len(pos) - 1:
        frame_indices.append(len(pos) - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, edgecolor="black", lw=2))
    goal_circle = Circle((goal[0, 0], goal[0, 1]), goal_tol, fill=False, color="red", ls="--", lw=1.3)
    ax.add_patch(goal_circle)

    line, = ax.plot([], [], "b-", lw=1.8, alpha=0.7)
    pt, = ax.plot([], [], "bo", markersize=8)
    goal_pt, = ax.plot([goal[0, 0]], [goal[0, 1]], "r*", markersize=12)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11)

    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.set_title("Low-level Continuous Policy")
    ax.grid(True, alpha=0.25)

    def init():
        line.set_data([], [])
        pt.set_data([], [])
        txt.set_text("")
        return line, pt, goal_pt, txt

    def animate(i):
        fi = frame_indices[i]
        line.set_data(pos[: fi + 1, 0], pos[: fi + 1, 1])
        pt.set_data([pos[fi, 0]], [pos[fi, 1]])
        txt.set_text(f"step={fi}\\ndist={dist[fi]:.3f}")
        return line, pt, goal_pt, txt

    ani = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=True,
    )
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize low-level continuous policy")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to latest.pth or final.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-mode", type=str, default="dual_pid", choices=["dual_pid", "direct"])
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--gif-frame-step", type=int, default=3, help="Use every Nth frame for faster GIF export")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    if hasattr(model, "rnn_type"):
        model.seq_len = int(ckpt.get("seq_len", 8))

    v_max = float(ckpt.get("v_max", 1.2))
    a_max = float(ckpt.get("a_max", 2.5))

    env = LowLevelTarget2DContinuousEnv(
        v_max=v_max,
        a_max=a_max,
        control_mode=args.control_mode,
        obs_mode=str(ckpt.get("obs_mode", "full")),
        enable_current_disturbance=bool(ckpt.get("enable_current_disturbance", False)),
        current_scale=float(ckpt.get("current_scale", 0.25)),
        current_ou_theta=float(ckpt.get("current_ou_theta", 0.12)),
        current_ou_sigma=float(ckpt.get("current_ou_sigma", 0.08)),
        current_drift_scale=float(ckpt.get("current_drift_scale", 0.10)),
        seed=args.seed,
    )

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", "ll_cont_vis", os.path.splitext(os.path.basename(args.ckpt))[0])
    os.makedirs(out_dir, exist_ok=True)

    print(f"Visualize ckpt={args.ckpt}, episodes={args.episodes}, mode={args.control_mode}")
    print(f"Output dir: {out_dir}")

    success = 0
    for ep in range(1, args.episodes + 1):
        traj = run_episode(model, env, device, deterministic=True)
        success += int(traj["success"])
        plot_episode(traj, out_dir, ep, env.world_size, env.goal_tolerance, env.dt)
        if not args.no_gif:
            gif_path = os.path.join(out_dir, f"ll_cont_ep{ep}.gif")
            save_gif(
                traj,
                gif_path,
                env.world_size,
                env.goal_tolerance,
                fps=args.fps,
                frame_step=args.gif_frame_step,
            )

    print("-" * 50)
    print(f"Done. Success: {success}/{args.episodes}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
