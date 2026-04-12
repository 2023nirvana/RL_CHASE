"""
2D 追逃 (low_level) 训练结果可视化
====================================

加载 train_pe_low_level_2d 训练得到的 checkpoint（best_model.pth），
在 LowLevelAUV2DPursuitEvasionEnv 中跑若干 episode，生成：
- 轨迹图 + 距离曲线 + 双方速度曲线 + 双方加速度曲线（2x2 子图）
- 每条 episode 的 GIF 动画（可选，默认生成）
- all_trajectories.png 汇总多条轨迹

使用：
  cd UnderwaterPursuitEvasion
  python training/visualize_pe_low_level_2d.py --experiment-name pe2d_small_easy --episodes 5
  python training/visualize_pe_low_level_2d.py --all --episodes 3
  python training/visualize_pe_low_level_2d.py --experiment-name pe2d_baseline --episodes 3 --no-gif --gif-fps 15
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_pursuit_evasion import LowLevelAUV2DPursuitEvasionEnv

# 与 run_pe_2d_parallel 一致：实验名 pe2d_<scenario> -> 环境参数
PE2D_ENV_PARAMS = {
    "pe2d_baseline": {"init_distance_range": (2.0, 6.0), "world_size": 10.0, "catch_radius": 0.5, "max_episode_seconds": 90.0},
    "pe2d_small_world": {"init_distance_range": (1.0, 3.0), "world_size": 5.0, "catch_radius": 0.5, "max_episode_seconds": 90.0},
    "pe2d_easy_catch": {"init_distance_range": (2.0, 4.0), "world_size": 10.0, "catch_radius": 0.8, "max_episode_seconds": 90.0},
    "pe2d_closer_start": {"init_distance_range": (1.0, 3.0), "world_size": 10.0, "catch_radius": 0.5, "max_episode_seconds": 90.0},
    "pe2d_short_ep": {"init_distance_range": (2.0, 6.0), "world_size": 10.0, "catch_radius": 0.5, "max_episode_seconds": 45.0},
    "pe2d_small_easy": {"init_distance_range": (1.0, 3.0), "world_size": 5.0, "catch_radius": 0.8, "max_episode_seconds": 60.0},
    "pe2d_ll_2m": {"init_distance_range": (2.0, 6.0), "world_size": 10.0, "catch_radius": 0.5, "max_episode_seconds": 90.0},
}


def build_actor(obs_dim=8, hidden_sizes=(128, 64), num_actions=6, device="cpu"):
    layers = []
    prev = obs_dim
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, num_actions))
    model = torch.nn.Sequential(*layers).to(device)
    return model


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = build_actor(obs_dim=8, hidden_sizes=[128, 64], num_actions=6, device=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def run_episode(actor, env: LowLevelAUV2DPursuitEvasionEnv, device: torch.device, deterministic: bool = True):
    obs, info = env.reset()
    pursuer_pos = [env.inner.dynamics.eta[:2].copy()]
    evader_pos = [env.evader_pos.copy()]
    pursuer_speeds = []  # 每步后的速度标量
    evader_speeds = []
    distances = [float(np.linalg.norm(pursuer_pos[0] - evader_pos[0]))]
    rewards = []
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(x)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = torch.distributions.Categorical(logits=logits).sample().item()
        obs, reward, term, trunc, info = env.step(action)
        nu = env.inner.dynamics.nu
        pursuer_speed = float(np.sqrt(nu[0] ** 2 + nu[1] ** 2))
        evader_speed = float(np.linalg.norm(env.evader_vel))
        pursuer_speeds.append(pursuer_speed)
        evader_speeds.append(evader_speed)
        pursuer_pos.append(env.inner.dynamics.eta[:2].copy())
        evader_pos.append(env.evader_pos.copy())
        dist = float(np.linalg.norm(pursuer_pos[-1] - evader_pos[-1]))
        distances.append(dist)
        rewards.append(reward)
        done = term or trunc
    success = info.get("termination_reason") == "success"
    return {
        "pursuer_pos": np.array(pursuer_pos),
        "evader_pos": np.array(evader_pos),
        "pursuer_speeds": np.array(pursuer_speeds),
        "evader_speeds": np.array(evader_speeds),
        "distances": np.array(distances),
        "rewards": np.array(rewards),
        "success": success,
        "steps": len(rewards),
    }


def _compute_acceleration(speeds: np.ndarray, dt: float) -> np.ndarray:
    """从速度序列计算加速度 |a| (前向差分)，长度与 speeds 一致，末尾重复。"""
    if len(speeds) < 2:
        return np.zeros_like(speeds)
    acc = np.diff(speeds) / dt
    return np.append(acc, acc[-1] if len(acc) else 0.0)


def plot_episode(traj: dict, save_path: str, ep_id: int, world_size: float, catch_radius: float,
                 dt: float, pursuer_v_max: float, evader_v_max: float, pursuer_a_max: float, evader_a_max: float):
    p = traj["pursuer_pos"]
    e = traj["evader_pos"]
    d = traj["distances"]
    p_speed = traj["pursuer_speeds"]
    e_speed = traj["evader_speeds"]
    success = traj["success"]
    p_acc = _compute_acceleration(p_speed, dt)
    e_acc = _compute_acceleration(e_speed, dt)
    steps_vel = np.arange(len(p_speed))
    steps_dist = np.arange(len(d))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(p[:, 0], p[:, 1], "b-", lw=1.5, label="Pursuer", alpha=0.8)
    ax.plot(e[:, 0], e[:, 1], "r-", lw=1.5, label="Evader", alpha=0.8)
    ax.scatter(p[0, 0], p[0, 1], c="blue", s=80, marker="o", zorder=5)
    ax.scatter(p[-1, 0], p[-1, 1], c="blue", s=80, marker="x", zorder=5)
    ax.scatter(e[0, 0], e[0, 1], c="red", s=80, marker="o", zorder=5)
    ax.scatter(e[-1, 0], e[-1, 1], c="red", s=80, marker="x", zorder=5)
    circle = Circle((e[-1, 0], e[-1, 1]), catch_radius, fill=False, edgecolor="green", linestyle="--", label="catch_radius")
    ax.add_patch(circle)
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"2D PE Trajectory (Ep{ep_id}) " + ("Caught" if success else "Timeout"))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps_dist, d, "g-", lw=2, label="Distance")
    ax.axhline(y=catch_radius, color="gray", ls="--", label=f"catch_radius={catch_radius}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.set_title("Distance to evader")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps_vel, p_speed, "b-", lw=2, label="Pursuer |v|", alpha=0.9)
    ax.plot(steps_vel, e_speed, "r-", lw=2, label="Evader |v|", alpha=0.9)
    ax.axhline(y=pursuer_v_max, color="blue", ls="--", alpha=0.5, label=f"P v_max={pursuer_v_max:.2f}")
    ax.axhline(y=evader_v_max, color="red", ls="--", alpha=0.5, label=f"E v_max={evader_v_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed |v|")
    ax.legend(fontsize=8)
    ax.set_title("Speed (Pursuer vs Evader)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_vel, p_acc, "b-", lw=2, label="Pursuer |a|", alpha=0.9)
    ax.plot(steps_vel, e_acc, "r-", lw=2, label="Evader |a|", alpha=0.9)
    ax.axhline(y=pursuer_a_max, color="blue", ls="--", alpha=0.5, label=f"P a_max={pursuer_a_max:.2f}")
    ax.axhline(y=evader_a_max, color="red", ls="--", alpha=0.5, label=f"E a_max={evader_a_max:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration |a|")
    ax.legend(fontsize=8)
    ax.set_title("Acceleration (Pursuer vs Evader)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_trajectory_gif(traj: dict, save_path: str, world_size: float, catch_radius: float, fps: int = 15, every_n: int = 2):
    """生成单条轨迹的 GIF 动画（追击者/逃脱者轨迹 + 当前距离）。"""
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
    except ImportError:
        print("    matplotlib.animation not available, skip GIF")
        return
    p_pos = np.array(traj["pursuer_pos"])
    e_pos = np.array(traj["evader_pos"])
    n_frames = len(p_pos)
    indices = list(range(0, n_frames, every_n))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6, label="Pursuer")
    e_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6, label="Evader")
    p_pt, = ax.plot([], [], "bo", markersize=10)
    e_pt, = ax.plot([], [], "ro", markersize=8)
    circle = Circle((0, 0), catch_radius, fill=False, edgecolor="green", linestyle="--")
    ax.add_patch(circle)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("2D PE Trajectory")
    ax.grid(True, alpha=0.3)

    def init():
        p_line.set_data([], [])
        e_line.set_data([], [])
        p_pt.set_data([], [])
        e_pt.set_data([], [])
        circle.center = (e_pos[0, 0], e_pos[0, 1])
        txt.set_text("")
        return p_line, e_line, p_pt, e_pt, txt

    def animate(i):
        idx = indices[min(i, len(indices) - 1)]
        p_line.set_data(p_pos[: idx + 1, 0], p_pos[: idx + 1, 1])
        e_line.set_data(e_pos[: idx + 1, 0], e_pos[: idx + 1, 1])
        p_pt.set_data([p_pos[idx, 0]], [p_pos[idx, 1]])
        e_pt.set_data([e_pos[idx, 0]], [e_pos[idx, 1]])
        circle.center = (e_pos[idx, 0], e_pos[idx, 1])
        dist = float(np.linalg.norm(p_pos[idx] - e_pos[idx]))
        txt.set_text(f"Step: {idx}\nDist: {dist:.2f}")
        return p_line, e_line, p_pt, e_pt, txt

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices), interval=1000 // fps, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"    GIF -> {save_path}")


def plot_all_episodes(trajectories: list, save_path: str, world_size: float, catch_radius: float):
    """一张图里画多条轨迹（每条不同颜色/线型）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    for i, traj in enumerate(trajectories):
        p = traj["pursuer_pos"]
        e = traj["evader_pos"]
        d = traj["distances"]
        success = traj["success"]
        c = colors[i % len(colors)]
        axes[0].plot(p[:, 0], p[:, 1], "-", color=c, lw=1.2, alpha=0.8, label=f"Ep{i+1} {'✓' if success else '✗'}")
        axes[0].plot(e[:, 0], e[:, 1], "--", color=c, lw=1.0, alpha=0.6)
        axes[1].plot(d, "-", color=c, lw=1.2, alpha=0.8, label=f"Ep{i+1}")

    axes[0].set_xlim(-world_size - 0.5, world_size + 0.5)
    axes[0].set_ylim(-world_size - 0.5, world_size + 0.5)
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[0].set_title("All trajectories (solid=Pursuer, dashed=Evader)")
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(y=catch_radius, color="gray", ls="--")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Distance")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize 2D PE (low_level) trained models")
    parser.add_argument("--experiment-name", type=str, default=None, help="e.g. pe2d_small_easy")
    parser.add_argument("--all", action="store_true", help="Run for all pe2d_* in checkpoints")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF frame rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    log_dir = os.path.join(PROJECT_ROOT, "visualization", "logs")

    if args.all:
        exp_names = [d for d in os.listdir(ckpt_dir) if d.startswith("pe2d_") and os.path.isdir(os.path.join(ckpt_dir, d))]
        exp_names = [x for x in exp_names if os.path.exists(os.path.join(ckpt_dir, x, "best_model.pth"))]
    elif args.experiment_name:
        if not os.path.exists(os.path.join(ckpt_dir, args.experiment_name, "best_model.pth")):
            print(f"Checkpoint not found: {ckpt_dir}/{args.experiment_name}/best_model.pth")
            return
        exp_names = [args.experiment_name]
    else:
        print("Use --experiment-name <name> or --all")
        return

    for exp_name in sorted(exp_names):
        ckpt_path = os.path.join(ckpt_dir, exp_name, "best_model.pth")
        params = PE2D_ENV_PARAMS.get(exp_name, PE2D_ENV_PARAMS["pe2d_baseline"])
        out_dir = args.out_dir or os.path.join(log_dir, exp_name, "plots")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[{exp_name}] Loading {ckpt_path} ...")
        actor = load_checkpoint(ckpt_path, device)
        env = LowLevelAUV2DPursuitEvasionEnv(seed=args.seed, **params)
        world_size = params["world_size"]
        catch_radius = params["catch_radius"]
        dt = env.dt
        pursuer_v_max = env.pursuer_v_max
        evader_v_max = env.evader_v_max
        pursuer_a_max = env.pursuer_a_max
        evader_a_max = env.evader_a_max

        gifs_dir = os.path.join(out_dir, "gifs")
        if not args.no_gif:
            os.makedirs(gifs_dir, exist_ok=True)

        trajectories = []
        for ep in range(args.episodes):
            traj = run_episode(actor, env, device, deterministic=True)
            trajectories.append(traj)
            plot_episode(
                traj,
                os.path.join(out_dir, f"trajectory_ep{ep+1}.png"),
                ep + 1,
                world_size,
                catch_radius,
                dt=dt,
                pursuer_v_max=pursuer_v_max,
                evader_v_max=evader_v_max,
                pursuer_a_max=pursuer_a_max,
                evader_a_max=evader_a_max,
            )
            print(f"  Ep{ep+1}: steps={traj['steps']}, success={traj['success']}")
            if not args.no_gif:
                save_trajectory_gif(
                    traj,
                    os.path.join(gifs_dir, f"{exp_name}_ep{ep+1}.gif"),
                    world_size,
                    catch_radius,
                    fps=args.gif_fps,
                )

        plot_all_episodes(trajectories, os.path.join(out_dir, "all_trajectories.png"), world_size, catch_radius)
        n_success = sum(1 for t in trajectories if t["success"])
        print(f"[{exp_name}] Done. Success {n_success}/{args.episodes}. Plots -> {out_dir}" + (" | GIFs -> " + gifs_dir if not args.no_gif else ""))
        env.close()

    print("Visualization done.")


if __name__ == "__main__":
    main()
