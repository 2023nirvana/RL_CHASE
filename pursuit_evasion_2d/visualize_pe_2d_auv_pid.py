"""
2D AUV+PID 追逃训练结果可视化
=============================

加载 pe2d_auv_pid 训练得到的 checkpoint，在 PursuitEvasion2DAUVPIDEnv 中跑若干 episode，
生成轨迹图、速度/加速度曲线、距离曲线及可选 GIF。

使用：
  cd UnderwaterPursuitEvasion
  python pursuit_evasion_2d/visualize_pe_2d_auv_pid.py --ckpt checkpoints/pe2d_auv_pid/pe2d_auv_10x10/best.pth --episodes 5
  python pursuit_evasion_2d/visualize_pe_2d_auv_pid.py --ckpt checkpoints/pe2d_auv_pid/pe2d_auv_10x10/best.pth --world-size 5 --episodes 3 --no-gif
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_pe_2d_auv_pid import PursuitEvasion2DAUVPIDEnv
from pursuit_evasion_2d.train_pe_2d_auv_pid import ActorCriticMLP, OBS_DIM, N_ACTIONS


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", OBS_DIM)
    n_actions = ckpt.get("n_actions", N_ACTIONS)
    model = ActorCriticMLP(obs_dim, n_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_episode(model, env: PursuitEvasion2DAUVPIDEnv, device, deterministic=True):
    obs, info = env.reset()
    trajectory = {
        "pursuer_pos": [info["pursuer_pos"].copy()],
        "evader_pos": [info["evader_pos"].copy()],
        "pursuer_vel": [],
        "evader_vel": [],
        "pursuer_acc": [],
        "distances": [info["distance"]],
        "rewards": [],
    }
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            action = torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(F.softmax(logits, -1)).sample().item()
        obs, reward, term, trunc, info = env.step(action)
        state = env.get_state()
        trajectory["pursuer_pos"].append(info["pursuer_pos"].copy())
        trajectory["evader_pos"].append(info["evader_pos"].copy())
        trajectory["pursuer_vel"].append(state["pursuer_vel"].copy())
        trajectory["evader_vel"].append(state["evader_vel"].copy())
        trajectory["pursuer_acc"].append(state["pursuer_acc"].copy())
        trajectory["distances"].append(info["distance"])
        trajectory["rewards"].append(reward)
        done = term or trunc
    trajectory["success"] = info.get("success", False)
    trajectory["steps"] = len(trajectory["pursuer_pos"]) - 1
    return trajectory


def plot_episode(trajectory, save_dir: str, episode_id: int, world_size: float, dt: float,
                 pursuer_v_max: float, evader_v_max: float, pursuer_a_max: float, evader_a_max: float, catch_radius: float):
    p_pos = np.array(trajectory["pursuer_pos"])
    e_pos = np.array(trajectory["evader_pos"])
    p_vel = np.array(trajectory["pursuer_vel"])
    e_vel = np.array(trajectory["evader_vel"])
    p_acc = np.array(trajectory["pursuer_acc"])
    dists = np.array(trajectory["distances"])
    steps_vel = np.arange(len(p_vel))
    steps_dist = np.arange(len(dists))
    success = trajectory.get("success", False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(p_pos[:, 0], p_pos[:, 1], "b-", lw=1.5, label="Pursuer", alpha=0.8)
    ax.plot(e_pos[:, 0], e_pos[:, 1], "r-", lw=1.5, label="Evader", alpha=0.8)
    ax.scatter(p_pos[0, 0], p_pos[0, 1], c="blue", s=80, marker="o", zorder=5)
    ax.scatter(p_pos[-1, 0], p_pos[-1, 1], c="blue", s=80, marker="x", zorder=5)
    ax.scatter(e_pos[0, 0], e_pos[0, 1], c="red", s=80, marker="o", zorder=5)
    ax.scatter(e_pos[-1, 0], e_pos[-1, 1], c="red", s=80, marker="x", zorder=5)
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, edgecolor="k", lw=1.5))
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"AUV+PID Trajectory (Ep{episode_id}) " + ("Caught" if success else "Timeout"))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    p_vn = np.linalg.norm(p_vel, axis=1)
    e_vn = np.linalg.norm(e_vel, axis=1)
    ax.plot(steps_vel, p_vn, "b-", lw=2, label="Pursuer $|v|$", alpha=0.9)
    ax.plot(steps_vel, e_vn, "r-", lw=2, label="Evader $|v|$", alpha=0.9)
    ax.axhline(y=pursuer_v_max, color="blue", ls="--", alpha=0.5, label=f"P $v_{{\\max}}$={pursuer_v_max}")
    ax.axhline(y=evader_v_max, color="red", ls="--", alpha=0.5, label=f"E $v_{{\\max}}$={evader_v_max}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed $|v|$")
    ax.legend(fontsize=8)
    ax.set_title("Speed (hard constraint)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    p_anorm = np.linalg.norm(p_acc, axis=1)
    e_acc_approx = np.zeros(len(e_vel))
    for i in range(1, len(e_vel)):
        e_acc_approx[i] = np.linalg.norm(e_vel[i] - e_vel[i - 1]) / dt
    ax.plot(steps_vel, p_anorm, "b-", lw=2, label="Pursuer $|a|$", alpha=0.9)
    ax.plot(steps_vel, e_acc_approx, "r-", lw=2, label="Evader $|a|$ (approx)", alpha=0.9)
    ax.axhline(y=pursuer_a_max, color="blue", ls="--", alpha=0.5, label=f"P $a_{{\\max}}$={pursuer_a_max}")
    ax.axhline(y=evader_a_max, color="red", ls="--", alpha=0.5, label=f"E $a_{{\\max}}$={evader_a_max}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration $|a|$")
    ax.legend(fontsize=8)
    ax.set_title("Acceleration (hard constraint)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_dist, dists, "green", lw=1.5)
    ax.axhline(y=catch_radius, color="r", ls="--", label="Catch radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Distance (Pursuer–Evader)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"pe2d_auv_pid_ep{episode_id}_traj_vel_acc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


def save_gif(trajectory, save_path: str, world_size: float, fps=20):
    p_pos = np.array(trajectory["pursuer_pos"])
    e_pos = np.array(trajectory["evader_pos"])
    n_frames = len(p_pos)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-world_size, -world_size), 2 * world_size, 2 * world_size, fill=False, edgecolor="k", lw=2))
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6)
    e_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6)
    p_pt, = ax.plot([], [], "bo", markersize=12)
    e_pt, = ax.plot([], [], "ro", markersize=10)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.set_title("PE2D AUV+PID")

    def init():
        p_line.set_data([], [])
        e_line.set_data([], [])
        p_pt.set_data([], [])
        e_pt.set_data([], [])
        txt.set_text("")
        return p_line, e_line, p_pt, e_pt, txt

    def animate(i):
        p_line.set_data(p_pos[: i + 1, 0], p_pos[: i + 1, 1])
        e_line.set_data(e_pos[: i + 1, 0], e_pos[: i + 1, 1])
        p_pt.set_data([p_pos[i, 0]], [p_pos[i, 1]])
        e_pt.set_data([e_pos[i, 0]], [e_pos[i, 1]])
        d = np.linalg.norm(p_pos[i] - e_pos[i])
        txt.set_text(f"Step: {i}\nDist: {d:.2f}")
        return p_line, e_line, p_pt, e_pt, txt

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PE2D AUV+PID trained model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pth or final.pth")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: visualization/logs/pe2d_auv_pid_vis)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--world-size", type=float, default=None, help="World size (default: from env)")
    parser.add_argument("--difficulty", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    world_size = args.world_size if args.world_size is not None else 5.0
    env = PursuitEvasion2DAUVPIDEnv(world_size=world_size, difficulty=args.difficulty, evader_mode="smart", seed=args.seed)
    pursuer_v_max = env.pursuer_v_max
    evader_v_max = env._effective_evader_v_max
    pursuer_a_max = env.pursuer_a_max
    evader_a_max = env._effective_evader_a_max
    catch_radius = env.catch_radius
    dt = env.dt

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", "pe2d_auv_pid_vis")
    os.makedirs(out_dir, exist_ok=True)
    print("AUV+PID visualization: checkpoint={}, world_size={}, episodes={}, out_dir={}".format(
        args.ckpt, world_size, args.episodes, out_dir))
    print("-" * 60)

    success_count = 0
    for ep in range(1, args.episodes + 1):
        trajectory = run_episode(model, env, device, deterministic=True)
        if trajectory["success"]:
            success_count += 1
        plot_episode(trajectory, out_dir, ep, world_size, dt,
                     pursuer_v_max, evader_v_max, pursuer_a_max, evader_a_max, catch_radius)
        if not args.no_gif:
            gif_path = os.path.join(out_dir, f"pe2d_auv_pid_ep{ep}.gif")
            save_gif(trajectory, gif_path, world_size)

    print("-" * 60)
    print("Done. Success rate: {}/{}".format(success_count, args.episodes))
    print("Outputs: {}".format(out_dir))


if __name__ == "__main__":
    main()
