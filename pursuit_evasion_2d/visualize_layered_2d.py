"""
2D 分层训练结果可视化：Stage1 追击者 + Stage2 追逃
====================================================

生成：轨迹图、速度曲线、加速度曲线、GIF 动画。
Stage1：追击者追移动目标；Stage2：追逃博弈。

使用：
  python pursuit_evasion_2d/visualize_layered_2d.py --both --episodes 3
  python pursuit_evasion_2d/visualize_layered_2d.py --stage1 --stage1-ckpt checkpoints/pursuer_stage1_dynamics/pursuer_const/best.pth
  python pursuit_evasion_2d/visualize_layered_2d.py --stage2 --stage2-ckpt checkpoints/pe2d_stage2_dynamics_accel/pe_from_stage1/best.pth
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
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_pursuer_low_level_dynamics import (
    PursuerLowLevelDynamicsEnv,
    make_pursuer_env,
    NUM_PURSUER_ACCEL_ACTIONS,
)
from pursuit_evasion_2d.env_asymmetric_pe_v2_dynamics_accel import (
    PursuitEvasion2DAsymmetricEnvV2DynamicsAccel,
)
from pursuit_evasion_2d.train_pursuer_stage1_dynamics import ActorCriticMLP as Stage1MLP
from pursuit_evasion_2d.train_pe_stage2_dynamics import ActorCriticMLP as Stage2MLP

OBS_DIM_STAGE1 = 18
OBS_DIM_STAGE2 = 22
WORLD_SIZE = 10.0
# 物理设定：追击者速度劣势、加速度优势（与 pe_2d_dynamics 一致）
PURSUER_V_MAX = 1.2
PURSUER_A_MAX = 2.5
EVADER_V_MAX = 1.4
EVADER_A_MAX = 0.6
DT = 0.05


def load_stage1_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", OBS_DIM_STAGE1)
    n_actions = ckpt.get("n_actions", NUM_PURSUER_ACCEL_ACTIONS)
    model = Stage1MLP(obs_dim, n_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_stage2_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", OBS_DIM_STAGE2)
    n_actions = ckpt.get("n_actions", 9)
    model = Stage2MLP(obs_dim, n_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_episode_stage1(model, env: PursuerLowLevelDynamicsEnv, device, deterministic=True):
    """跑一局 Stage1，记录 pos/vel/acc（从 env 内部读取）"""
    obs, info = env.reset()
    trajectory = {
        "pursuer_pos": [env._pos.copy()],
        "target_pos": [env._target_pos.copy()],
        "pursuer_vel": [env._vel.copy()],
        "pursuer_acc": [env._acc.copy()],
        "target_vel": [env._target_vel.copy()],
        "distances": [float(np.linalg.norm(env._target_pos - env._pos))],
        "rewards": [],
    }
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            action = torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(F.softmax(logits, -1)).sample().item()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        trajectory["pursuer_pos"].append(env._pos.copy())
        trajectory["target_pos"].append(env._target_pos.copy())
        trajectory["pursuer_vel"].append(env._vel.copy())
        trajectory["pursuer_acc"].append(env._acc.copy())
        trajectory["target_vel"].append(env._target_vel.copy())
        trajectory["distances"].append(info["distance"])
        trajectory["rewards"].append(reward)
    trajectory["success"] = info.get("success", False)
    trajectory["steps"] = len(trajectory["pursuer_pos"]) - 1
    return trajectory


def run_episode_stage2(model, env: PursuitEvasion2DAsymmetricEnvV2DynamicsAccel, device, deterministic=True):
    """跑一局 Stage2，记录 pos/vel/acc"""
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


def plot_stage1_episode(trajectory, save_dir: str, episode_id: int):
    """Stage1：轨迹、速度、加速度、GIF"""
    p_pos = np.array(trajectory["pursuer_pos"])
    t_pos = np.array(trajectory["target_pos"])
    p_vel = np.array(trajectory["pursuer_vel"])
    t_vel = np.array(trajectory["target_vel"])
    p_acc = np.array(trajectory["pursuer_acc"])
    dists = np.array(trajectory["distances"])
    steps = np.arange(len(p_pos))
    success = trajectory.get("success", False)
    p_vnorm = np.linalg.norm(p_vel, axis=1)
    t_vnorm = np.linalg.norm(t_vel, axis=1)
    p_anorm = np.linalg.norm(p_acc, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(p_pos[:, 0], p_pos[:, 1], "b-", lw=1.5, label="Pursuer", alpha=0.8)
    ax.plot(t_pos[:, 0], t_pos[:, 1], "r-", lw=1.5, label="Target", alpha=0.8)
    ax.scatter(p_pos[0, 0], p_pos[0, 1], c="blue", s=80, marker="o", zorder=5)
    ax.scatter(p_pos[-1, 0], p_pos[-1, 1], c="blue", s=80, marker="x", zorder=5)
    ax.scatter(t_pos[0, 0], t_pos[0, 1], c="red", s=80, marker="o", zorder=5)
    ax.scatter(t_pos[-1, 0], t_pos[-1, 1], c="red", s=80, marker="x", zorder=5)
    ax.add_patch(Rectangle((-WORLD_SIZE, -WORLD_SIZE), 2 * WORLD_SIZE, 2 * WORLD_SIZE, fill=False, edgecolor="k", lw=1.5))
    ax.set_xlim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_ylim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Stage1 Trajectory (Ep{episode_id}) " + ("Success" if success else "Fail"))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, p_vnorm, "b-", lw=2, label="Pursuer $|v|$", alpha=0.9)
    ax.plot(steps, t_vnorm, "r-", lw=2, label="Target $|v|$", alpha=0.9)
    ax.axhline(y=1.2, color="blue", ls="--", alpha=0.5, label="Pursuer $v_{max}$=1.2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed $|v|$")
    ax.legend()
    ax.set_title("Speed (absolute value)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, p_anorm, "b-", lw=2, label="Pursuer $|a|$", alpha=0.9)
    ax.axhline(y=2.5, color="blue", ls="--", alpha=0.5, label="Pursuer $a_{max}$=2.5")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration $|a|$")
    ax.legend()
    ax.set_title("Acceleration (absolute value)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, dists, "green", lw=1.5)
    ax.axhline(y=0.5, color="r", ls="--", label="Catch radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Distance to Target")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"stage1_ep{episode_id}_traj_vel_acc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    gif_path = os.path.join(save_dir, f"stage1_ep{episode_id}.gif")
    _save_stage1_gif(trajectory, gif_path)
    print(f"  Saved {gif_path}")


def _save_stage1_gif(trajectory, save_path, fps=20):
    p_pos = np.array(trajectory["pursuer_pos"])
    t_pos = np.array(trajectory["target_pos"])
    n_frames = len(p_pos)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-WORLD_SIZE, -WORLD_SIZE), 2 * WORLD_SIZE, 2 * WORLD_SIZE, fill=False, edgecolor="k", lw=2))
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6)
    t_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6)
    p_pt, = ax.plot([], [], "bo", markersize=14)
    t_pt, = ax.plot([], [], "ro", markersize=10)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_ylim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_aspect("equal")
    ax.set_title("Stage1: Pursuer vs Target")

    def init():
        p_line.set_data([], [])
        t_line.set_data([], [])
        p_pt.set_data([], [])
        t_pt.set_data([], [])
        txt.set_text("")
        return p_line, t_line, p_pt, t_pt, txt

    def animate(i):
        p_line.set_data(p_pos[: i + 1, 0], p_pos[: i + 1, 1])
        t_line.set_data(t_pos[: i + 1, 0], t_pos[: i + 1, 1])
        p_pt.set_data([p_pos[i, 0]], [p_pos[i, 1]])
        t_pt.set_data([t_pos[i, 0]], [t_pos[i, 1]])
        d = np.linalg.norm(p_pos[i] - t_pos[i])
        txt.set_text(f"Step: {i}\nDist: {d:.2f}")
        return p_line, t_line, p_pt, t_pt, txt

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()


def plot_stage2_episode(trajectory, save_dir: str, episode_id: int):
    """Stage2：轨迹、速度、加速度、GIF"""
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
    ax.add_patch(Rectangle((-WORLD_SIZE, -WORLD_SIZE), 2 * WORLD_SIZE, 2 * WORLD_SIZE, fill=False, edgecolor="k", lw=1.5))
    ax.set_xlim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_ylim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Stage2 Trajectory (Ep{episode_id}) " + ("Caught" if success else "Timeout"))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    p_vn = np.linalg.norm(p_vel, axis=1)
    e_vn = np.linalg.norm(e_vel, axis=1)
    ax.plot(steps_vel, p_vn, "b-", lw=2, label="Pursuer $|v|$", alpha=0.9)
    ax.plot(steps_vel, e_vn, "r-", lw=2, label="Evader $|v|$", alpha=0.9)
    ax.axhline(y=PURSUER_V_MAX, color="blue", ls="--", alpha=0.5, label=f"P $v_{{\\max}}$={PURSUER_V_MAX}")
    ax.axhline(y=EVADER_V_MAX, color="red", ls="--", alpha=0.5, label=f"E $v_{{\\max}}$={EVADER_V_MAX}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed $|v|$")
    ax.legend(fontsize=8)
    ax.set_title("Speed (absolute): P slower, E faster")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    p_anorm = np.linalg.norm(p_acc, axis=1)
    e_acc_approx = np.zeros(len(e_vel))
    for i in range(1, len(e_vel)):
        e_acc_approx[i] = np.linalg.norm(e_vel[i] - e_vel[i - 1]) / DT
    ax.plot(steps_vel, p_anorm, "b-", lw=2, label="Pursuer $|a|$", alpha=0.9)
    ax.plot(steps_vel, e_acc_approx, "r-", lw=2, label="Evader $|a|$ (approx)", alpha=0.9)
    ax.axhline(y=PURSUER_A_MAX, color="blue", ls="--", alpha=0.5, label=f"P $a_{{\\max}}$={PURSUER_A_MAX}")
    ax.axhline(y=EVADER_A_MAX, color="red", ls="--", alpha=0.5, label=f"E $a_{{\\max}}$={EVADER_A_MAX}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration $|a|$")
    ax.legend(fontsize=8)
    ax.set_title("Acceleration (absolute): P higher, E lower")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_dist, dists, "green", lw=1.5)
    ax.axhline(y=0.5, color="r", ls="--", label="Catch radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Distance (Pursuer–Evader)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"stage2_ep{episode_id}_traj_vel_acc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    gif_path = os.path.join(save_dir, f"stage2_ep{episode_id}.gif")
    _save_stage2_gif(trajectory, gif_path)
    print(f"  Saved {gif_path}")


def _save_stage2_gif(trajectory, save_path, fps=20):
    p_pos = np.array(trajectory["pursuer_pos"])
    e_pos = np.array(trajectory["evader_pos"])
    n_frames = len(p_pos)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((-WORLD_SIZE, -WORLD_SIZE), 2 * WORLD_SIZE, 2 * WORLD_SIZE, fill=False, edgecolor="k", lw=2))
    catch = Circle((0, 0), 0.5, fill=False, edgecolor="green", ls="--")
    ax.add_patch(catch)
    p_line, = ax.plot([], [], "b-", lw=1.5, alpha=0.6)
    e_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.6)
    p_pt, = ax.plot([], [], "bo", markersize=14)
    e_pt, = ax.plot([], [], "ro", markersize=10)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_ylim(-WORLD_SIZE - 1, WORLD_SIZE + 1)
    ax.set_aspect("equal")
    ax.set_title("Stage2: Pursuit-Evasion")

    def init():
        p_line.set_data([], [])
        e_line.set_data([], [])
        p_pt.set_data([], [])
        e_pt.set_data([], [])
        catch.center = (0, 0)
        txt.set_text("")
        return p_line, e_line, p_pt, e_pt, catch, txt

    def animate(i):
        p_line.set_data(p_pos[: i + 1, 0], p_pos[: i + 1, 1])
        e_line.set_data(e_pos[: i + 1, 0], e_pos[: i + 1, 1])
        p_pt.set_data([p_pos[i, 0]], [p_pos[i, 1]])
        e_pt.set_data([e_pos[i, 0]], [e_pos[i, 1]])
        catch.center = (p_pos[i, 0], p_pos[i, 1])
        d = np.linalg.norm(p_pos[i] - e_pos[i])
        txt.set_text(f"Step: {i}\nDist: {d:.2f}")
        return p_line, e_line, p_pt, e_pt, catch, txt

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize 2D layered training (Stage1 & Stage2)")
    parser.add_argument("--stage1", action="store_true", help="Visualize Stage1 only")
    parser.add_argument("--stage2", action="store_true", help="Visualize Stage2 only")
    parser.add_argument("--both", action="store_true", help="Visualize both (default)")
    parser.add_argument("--stage1-ckpt", type=str, default="checkpoints/pursuer_stage1_dynamics/pursuer_const/best.pth")
    parser.add_argument("--stage2-ckpt", type=str, default="checkpoints/pe2d_stage2_dynamics_accel/pe_from_stage1/best.pth")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize per stage")
    parser.add_argument("--out-dir", type=str, default="visualization/logs/2d_layered_visualization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.stage1 and not args.stage2:
        args.both = True
    if args.both:
        args.stage1 = args.stage2 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.stage1:
        if not os.path.isfile(args.stage1_ckpt):
            print(f"Stage1 checkpoint not found: {args.stage1_ckpt}")
        else:
            print("=" * 60)
            print("Stage1: Pursuer vs Moving Target")
            print("=" * 60)
            model = load_stage1_model(args.stage1_ckpt, device)
            env = make_pursuer_env(target_motion="constant", target_speed=0.5, seed=args.seed)
            stage1_dir = os.path.join(args.out_dir, "stage1")
            os.makedirs(stage1_dir, exist_ok=True)
            for ep in range(args.episodes):
                env.reset(seed=args.seed + ep + 100)
                traj = run_episode_stage1(model, env, device)
                print(f"  Episode {ep + 1}: steps={traj['steps']}, success={traj['success']}")
                plot_stage1_episode(traj, stage1_dir, ep + 1)

    if args.stage2:
        if not os.path.isfile(args.stage2_ckpt):
            print(f"Stage2 checkpoint not found: {args.stage2_ckpt}")
        else:
            print("=" * 60)
            print("Stage2: Pursuit-Evasion")
            print("=" * 60)
            model = load_stage2_model(args.stage2_ckpt, device)
            env = PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(difficulty=0.6, evader_mode="smart", seed=args.seed)
            stage2_dir = os.path.join(args.out_dir, "stage2")
            os.makedirs(stage2_dir, exist_ok=True)
            for ep in range(args.episodes):
                env.reset(seed=args.seed + ep + 200)
                traj = run_episode_stage2(model, env, device)
                print(f"  Episode {ep + 1}: steps={traj['steps']}, success={traj['success']}")
                plot_stage2_episode(traj, stage2_dir, ep + 1)

    print("\nDone. Outputs under:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
