"""
3D 模型轨迹与 GIF 可视化 - 类比 2D 效果，带航向标注与三维轨迹表现

为 3d_baseline / 3d_combined / 3d_long_steps 等生成：
- 三维轨迹图（真实 3D 立体效果 + 航向箭头）
- 轨迹 GIF 动画（过程中标注航向）

使用：
  python evaluation/eval_3d_trajectory_visualize.py --models 3d_combined 3d_long_steps --episodes 5 --save-gif
  python evaluation/eval_3d_trajectory_visualize.py --all --save-gif
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from core.environment.low_level_env_3d import LowLevelAUV3DEnv

MODEL_CONFIGS = {
    "3d_baseline": {"checkpoint": "checkpoints/3d_baseline/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D Baseline"},
    "3d_combined": {"checkpoint": "checkpoints/3d_combined/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D Combined"},
    "3d_long_steps": {"checkpoint": "checkpoints/3d_long_steps/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D Long Steps"},
    "3d_large_net": {"checkpoint": "checkpoints/3d_large_net/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D Large Net"},
    "3d_deep_net": {"checkpoint": "checkpoints/3d_deep_net/best_model.pth", "hidden_sizes": [128, 128, 64, 32], "target_distance": 5.0, "description": "3D Deep Net"},
    "3d_high_lr": {"checkpoint": "checkpoints/3d_high_lr/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D High LR"},
    "3d_low_clip": {"checkpoint": "checkpoints/3d_low_clip/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D Low Clip"},
    "3d_high_ent": {"checkpoint": "checkpoints/3d_high_ent/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D High Ent"},
    "3d_more_envs": {"checkpoint": "checkpoints/3d_more_envs/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D More Envs"},
    "3d_large_batch": {"checkpoint": "checkpoints/3d_large_batch/best_model.pth", "hidden_sizes": [128, 64], "target_distance": 5.0, "description": "3D Large Batch"},
    # 朝向消融 8 组（与 train_3d_heading_ablation.py 一致，训练完可生成轨迹图+GIF）
    "3d_habl_h03": {"checkpoint": "checkpoints/3d_habl_h03/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.3"},
    "3d_habl_h05": {"checkpoint": "checkpoints/3d_habl_h05/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.5"},
    "3d_habl_h07": {"checkpoint": "checkpoints/3d_habl_h07/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.7"},
    "3d_habl_h06_d08": {"checkpoint": "checkpoints/3d_habl_h06_d08/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.6 d=0.8"},
    "3d_habl_h05_strict": {"checkpoint": "checkpoints/3d_habl_h05_strict/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.5 strict"},
    "3d_habl_h07_strict": {"checkpoint": "checkpoints/3d_habl_h07_strict/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.7 strict"},
    "3d_habl_h05_penalty": {"checkpoint": "checkpoints/3d_habl_h05_penalty/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.5 penalty"},
    "3d_habl_h07_penalty": {"checkpoint": "checkpoints/3d_habl_h07_penalty/best_model.pth", "hidden_sizes": [256, 128, 64], "target_distance": 5.0, "description": "3D HABL h=0.7 penalty"},
}


def build_actor(obs_dim, num_actions, hidden_sizes, device):
    layers = []
    prev = obs_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, num_actions))
    return nn.Sequential(*layers).to(device)


class Evaluator3D:
    def __init__(self, checkpoint_path, hidden_sizes, target_distance=5.0, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = LowLevelAUV3DEnv(target_distance=target_distance)
        self.obs_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.actor = build_actor(self.obs_dim, self.num_actions, hidden_sizes, self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"], strict=True)
        self.actor.eval()

    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            logits = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1).item() if deterministic else torch.distributions.Categorical(probs).sample().item()

    def run_episode(self, verbose=False, max_steps=None):
        obs, info = self.env.reset()
        trajectory = {"positions": [], "z": [], "yaw_angles": [], "target": info["target_pos"].copy(), "actions": [], "rewards": []}
        eta = self.env.inner.dynamics.eta
        trajectory["positions"].append(eta[:2].tolist())
        trajectory["z"].append(float(eta[2]))
        trajectory["yaw_angles"].append(float(eta[5]))
        done, total_reward, step = False, 0, 0
        while not done and (max_steps is None or step < max_steps):
            action = self.get_action(obs)
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            total_reward += reward
            step += 1
            eta = self.env.inner.dynamics.eta
            trajectory["positions"].append(eta[:2].tolist())
            trajectory["z"].append(float(eta[2]))
            trajectory["yaw_angles"].append(float(eta[5]))
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            if verbose and step % 100 == 0:
                print("    Step %d: pos_error=%.3fm" % (step, info.get("pos_error", 0)))
        success = info.get("termination_reason") == "success"
        return {"success": success, "steps": step, "total_reward": total_reward, "final_pos_error": info.get("pos_error", 0), "trajectory": trajectory}

    def evaluate(self, n_episodes=5, verbose=True, max_steps=None):
        results = []
        for ep in range(n_episodes):
            if verbose:
                print("  Episode %d/%d ..." % (ep + 1, n_episodes), end=" ", flush=True)
            r = self.run_episode(verbose=False, max_steps=max_steps)
            results.append(r)
            if verbose:
                print("OK" if r["success"] else "Fail", "steps=%d, reward=%.1f" % (r["steps"], r["total_reward"]))
        return results


def _get_xyz(trajectory):
    pos = np.array(trajectory["positions"])
    z = np.array(trajectory["z"])
    return pos[:, 0], pos[:, 1], z


def plot_trajectory_3d(result, save_path=None, title_prefix="", arrow_interval=None, arrow_scale=0.4):
    traj = result["trajectory"]
    x, y, z = _get_xyz(traj)
    yaw = np.array(traj["yaw_angles"])
    target = np.array(traj["target"])
    success = result["success"]
    n_pts = len(x)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_pts - 1))
    for i in range(n_pts - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], lw=2.5, alpha=0.9)
    ax.scatter([x[0]], [y[0]], [z[0]], c="green", s=120, marker="o", label="Start", edgecolors="darkgreen", linewidths=2)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c="blue", s=100, marker="s", label="End", edgecolors="navy", linewidths=2)
    ax.scatter([target[0]], [target[1]], [target[2]], c="red", s=200, marker="*", label="Target", edgecolors="darkred", linewidths=1)
    interval = arrow_interval if arrow_interval is not None else max(1, n_pts // 14)
    for i in range(0, n_pts, interval):
        u = arrow_scale * np.cos(yaw[i])
        v = arrow_scale * np.sin(yaw[i])
        ax.quiver(x[i], y[i], z[i], u, v, 0.0, color="orange", arrow_length_ratio=0.25, linewidth=1.5, alpha=0.85)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("%s3D Trajectory | 航向(heading) 橙色箭头 - %s" % (title_prefix, "Success" if success else "Fail"))
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="orange", lw=3, label="航向 (heading)"))
    labels.append("航向 (heading)")
    ax.legend(handles, labels, loc="upper left")
    margin = 0.8
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(z.min() - margin, z.max() + margin)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved:", save_path)
    return fig


def plot_trajectory_3d_panel(result, save_path=None, title_prefix=""):
    traj = result["trajectory"]
    x, y, z = _get_xyz(traj)
    yaw = np.array(traj["yaw_angles"])
    target = np.array(traj["target"])
    success = result["success"]
    n_pts = len(x)
    arrow_interval = max(1, n_pts // 12)
    al = 0.35
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_pts - 1))
    for i in range(n_pts - 1):
        ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], lw=2, alpha=0.9)
    ax1.scatter([x[0]], [y[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c="blue", s=80, marker="s", label="End")
    ax1.scatter([target[0]], [target[1]], [target[2]], c="red", s=150, marker="*", label="Target")
    for i in range(0, n_pts, arrow_interval):
        ax1.quiver(x[i], y[i], z[i], al*np.cos(yaw[i]), al*np.sin(yaw[i]), 0, color="orange", arrow_length_ratio=0.2, linewidth=1.2)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
    ax1.set_title("3D (heading) - %s" % ("Success" if success else "Fail"))
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y, "b-", lw=2, alpha=0.7)
    ax2.plot(x[0], y[0], "go", markersize=12, label="Start")
    ax2.plot(x[-1], y[-1], "bs", markersize=10, label="End")
    ax2.plot(target[0], target[1], "r*", markersize=14, label="Target")
    for i in range(0, n_pts, arrow_interval):
        dx, dy = al * np.cos(yaw[i]), al * np.sin(yaw[i])
        ax2.arrow(x[i], y[i], dx, dy, head_width=0.08, head_length=0.04, fc="orange", ec="orange", alpha=0.7)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("XY (heading)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    ax3 = fig.add_subplot(133)
    ax3.plot(range(len(z)), z, "b-", lw=2, label="AUV depth")
    ax3.axhline(y=target[2], color="r", linestyle="--", label="Target depth")
    ax3.set_xlabel("Step"); ax3.set_ylabel("Z (m)")
    ax3.set_title("Depth over Time")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.suptitle("%s3D Trajectory with Heading - %s" % (title_prefix, "Success" if success else "Fail"), fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved:", save_path)
    return fig


def save_trajectory_gif_3d(result, save_path, title_prefix="", fps=12, every_n=3):
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
    except ImportError:
        print("  matplotlib.animation not available, skip GIF")
        return
    traj = result["trajectory"]
    x, y, z = _get_xyz(traj)
    yaw = np.array(traj["yaw_angles"])
    target = np.array(traj["target"])
    success = result["success"]
    n_pts = len(x)
    indices = list(range(0, n_pts, every_n))
    if indices[-1] != n_pts - 1:
        indices.append(n_pts - 1)
    margin = 1.0
    arrow_scale = 0.4
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(z.min() - margin, z.max() + margin)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.scatter([target[0]], [target[1]], [target[2]], c="red", s=150, marker="*", label="Target")
    line, = ax.plot([], [], [], "b-", lw=2, alpha=0.8)
    point, = ax.plot([], [], [], "go", markersize=12)
    title_text = ax.set_title("")
    qv_holder = [None]

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        title_text.set_text("")
        return line, point, title_text

    def animate(i):
        idx = indices[min(i, len(indices) - 1)]
        line.set_data(x[:idx+1], y[:idx+1])
        line.set_3d_properties(z[:idx+1])
        point.set_data([x[idx]], [y[idx]])
        point.set_3d_properties([z[idx]])
        if qv_holder[0] is not None:
            try:
                qv_holder[0].remove()
            except Exception:
                pass
        u = arrow_scale * np.cos(yaw[idx])
        v = arrow_scale * np.sin(yaw[idx])
        qv_holder[0] = ax.quiver(x[idx], y[idx], z[idx], u, v, 0.0, color="orange", arrow_length_ratio=0.25, linewidth=2)
        title_text.set_text("%sStep %d/%d | heading | %s" % (title_prefix, idx, n_pts - 1, "Success" if success else "Fail"))
        return line, point, title_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices), interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print("  GIF saved:", save_path)


def main():
    parser = argparse.ArgumentParser(description="3D trajectory and GIF visualization with heading")
    parser.add_argument("--models", type=str, nargs="+", help="Model names, e.g. 3d_combined 3d_long_steps")
    parser.add_argument("--all", action="store_true", help="Run all 3D models that have checkpoints")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--max-plots", type=int, default=3)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps per episode for faster viz (e.g. 800)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    if args.models:
        model_names = [m for m in args.models if m in MODEL_CONFIGS]
    elif args.all:
        model_names = list(MODEL_CONFIGS.keys())
    else:
        model_names = ["3d_combined", "3d_long_steps", "3d_baseline"]
    existing = [name for name in model_names if os.path.exists(os.path.join(PROJECT_ROOT, MODEL_CONFIGS[name]["checkpoint"]))]
    for name in model_names:
        if name not in existing and args.verbose:
            print("[SKIP] %s: checkpoint not found" % name)
    if not existing:
        print("No models to run.")
        return
    for name in existing:
        cfg = MODEL_CONFIGS[name]
        ckpt_path = os.path.join(PROJECT_ROOT, cfg["checkpoint"])
        out_base = args.output_dir or os.path.join(PROJECT_ROOT, "visualization", "logs", name, "plots")
        os.makedirs(out_base, exist_ok=True)
        print("\n>>> %s (%s)" % (cfg["description"], name))
        evaluator = Evaluator3D(ckpt_path, cfg["hidden_sizes"], target_distance=cfg["target_distance"])
        results = evaluator.evaluate(n_episodes=args.episodes, verbose=args.verbose, max_steps=args.max_steps)
        prefix = "[%s] " % name
        if args.save_plots:
            for i, r in enumerate(results[: args.max_plots]):
                path_3d = os.path.join(out_base, "trajectory_3d_ep%d.png" % (i + 1))
                plot_trajectory_3d(r, path_3d, title_prefix=prefix)
                path_panel = os.path.join(out_base, "trajectory_3d_panel_ep%d.png" % (i + 1))
                plot_trajectory_3d_panel(r, path_panel, title_prefix=prefix)
        if args.save_gif:
            gif_dir = os.path.join(out_base, "gifs")
            os.makedirs(gif_dir, exist_ok=True)
            for i, r in enumerate(results[: args.max_plots]):
                gif_path = os.path.join(gif_dir, "%s_trajectory_3d_ep%d.gif" % (name, i + 1))
                save_trajectory_gif_3d(r, gif_path, title_prefix=prefix, fps=args.gif_fps)
    print("\nDone. Outputs under visualization/logs/<model>/plots/ (and gifs/ if --save-gif).")


if __name__ == "__main__":
    main()
