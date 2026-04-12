"""
非对称追逃评估与可视化 - 动力学版（基于 low_level_env 动力学）
============================================================

用于评估在 env_asymmetric_pe_v2_dynamics 上训练的模型（观测 22 维，含加速度）。
与 eval_pe_v2.py 对应简化版；本脚本不修改原文件。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pursuit_evasion_2d.env_asymmetric_pe_v2_dynamics import (
    PursuitEvasion2DAsymmetricEnvV2Dynamics,
    make_easy_env,
    make_medium_env,
    make_hard_env,
    make_expert_env,
    NUM_ACCEL_ACTIONS,
)
from pursuit_evasion_2d.env_asymmetric_pe_v2_pid import PursuitEvasion2DAsymmetricEnvV2PID
from pursuit_evasion_2d.train_pe_v2_dynamics import ActorCriticMLP, OBS_DIM_DYNAMICS


def load_model(checkpoint_path: str, device: torch.device):
    """加载动力学版模型（obs_dim=22）"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    obs_dim = checkpoint.get("obs_dim", OBS_DIM_DYNAMICS)
    if obs_dim != OBS_DIM_DYNAMICS:
        print(f"Warning: checkpoint obs_dim={obs_dim}, using {OBS_DIM_DYNAMICS} for dynamics env")
        obs_dim = OBS_DIM_DYNAMICS

    n_actions = NUM_ACCEL_ACTIONS
    model = ActorCriticMLP(obs_dim, n_actions, hidden_sizes=(256, 128, 64)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded dynamics model from {checkpoint_path} (obs_dim={obs_dim})")
    if "success_rate" in checkpoint:
        print(f"  Training success rate: {checkpoint['success_rate']:.1%}")
    if "difficulty" in checkpoint:
        print(f"  Training difficulty: {checkpoint['difficulty']:.2f}")

    return model


def run_episode(model, env, device, deterministic=True, render_data=False):
    """运行一个回合（动力学环境）"""
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    trajectory = {
        "pursuer_pos": [info["pursuer_pos"].copy()],
        "evader_pos": [info["evader_pos"].copy()],
        "pursuer_vel": [],
        "evader_vel": [],
        "actions": [],
        "rewards": [],
        "distances": [info["distance"]],
    }

    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model.forward(x)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample().item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if render_data:
            trajectory["pursuer_pos"].append(info["pursuer_pos"].copy())
            trajectory["evader_pos"].append(info["evader_pos"].copy())
            trajectory["pursuer_vel"].append(info.get("pursuer_vel", np.zeros(2)).copy())
            trajectory["evader_vel"].append(info.get("evader_vel", np.zeros(2)).copy())
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["distances"].append(info["distance"])

    result = {
        "success": info.get("success", False),
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": info["distance"],
    }

    if render_data:
        result["trajectory"] = trajectory

    return result


def _make_env(difficulty, evader_mode, env_type="dynamics"):
    EnvClass = PursuitEvasion2DAsymmetricEnvV2PID if env_type == "pid" else PursuitEvasion2DAsymmetricEnvV2Dynamics
    return EnvClass(difficulty=difficulty, evader_mode=evader_mode)


def evaluate_model(model, device, n_episodes=50, difficulties=None, env_type="dynamics"):
    """在多种难度下评估模型（env_type: dynamics 或 pid）"""
    if difficulties is None:
        difficulties = [
            ("easy", 0.3, "simple"),
            ("medium", 0.6, "smart"),
            ("hard", 1.0, "smart"),
            ("expert", 1.0, "expert"),
        ]
    label = "PID env" if env_type == "pid" else "Dynamics env"
    results = {}

    for name, difficulty, evader_mode in difficulties:
        print(f"\nEvaluating on {name} (difficulty={difficulty}, evader={evader_mode}) [{label}]...")

        env = _make_env(difficulty, evader_mode, env_type)

        successes = []
        rewards = []
        lengths = []

        for i in range(n_episodes):
            result = run_episode(model, env, device, deterministic=True)
            successes.append(result["success"])
            rewards.append(result["total_reward"])
            lengths.append(result["steps"])

        results[name] = {
            "difficulty": difficulty,
            "evader_mode": evader_mode,
            "success_rate": np.mean(successes),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
        }

        print(f"  Success rate: {results[name]['success_rate']:.1%}")
        print(f"  Mean reward: {results[name]['mean_reward']:.1f} ± {results[name]['std_reward']:.1f}")
        print(f"  Mean length: {results[name]['mean_length']:.0f} ± {results[name]['std_length']:.0f}")

    return results


def visualize_trajectory(trajectory, save_path=None, show=True, title=None):
    """可视化单个追逃轨迹"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    p_pos = np.array(trajectory["pursuer_pos"])
    e_pos = np.array(trajectory["evader_pos"])

    ax.plot(p_pos[:, 0], p_pos[:, 1], "b-", linewidth=1.5, label="Pursuer", alpha=0.7)
    ax.plot(e_pos[:, 0], e_pos[:, 1], "r-", linewidth=1.5, label="Evader", alpha=0.7)
    ax.scatter(p_pos[0, 0], p_pos[0, 1], c="blue", s=100, marker="o", zorder=5, label="P start")
    ax.scatter(p_pos[-1, 0], p_pos[-1, 1], c="blue", s=100, marker="x", zorder=5, label="P end")
    ax.scatter(e_pos[0, 0], e_pos[0, 1], c="red", s=100, marker="o", zorder=5, label="E start")
    ax.scatter(e_pos[-1, 0], e_pos[-1, 1], c="red", s=100, marker="x", zorder=5, label="E end")

    world_size = 10.0
    ax.add_patch(
        Rectangle(
            (-world_size, -world_size), 2 * world_size, 2 * world_size,
            fill=False, edgecolor="black", linewidth=2,
        )
    )
    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trajectory (Dynamics)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    distances = trajectory["distances"]
    ax.plot(distances, "g-", linewidth=1.5)
    ax.axhline(y=0.5, color="r", linestyle="--", label="Catch radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Distance over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    rewards = trajectory["rewards"]
    cumulative_rewards = np.cumsum(rewards)
    ax.plot(cumulative_rewards, "purple", linewidth=1.5, label="Cumulative")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_animation(trajectory, save_path=None, fps=20):
    """创建追逃动画"""
    p_pos = np.array(trajectory["pursuer_pos"])
    e_pos = np.array(trajectory["evader_pos"])
    n_frames = len(p_pos)

    fig, ax = plt.subplots(figsize=(8, 8))
    world_size = 10.0

    ax.add_patch(
        Rectangle(
            (-world_size, -world_size), 2 * world_size, 2 * world_size,
            fill=False, edgecolor="black", linewidth=2,
        )
    )

    pursuer_line, = ax.plot([], [], "b-", linewidth=1.5, alpha=0.5)
    evader_line, = ax.plot([], [], "r-", linewidth=1.5, alpha=0.5)
    pursuer_point, = ax.plot([], [], "bo", markersize=12)
    evader_point, = ax.plot([], [], "ro", markersize=12)

    catch_circle = Circle((0, 0), 0.5, fill=False, edgecolor="green", linestyle="--")
    ax.add_patch(catch_circle)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top")

    ax.set_xlim(-world_size - 1, world_size + 1)
    ax.set_ylim(-world_size - 1, world_size + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Pursuit-Evasion Animation (Dynamics)")

    def init():
        pursuer_line.set_data([], [])
        evader_line.set_data([], [])
        pursuer_point.set_data([], [])
        evader_point.set_data([], [])
        catch_circle.center = (0, 0)
        step_text.set_text("")
        return pursuer_line, evader_line, pursuer_point, evader_point, catch_circle, step_text

    def animate(frame):
        pursuer_line.set_data(p_pos[: frame + 1, 0], p_pos[: frame + 1, 1])
        evader_line.set_data(e_pos[: frame + 1, 0], e_pos[: frame + 1, 1])
        pursuer_point.set_data([p_pos[frame, 0]], [p_pos[frame, 1]])
        evader_point.set_data([e_pos[frame, 0]], [e_pos[frame, 1]])
        catch_circle.center = (p_pos[frame, 0], p_pos[frame, 1])
        dist = np.linalg.norm(p_pos[frame] - e_pos[frame])
        step_text.set_text(f"Step: {frame}/{n_frames - 1}\nDistance: {dist:.2f}")
        return pursuer_line, evader_line, pursuer_point, evader_point, catch_circle, step_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000 / fps, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_report(model, device, save_dir, n_episodes=100, env_type="dynamics"):
    """生成完整评估报告（env_type: dynamics 或 pid）"""
    os.makedirs(save_dir, exist_ok=True)
    label = "PID env" if env_type == "pid" else "Dynamics env"
    env_name = "PursuitEvasion2DAsymmetricEnvV2PID" if env_type == "pid" else "PursuitEvasion2DAsymmetricEnvV2Dynamics"

    print("=" * 60)
    print(f"Generating Evaluation Report ({label})")
    print("=" * 60)

    results = evaluate_model(model, device, n_episodes=n_episodes, env_type=env_type)

    report = {
        "timestamp": datetime.now().isoformat(),
        "env": env_name,
        "obs_dim": OBS_DIM_DYNAMICS,
        "n_episodes": n_episodes,
        "results": results,
    }

    with open(os.path.join(save_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    names = list(results.keys())
    success_rates = [results[n]["success_rate"] for n in names]
    mean_rewards = [results[n]["mean_reward"] for n in names]
    mean_lengths = [results[n]["mean_length"] for n in names]

    axes[0].bar(names, success_rates, color=["green", "yellow", "orange", "red"])
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title(f"Success Rate by Difficulty ({label})")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center")

    axes[1].bar(names, mean_rewards, color=["green", "yellow", "orange", "red"])
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Mean Reward by Difficulty")

    axes[2].bar(names, mean_lengths, color=["green", "yellow", "orange", "red"])
    axes[2].set_ylabel("Mean Episode Length")
    axes[2].set_title("Mean Length by Difficulty")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluation_summary.png"), dpi=150)
    plt.close()

    print("\nGenerating example trajectories (PNG + GIF)...")

    for name, difficulty, evader_mode in [("easy", 0.3, "simple"), ("hard", 1.0, "smart")]:
        env = _make_env(difficulty, evader_mode, env_type)

        for attempt in range(20):
            result = run_episode(model, env, device, deterministic=True, render_data=True)
            if result["success"]:
                visualize_trajectory(
                    result["trajectory"],
                    save_path=os.path.join(save_dir, f"trajectory_{name}_success.png"),
                    show=False,
                    title=f"Success ({name}, {label}): {result['steps']} steps",
                )
                create_animation(
                    result["trajectory"],
                    save_path=os.path.join(save_dir, f"trajectory_{name}_success.gif"),
                    fps=20,
                )
                break

        for attempt in range(20):
            result = run_episode(model, env, device, deterministic=True, render_data=True)
            if not result["success"]:
                visualize_trajectory(
                    result["trajectory"],
                    save_path=os.path.join(save_dir, f"trajectory_{name}_failure.png"),
                    show=False,
                    title=f"Failure ({name}, {label}): dist={result['final_distance']:.2f}",
                )
                create_animation(
                    result["trajectory"],
                    save_path=os.path.join(save_dir, f"trajectory_{name}_failure.gif"),
                    fps=20,
                )
                break

    print(f"\nReport saved to {save_dir} ({label})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PE2D V2 model (dynamics or PID env)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--env", type=str, default="dynamics", choices=["dynamics", "pid"],
                        help="Evaluate on this env (must match training env for correct behavior)")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of episodes per difficulty")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--visualize", action="store_true", help="Show trajectory visualization")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--difficulty", type=float, default=0.6, help="Difficulty for visualization")
    parser.add_argument("--evader-mode", type=str, default="smart", help="Evader mode for visualization")

    args = parser.parse_args()
    env_type = args.env.lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluation env: {env_type}")

    model = load_model(args.checkpoint, device)

    if args.save_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.save_dir = os.path.join(checkpoint_dir, "evaluation")

    if args.visualize or args.animate:
        env = _make_env(args.difficulty, args.evader_mode, env_type)
        result = run_episode(model, env, device, deterministic=True, render_data=True)

        print(f"\nEpisode result ({env_type} env):")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Total reward: {result['total_reward']:.1f}")

        if args.visualize:
            visualize_trajectory(
                result["trajectory"],
                title=f"{'Success' if result['success'] else 'Failure'} - {result['steps']} steps ({env_type})",
            )

        if args.animate:
            os.makedirs(args.save_dir, exist_ok=True)
            create_animation(
                result["trajectory"],
                save_path=os.path.join(args.save_dir, "animation.gif"),
            )
    else:
        generate_report(model, device, args.save_dir, n_episodes=args.n_episodes, env_type=env_type)


if __name__ == "__main__":
    main()
