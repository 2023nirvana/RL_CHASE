#!/usr/bin/env python3
"""V3 增强版可视化：轨迹GIF(含速度矢量+航向) + 多维统计面板 + 跨实验对比。

用法：
    # 单实验
    python low_near/visualize_v3_enhanced.py --exp-dir low_near/v4_tight_baseline
    # 多实验对比
    python low_near/visualize_v3_enhanced.py --compare \
        low_near/v4_tight_baseline \
        low_near/v4_tight_vel015_v2 \
        low_near/v4_tight_vel06_v2
"""
import os, sys, argparse, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib import cm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous
from training.train_2d_v3_continuous import GaussianActorCritic


# ──────────────────── 工具函数 ────────────────────

def load_model(exp_dir, checkpoint_name="best"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(exp_dir, "checkpoints", f"{checkpoint_name}.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden = ckpt.get("config", {}).get("network", {}).get(
        "feature_extractor", {}
    ).get("hidden_sizes", [128, 64])
    model = GaussianActorCritic(obs_dim, act_dim, hidden=tuple(hidden)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def rollout(model, env, device, deterministic=True):
    """跑一轮 episode，返回丰富的轨迹数据。"""
    obs, info = env.reset()
    data = {
        "pos": [env.inner.dynamics.eta[:2].copy()],
        "yaw": [float(env.inner.dynamics.eta[5])],
        "vel_surge": [float(env.inner.dynamics.nu[0])],
        "vel_sway": [float(env.inner.dynamics.nu[1])],
        "yaw_rate": [float(env.inner.dynamics.nu[5])],
        "speed": [float(np.linalg.norm(env.inner.dynamics.nu[:2]))],
        "actions": [],
        "errors": [],
        "headings": [],
        "rewards": [],
        "target": np.array(info["target_pos"]),
    }
    done = False
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(x, deterministic=deterministic)
        a_np = action[0].cpu().numpy()
        obs, reward, term, trunc, info = env.step(a_np)
        eta = env.inner.dynamics.eta
        nu = env.inner.dynamics.nu
        data["pos"].append(eta[:2].copy())
        data["yaw"].append(float(eta[5]))
        data["vel_surge"].append(float(nu[0]))
        data["vel_sway"].append(float(nu[1]))
        data["yaw_rate"].append(float(nu[5]))
        data["speed"].append(float(np.linalg.norm(nu[:2])))
        data["actions"].append(a_np.copy())
        data["errors"].append(info.get("pos_error", float("nan")))
        data["headings"].append(info.get("heading_error", float("nan")))
        data["rewards"].append(reward)
        done = term or trunc

    data["pos"] = np.array(data["pos"])
    data["actions"] = np.array(data["actions"]) if data["actions"] else np.zeros((0, 3))
    for k in ("yaw", "vel_surge", "vel_sway", "yaw_rate", "speed", "errors", "headings", "rewards"):
        data[k] = np.array(data[k])
    data["success"] = info.get("termination_reason") == "success"
    data["reason"] = info.get("termination_reason", "unknown")
    data["steps"] = len(data["errors"])
    return data


# ──────────────────── 增强版轨迹 GIF ────────────────────

def save_enhanced_gif(ep, out_path, fps=20, reach_threshold=0.05):
    """带速度矢量、航向箭头、速度色带、误差曲线的轨迹动画。"""
    traj = ep["pos"]
    target = ep["target"]
    n = len(traj)

    fig = plt.figure(figsize=(14, 6))
    ax_traj = fig.add_axes([0.04, 0.08, 0.42, 0.84])
    ax_err  = fig.add_axes([0.54, 0.55, 0.42, 0.38])
    ax_vel  = fig.add_axes([0.54, 0.08, 0.42, 0.38])

    # --- 轨迹面板 ---
    margin = 0.25
    all_pts = np.vstack([traj, target.reshape(1, 2)])
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    ax_traj.set_xlim(lo[0], hi[0])
    ax_traj.set_ylim(lo[1], hi[1])
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.2)
    ax_traj.set_title("Trajectory + Velocity", fontsize=11)

    # 到达阈值圈
    circle = Circle(target, reach_threshold, fill=False, ls="--", color="red", alpha=0.5, lw=1.5)
    ax_traj.add_patch(circle)
    ax_traj.plot(target[0], target[1], "r*", ms=16, zorder=10)
    # 起点
    ax_traj.plot(traj[0, 0], traj[0, 1], "gs", ms=10, zorder=10, label="Start")

    trail, = ax_traj.plot([], [], "b-", lw=1.5, alpha=0.6)
    dot = ax_traj.scatter([], [], c="blue", s=60, zorder=8)
    vel_arrow = ax_traj.annotate("", xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    heading_arrow = ax_traj.annotate("", xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    txt_traj = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes,
                             va="top", fontsize=8, family="monospace",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- 误差面板 ---
    ax_err.set_xlim(0, n)
    err_max = max(0.3, np.nanmax(ep["errors"]) * 1.1) if len(ep["errors"]) > 0 else 1.0
    ax_err.set_ylim(0, err_max)
    ax_err.set_title("Position Error", fontsize=10)
    ax_err.set_ylabel("m")
    ax_err.axhline(reach_threshold, color="red", ls="--", alpha=0.5, lw=1, label=f"threshold={reach_threshold}m")
    ax_err.grid(True, alpha=0.2)
    ax_err.legend(fontsize=7, loc="upper right")
    err_line, = ax_err.plot([], [], "steelblue", lw=1.5)

    # --- 速度面板 ---
    ax_vel.set_xlim(0, n)
    spd_max = max(0.5, np.nanmax(ep["speed"]) * 1.2)
    ax_vel.set_ylim(-spd_max, spd_max)
    ax_vel.set_title("Velocity Components (body)", fontsize=10)
    ax_vel.set_xlabel("step")
    ax_vel.set_ylabel("m/s")
    ax_vel.grid(True, alpha=0.2)
    surge_line, = ax_vel.plot([], [], "tab:blue", lw=1.2, label="surge")
    sway_line, = ax_vel.plot([], [], "tab:orange", lw=1.2, label="sway")
    speed_line, = ax_vel.plot([], [], "tab:red", lw=1.8, alpha=0.7, label="|v|")
    ax_vel.legend(fontsize=7, loc="upper right")

    def update(i):
        # 轨迹
        trail.set_data(traj[:i+1, 0], traj[:i+1, 1])
        dot.set_offsets([traj[i]])
        # 速度矢量（体→世界）
        yaw_i = ep["yaw"][i]
        vs = ep["vel_surge"][i]
        vw = ep["vel_sway"][i]
        vx_w = vs * math.cos(yaw_i) - vw * math.sin(yaw_i)
        vy_w = vs * math.sin(yaw_i) + vw * math.cos(yaw_i)
        scale = 0.3
        vel_arrow.xy = (traj[i, 0] + vx_w * scale, traj[i, 1] + vy_w * scale)
        vel_arrow.set_position((traj[i, 0], traj[i, 1]))
        # 航向箭头
        hlen = 0.12
        heading_arrow.xy = (traj[i, 0] + hlen * math.cos(yaw_i),
                            traj[i, 1] + hlen * math.sin(yaw_i))
        heading_arrow.set_position((traj[i, 0], traj[i, 1]))
        # 文字
        ei = min(i, len(ep["errors"]) - 1) if len(ep["errors"]) > 0 else 0
        err_val = ep["errors"][ei] if len(ep["errors"]) > 0 else 0
        spd_val = ep["speed"][i]
        status = "SUCCESS" if ep["success"] and i >= n - 2 else ep["reason"] if i >= n - 2 else ""
        txt_traj.set_text(
            f"step={i:4d}  err={err_val:.4f}m\n"
            f"spd={spd_val:.3f}m/s  yaw={math.degrees(yaw_i):+.1f}\u00b0\n"
            f"{status}"
        )
        # 误差曲线
        err_line.set_data(range(min(i+1, len(ep["errors"]))), ep["errors"][:i+1])
        # 速度曲线
        surge_line.set_data(range(i+1), ep["vel_surge"][:i+1])
        sway_line.set_data(range(i+1), ep["vel_sway"][:i+1])
        speed_line.set_data(range(i+1), ep["speed"][:i+1])
        return []

    ani = FuncAnimation(fig, update, frames=n, interval=int(1000 / fps), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ──────────────────── 静态轨迹 PNG（单 episode 详情） ────────────────────

def save_episode_detail_png(ep, out_path, reach_threshold=0.05):
    """6 子图详细面板：轨迹+速度色带 / 误差 / 速度分量 / 动作 / 航向 / yaw_rate。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) 轨迹 + 速度色带
    ax = axes[0, 0]
    traj = ep["pos"]
    target = ep["target"]
    speeds = ep["speed"][:len(traj)-1] if len(ep["speed"]) > 1 else np.zeros(max(1, len(traj)-1))
    if len(traj) > 1:
        for j in range(len(traj) - 1):
            c = cm.viridis(speeds[j] / max(np.max(speeds), 1e-6))
            ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], color=c, lw=2)
    cax = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, max(np.max(speeds), 0.01)), cmap="viridis"), ax=ax)
    cax.set_label("speed (m/s)")
    circle = Circle(target, reach_threshold, fill=False, ls="--", color="red", alpha=0.6, lw=1.5)
    ax.add_patch(circle)
    ax.plot(target[0], target[1], "r*", ms=14)
    ax.plot(traj[0, 0], traj[0, 1], "gs", ms=10)
    # 每隔一定步画航向箭头
    arrow_step = max(1, len(traj) // 12)
    for j in range(0, len(traj), arrow_step):
        yaw_j = ep["yaw"][j]
        ax.annotate("", xy=(traj[j, 0] + 0.06 * math.cos(yaw_j), traj[j, 1] + 0.06 * math.sin(yaw_j)),
                     xytext=(traj[j, 0], traj[j, 1]),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    result = "SUCCESS" if ep["success"] else ep["reason"]
    ax.set_title(f"Trajectory (speed colormap) [{result}]", fontsize=10)

    # (0,1) 位置误差
    ax = axes[0, 1]
    ax.plot(ep["errors"], "steelblue", lw=1.5)
    ax.axhline(reach_threshold, color="red", ls="--", alpha=0.5, label=f"thr={reach_threshold}")
    ax.set_title("Position Error")
    ax.set_ylabel("m"); ax.set_xlabel("step")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (0,2) 速度分量
    ax = axes[0, 2]
    ax.plot(ep["vel_surge"], label="surge", lw=1.2)
    ax.plot(ep["vel_sway"], label="sway", lw=1.2)
    ax.plot(ep["speed"], label="|v|", lw=1.8, alpha=0.6, color="red")
    ax.set_title("Velocity (body frame)")
    ax.set_ylabel("m/s"); ax.set_xlabel("step")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (1,0) 动作输出
    ax = axes[1, 0]
    if len(ep["actions"]) > 0:
        ax.plot(ep["actions"][:, 0], label="surge_cmd", lw=1)
        ax.plot(ep["actions"][:, 1], label="sway_cmd", lw=1)
        ax.plot(ep["actions"][:, 2], label="yaw_cmd", lw=1)
    ax.set_title("Actions")
    ax.set_ylabel("value"); ax.set_xlabel("step")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (1,1) 航向误差
    ax = axes[1, 1]
    ax.plot(np.degrees(ep["headings"]), "tab:purple", lw=1.2)
    ax.set_title("Heading Error")
    ax.set_ylabel("deg"); ax.set_xlabel("step")
    ax.grid(True, alpha=0.2)

    # (1,2) yaw_rate
    ax = axes[1, 2]
    ax.plot(ep["yaw_rate"], "tab:brown", lw=1.2)
    ax.set_title("Yaw Rate")
    ax.set_ylabel("rad/s"); ax.set_xlabel("step")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────── 统计汇总面板 ────────────────────

def save_statistics_panel(episodes, out_path, exp_name="", reach_threshold=0.05):
    """8 子图统计面板。"""
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    final_errors = [e["errors"][-1] if len(e["errors"]) > 0 else np.nan for e in episodes]
    step_counts = [e["steps"] for e in episodes]
    total_rewards = [float(np.sum(e["rewards"])) for e in episodes]
    init_speeds = [e["speed"][0] for e in episodes]
    max_speeds = [float(np.max(e["speed"])) for e in episodes]
    mean_speeds = [float(np.mean(e["speed"])) for e in episodes]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(f"{exp_name}  —  {successes}/{n} success ({successes/n:.0%})", fontsize=13, fontweight="bold")

    # (0,0) 成功率
    ax = axes[0, 0]
    ax.bar(["Success", "Fail"], [successes/n, 1-successes/n], color=["#2ecc71", "#e74c3c"], alpha=0.8)
    ax.set_ylim(0, 1.05); ax.set_title("Success Rate"); ax.grid(axis="y", alpha=0.2)

    # (0,1) 最终误差
    ax = axes[0, 1]
    ax.hist(final_errors, bins=20, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(reach_threshold, color="red", ls="--", lw=1.5, label=f"thr={reach_threshold}")
    ax.set_title(f"Final Error (med={np.nanmedian(final_errors):.4f}m)")
    ax.set_xlabel("m"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # (0,2) 步数
    ax = axes[0, 2]
    ax.hist(step_counts, bins=20, color="orange", alpha=0.75, edgecolor="white")
    ax.set_title(f"Episode Steps (med={int(np.median(step_counts))})")
    ax.set_xlabel("steps"); ax.grid(alpha=0.2)

    # (0,3) 总奖励
    ax = axes[0, 3]
    ax.hist(total_rewards, bins=20, color="purple", alpha=0.75, edgecolor="white")
    ax.set_title(f"Total Reward (med={np.median(total_rewards):.1f})")
    ax.set_xlabel("reward"); ax.grid(alpha=0.2)

    # (1,0) 初始速度分布
    ax = axes[1, 0]
    ax.hist(init_speeds, bins=20, color="teal", alpha=0.75, edgecolor="white")
    ax.set_title(f"Init Speed (mean={np.mean(init_speeds):.3f})")
    ax.set_xlabel("m/s"); ax.grid(alpha=0.2)

    # (1,1) 最大速度
    ax = axes[1, 1]
    ax.hist(max_speeds, bins=20, color="coral", alpha=0.75, edgecolor="white")
    ax.set_title(f"Max Speed (mean={np.mean(max_speeds):.3f})")
    ax.set_xlabel("m/s"); ax.grid(alpha=0.2)

    # (1,2) 所有 episode 误差曲线叠加
    ax = axes[1, 2]
    for ep in episodes:
        color = "#2ecc71" if ep["success"] else "#e74c3c"
        ax.plot(ep["errors"], color=color, alpha=0.25, lw=0.8)
    ax.axhline(reach_threshold, color="red", ls="--", alpha=0.5)
    ax.set_title("All Error Curves"); ax.set_xlabel("step"); ax.set_ylabel("m"); ax.grid(alpha=0.2)

    # (1,3) 所有 episode 速度曲线叠加
    ax = axes[1, 3]
    for ep in episodes:
        color = "#2ecc71" if ep["success"] else "#e74c3c"
        ax.plot(ep["speed"], color=color, alpha=0.25, lw=0.8)
    ax.set_title("All Speed Curves"); ax.set_xlabel("step"); ax.set_ylabel("m/s"); ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────── 跨实验对比 ────────────────────

def save_comparison_plot(all_results, out_path):
    """多实验横向对比条形图。"""
    names = list(all_results.keys())
    n_exp = len(names)
    succ_rates = []
    med_errors = []
    med_steps = []
    med_rewards = []
    mean_init_spd = []
    for nm in names:
        eps = all_results[nm]
        succ_rates.append(sum(1 for e in eps if e["success"]) / len(eps))
        fe = [e["errors"][-1] for e in eps if len(e["errors"]) > 0]
        med_errors.append(np.nanmedian(fe) if fe else np.nan)
        med_steps.append(np.median([e["steps"] for e in eps]))
        med_rewards.append(np.median([float(np.sum(e["rewards"])) for e in eps]))
        mean_init_spd.append(np.mean([e["speed"][0] for e in eps]))

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    x = np.arange(n_exp)
    short_names = [n.replace("v4_tight_", "").replace("low_near/", "") for n in names]

    ax = axes[0]
    bars = ax.bar(x, succ_rates, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax.set_ylim(0, 1.05); ax.set_title("Success Rate", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    for b, v in zip(bars, succ_rates):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    ax.bar(x, med_errors, color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_title("Median Final Error (m)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    ax.bar(x, med_steps, color="orange", alpha=0.8, edgecolor="white")
    ax.set_title("Median Steps", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[3]
    ax.bar(x, med_rewards, color="purple", alpha=0.8, edgecolor="white")
    ax.set_title("Median Reward", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[4]
    ax.bar(x, mean_init_spd, color="teal", alpha=0.8, edgecolor="white")
    ax.set_title("Mean Init Speed (m/s)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────── main ────────────────────

def run_single_experiment(exp_dir, episodes=50, gif_eps=3, fps=20, seed=1234,
                          reach_threshold=0.05, init_velocity_range=0.0,
                          init_yaw_rate_range=0.0):
    """对单个实验生成全部可视化。"""
    exp_name = os.path.basename(exp_dir)
    print(f"\n{'='*60}")
    print(f"  Evaluating: {exp_name}  (vel_range={init_velocity_range})")
    print(f"{'='*60}")

    model, device = load_model(exp_dir)
    env = LowLevelAUV2DEnvV3Continuous(
        reach_threshold=reach_threshold,
        target_distance_min=0.3,
        target_distance_max=1.0,
        max_episode_seconds=30.0,
        init_velocity_range=init_velocity_range,
        init_yaw_rate_range=init_yaw_rate_range,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_eps = []
    for i in range(episodes):
        ep = rollout(model, env, device, deterministic=True)
        all_eps.append(ep)
        if (i + 1) % 10 == 0:
            succ = sum(1 for e in all_eps if e["success"])
            print(f"  [{i+1}/{episodes}] succ={succ}/{i+1}={succ/(i+1):.0%}")

    env.close()

    # --- 输出目录 ---
    out_dir = os.path.join(exp_dir, "enhanced_vis")
    gif_dir = os.path.join(out_dir, "gifs")
    detail_dir = os.path.join(out_dir, "details")
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    # --- 统计面板 ---
    save_statistics_panel(all_eps, os.path.join(out_dir, "statistics_panel.png"),
                          exp_name=exp_name, reach_threshold=reach_threshold)
    print(f"  Saved statistics_panel.png")

    # --- best/worst GIF + 详情 PNG ---
    sorted_eps = sorted(all_eps, key=lambda e: (not e["success"], e["errors"][-1] if e["errors"].size else 999))
    best_eps = sorted_eps[:gif_eps]
    worst_eps = sorted_eps[-gif_eps:]

    for idx, ep in enumerate(best_eps):
        save_enhanced_gif(ep, os.path.join(gif_dir, f"best_{idx+1}.gif"), fps=fps,
                          reach_threshold=reach_threshold)
        save_episode_detail_png(ep, os.path.join(detail_dir, f"best_{idx+1}_detail.png"),
                                reach_threshold=reach_threshold)
        print(f"  Saved best_{idx+1}.gif + detail")

    for idx, ep in enumerate(worst_eps):
        save_enhanced_gif(ep, os.path.join(gif_dir, f"worst_{idx+1}.gif"), fps=fps,
                          reach_threshold=reach_threshold)
        save_episode_detail_png(ep, os.path.join(detail_dir, f"worst_{idx+1}_detail.png"),
                                reach_threshold=reach_threshold)
        print(f"  Saved worst_{idx+1}.gif + detail")

    # --- summary ---
    succ = sum(1 for e in all_eps if e["success"])
    fe = [e["errors"][-1] for e in all_eps if e["errors"].size > 0]
    summary = {
        "experiment": exp_name,
        "episodes": episodes,
        "success_rate": succ / episodes,
        "median_final_error": float(np.nanmedian(fe)),
        "mean_final_error": float(np.nanmean(fe)),
        "median_steps": int(np.median([e["steps"] for e in all_eps])),
        "mean_init_speed": float(np.mean([e["speed"][0] for e in all_eps])),
        "init_velocity_range": init_velocity_range,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: succ={succ}/{episodes}={succ/episodes:.0%} med_err={np.nanmedian(fe):.4f}m")

    return all_eps


# 实验名 → init_velocity_range 映射
EXP_VEL_MAP = {
    "v4_tight_baseline": 0.0,
    "v4_tight_vel015_v2": 0.15,
    "v4_tight_vel030_v2": 0.3,
    "v4_tight_vel06_v2": 0.6,
    "v4_tight_vel08_v2": 0.8,
    "v4_tight_vel10_v2": 1.0,
}

EXP_YAW_MAP = {
    "v4_tight_baseline": 0.0,
    "v4_tight_vel015_v2": 0.075,
    "v4_tight_vel030_v2": 0.2,
    "v4_tight_vel06_v2": 0.3,
    "v4_tight_vel08_v2": 0.4,
    "v4_tight_vel10_v2": 0.5,
}


def main():
    parser = argparse.ArgumentParser(description="V3 Enhanced Visualization")
    parser.add_argument("--exp-dir", type=str, default=None, help="单实验目录")
    parser.add_argument("--compare", nargs="+", default=None, help="多实验对比")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gif-episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reach-threshold", type=float, default=0.05)
    parser.add_argument("--init-velocity-range", type=float, default=None,
                        help="覆盖默认 init_velocity_range")
    args = parser.parse_args()

    if args.compare:
        # 多实验对比模式
        all_results = {}
        for exp_path in args.compare:
            exp_name = os.path.basename(exp_path)
            vel = args.init_velocity_range if args.init_velocity_range is not None else EXP_VEL_MAP.get(exp_name, 0.0)
            yaw = EXP_YAW_MAP.get(exp_name, vel * 0.5)
            eps = run_single_experiment(
                exp_path, episodes=args.episodes, gif_eps=args.gif_episodes,
                fps=args.fps, seed=args.seed, reach_threshold=args.reach_threshold,
                init_velocity_range=vel, init_yaw_rate_range=yaw,
            )
            all_results[exp_name] = eps

        out_path = os.path.join("low_near", "comparison_enhanced.png")
        save_comparison_plot(all_results, out_path)
        print(f"\nSaved comparison plot: {out_path}")

    elif args.exp_dir:
        exp_name = os.path.basename(args.exp_dir)
        vel = args.init_velocity_range if args.init_velocity_range is not None else EXP_VEL_MAP.get(exp_name, 0.0)
        yaw = EXP_YAW_MAP.get(exp_name, vel * 0.5)
        run_single_experiment(
            args.exp_dir, episodes=args.episodes, gif_eps=args.gif_episodes,
            fps=args.fps, seed=args.seed, reach_threshold=args.reach_threshold,
            init_velocity_range=vel, init_yaw_rate_range=yaw,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
