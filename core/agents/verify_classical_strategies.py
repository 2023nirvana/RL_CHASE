"""
经典追击策略 - 简单几何验证
============================
不启动真实环境，仅用 2D 质点模型验证各策略的 subgoal 方向是否合理：
  场景1: 静态目标 — 所有策略应产生朝目标方向的 subgoal
  场景2: 匀速直线逃脱 — 带预判的策略应偏向拦截方向
  场景3: 圆弧机动逃脱 — APN 应优于 PN
  场景4: 多步追击模拟 — 验证追击者能接近目标

输出: 每个策略的定量结果 + 轨迹图 (PNG)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.agents.classical_pursuit_strategies import (
    make_strategy, list_strategies, BaseStrategy
)

np.set_printoptions(precision=4, suppress=True)


def simulate_pursuit(
    strategy: BaseStrategy,
    pursuer_pos0: np.ndarray,
    pursuer_yaw0: float,
    pursuer_speed: float,
    evader_pos0: np.ndarray,
    evader_vel: np.ndarray,  # constant evader velocity (or callable)
    evader_accel: float = 0.0,  # angular acceleration for circular motion
    dt: float = 0.02,
    max_steps: int = 5000,
    catch_radius: float = 0.5,
    subgoal_radius: float = 2.0,
    world_size: float = 15.0,
):
    """
    简化 2D 追击模拟: 追击者以恒定速度朝 subgoal 方向运动。
    返回追击者和逃脱者轨迹以及是否捕获。
    """
    strategy.reset()
    strategy.set_dt(dt)

    p_pos = np.array(pursuer_pos0, dtype=np.float64)
    p_yaw = float(pursuer_yaw0)
    e_pos = np.array(evader_pos0, dtype=np.float64)
    e_vel = np.array(evader_vel, dtype=np.float64)

    p_traj = [p_pos.copy()]
    e_traj = [e_pos.copy()]
    subgoals = []

    caught = False
    for step in range(max_steps):
        # Pursuer velocity (world frame) from current yaw & speed
        p_vel = pursuer_speed * np.array([np.cos(p_yaw), np.sin(p_yaw)])

        # Get subgoal from strategy
        sg = strategy.compute_subgoal(
            p_pos, p_vel, p_yaw, e_pos, e_vel,
            subgoal_radius=subgoal_radius, world_size=world_size,
        )
        subgoals.append(sg.copy())

        # Move pursuer toward subgoal
        direction = sg - p_pos
        dist_to_sg = np.linalg.norm(direction)
        if dist_to_sg > 1e-6:
            direction = direction / dist_to_sg
        p_pos = p_pos + direction * pursuer_speed * dt
        p_yaw = np.arctan2(direction[1], direction[0])

        # Move evader
        if evader_accel != 0.0:
            # Circular motion: rotate velocity
            angle = evader_accel * dt
            c, s = np.cos(angle), np.sin(angle)
            e_vel = np.array([c * e_vel[0] - s * e_vel[1],
                              s * e_vel[0] + c * e_vel[1]])
        e_pos = e_pos + e_vel * dt

        # Clip to world
        p_pos = np.clip(p_pos, -world_size, world_size)
        e_pos = np.clip(e_pos, -world_size, world_size)

        p_traj.append(p_pos.copy())
        e_traj.append(e_pos.copy())

        # Check catch
        if np.linalg.norm(p_pos - e_pos) < catch_radius:
            caught = True
            break

    return {
        "caught": caught,
        "steps": step + 1 if caught else max_steps,
        "time": (step + 1 if caught else max_steps) * dt,
        "p_traj": np.array(p_traj),
        "e_traj": np.array(e_traj),
        "subgoals": np.array(subgoals),
        "final_dist": np.linalg.norm(p_pos - e_pos),
    }


# ──────────── Test Scenarios ────────────

def run_all_tests():
    strategies = {
        "PP":   make_strategy("pp"),
        "Lead": make_strategy("lead", pursuer_max_speed=2.0),
        "PN":   make_strategy("pn", N=4),
        "APN":  make_strategy("apn", N=4),
        "CB":   make_strategy("cb", K=8),
        "DP":   make_strategy("dp", deviation_deg=20),
    }

    scenarios = {
        "S1: Static Target": dict(
            pursuer_pos0=[0, 0], pursuer_yaw0=0, pursuer_speed=2.0,
            evader_pos0=[5, 3], evader_vel=[0, 0], evader_accel=0.0,
            max_steps=3000,
        ),
        "S2: Linear Escape (fast)": dict(
            pursuer_pos0=[0, 0], pursuer_yaw0=0, pursuer_speed=2.0,
            evader_pos0=[4, 0], evader_vel=[1.2, 0.6], evader_accel=0.0,
            max_steps=5000,
        ),
        "S3: Circular Maneuver": dict(
            pursuer_pos0=[0, 0], pursuer_yaw0=0, pursuer_speed=2.0,
            evader_pos0=[4, 0], evader_vel=[1.0, 0], evader_accel=0.8,
            max_steps=5000,
        ),
        "S4: Fast Evader (speed ratio 0.9)": dict(
            pursuer_pos0=[0, 0], pursuer_yaw0=0, pursuer_speed=2.0,
            evader_pos0=[3, 2], evader_vel=[1.5, 0.8], evader_accel=0.0,
            max_steps=8000,
        ),
    }

    out_dir = "core/agents/strategy_verification"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    print("=" * 70)
    print("  Classical Pursuit Strategies — Geometric Verification")
    print("=" * 70)

    for sc_name, sc_kwargs in scenarios.items():
        print(f"\n{'─' * 50}")
        print(f"  Scenario: {sc_name}")
        print(f"{'─' * 50}")
        results[sc_name] = {}
        for st_name, strategy in strategies.items():
            res = simulate_pursuit(strategy, **sc_kwargs)
            results[sc_name][st_name] = res
            status = "CAUGHT" if res["caught"] else "MISSED"
            print(f"  {st_name:6s}: {status} | steps={res['steps']:5d} "
                  f"| time={res['time']:.2f}s | final_dist={res['final_dist']:.3f}m")

    # ──────────── Plot ────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = {"PP": "tab:blue", "Lead": "tab:orange", "PN": "tab:green",
              "APN": "tab:red", "CB": "tab:purple", "DP": "tab:brown"}

    for ax, (sc_name, sc_results) in zip(axes.flat, results.items()):
        for st_name, res in sc_results.items():
            pt = res["p_traj"]
            ax.plot(pt[:, 0], pt[:, 1], color=colors[st_name], linewidth=1.2,
                    label=st_name, alpha=0.85)
            ax.plot(pt[0, 0], pt[0, 1], 'o', color=colors[st_name], ms=5)
            if res["caught"]:
                ax.plot(pt[-1, 0], pt[-1, 1], '*', color=colors[st_name], ms=10)

        # Evader trajectory (same for all strategies in same scenario)
        et = list(sc_results.values())[0]["e_traj"]
        ax.plot(et[:, 0], et[:, 1], 'k--', linewidth=2, label="Evader", alpha=0.7)
        ax.plot(et[0, 0], et[0, 1], 'ks', ms=8)

        ax.set_title(sc_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Classical Pursuit Strategies — Trajectory Comparison", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(out_dir, "trajectory_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Trajectory comparison saved: {fig_path}")

    # ──────────── Summary bar chart: catch time ────────────
    fig2, axes2 = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes2 = [axes2]
    for ax, (sc_name, sc_results) in zip(axes2, results.items()):
        names = list(sc_results.keys())
        times = [sc_results[n]["time"] for n in names]
        caught = [sc_results[n]["caught"] for n in names]
        bar_colors = [colors[n] if c else "lightgray" for n, c in zip(names, caught)]
        bars = ax.bar(names, times, color=bar_colors, edgecolor="black", linewidth=0.5)
        for bar, c in zip(bars, caught):
            if not c:
                bar.set_hatch("//")
        ax.set_title(sc_name, fontsize=10)
        ax.set_ylabel("Time to catch (s)")
        ax.tick_params(axis="x", rotation=30)
    fig2.suptitle("Catch Time Comparison (hatched = not caught)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig2_path = os.path.join(out_dir, "catch_time_comparison.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Catch time comparison saved: {fig2_path}")

    # ──────────── Single-step direction check ────────────
    print(f"\n{'=' * 70}")
    print("  Single-Step Direction Sanity Check")
    print(f"{'=' * 70}")
    p_pos = np.array([0.0, 0.0])
    p_vel = np.array([1.0, 0.0])
    p_yaw = 0.0
    e_pos = np.array([5.0, 2.0])
    e_vel = np.array([0.8, 0.5])
    radius = 2.0

    expected_los = np.arctan2(2.0, 5.0)
    print(f"  LOS angle to evader: {np.degrees(expected_los):.1f}°")
    print(f"  Evader vel direction: {np.degrees(np.arctan2(0.5, 0.8)):.1f}°\n")

    for st_name, strategy in strategies.items():
        strategy.reset()
        strategy.set_dt(0.02)
        sg = strategy.compute_subgoal(p_pos, p_vel, p_yaw, e_pos, e_vel, radius)
        sg_dir = sg - p_pos
        sg_angle = np.degrees(np.arctan2(sg_dir[1], sg_dir[0]))
        print(f"  {st_name:6s}: subgoal=({sg[0]:+.3f}, {sg[1]:+.3f}) "
              f"dir={sg_angle:+.1f}° dist={np.linalg.norm(sg_dir):.3f}m")

    print(f"\n{'=' * 70}")
    print("  Verification complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all_tests()
