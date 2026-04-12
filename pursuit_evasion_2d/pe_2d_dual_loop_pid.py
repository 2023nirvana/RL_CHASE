"""
2D 双环 PID - 与 low_level 的 IntegratedController 平移部分一致
================================================================

从 configs/low_level_config_2d.yaml 读取：
- action_space.velocity_controller: kp, kd, max_force（速度环 PD）
- action_space.goal_slowdown: enabled, radius, min_scale
- action_space.action_distance: surge, sway（每步位移，translation_mode=distance 时 desired_vel = linear_cmd/Ts）
- action_space.translation_mode: "distance"

双环逻辑（与 core/control/dual_loop_pid.py 中 IntegratedController 一致）：
1. 外环（位置/位移 -> 期望速度）：linear_cmd = action 对应的 [dx, dy]，desired_vel = linear_cmd / Ts；若提供 target_position 则按 goal_slowdown 缩放。
2. 内环（速度环）：force = Kp_vel * (desired_vel - current_vel) - Kd_vel * current_vel，再按 max_force 限幅。

2D 下仅保留 x-y（surge, sway），无姿态环。
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import yaml


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return x
    n = np.linalg.norm(x)
    if n <= max_norm:
        return x
    return x * (max_norm / n)


def _load_low_level_config_2d(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(proj_root, "configs", "low_level_config_2d.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _action_to_linear_cmd_2d(action: int, surge: float, sway: float) -> np.ndarray:
    """
    离散动作 0~8 -> 2D 位移 [dx, dy]（世界系 x,y）。
    0: 零位移
    1~8: 8 个方向，从 0°（东/x+）逆时针，与 low_level 的 surge/sway 对应。
    """
    if action == 0:
        return np.zeros(2, dtype=np.float64)
    angles = np.arange(8) * (2 * np.pi / 8)  # 0, 45, 90, ...
    idx = action - 1
    # 每个方向：x = surge*cos, y = sway*sin（与 body 的 surge=前、sway=左 对应：世界 x 前，y 左）
    dx = surge * np.cos(angles[idx])
    dy = sway * np.sin(angles[idx])
    return np.array([dx, dy], dtype=np.float64)


class DualLoopPID2D:
    """
    2D 双环 PID：与 low_level 的 IntegratedController 平移部分同构。
    外环：位移/目标 -> 期望速度；内环：速度环 PD -> 力。
    """

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config is None:
            config = _load_low_level_config_2d(config_path)
        ac = config.get("action_space", {})
        vc = ac.get("velocity_controller", {})
        gs = ac.get("goal_slowdown", {})
        ad = ac.get("action_distance", {})

        self.translation_mode = ac.get("translation_mode", "distance")
        self.surge = float(ad.get("surge", 0.03))
        self.sway = float(ad.get("sway", 0.03))

        kp = vc.get("kp", [15.0, 15.0, 0.0])
        kd = vc.get("kd", [5.0, 5.0, 0.0])
        mf = vc.get("max_force", [20.0, 20.0, 0.0])
        self.Kp_vel = np.array([float(kp[0]), float(kp[1])], dtype=np.float64)
        self.Kd_vel = np.array([float(kd[0]), float(kd[1])], dtype=np.float64)
        self.max_force = np.array([float(mf[0]), float(mf[1])], dtype=np.float64)

        self.goal_slowdown_enabled = gs.get("enabled", True)
        self.goal_slowdown_radius = float(gs.get("radius", 0.5))
        self.goal_slowdown_min_scale = float(gs.get("min_scale", 0.3))

    def reset(self) -> None:
        pass

    def _compute_thrust_from_linear_cmd(
        self,
        linear_cmd: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        Ts: float,
        target_position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:2]
        current_vel = np.asarray(current_vel, dtype=np.float64).ravel()[:2]
        linear_cmd = np.asarray(linear_cmd, dtype=np.float64).ravel()[:2]

        if self.translation_mode == "distance":
            desired_vel = linear_cmd / max(Ts, 1e-6)
            slowdown_scale = 1.0
            if (
                target_position is not None
                and self.goal_slowdown_enabled
                and self.goal_slowdown_radius > 1e-6
            ):
                target_position = np.asarray(target_position, dtype=np.float64).ravel()[:2]
                pos_error_norm = np.linalg.norm(target_position - current_pos)
                slowdown_scale = max(
                    self.goal_slowdown_min_scale,
                    min(1.0, pos_error_norm / self.goal_slowdown_radius),
                )
                desired_vel = desired_vel * slowdown_scale
            vel_error = desired_vel - current_vel
            force_cmd = self.Kp_vel * vel_error - self.Kd_vel * current_vel
            force_cmd = np.clip(
                force_cmd,
                -self.max_force,
                self.max_force,
            )
        else:
            desired_vel = np.zeros(2, dtype=np.float64)
            vel_error = np.zeros(2, dtype=np.float64)
            force_cmd = linear_cmd
            slowdown_scale = 1.0

        debug_info = {
            "linear_cmd": linear_cmd.copy(),
            "desired_vel": desired_vel.copy(),
            "vel_error": vel_error.copy(),
            "slowdown_scale": float(slowdown_scale),
            "force_cmd": force_cmd.copy(),
        }
        return force_cmd.astype(np.float32), debug_info

    def compute_thrust(
        self,
        action: int,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        Ts: float,
        target_position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        与 IntegratedController.compute_thrust 的平移部分一致（2D 版）。

        Args:
            action: 0=保持，1~8=8 方向
            current_pos: [x, y]
            current_vel: [vx, vy]
            Ts: 采样周期
            target_position: 可选，用于 goal_slowdown（例如追击时传 evader 位置）

        Returns:
            force_2d: [Fx, Fy] 世界系力，已按 max_force 限幅
            debug_info: 调试信息
        """
        linear_cmd = _action_to_linear_cmd_2d(action, self.surge, self.sway)
        thrust, debug_info = self._compute_thrust_from_linear_cmd(
            linear_cmd=linear_cmd,
            current_pos=current_pos,
            current_vel=current_vel,
            Ts=Ts,
            target_position=target_position,
        )
        debug_info["action"] = int(action)
        return thrust, debug_info

    def compute_thrust_from_continuous_action(
        self,
        action_xy: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        Ts: float,
        target_position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        连续动作接口：action_xy ∈ [-1, 1]^2，映射为连续每步位移 linear_cmd。
        与离散接口一致地复用双环 PID（外环位移->期望速度，内环速度 PD）。
        """
        a = np.asarray(action_xy, dtype=np.float64).ravel()[:2]
        a = np.clip(a, -1.0, 1.0)
        linear_cmd = np.array([a[0] * self.surge, a[1] * self.sway], dtype=np.float64)
        thrust, debug_info = self._compute_thrust_from_linear_cmd(
            linear_cmd=linear_cmd,
            current_pos=current_pos,
            current_vel=current_vel,
            Ts=Ts,
            target_position=target_position,
        )
        debug_info["continuous_action"] = a.astype(np.float32)
        return thrust, debug_info


def action_to_target_position_2d_from_config(
    action: int,
    current_pos: np.ndarray,
    world_size: float,
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
) -> np.ndarray:
    """
    与 low_level 的「每步位移」一致：目标点 = 当前点 + linear_cmd。
    用于需要「目标点」语义时（例如与原有 pe_2d_pid 接口兼容）。
    """
    if config is None:
        config = _load_low_level_config_2d(config_path)
    ad = config.get("action_space", {}).get("action_distance", {})
    surge = float(ad.get("surge", 0.03))
    sway = float(ad.get("sway", 0.03))
    linear_cmd = _action_to_linear_cmd_2d(action, surge, sway)
    current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:2]
    target = current_pos + linear_cmd
    target = np.clip(target, -world_size, world_size)
    return target.astype(np.float32)
