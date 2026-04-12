"""
二维位置/速度 PID 控制器 - 与 low_level 双环逻辑一致
RL 输出目标点，本控制器算 τ，再交给 Dynamics2D 积分；观测可含速度、加速度。
"""

from typing import Tuple, Dict, Any
import numpy as np


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return x
    n = np.linalg.norm(x)
    if n <= max_norm:
        return x
    return x * (max_norm / n)


class PIDController2D:
    """
    2D 位置-速度双环：目标位置 -> 期望速度(位置环 P 限幅) -> 速度环 PD -> 推力 τ
    """

    def __init__(
        self,
        Kp_pos: float = 2.0,
        Kp_vel: float = 20.0,
        Kd_vel: float = 6.0,
        v_max: float = 1.2,
        max_force: float = 50.0,
        goal_slowdown_radius: float = 1.0,
        goal_slowdown_min_scale: float = 0.2,
    ):
        self.Kp_pos = Kp_pos
        self.Kp_vel = Kp_vel
        self.Kd_vel = Kd_vel
        self.v_max = v_max
        self.max_force = max_force
        self.goal_slowdown_radius = goal_slowdown_radius
        self.goal_slowdown_min_scale = goal_slowdown_min_scale

    def reset(self) -> None:
        pass

    def compute_thrust(
        self,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        target_pos = np.asarray(target_pos, dtype=np.float64).ravel()[:2]
        current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:2]
        current_vel = np.asarray(current_vel, dtype=np.float64).ravel()[:2]
        pos_error = target_pos - current_pos
        dist = np.linalg.norm(pos_error)
        if dist < 1e-8:
            desired_vel = np.zeros(2, dtype=np.float64)
        else:
            desired_vel = self.Kp_pos * pos_error
            speed_desired = np.linalg.norm(desired_vel)
            if speed_desired > 1e-10:
                if self.goal_slowdown_radius > 1e-6 and dist < self.goal_slowdown_radius:
                    scale = np.clip(
                        dist / self.goal_slowdown_radius,
                        self.goal_slowdown_min_scale,
                        1.0,
                    )
                    speed_desired = min(speed_desired, self.v_max * scale)
                else:
                    speed_desired = min(speed_desired, self.v_max)
                desired_vel = desired_vel * (speed_desired / np.linalg.norm(desired_vel))
        vel_error = desired_vel - current_vel
        force_cmd = self.Kp_vel * vel_error - self.Kd_vel * current_vel
        thrust = _clip_norm(force_cmd, self.max_force).astype(np.float32)
        debug_info = {
            "desired_vel": desired_vel.copy(),
            "pos_error_norm": float(dist),
        }
        return thrust, debug_info


def action_to_target_position_2d(
    action: int,
    current_pos: np.ndarray,
    world_size: float,
    step_length: float = 2.0,
) -> np.ndarray:
    """离散动作 0~8 -> 目标点。0=保持，1~8=当前点 + step_length*方向。"""
    current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:2]
    if action == 0:
        return current_pos.copy()
    angles = np.arange(8) * (2 * np.pi / 8)
    idx = action - 1
    direction = np.array([np.cos(angles[idx]), np.sin(angles[idx])], dtype=np.float64)
    target = current_pos + step_length * direction
    target = np.clip(target, -world_size, world_size)
    return target.astype(np.float32)
