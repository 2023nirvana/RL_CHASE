"""
低层连续控制环境：2D 目标到达（含加速度观测）
================================================

用于阶段一：先训练“连续动作”底层控制器，验证目标到达能力。
动作空间：Box([-1, -1], [1, 1])，映射为连续加速度命令。
观测空间：pos_error(2), vel(2), acc(2) + 可选历史（默认无历史）。
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .pe_2d_dual_loop_pid import DualLoopPID2D
from .pe_2d_auv_config import create_pursuer_dynamics_from_config


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(x)
    if n <= max_norm or max_norm <= 0:
        return x
    if n < 1e-12:
        return x
    return x * (max_norm / n)


class LowLevelTarget2DContinuousEnv(gym.Env):
    """2D 连续动作目标到达环境（双积分器 + v/a 硬约束）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        v_max: float = 1.2,
        a_max: float = 2.5,
        control_mode: str = "dual_pid",
        low_level_config_path: Optional[str] = None,
        robot_config_path: Optional[str] = None,
        world_size: float = 10.0,
        dt: float = 0.05,
        max_steps: int = 400,
        goal_tolerance: float = 0.2,
        target_distance_range: Tuple[float, float] = (1.0, 5.0),
        obs_scale_pos: float = 1.0 / 10.0,
        obs_scale_vel: float = 1.0 / 2.0,
        obs_scale_acc: float = 1.0 / 3.0,
        reward_goal: float = 50.0,
        reward_distance_weight: float = 0.2,
        reward_improve_weight: float = 5.0,
        reward_closing_speed_weight: float = 0.5,
        reward_near_goal_weight: float = 0.2,
        near_goal_scale: float = 0.8,
        penalty_time: float = 0.01,
        penalty_action: float = 0.0005,
        obs_mode: str = "full",
        enable_current_disturbance: bool = False,
        current_scale: float = 0.25,
        current_ou_theta: float = 0.12,
        current_ou_sigma: float = 0.08,
        current_drift_scale: float = 0.10,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.control_mode = str(control_mode).lower()
        if self.control_mode not in ("dual_pid", "direct"):
            raise ValueError(f"Unsupported control_mode={control_mode}, expected dual_pid or direct")
        self.low_level_config_path = low_level_config_path
        self.robot_config_path = robot_config_path
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_tolerance = float(goal_tolerance)
        self.target_distance_range = target_distance_range
        self.obs_scale_pos = float(obs_scale_pos)
        self.obs_scale_vel = float(obs_scale_vel)
        self.obs_scale_acc = float(obs_scale_acc)

        self.reward_goal = float(reward_goal)
        self.reward_distance_weight = float(reward_distance_weight)
        self.reward_improve_weight = float(reward_improve_weight)
        self.reward_closing_speed_weight = float(reward_closing_speed_weight)
        self.reward_near_goal_weight = float(reward_near_goal_weight)
        self.near_goal_scale = float(max(1e-6, near_goal_scale))
        self.penalty_time = float(penalty_time)
        self.penalty_action = float(penalty_action)
        self.obs_mode = str(obs_mode).lower()
        if self.obs_mode not in ("full", "no_acc", "pos_only"):
            raise ValueError(f"Unsupported obs_mode={obs_mode}, expected full/no_acc/pos_only")
        self.enable_current_disturbance = bool(enable_current_disturbance)
        self.current_scale = float(current_scale)
        self.current_ou_theta = float(current_ou_theta)
        self.current_ou_sigma = float(current_ou_sigma)
        self.current_drift_scale = float(current_drift_scale)

        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._current = np.zeros(2, dtype=np.float32)
        self._current_bias = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_distance = 0.0
        self._last_thrust = np.zeros(2, dtype=np.float32)

        self.np_random = np.random.default_rng(seed)

        self._dual_pid = None
        self._dynamics = None
        if self.control_mode == "dual_pid":
            self._dual_pid = DualLoopPID2D(config_path=low_level_config_path)
            self._dynamics = create_pursuer_dynamics_from_config(
                config_path=robot_config_path,
                dt=self.dt,
                v_max_override=self.v_max,
                a_max_override=self.a_max,
            )

        # obs dimension depends on obs_mode
        if self.obs_mode == "full":
            obs_dim = 6  # pos_error(2) + vel(2) + acc(2)
        elif self.obs_mode == "no_acc":
            obs_dim = 4  # pos_error(2) + vel(2)
        else:  # pos_only
            obs_dim = 2  # pos_error(2)
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 10.0,
            high=np.ones(obs_dim, dtype=np.float32) * 10.0,
            dtype=np.float32,
        )

        # continuous action in [-1,1]^2 -> desired acceleration scaled by a_max
        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

    def _step_dynamics(self, acc_cmd: np.ndarray):
        acc = np.asarray(acc_cmd, dtype=np.float32).reshape(2)
        acc = np.clip(acc, -self.a_max, self.a_max)
        acc = _clip_norm(acc, self.a_max)

        current_push = self._current if self.enable_current_disturbance else 0.0
        vel_new = self._vel + (acc + current_push) * self.dt
        vel_new = _clip_norm(vel_new, self.v_max)

        pos_new = self._pos + vel_new * self.dt
        pos_new = np.clip(pos_new, -self.world_size, self.world_size)

        self._acc = acc
        self._vel = vel_new.astype(np.float32)
        self._pos = pos_new.astype(np.float32)

    def _step_with_dual_pid(self, action: np.ndarray):
        thrust, _ = self._dual_pid.compute_thrust_from_continuous_action(
            action_xy=action,
            current_pos=self._pos.copy(),
            current_vel=self._vel.copy(),
            Ts=self.dt,
            target_position=self._goal.copy(),
        )
        pos_new, vel_new, acc = self._dynamics.step(
            self._pos.copy(),
            self._vel.copy(),
            thrust,
            v_max=self.v_max,
            a_max=self.a_max,
        )

        if self.enable_current_disturbance:
            vel_new = vel_new + self._current * self.dt
            vel_new = _clip_norm(vel_new, self.v_max)
            pos_new = pos_new + self._current * self.dt

        pos_new = np.clip(pos_new, -self.world_size, self.world_size)
        self._pos = pos_new.astype(np.float32)
        self._vel = vel_new.astype(np.float32)
        self._acc = acc.astype(np.float32)
        self._last_thrust = thrust.astype(np.float32)

    def _update_current(self):
        if not self.enable_current_disturbance:
            self._current[:] = 0.0
            return
        noise = self.np_random.normal(0.0, self.current_ou_sigma, size=2).astype(np.float32)
        self._current += self.current_ou_theta * (self._current_bias - self._current) + noise
        self._current = _clip_norm(self._current, self.current_scale).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        pos_error = self._goal - self._pos
        parts = [
            pos_error[0] * self.obs_scale_pos,
            pos_error[1] * self.obs_scale_pos,
        ]
        if self.obs_mode in ("full", "no_acc"):
            parts.extend([
                self._vel[0] * self.obs_scale_vel,
                self._vel[1] * self.obs_scale_vel,
            ])
        if self.obs_mode == "full":
            parts.extend([
                self._acc[0] * self.obs_scale_acc,
                self._acc[1] * self.obs_scale_acc,
            ])
        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)

        self._pos = self.np_random.uniform(
            -self.world_size * 0.5, self.world_size * 0.5, size=2
        ).astype(np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)
        self._current = np.zeros(2, dtype=np.float32)
        self._last_thrust = np.zeros(2, dtype=np.float32)
        if self._dual_pid is not None:
            self._dual_pid.reset()
        if self._dynamics is not None:
            self._dynamics.reset(self._pos.copy(), self._vel.copy())

        if self.enable_current_disturbance:
            ang = self.np_random.uniform(0, 2 * np.pi)
            mag = self.np_random.uniform(0.0, self.current_drift_scale)
            self._current_bias = (mag * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)).astype(np.float32)
            self._current = self._current_bias.copy()
        else:
            self._current_bias = np.zeros(2, dtype=np.float32)

        dist = self.np_random.uniform(
            self.target_distance_range[0], self.target_distance_range[1]
        )
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._goal = (
            self._pos + dist * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        )
        self._goal = np.clip(self._goal, -self.world_size, self.world_size)

        self._step_count = 0
        self._prev_distance = float(np.linalg.norm(self._goal - self._pos))

        info = {
            "goal": self._goal.copy(),
            "distance": self._prev_distance,
            "current": self._current.copy(),
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        self._update_current()
        if self.control_mode == "dual_pid":
            self._step_with_dual_pid(action)
            acc_cmd = self._acc.copy()
        else:
            acc_cmd = action * self.a_max
            self._step_dynamics(acc_cmd)
            self._last_thrust = np.zeros(2, dtype=np.float32)
        self._step_count += 1

        distance = float(np.linalg.norm(self._goal - self._pos))
        improve = self._prev_distance - distance
        goal_dir = (self._goal - self._pos) / (distance + 1e-6)
        closing_speed = float(np.dot(self._vel, goal_dir))
        near_goal_bonus = float(np.exp(-distance / self.near_goal_scale))

        reward = 0.0
        reward += self.reward_improve_weight * improve
        reward += -self.reward_distance_weight * distance
        reward += self.reward_closing_speed_weight * closing_speed
        reward += self.reward_near_goal_weight * near_goal_bonus
        reward += -self.penalty_time
        reward += -self.penalty_action * float(np.linalg.norm(action) ** 2)

        terminated = distance < self.goal_tolerance
        if terminated:
            reward += self.reward_goal

        truncated = self._step_count >= self.max_steps
        self._prev_distance = distance

        info = {
            "distance": distance,
            "success": bool(terminated),
            "acc_cmd": acc_cmd.astype(np.float32),
            "control_mode": self.control_mode,
            "thrust_cmd": self._last_thrust.copy(),
            "speed": float(np.linalg.norm(self._vel)),
            "thrust_norm": float(np.linalg.norm(self._last_thrust)),
            "closing_speed": closing_speed,
            "near_goal_bonus": near_goal_bonus,
            "current": self._current.copy(),
            "current_norm": float(np.linalg.norm(self._current)),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass
