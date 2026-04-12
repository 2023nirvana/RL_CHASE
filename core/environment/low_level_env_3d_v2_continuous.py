"""
三维环境 V2 连续动作版本（低层控制拓展）
=========================================

目标：保留 `low_level_env_3d_v2.py` 的任务定义、观测、奖励与终止逻辑，
仅将动作从离散 8 动作改为连续 4 维：

  action = [surge_cmd, sway_cmd, heave_cmd, yaw_cmd] ∈ [-1, 1]^4

执行逻辑：
- surge/sway/heave 通过 action_distance 映射到每步位移命令
- yaw_cmd 通过 action_angle_delta.yaw 映射到航向增量
- 经 integrated_controller + dynamics 推进一步
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV3DEnvV2Continuous(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 target_distance: float = 12.0,
                 z_range: Tuple[float, float] = (2.0, 4.0),
                 heading_coef: float = 0.7,
                 direction_coef: float = 0.6,
                 strict_direction_heading: bool = True,
                 heading_penalty_wrong: float = 0.0,
                 backward_facing_target_penalty: float = 0.5,
                 z_coef: float = 1.0,
                 z_boost_when_xy_close: float = 1.5,
                 xy_close_threshold: float = 1.0,
                 relaxed_heading_when_xy_close: bool = True,
                 success_radius: float = 0.5,
                 heave_step_override: Optional[float] = None,
                 yaw_delta_override: Optional[float] = None,
                 yaw_rate_penalty_coef: float = 0.03,
                 action_smooth_coef: float = 0.02,
                 z_lock_coef: float = 0.6,
                 improvement_coef: float = 8.0,
                 improvement_clip: float = 0.12,
                 heading_bonus_coef: float = 0.08,
                 direction_use_closing_speed: bool = True,
                 yaw_action_penalty_coef: float = 0.02,
                 yaw_action_flip_penalty_coef: float = 0.03,
                 enable_pitch_control: bool = True,
                 pitch_delta_override: Optional[float] = None,
                 pitch_limit_deg: float = 25.0,
                 min_target_z_gap: float = 1.0,
                 z_progress_coef: float = 1.2,
                 z_progress_clip: float = 0.08,
                 xy_gate_strictness: float = 1.0,
                 spiral_penalty_coef: float = 0.25,
                 max_steps: int = 4000,
                 # ---------- 初始速度随机化 ----------
                 init_velocity_range: float = 0.0,
                 init_yaw_rate_range: float = 0.0,
                 ):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")

        self.inner = LowLevelAUVEnv(
            config_path=config_path,
            render_mode=render_mode,
            curriculum_stage=curriculum_stage,
            enable_domain_randomization=enable_domain_randomization,
            seed=seed,
        )

        self.target_distance = target_distance
        self.z_range = z_range
        self.heading_coef = heading_coef
        self.direction_coef = direction_coef
        self.strict_direction_heading = strict_direction_heading
        self.heading_penalty_wrong = heading_penalty_wrong
        self.backward_facing_target_penalty = backward_facing_target_penalty
        self.z_coef = z_coef
        self.z_boost_when_xy_close = z_boost_when_xy_close
        self.xy_close_threshold = xy_close_threshold
        self.relaxed_heading_when_xy_close = relaxed_heading_when_xy_close
        self.success_radius = float(success_radius)
        self.yaw_rate_penalty_coef = float(yaw_rate_penalty_coef)
        self.action_smooth_coef = float(action_smooth_coef)
        self.z_lock_coef = float(z_lock_coef)
        self.improvement_coef = float(improvement_coef)
        self.improvement_clip = float(improvement_clip)
        self.heading_bonus_coef = float(heading_bonus_coef)
        self.direction_use_closing_speed = bool(direction_use_closing_speed)
        self.yaw_action_penalty_coef = float(yaw_action_penalty_coef)
        self.yaw_action_flip_penalty_coef = float(yaw_action_flip_penalty_coef)
        self.max_steps = max_steps

        self.enable_pitch_control = bool(enable_pitch_control)
        self.pitch_limit = np.deg2rad(float(max(1.0, pitch_limit_deg)))
        self.min_target_z_gap = float(max(0.0, min_target_z_gap))
        self.z_progress_coef = float(z_progress_coef)
        self.z_progress_clip = float(max(1e-6, z_progress_clip))
        self.xy_gate_strictness = float(max(1e-3, xy_gate_strictness))
        self.spiral_penalty_coef = float(max(0.0, spiral_penalty_coef))

        # ---------- 初始速度随机化 ----------
        self.init_velocity_range = float(init_velocity_range)
        self.init_yaw_rate_range = float(init_yaw_rate_range) if init_yaw_rate_range > 0 else self.init_velocity_range * 0.5

        self.act_dim = 5 if self.enable_pitch_control else 4
        self.action_space = spaces.Box(
            low=-np.ones(self.act_dim, dtype=np.float32),
            high=np.ones(self.act_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.obs_dim = 10
        obs_high = np.ones(self.obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        action_cfg = self.inner.config.get("action_space", {})
        distance_cfg = action_cfg.get("action_distance", {})
        angle_cfg = action_cfg.get("action_angle_delta", {})
        self.surge_step = float(distance_cfg.get("surge", 0.03))
        self.sway_step = float(distance_cfg.get("sway", 0.03))
        raw_heave = float(distance_cfg.get("heave", self.sway_step))
        raw_yaw_delta = float(angle_cfg.get("yaw", 0.0))

        if heave_step_override is not None:
            self.heave_step = float(heave_step_override)
        else:
            self.heave_step = raw_heave if raw_heave > 1e-6 else self.sway_step

        if yaw_delta_override is not None:
            self.yaw_delta = float(yaw_delta_override)
        else:
            self.yaw_delta = raw_yaw_delta if raw_yaw_delta > 1e-6 else 0.08

        raw_pitch_delta = float(angle_cfg.get("pitch", self.yaw_delta))
        if pitch_delta_override is not None:
            self.pitch_delta = float(pitch_delta_override)
        else:
            self.pitch_delta = raw_pitch_delta if raw_pitch_delta > 1e-6 else self.yaw_delta

        if raw_heave <= 1e-6 and heave_step_override is None:
            print(f"[3DContEnv] heave step in config is {raw_heave:.4f}, fallback to sway_step={self.heave_step:.4f}")
        if raw_yaw_delta <= 1e-6 and yaw_delta_override is None:
            print(f"[3DContEnv] yaw delta in config is {raw_yaw_delta:.4f}, fallback to yaw_delta={self.yaw_delta:.4f}")

        self.episode_count = 0
        self.success_count = 0
        self.prev_pos_error = None
        self.step_count = 0
        self.current_action = None
        self.prev_action = None
        self.prev_yaw_action = None
        self.prev_abs_z_error = None

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_3d_constraints(self):
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            eta[3] = 0.0
            if self.enable_pitch_control:
                eta[4] = np.clip(eta[4], -self.pitch_limit, self.pitch_limit)
            else:
                eta[4] = 0.0
            nu[3] = 0.0
            if not self.enable_pitch_control:
                nu[4] = 0.0
            eta[2] = np.clip(eta[2], self.z_range[0], self.z_range[1])
        except Exception:
            pass

    def _sample_3d_target(self):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.target_distance * 0.5, self.target_distance)
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        init_z = float(np.mean(self.z_range))
        target_z = np.random.uniform(self.z_range[0], self.z_range[1])
        if self.min_target_z_gap > 0:
            for _ in range(24):
                if abs(float(target_z - init_z)) >= self.min_target_z_gap:
                    break
                target_z = np.random.uniform(self.z_range[0], self.z_range[1])
            if abs(float(target_z - init_z)) < self.min_target_z_gap:
                target_z = float(np.clip(
                    init_z + np.sign(np.random.uniform(-1.0, 1.0) + 1e-6) * self.min_target_z_gap,
                    self.z_range[0],
                    self.z_range[1],
                ))
        return np.array([target_x, target_y, target_z], dtype=np.float64)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.inner.reset(seed=seed, options=options)

        self.inner.dynamics.eta[:3] = np.array([0.0, 0.0, np.mean(self.z_range)], dtype=np.float64)
        self.inner.dynamics.eta[3:] = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)], dtype=np.float64)
        self.inner.dynamics.nu[:] = 0.0

        # ---------- 初始速度随机化 ----------
        if self.init_velocity_range > 0:
            nu = self.inner.dynamics.nu
            nu[0] = np.random.uniform(-self.init_velocity_range, self.init_velocity_range)  # surge
            nu[1] = np.random.uniform(-self.init_velocity_range, self.init_velocity_range)  # sway
            nu[2] = np.random.uniform(-self.init_velocity_range, self.init_velocity_range)  # heave
            nu[5] = np.random.uniform(-self.init_yaw_rate_range, self.init_yaw_rate_range)  # yaw_rate

        target_pos = self._sample_3d_target()
        self.inner.target_state = np.zeros(12, dtype=np.float64)
        self.inner.target_state[:3] = target_pos

        self._enforce_3d_constraints()

        self.episode_count += 1
        self.prev_pos_error = None
        self.step_count = 0
        self.current_action = None
        self.prev_action = None
        self.prev_yaw_action = None
        self.prev_abs_z_error = None

        obs = self._get_obs()
        info = {
            "target_pos": target_pos.tolist(),
            "init_pos": self.inner.dynamics.eta[:3].tolist(),
            "init_yaw": float(self.inner.dynamics.eta[5]),
            "init_distance": float(np.linalg.norm(target_pos - self.inner.dynamics.eta[:3])),
            "mode": "3d_v2_continuous",
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 4 and self.enable_pitch_control:
            action = np.concatenate([action, np.zeros(1, dtype=np.float32)], axis=0)
        if action.size != self.act_dim:
            raise ValueError(f"Expected action dim {self.act_dim}, got {action.size}")
        action = np.clip(action, -1.0, 1.0)
        self.current_action = action.copy()

        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        dt = max(self.inner.dt, 1e-6)

        linear_cmd = np.array([
            float(action[0]) * self.surge_step,
            float(action[1]) * self.sway_step,
            float(action[2]) * self.heave_step,
        ], dtype=np.float64)
        desired_vel_body = linear_cmd / dt

        action_cfg = self.inner.config.get("action_space", {})
        slowdown_cfg = action_cfg.get("goal_slowdown", {})
        if slowdown_cfg.get("enabled", True):
            radius = float(slowdown_cfg.get("radius", 1.0))
            min_scale = float(slowdown_cfg.get("min_scale", 0.2))
            pos_error = np.linalg.norm(self.inner.target_state[:3] - eta[:3])
            if radius > 1e-6:
                scale = max(min_scale, min(1.0, pos_error / radius))
                desired_vel_body = desired_vel_body * scale

        current_vel_body = nu[:3]
        vel_error = desired_vel_body - current_vel_body
        ctl = self.inner.integrated_controller
        force_cmd = ctl.vel_kp * vel_error - ctl.vel_kd * current_vel_body
        force_cmd = np.clip(force_cmd, -ctl.max_force, ctl.max_force)

        roll_current, pitch_current, yaw_current = eta[3], eta[4], eta[5]
        p, q, r = nu[3], nu[4], nu[5]
        target_yaw = np.arctan2(
            np.sin(yaw_current + float(action[3]) * self.yaw_delta),
            np.cos(yaw_current + float(action[3]) * self.yaw_delta),
        )
        if self.enable_pitch_control:
            target_pitch = float(np.clip(
                pitch_current + float(action[4]) * self.pitch_delta,
                -self.pitch_limit,
                self.pitch_limit,
            ))
        else:
            target_pitch = 0.0
        angle_target = np.array([target_yaw, target_pitch, 0.0], dtype=np.float64)
        angle_current = np.array([yaw_current, pitch_current, roll_current], dtype=np.float64)
        omega_current = np.array([r, q, p], dtype=np.float64)

        tau_attitude, _ = ctl.pid_controller.compute(angle_target, angle_current, omega_current, self.inner.dt)
        thrust_command = ctl.thrust_allocator.allocate_with_attitude(force_cmd, tau_attitude)
        self.inner.last_thrust_command = thrust_command.copy()

        external_force = self.inner._compute_external_force()
        self.inner.dynamics.step(thrust_command, external_force)
        self.inner.water_current.step(self.inner.dt)
        self.inner.current_step += 1

        self._enforce_3d_constraints()
        self.step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()

        info = {
            **reward_info,
            **term_info,
            "action_continuous": action.tolist(),
            "mode": "3d_v2_continuous",
            "step": self.step_count,
            "success_rate": self.success_count / max(1, self.episode_count),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error_world = target[:3] - eta[:3]
        distance = np.linalg.norm(pos_error_world)

        target_angle_xy = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = np.arctan2(np.sin(target_angle_xy - yaw), np.cos(target_angle_xy - yaw))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        pos_error_body_z = pos_error_world[2]

        obs = np.array([
            pos_error_body_x / 10.0,
            pos_error_body_y / 10.0,
            pos_error_body_z / 5.0,
            heading_error / np.pi,
            target_angle_xy / np.pi,
            nu[0] / 2.0,
            nu[1] / 2.0,
            nu[2] / 2.0,
            nu[5] / 2.0,
            distance / 15.0,
        ], dtype=np.float32)

        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self) -> Tuple[float, Dict]:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error = np.linalg.norm(eta[:3] - target[:3])
        xy_error = np.linalg.norm(eta[:2] - target[:2])
        z_error = target[2] - eta[2]
        abs_z_error = abs(z_error)

        pos_error_world_xy = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world_xy[1], pos_error_world_xy[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))

        reward_info: Dict[str, float] = {}

        position_reward = -0.22 * pos_error
        reward_info["position"] = position_reward

        improvement_reward = 0.0
        if self.prev_pos_error is not None:
            delta_err = float(self.prev_pos_error - pos_error)
            clipped_delta = float(np.clip(delta_err, -self.improvement_clip, self.improvement_clip))
            improvement_reward = self.improvement_coef * clipped_delta
        self.prev_pos_error = pos_error
        reward_info["improvement"] = improvement_reward

        gate_xy = float(np.clip((xy_error / max(1e-6, self.xy_close_threshold)) ** self.xy_gate_strictness, 0.0, 1.0))

        heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
        if heading_error < np.pi / 6:
            heading_reward += self.heading_bonus_coef
        elif heading_error < np.pi / 3:
            heading_reward += 0.5 * self.heading_bonus_coef
        heading_reward *= (0.25 + 0.75 * gate_xy)
        reward_info["heading"] = heading_reward

        direction_reward = 0.0
        if pos_error > 0.1:
            yaw_now = float(eta[5])
            pitch_now = float(eta[4])
            cos_yaw = np.cos(yaw_now)
            sin_yaw = np.sin(yaw_now)
            cos_pitch = np.cos(pitch_now)
            sin_pitch = np.sin(pitch_now)
            vel_world = np.array([
                cos_yaw * cos_pitch * nu[0] - sin_yaw * nu[1] + cos_yaw * sin_pitch * nu[2],
                sin_yaw * cos_pitch * nu[0] + cos_yaw * nu[1] + sin_yaw * sin_pitch * nu[2],
                -sin_pitch * nu[0] + cos_pitch * nu[2],
            ], dtype=np.float64)

            if self.direction_use_closing_speed:
                tgt_vec3 = (target[:3] - eta[:3]).astype(np.float64)
                tgt_norm3 = np.linalg.norm(tgt_vec3)
                if tgt_norm3 > 1e-8:
                    tgt_dir3 = tgt_vec3 / tgt_norm3
                    closing_speed3 = float(np.dot(vel_world, tgt_dir3))
                    direction_reward = self.direction_coef * max(0.0, closing_speed3)
            else:
                forward_vel = nu[0]
                if self.strict_direction_heading:
                    if self.relaxed_heading_when_xy_close and xy_error < 0.5:
                        align_ok = heading_error < np.pi / 3
                    else:
                        align_ok = heading_error < np.pi / 6
                else:
                    align_ok = heading_error < 0.5
                if align_ok and forward_vel > 0:
                    direction_reward = self.direction_coef * forward_vel
            direction_reward *= (0.2 + 0.8 * gate_xy)
        reward_info["direction"] = direction_reward

        wrong_way_penalty = 0.0
        if self.heading_penalty_wrong > 0 and pos_error > 0.1 and nu[0] > 0 and heading_error > np.pi / 2:
            wrong_way_penalty = -self.heading_penalty_wrong
        reward_info["wrong_way_penalty"] = wrong_way_penalty

        backward_penalty = 0.0
        if self.backward_facing_target_penalty > 0 and pos_error > 0.5:
            if nu[0] < -0.05 and heading_error < np.pi / 2:
                backward_penalty = -self.backward_facing_target_penalty
        reward_info["backward_penalty"] = backward_penalty

        z_vel = vel_world[2] if "vel_world" in locals() else nu[2]
        z_direction_reward = 0.0
        if abs(z_error) > 0.1:
            if z_error > 0 and z_vel > 0:
                z_direction_reward = self.z_coef * z_vel
            elif z_error < 0 and z_vel < 0:
                z_direction_reward = self.z_coef * abs(z_vel)
        if xy_error < self.xy_close_threshold and z_direction_reward > 0:
            z_direction_reward *= self.z_boost_when_xy_close
        reward_info["z_direction"] = z_direction_reward

        z_progress_reward = 0.0
        if self.prev_abs_z_error is not None:
            z_delta = float(np.clip(self.prev_abs_z_error - abs_z_error, -self.z_progress_clip, self.z_progress_clip))
            z_progress_reward = self.z_progress_coef * z_delta
        self.prev_abs_z_error = abs_z_error
        if xy_error < self.xy_close_threshold:
            z_progress_reward *= self.z_boost_when_xy_close
        reward_info["z_progress"] = z_progress_reward

        z_lock_penalty = 0.0
        if xy_error < self.xy_close_threshold:
            z_lock_penalty = -self.z_lock_coef * abs(z_error)
        reward_info["z_lock_penalty"] = z_lock_penalty

        yaw_rate_penalty = -self.yaw_rate_penalty_coef * abs(nu[5])
        reward_info["yaw_rate_penalty"] = yaw_rate_penalty

        spiral_penalty = 0.0
        if self.spiral_penalty_coef > 0 and xy_error < self.xy_close_threshold and abs_z_error > 0.25:
            to_target_xy = (target[:2] - eta[:2]).astype(np.float64)
            norm_xy = np.linalg.norm(to_target_xy)
            if norm_xy > 1e-8 and "vel_world" in locals():
                t_hat = to_target_xy / norm_xy
                radial = float(np.dot(vel_world[:2], t_hat))
                tangential = float(np.sqrt(max(0.0, np.dot(vel_world[:2], vel_world[:2]) - radial * radial)))
                spiral_penalty = -self.spiral_penalty_coef * tangential
        reward_info["spiral_penalty"] = spiral_penalty

        yaw_action_penalty = 0.0
        yaw_flip_penalty = 0.0
        if self.current_action is not None:
            yaw_act = float(self.current_action[3])
            yaw_action_penalty = -self.yaw_action_penalty_coef * abs(yaw_act)
            if self.prev_yaw_action is not None:
                if yaw_act * self.prev_yaw_action < -0.04:
                    yaw_flip_penalty = -self.yaw_action_flip_penalty_coef * abs(yaw_act - self.prev_yaw_action)
            self.prev_yaw_action = yaw_act
        reward_info["yaw_action_penalty"] = yaw_action_penalty
        reward_info["yaw_flip_penalty"] = yaw_flip_penalty

        action_smooth_penalty = 0.0
        if self.prev_action is not None and self.current_action is not None:
            action_delta = np.asarray(self.current_action) - np.asarray(self.prev_action)
            action_smooth_penalty = -self.action_smooth_coef * float(np.sum(action_delta ** 2))
        reward_info["action_smooth_penalty"] = action_smooth_penalty
        if self.current_action is not None:
            self.prev_action = np.asarray(self.current_action, dtype=np.float64)

        proximity_reward = 3.0 * (self.success_radius - pos_error) if pos_error < self.success_radius else 0.0
        reward_info["proximity"] = proximity_reward

        goal_bonus = 100.0 if pos_error < self.success_radius else 0.0
        reward_info["success"] = bool(pos_error < self.success_radius)
        reward_info["goal"] = goal_bonus

        time_penalty = -0.02
        reward_info["time"] = time_penalty

        total_reward = (
            position_reward + improvement_reward + heading_reward +
            direction_reward + wrong_way_penalty + backward_penalty +
            z_direction_reward + z_progress_reward + z_lock_penalty + yaw_rate_penalty +
            spiral_penalty +
            yaw_action_penalty + yaw_flip_penalty + action_smooth_penalty +
            proximity_reward + goal_bonus + time_penalty
        )
        reward_info["total"] = total_reward
        reward_info["pos_error"] = pos_error
        reward_info["heading_error"] = heading_error
        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        info: Dict[str, Any] = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        pos_error = np.linalg.norm(eta[:3] - target[:3])

        if pos_error < self.success_radius:
            info["termination_reason"] = "success"
            self.success_count += 1
            return True, False, info

        if abs(eta[0]) > 25 or abs(eta[1]) > 25 or eta[2] < 0 or eta[2] > 10:
            info["termination_reason"] = "out_of_bounds"
            return True, False, info

        if self.step_count >= self.max_steps:
            info["termination_reason"] = "timeout"
            return False, True, info

        info["termination_reason"] = None
        info["pos_error"] = pos_error
        return False, False, info

    def render(self):
        return self.inner.render()

    def close(self):
        return self.inner.close()


class VecLowLevelAUV3DEnvV2Continuous:
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV3DEnvV2Continuous(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            obs, info = env.reset(seed=s)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(term_list),
            np.array(trunc_list),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()
