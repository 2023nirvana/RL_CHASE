"""
三维环境 - 8个离散动作
===========================

方案A：从2D扩展到3D

动作空间（8个）：
  0: forward      - 前进
  1: backward     - 后退
  2: left         - 左平移
  3: right        - 右平移
  4: yaw_left     - 左转
  5: yaw_right    - 右转
  6: heave_up     - 上升 (新增)
  7: heave_down   - 下潜 (新增)

观测空间（10维）：
  - pos_error_x, pos_error_y, pos_error_z (体坐标系)
  - heading_error, target_angle
  - vx, vy, vz, yaw_rate
  - distance
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


class LowLevelAUV3DEnv(gym.Env):
    """
    三维八动作环境
    
    相比 2D V2 的改进：
    - 增加 heave_up/heave_down 深度控制动作
    - 目标点扩展到 3D 空间
    - 观测增加 z 轴信息
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right", "heave_up", "heave_down"]
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 target_distance: float = 3.0,
                 z_range: Tuple[float, float] = (2.0, 4.0),
                 heading_coef: float = 0.5,
                 direction_coef: float = 0.6,
                 strict_direction_heading: bool = False,
                 heading_penalty_wrong: float = 0.0):
        """
        Args:
            target_distance: 水平面目标距离范围
            z_range: 深度范围 (min_z, max_z)
            heading_coef: 朝向目标奖励系数，越大越强调对准目标再前进
            direction_coef: 对准且前进时的速度奖励系数
            strict_direction_heading: True 时仅当朝向误差<30°才给前进奖励，促先对准再走
            heading_penalty_wrong: 朝向误差>90°且仍在前进时的小惩罚，抑制“反着走”
        """
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

        # 8 动作映射
        self.action_map = {
            0: 1,   # forward  -> surge_forward
            1: 2,   # backward -> surge_backward
            2: 3,   # left     -> sway_left
            3: 4,   # right    -> sway_right
            4: 7,   # yaw_left  -> yaw_left
            5: 8,   # yaw_right -> yaw_right
            6: 5,   # heave_up   -> heave_up (新增)
            7: 6,   # heave_down -> heave_down (新增)
        }

        self.action_space = spaces.Discrete(8)
        
        # 观测空间：10 维
        self.obs_dim = 10
        obs_high = np.ones(self.obs_dim) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        
        self.episode_count = 0
        self.success_count = 0
        self.prev_pos_error = None
        self.step_count = 0

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_3d_constraints(self):
        """强制 3D 约束（保持 roll/pitch 稳定）"""
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            
            # 保持 roll 和 pitch 为 0（简化问题）
            eta[3] = 0.0  # roll
            eta[4] = 0.0  # pitch
            
            nu[3] = 0.0   # p (roll rate)
            nu[4] = 0.0   # q (pitch rate)
            
            # 限制 z 范围
            eta[2] = np.clip(eta[2], self.z_range[0], self.z_range[1])
        except Exception:
            pass

    def _sample_3d_target(self):
        """采样 3D 目标点"""
        # 水平面随机方向和距离
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.target_distance * 0.5, self.target_distance)
        
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        
        # 深度随机
        target_z = np.random.uniform(self.z_range[0], self.z_range[1])
        
        return np.array([target_x, target_y, target_z])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        obs, info = self.inner.reset(seed=seed, options=options)
        
        # 初始化 AUV 位置在原点附近
        self.inner.dynamics.eta[:3] = np.array([0.0, 0.0, np.mean(self.z_range)])
        self.inner.dynamics.eta[3:] = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])  # 随机 yaw
        self.inner.dynamics.nu[:] = 0.0
        
        # 设置 3D 目标
        target_pos = self._sample_3d_target()
        self.inner.target_state = np.zeros(6)
        self.inner.target_state[:3] = target_pos
        
        self._enforce_3d_constraints()
        
        self.episode_count += 1
        self.prev_pos_error = None
        self.step_count = 0
        
        obs = self._get_obs()
        info = {
            'target_pos': target_pos.tolist(),
            'init_pos': self.inner.dynamics.eta[:3].tolist()
        }
        
        return obs, info

    def step(self, action: int):
        """执行动作"""
        mapped_action = self.action_map[action]
        _, _, _, _, info = self.inner.step(mapped_action)
        
        self._enforce_3d_constraints()
        self.step_count += 1
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()
        
        info.update(reward_info)
        info.update(term_info)
        info['action_name'] = self.ACTION_NAMES[action]
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """获取 10 维观测"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 3D 位置误差（世界坐标系）
        pos_error_world = target[:3] - eta[:3]
        distance = np.linalg.norm(pos_error_world)
        
        # 水平面目标角度
        target_angle_xy = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        
        # 朝向误差
        heading_error = np.arctan2(np.sin(target_angle_xy - yaw), np.cos(target_angle_xy - yaw))
        
        # 位置误差旋转到体坐标系
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        pos_error_body_z = pos_error_world[2]  # z 不受 yaw 影响
        
        # 速度
        vx = nu[0]  # surge
        vy = nu[1]  # sway
        vz = nu[2]  # heave
        yaw_rate = nu[5]
        
        # 归一化
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            pos_error_body_z / 5.0,
            heading_error / np.pi,
            target_angle_xy / np.pi,
            vx / 2.0,
            vy / 2.0,
            vz / 2.0,
            yaw_rate / 2.0,
            distance / 5.0
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self) -> Tuple[float, Dict]:
        """计算奖励"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 3D 位置误差
        pos_error = np.linalg.norm(eta[:3] - target[:3])
        
        # 水平面朝向误差
        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))
        
        reward_info = {}
        
        # 1. 位置误差惩罚
        position_reward = -0.3 * pos_error
        reward_info['position'] = position_reward
        
        # 2. 位置改进奖励
        improvement_reward = 0.0
        if self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward
        
        # 3. 朝向奖励（权重可调，加强“对准目标”的激励）
        heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
        # 3.5 小角度对准奖励：朝向误差 < 30° 时额外奖励，鼓励先对准再前进
        if heading_error < np.pi / 6:
            heading_reward += 0.15
        elif heading_error < np.pi / 3:
            heading_reward += 0.05
        reward_info['heading'] = heading_reward

        # 4. 方向引导（水平面）：对准且前进时给更多奖励
        direction_reward = 0.0
        if pos_error > 0.1:
            forward_vel = nu[0]
            if self.strict_direction_heading:
                align_ok = heading_error < np.pi / 6  # 仅 30° 内才奖励前进
            else:
                align_ok = heading_error < 0.5
            if align_ok and forward_vel > 0:
                direction_reward = self.direction_coef * forward_vel
        reward_info['direction'] = direction_reward

        # 4.2 反朝向惩罚：朝向误差>90°却还在前进，小惩罚
        wrong_way_penalty = 0.0
        if self.heading_penalty_wrong > 0 and pos_error > 0.1 and nu[0] > 0 and heading_error > np.pi / 2:
            wrong_way_penalty = -self.heading_penalty_wrong
        reward_info['wrong_way_penalty'] = wrong_way_penalty
        
        # 4.5 Z 轴方向引导（新增）
        z_error = target[2] - eta[2]
        z_vel = nu[2]  # heave velocity
        z_direction_reward = 0.0
        if abs(z_error) > 0.1:
            # 如果 z 方向移动正确，给奖励
            if z_error > 0 and z_vel > 0:  # 需要上升且正在上升
                z_direction_reward = 0.5 * z_vel
            elif z_error < 0 and z_vel < 0:  # 需要下潜且正在下潜
                z_direction_reward = 0.5 * abs(z_vel)
        reward_info['z_direction'] = z_direction_reward
        
        # 5. 接近目标奖励
        if pos_error < 0.5:
            proximity_reward = 3.0 * (0.5 - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward
        
        # 6. 到达目标大奖励（3D 阈值放宽到 0.5m）
        goal_bonus = 0.0
        if pos_error < 0.5:
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus
        
        # 7. 时间惩罚
        time_penalty = -0.02
        reward_info['time'] = time_penalty
        
        total_reward = (position_reward + improvement_reward + heading_reward +
                       direction_reward + wrong_way_penalty + z_direction_reward +
                       proximity_reward + goal_bonus + time_penalty)
        reward_info['total'] = total_reward
        reward_info['pos_error'] = pos_error
        reward_info['heading_error'] = heading_error
        
        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        """检查终止条件"""
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:3] - target[:3])
        
        # 成功（3D 阈值放宽到 0.5m）
        if pos_error < 0.5:
            info['termination_reason'] = 'success'
            self.success_count += 1
            return True, False, info
        
        # 出界
        if abs(eta[0]) > 20 or abs(eta[1]) > 20 or eta[2] < 0 or eta[2] > 10:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info
        
        # 超时
        if self.step_count >= 3000:
            info['termination_reason'] = 'timeout'
            return False, True, info
        
        info['termination_reason'] = None
        info['pos_error'] = pos_error
        return False, False, info

    def render(self):
        return self.inner.render()

    def close(self):
        return self.inner.close()


class VecLowLevelAUV3DEnv:
    """向量化 3D 环境"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV3DEnv(**kwargs) for _ in range(num_envs)]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list
    
    def step(self, actions):
        obs_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, reset_info = env.reset()
                info['reset_info'] = reset_info
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (np.array(obs_list), np.array(reward_list), 
                np.array(terminated_list), np.array(truncated_list), info_list)
