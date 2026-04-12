"""
二维环境 V2 - 6个离散动作（含转向）
=====================================

Stage 1 复杂化：在 4 动作基础上增加 yaw_left 和 yaw_right

动作空间（6个）：
  0: forward      - 前进
  1: backward     - 后退
  2: left         - 左平移
  3: right        - 右平移
  4: yaw_left     - 左转 (新增)
  5: yaw_right    - 右转 (新增)

新增特性：
  - 转向动作，需要学会面朝目标
  - 奖励函数增加朝向引导
  - 观测空间增加目标方向信息
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


class LowLevelAUV2DEnvV2(gym.Env):
    """
    二维六动作环境（含转向）
    
    相比 V1 的改进：
    - 增加 yaw_left/yaw_right 转向动作
    - 奖励函数鼓励面朝目标
    - 观测增加目标方向角信息
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 target_distance: float = 1.0):
        """
        Args:
            config_path: 配置文件路径
            fixed_z: 固定的 z 坐标（深度）
            target_distance: 目标距离范围
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

        self.fixed_z = fixed_z
        self.target_distance = target_distance
        
        # 6 动作映射到原始环境
        # 原始: 0=hover, 1=surge_forward, 2=surge_backward, 3=sway_left, 4=sway_right, 
        #       5=heave_up, 6=heave_down, 7=yaw_left, 8=yaw_right, ...
        self.action_map = {
            0: 1,  # forward  -> surge_forward
            1: 2,  # backward -> surge_backward
            2: 3,  # left     -> sway_left
            3: 4,  # right    -> sway_right
            4: 7,  # yaw_left  -> yaw_left (新增)
            5: 8,  # yaw_right -> yaw_right (新增)
        }

        self.action_space = spaces.Discrete(6)
        
        # 观测空间：8 维
        # [pos_error_x, pos_error_y, heading_error, target_angle, vx, vy, yaw_rate, distance]
        self.obs_dim = 8
        obs_high = np.ones(self.obs_dim) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        
        # 统计
        self.episode_count = 0
        self.success_count = 0

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_2d_state(self):
        """强制 2D 约束"""
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            
            eta[2] = float(self.fixed_z)
            eta[3] = 0.0  # roll
            eta[4] = 0.0  # pitch
            
            nu[2] = 0.0  # w
            nu[3] = 0.0  # p
            nu[4] = 0.0  # q
        except Exception:
            pass

    def _sample_2d_target(self):
        """在 2D 平面内采样目标"""
        angle = self.inner.np_random.uniform(0, 2 * np.pi)
        distance = self.inner.np_random.uniform(0.5, self.target_distance)
        
        init_pos = self.inner.dynamics.eta[:3].copy()
        target_x = init_pos[0] + distance * np.cos(angle)
        target_y = init_pos[1] + distance * np.sin(angle)
        
        self.inner.target_state = np.array([
            target_x, target_y, self.fixed_z,
            0.0, 0.0, 0.0,  # roll, pitch, yaw
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # velocities
        ])

    def _get_observation(self) -> np.ndarray:
        """获取 8 维观测"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 位置误差（世界坐标系）
        pos_error_world = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error_world)
        
        # 当前 yaw
        yaw = eta[5]
        
        # 目标方向角（世界坐标系）
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        
        # 朝向误差：当前 yaw 与目标方向的差
        heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
        
        # 位置误差旋转到体坐标系
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        
        # 速度
        vx = nu[0]  # surge
        vy = nu[1]  # sway
        yaw_rate = nu[5]
        
        # 归一化
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,      # 朝向误差 [-1, 1]
            target_angle / np.pi,       # 目标方向 [-1, 1]
            vx / 2.0,
            vy / 2.0,
            yaw_rate / 2.0,
            distance / 5.0              # 距离
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self) -> Tuple[float, Dict]:
        """计算奖励 - 增加朝向引导"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        # 计算朝向误差
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
        if hasattr(self, 'prev_pos_error') and self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward
        
        # 3. 朝向奖励（新增）- 鼓励面朝目标
        # heading_error 在 [0, pi]，越小越好
        heading_reward = 0.3 * (1.0 - heading_error / np.pi)  # [0, 0.3]
        reward_info['heading'] = heading_reward
        
        # 4. 方向引导（前进速度朝向目标）
        direction_reward = 0.0
        if pos_error > 0.1:
            # 如果朝向正确且在前进，给奖励
            forward_vel = nu[0]  # surge velocity
            if heading_error < 0.5 and forward_vel > 0:  # 朝向大致正确且前进
                direction_reward = 0.5 * forward_vel
        reward_info['direction'] = direction_reward
        
        # 5. 接近目标奖励
        if pos_error < 0.3:
            proximity_reward = 3.0 * (0.3 - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward
        
        # 6. 到达目标大奖励
        goal_bonus = 0.0
        if pos_error < 0.2:
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus
        
        # 7. 时间惩罚
        time_penalty = -0.02
        reward_info['time'] = time_penalty
        
        # 8. 原地转圈惩罚（防止一直转向不前进）
        spin_penalty = 0.0
        if abs(nu[5]) > 0.5 and abs(nu[0]) < 0.1:  # 高角速度但低前进速度
            spin_penalty = -0.1 * abs(nu[5])
        reward_info['spin_penalty'] = spin_penalty
        
        total_reward = (position_reward + improvement_reward + heading_reward + 
                       direction_reward + proximity_reward + goal_bonus + 
                       time_penalty + spin_penalty)
        reward_info['total'] = total_reward
        reward_info['pos_error'] = pos_error
        reward_info['heading_error'] = heading_error
        
        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        """检查终止条件"""
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        if pos_error < 0.2:
            info['termination_reason'] = 'success'
            return True, False, info
        
        if abs(eta[0]) > 20 or abs(eta[1]) > 20:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info
        
        max_steps = int(90.0 / self.inner.dt)  # 增加到 90 秒（转向需要更多时间）
        if self.inner.current_step >= max_steps:
            info['termination_reason'] = 'timeout'
            return False, True, info
        
        return False, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.inner.reset(seed=seed, options=options)
        
        self._enforce_2d_state()
        self._sample_2d_target()
        self._enforce_2d_state()
        
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        self.prev_pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        obs = self._get_observation()
        
        info = {
            'simplified_mode': '2d_6actions',
            'target_distance': self.prev_pos_error,
            'target_pos': target[:2].tolist(),
            'init_pos': eta[:2].tolist()
        }
        
        self.episode_count += 1
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = int(action)
        mapped_action = self.action_map.get(action, 1)
        
        _, _, _, _, inner_info = self.inner.step(mapped_action)
        
        self._enforce_2d_state()
        
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()
        
        obs = self._get_observation()
        
        if terminated and term_info.get('termination_reason') == 'success':
            self.success_count += 1
        
        info = {
            **reward_info,
            **term_info,
            'action_2d': action,
            'action_name': self.ACTION_NAMES[action],
            'action_mapped': mapped_action,
            'simplified_mode': '2d_6actions',
            'step': self.inner.current_step,
            'success_rate': self.success_count / max(1, self.episode_count)
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()

    def get_episode_statistics(self):
        stats = self.inner.get_episode_statistics() if hasattr(self.inner, 'get_episode_statistics') else {}
        stats['success_rate_2d'] = self.success_count / max(1, self.episode_count)
        stats['episode_count_2d'] = self.episode_count
        return stats


class VecLowLevelAUV2DEnvV2:
    """向量化 2D V2 环境"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DEnvV2(**kwargs) for _ in range(num_envs)]
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
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
        return np.array(obs_list), np.array(reward_list), np.array(term_list), np.array(trunc_list), info_list
    
    def close(self):
        for env in self.envs:
            env.close()
