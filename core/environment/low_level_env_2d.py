"""
简化的二维低层环境（4个离散动作）
=====================================

在不改动底层动力学和控制器（PID+推力分配）的前提下，将任务简化为 x-y 平面内的导航。

动作空间（4个）：
  0: surge_forward  - 前进
  1: surge_backward - 后退
  2: sway_left      - 左平移
  3: sway_right     - 右平移

状态约束：
  - z 固定在 fixed_z（如 2.0m）
  - roll, pitch 置零（只保留 yaw）
  - 垂直速度 w, p, q 置零

设计目标：快速验证 PID+RL 架构在最简单场景下能否收敛。
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV2DEnv(gym.Env):
    """
    二维四动作封装器
    
    这是对 LowLevelAUVEnv 的轻量包装：
    - 动作空间缩减为 4 个（只保留水平平移）
    - 状态强制约束到 2D 平面
    - 目标只在 2D 平面内采样
    - 观测空间简化（可选）
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right"]
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,  # 2D 先关闭域随机化
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 use_simplified_obs: bool = True,
                 target_distance: float = 1.0,
                 min_target_distance: float = None):  # 新增：最小目标距离
        """
        Args:
            config_path: 配置文件路径（使用 2D 专用配置）
            fixed_z: 固定的 z 坐标（深度）
            use_simplified_obs: 是否使用简化的观测（只保留 2D 相关）
            target_distance: 目标距离范围
        """
        # 使用 2D 专用配置
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                # 如果 2D 配置不存在，使用默认配置
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(
            config_path=config_path,
            render_mode=render_mode,
            curriculum_stage=curriculum_stage,
            enable_domain_randomization=enable_domain_randomization,
            seed=seed,
        )

        self.fixed_z = fixed_z
        self.use_simplified_obs = use_simplified_obs
        self.target_distance = target_distance
        # 最小目标距离，默认为 target_distance * 0.5
        self.min_target_distance = min_target_distance if min_target_distance is not None else target_distance * 0.5
        
        # 映射到原始环境中水平平移动作的索引
        # 参考 LowLevelAUVEnv.ACTION_NAMES: 1=surge_forward, 2=surge_backward, 3=sway_left, 4=sway_right
        self.action_map = {
            0: 1,  # forward  -> surge_forward
            1: 2,  # backward -> surge_backward
            2: 3,  # left     -> sway_left
            3: 4,  # right    -> sway_right
        }

        self.action_space = spaces.Discrete(4)
        
        # 观测空间
        if use_simplified_obs:
            # 简化观测：[pos_error_x, pos_error_y, yaw_error, vx, vy, yaw_rate] = 6 维
            # 或者更简单：[pos_error_x, pos_error_y, vx, vy] = 4 维
            self.obs_dim = 6
            obs_high = np.ones(self.obs_dim) * 10.0
            self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        else:
            # 复用底层观测空间
            self.observation_space = self.inner.observation_space
            self.obs_dim = self.inner.obs_dim
        
        # 统计
        self.episode_count = 0
        self.success_count = 0

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_2d_state(self):
        """强制 2D 约束（就地修改动力学状态）"""
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            
            # 固定高度
            eta[2] = float(self.fixed_z)
            # 置零 roll, pitch（保留 yaw）
            eta[3] = 0.0
            eta[4] = 0.0
            
            # 取消垂直速度和姿态角速度（保留 yaw rate）
            nu[2] = 0.0  # w
            nu[3] = 0.0  # p (roll rate)
            nu[4] = 0.0  # q (pitch rate)
        except Exception:
            pass

    def _sample_2d_target(self):
        """在 2D 平面内采样目标"""
        # 随机角度
        angle = self.inner.np_random.uniform(0, 2 * np.pi)
        # 随机距离（在 min_target_distance ~ target_distance 范围内）
        distance = self.inner.np_random.uniform(self.min_target_distance, self.target_distance)
        
        # 计算目标位置（相对于初始位置）
        init_pos = self.inner.dynamics.eta[:3].copy()
        target_x = init_pos[0] + distance * np.cos(angle)
        target_y = init_pos[1] + distance * np.sin(angle)
        target_z = self.fixed_z
        
        # 目标姿态：只关心 yaw，可以设为 0 或朝向目标方向
        target_roll = 0.0
        target_pitch = 0.0
        target_yaw = 0.0
        
        # 目标速度为零
        target_vel = np.zeros(6)
        
        self.inner.target_state = np.array([
            target_x, target_y, target_z,
            target_roll, target_pitch, target_yaw,
            *target_vel
        ])

    def _get_simplified_obs(self) -> np.ndarray:
        """获取简化的 2D 观测"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 位置误差（体坐标系）
        pos_error_world = target[:2] - eta[:2]  # 只取 x, y
        yaw = eta[5]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        # 旋转到体坐标系
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        
        # yaw 误差
        target_yaw = target[5]
        yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
        
        # 速度（体坐标系）
        vx = nu[0]  # surge
        vy = nu[1]  # sway
        yaw_rate = nu[5]
        
        # 归一化
        obs = np.array([
            pos_error_body_x / 5.0,  # 归一化
            pos_error_body_y / 5.0,
            yaw_error / np.pi,
            vx / 2.0,
            vy / 2.0,
            yaw_rate / 2.0
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)

    def _compute_2d_reward(self) -> Tuple[float, Dict]:
        """计算简化的 2D 奖励 - 重点是清晰的梯度信号"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 2D 位置误差
        pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        reward_info = {}
        
        # 1. 位置误差惩罚（主要信号）
        position_reward = -0.3 * pos_error
        reward_info['position'] = position_reward
        
        # 2. 位置改进奖励（最重要的梯度信号）
        improvement_reward = 0.0
        if hasattr(self, 'prev_pos_error') and self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            # 强化改进信号：靠近得 +10，远离得 -10（相对于 0.01m 的改进）
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward
        
        # 3. 方向引导奖励（鼓励速度朝向目标）
        direction_reward = 0.0
        target_vec = target[:2] - eta[:2]  # 世界坐标系下的目标向量
        target_dist = np.linalg.norm(target_vec)
        if target_dist > 0.01:
            target_dir = target_vec / target_dist
            vel_world = nu[:2]  # 假设速度已在世界坐标系（简化）
            # 速度在目标方向的投影
            vel_toward = np.dot(vel_world, target_dir)
            direction_reward = 0.5 * vel_toward  # 朝目标移动得正奖励
        reward_info['direction'] = direction_reward
        
        # 4. 接近目标奖励
        if pos_error < 0.3:
            proximity_reward = 3.0 * (0.3 - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward
        
        # 5. 到达目标大奖励
        goal_bonus = 0.0
        if pos_error < 0.15:  # 成功阈值（0.15m，约机器人宽度一半）
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus
        
        # 6. 时间惩罚（鼓励快速完成）
        time_penalty = -0.02
        reward_info['time'] = time_penalty
        
        total_reward = (position_reward + improvement_reward + direction_reward + 
                       proximity_reward + goal_bonus + time_penalty)
        reward_info['total'] = total_reward
        reward_info['pos_error'] = pos_error
        
        return total_reward, reward_info

    def _check_2d_termination(self) -> Tuple[bool, bool, Dict]:
        """检查 2D 终止条件"""
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        # 成功（阈值 0.15m）
        if pos_error < 0.15:
            info['termination_reason'] = 'success'
            return True, False, info
        
        # 超出边界（简化）
        if abs(eta[0]) > 20 or abs(eta[1]) > 20:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info
        
        # 超时
        max_steps = int(60.0 / self.inner.dt)  # 60 秒
        if self.inner.current_step >= max_steps:
            info['termination_reason'] = 'timeout'
            return False, True, info
        
        return False, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.inner.reset(seed=seed, options=options)
        
        # 强制 2D 约束
        self._enforce_2d_state()
        
        # 重新采样 2D 目标
        self._sample_2d_target()
        
        # 再次强制约束（确保目标也在 2D）
        self._enforce_2d_state()
        
        # 初始化误差记录
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        self.prev_pos_error = np.linalg.norm(eta[:2] - target[:2])
        
        # 获取观测
        if self.use_simplified_obs:
            obs = self._get_simplified_obs()
        else:
            obs = self.inner._get_observation()
        
        info = {
            'simplified_mode': '2d_4actions',
            'target_distance': self.prev_pos_error,
            'target_pos': target[:2].tolist(),
            'init_pos': eta[:2].tolist()
        }
        
        self.episode_count += 1
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 映射动作
        action = int(action)
        mapped_action = self.action_map.get(action, 1)
        
        # 执行内层环境步
        _, _, _, _, inner_info = self.inner.step(mapped_action)
        
        # 强制 2D 约束
        self._enforce_2d_state()
        
        # 计算 2D 奖励
        reward, reward_info = self._compute_2d_reward()
        
        # 检查 2D 终止
        terminated, truncated, term_info = self._check_2d_termination()
        
        # 获取观测
        if self.use_simplified_obs:
            obs = self._get_simplified_obs()
        else:
            obs = self.inner._get_observation()
        
        # 统计成功率
        if terminated and term_info.get('termination_reason') == 'success':
            self.success_count += 1
        
        info = {
            **reward_info,
            **term_info,
            'action_2d': action,
            'action_name': self.ACTION_NAMES[action],
            'action_mapped': mapped_action,
            'simplified_mode': '2d_4actions',
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


class VecLowLevelAUV2DEnv:
    """向量化 2D 环境（简化版）"""
    
    def __init__(self, num_envs: int = 4, target_distance: float = 1.0, 
                 min_target_distance: float = None, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DEnv(
            target_distance=target_distance,
            min_target_distance=min_target_distance,
            **kwargs
        ) for _ in range(num_envs)]
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
