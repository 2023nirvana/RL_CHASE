"""
带避障功能的二维环境
======================

在简化奖励环境基础上添加随机障碍物和避障奖励

特点：
1. 随机生成障碍物（圆形）
2. 避障奖励：远离障碍物得正奖励，碰撞得负奖励
3. 任务完成仍为主要目标

动作空间（6个）：forward, backward, left, right, yaw_left, yaw_right
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class Obstacle:
    """障碍物类"""
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.position = np.array([x, y])
    
    def distance_to(self, pos: np.ndarray) -> float:
        """计算到障碍物表面的距离（负数表示碰撞）"""
        return np.linalg.norm(pos - self.position) - self.radius
    
    def is_collision(self, pos: np.ndarray, auv_radius: float = 0.2) -> bool:
        """检查是否碰撞"""
        return self.distance_to(pos) < auv_radius


class LowLevelAUV2DObstacleEnv(gym.Env):
    """
    带避障功能的2D环境
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 target_distance: float = 5.0,
                 # 障碍物参数
                 num_obstacles: int = 5,
                 current_num_obstacles: int = None,  # 当前实际使用的障碍物数量（用于课程学习）
                 min_obstacle_radius: float = 0.3,
                 max_obstacle_radius: float = 0.8,
                 min_obstacle_distance: float = 1.0,  # 障碍物与起点/终点的最小距离
                 auv_radius: float = 0.2,
                 # 奖励参数
                 goal_reward: float = 100.0,
                 collision_penalty: float = -50.0,
                 obstacle_avoidance_coef: float = 0.5,  # 避障奖励系数
                 distance_coef: float = 0.5,
                 heading_coef: float = 0.3,
                 time_penalty: float = 0.01,
                 success_threshold: float = 0.3,
                 max_steps: int = 2000,
                 # 避障奖励类型（已废弃，使用简化逻辑）
                 use_dense_obstacle_reward: bool = True):
        """
        Args:
            num_obstacles: 障碍物数量
            min_obstacle_radius: 最小障碍物半径
            max_obstacle_radius: 最大障碍物半径
            min_obstacle_distance: 障碍物与起点/终点的最小距离
            auv_radius: AUV的碰撞半径
            collision_penalty: 碰撞惩罚
            obstacle_avoidance_coef: 避障奖励系数
            use_dense_obstacle_reward: 是否使用密集避障奖励
        """
        # 配置路径
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(config_path=config_path, render_mode=render_mode, seed=seed)
        
        self.fixed_z = fixed_z
        self.target_distance = target_distance
        
        # 障碍物配置
        self.num_obstacles = num_obstacles  # 最大障碍物数量
        self.current_num_obstacles = current_num_obstacles if current_num_obstacles is not None else num_obstacles  # 当前使用的数量
        self.min_obstacle_radius = min_obstacle_radius
        self.max_obstacle_radius = max_obstacle_radius
        self.min_obstacle_distance = min_obstacle_distance
        self.auv_radius = auv_radius
        self.obstacles: List[Obstacle] = []
        
        # 奖励配置
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.obstacle_avoidance_coef = obstacle_avoidance_coef
        self.distance_coef = distance_coef
        self.heading_coef = heading_coef
        self.time_penalty = time_penalty
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.use_dense_obstacle_reward = use_dense_obstacle_reward
        
        # 动作映射
        self.action_map = {
            0: 1,  # forward
            1: 2,  # backward
            2: 3,  # left
            3: 4,  # right
            4: 7,  # yaw_left
            5: 8,  # yaw_right
        }
        
        self.action_space = spaces.Discrete(6)
        
        # 观测空间：8 + 2*num_obstacles（基础obs + 每个障碍物的相对距离和方向）
        # 简化：只用最近1个障碍物的信息（减少观测维度，降低学习难度）
        self.k_nearest = 1  # 只考虑最近1个障碍物
        self.obs_dim = 8 + 2 * self.k_nearest  # 每个障碍物：距离 + 方向角 = 10维
        self.observation_space = spaces.Box(
            low=-10 * np.ones(self.obs_dim), 
            high=10 * np.ones(self.obs_dim), 
            dtype=np.float32
        )
        
        # 状态
        self.step_count = 0
        self.prev_distance = None
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0

    def _enforce_2d(self):
        """强制2D约束"""
        try:
            self.inner.dynamics.eta[2] = self.fixed_z
            self.inner.dynamics.eta[3:5] = 0.0
            self.inner.dynamics.nu[2:5] = 0.0
        except:
            pass

    def _sample_obstacles(self):
        """采样障碍物位置（根据current_num_obstacles）"""
        self.obstacles = []
        
        # 如果当前障碍物数量为0，直接返回（课程学习：无障碍物阶段）
        if self.current_num_obstacles == 0:
            return
        
        init_pos = self.inner.dynamics.eta[:2]
        target_pos = self.inner.target_state[:2]
        
        # 在起点和终点之间的区域生成障碍物
        center = (init_pos + target_pos) / 2
        spread = max(self.target_distance * 0.8, 3.0)
        
        max_attempts = 100
        for _ in range(self.current_num_obstacles):
            for attempt in range(max_attempts):
                # 随机位置
                x = center[0] + np.random.uniform(-spread, spread)
                y = center[1] + np.random.uniform(-spread, spread)
                radius = np.random.uniform(self.min_obstacle_radius, self.max_obstacle_radius)
                
                # 检查是否与起点/终点太近
                dist_to_start = np.linalg.norm(np.array([x, y]) - init_pos) - radius
                dist_to_target = np.linalg.norm(np.array([x, y]) - target_pos) - radius
                
                if dist_to_start > self.min_obstacle_distance and dist_to_target > self.min_obstacle_distance:
                    # 检查是否与其他障碍物重叠
                    valid = True
                    for obs in self.obstacles:
                        if np.linalg.norm(np.array([x, y]) - obs.position) < radius + obs.radius + 0.5:
                            valid = False
                            break
                    
                    if valid:
                        self.obstacles.append(Obstacle(x, y, radius))
                        break

    def _sample_target(self):
        """采样目标位置"""
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(self.target_distance * 0.8, self.target_distance)
        
        init_pos = self.inner.dynamics.eta[:2]
        target_x = init_pos[0] + dist * np.cos(angle)
        target_y = init_pos[1] + dist * np.sin(angle)
        
        self.inner.target_state = np.zeros(12)
        self.inner.target_state[:2] = [target_x, target_y]
        self.inner.target_state[2] = self.fixed_z

    def _get_obstacle_obs(self) -> np.ndarray:
        """获取障碍物观测（最近K个障碍物的距离和方向）"""
        pos = self.inner.dynamics.eta[:2]
        yaw = self.inner.dynamics.eta[5]
        
        # 计算到所有障碍物的距离
        obstacle_info = []
        for obs in self.obstacles:
            dist = obs.distance_to(pos)
            rel_pos = obs.position - pos
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            # 转换到体坐标系
            rel_angle = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))
            obstacle_info.append((dist, rel_angle))
        
        # 按距离排序，取最近K个
        obstacle_info.sort(key=lambda x: x[0])
        
        obs_array = []
        for i in range(self.k_nearest):
            if i < len(obstacle_info):
                dist, angle = obstacle_info[i]
                obs_array.extend([dist / 5.0, angle / np.pi])  # 归一化
            else:
                obs_array.extend([2.0, 0.0])  # 无障碍物时的默认值
        
        return np.array(obs_array, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        # 位置误差
        pos_error = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error)
        
        # 目标方向角
        target_angle = np.arctan2(pos_error[1], pos_error[0])
        
        # 朝向误差
        yaw = eta[5]
        heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
        
        # 体坐标系位置误差
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error[0] + sin_yaw * pos_error[1]
        pos_error_body_y = -sin_yaw * pos_error[0] + cos_yaw * pos_error[1]
        
        base_obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            nu[0] / 2.0,
            nu[1] / 2.0,
            nu[5] / 2.0,
            distance / 5.0
        ], dtype=np.float32)
        
        # 障碍物观测
        obstacle_obs = self._get_obstacle_obs()
        
        obs = np.concatenate([base_obs, obstacle_obs])
        return np.clip(obs, -10.0, 10.0)

    def _compute_obstacle_reward(self) -> Tuple[float, Dict]:
        """计算避障奖励（简化版）"""
        pos = self.inner.dynamics.eta[:2]
        
        min_dist = float('inf')
        collision = False
        
        for obs in self.obstacles:
            dist = obs.distance_to(pos)
            min_dist = min(min_dist, dist)
            if obs.is_collision(pos, self.auv_radius):
                collision = True
        
        reward_info = {
            'min_obstacle_dist': min_dist,
            'collision': collision
        }
        
        if collision:
            # 碰撞：大惩罚
            reward = self.collision_penalty
            reward_info['obstacle_reward'] = self.collision_penalty
        else:
            # 简化：只给安全距离奖励（距离障碍物越远越好，但奖励很小，主要靠碰撞惩罚）
            # 只在非常近时给轻微惩罚，避免过度复杂
            if min_dist < 0.5:
                # 非常近时给轻微惩罚
                reward = -self.obstacle_avoidance_coef * 0.5 * (0.5 - min_dist)
            else:
                reward = 0.0
            reward_info['obstacle_reward'] = reward
        
        return reward, reward_info

    def _compute_reward(self) -> Tuple[float, Dict]:
        """计算总奖励"""
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        pos_error = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error)
        
        # 朝向误差
        target_angle = np.arctan2(pos_error[1], pos_error[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))
        
        reward_info = {'pos_error': distance, 'heading_error': heading_error}
        
        # 成功奖励
        success = distance < self.success_threshold
        goal_bonus = self.goal_reward if success else 0.0
        reward_info['success'] = success
        reward_info['goal'] = goal_bonus
        
        # 距离奖励（势能塑形）
        if self.prev_distance is not None:
            shaping = 5.0 * (self.prev_distance - distance)
        else:
            shaping = 0.0
        self.prev_distance = distance
        
        # 朝向奖励
        heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
        
        # 避障奖励
        obstacle_reward, obstacle_info = self._compute_obstacle_reward()
        reward_info.update(obstacle_info)
        
        # 总奖励 = 目标奖励 + 距离塑形 + 朝向奖励 + 避障奖励 - 时间惩罚
        reward = goal_bonus + shaping + heading_reward + obstacle_reward - self.time_penalty
        
        reward_info['shaping'] = shaping
        reward_info['heading'] = heading_reward
        reward_info['time'] = -self.time_penalty
        reward_info['total'] = reward
        
        return reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        """检查终止条件"""
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        distance = np.linalg.norm(eta[:2] - target[:2])
        
        # 成功
        if distance < self.success_threshold:
            return True, False, {'termination_reason': 'success', 'pos_error': distance}
        
        # 碰撞
        for obs in self.obstacles:
            if obs.is_collision(eta[:2], self.auv_radius):
                self.collision_count += 1
                return True, False, {'termination_reason': 'collision', 'pos_error': distance}
        
        # 超出边界
        if abs(eta[0]) > 25 or abs(eta[1]) > 25:
            return True, False, {'termination_reason': 'out_of_bounds', 'pos_error': distance}
        
        # 超时
        if self.step_count >= self.max_steps:
            return False, True, {'termination_reason': 'timeout', 'pos_error': distance}
        
        return False, False, {'termination_reason': None, 'pos_error': distance}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        self.inner.reset(seed=seed, options=options)
        
        # 初始化位置
        self.inner.dynamics.eta[:3] = [0, 0, self.fixed_z]
        self.inner.dynamics.eta[3:] = [0, 0, np.random.uniform(-np.pi, np.pi)]
        self.inner.dynamics.nu[:] = 0
        
        self._sample_target()
        self._sample_obstacles()
        self._enforce_2d()
        
        self.step_count = 0
        self.prev_distance = np.linalg.norm(self.inner.dynamics.eta[:2] - self.inner.target_state[:2])
        self.episode_count += 1
        
        obs = self._get_obs()
        info = {
            'target_pos': self.inner.target_state[:2].tolist(),
            'init_distance': self.prev_distance,
            'num_obstacles': len(self.obstacles),
            'obstacles': [(o.x, o.y, o.radius) for o in self.obstacles]
        }
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = int(action)
        mapped_action = self.action_map.get(action, 1)
        
        self.inner.step(mapped_action)
        self._enforce_2d()
        self.step_count += 1
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()
        
        if terminated and term_info.get('termination_reason') == 'success':
            self.success_count += 1
        
        info = {
            **reward_info,
            **term_info,
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'step': self.step_count,
            'success_rate': self.success_count / max(1, self.episode_count),
            'collision_rate': self.collision_count / max(1, self.episode_count)
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()
    
    def get_obstacles(self) -> List[Tuple[float, float, float]]:
        """获取障碍物列表 [(x, y, radius), ...]"""
        return [(o.x, o.y, o.radius) for o in self.obstacles]
    
    def set_num_obstacles(self, num: int):
        """设置当前障碍物数量（用于课程学习）"""
        self.current_num_obstacles = min(num, self.num_obstacles)


class VecLowLevelAUV2DObstacleEnv:
    """向量化避障环境"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DObstacleEnv(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        results = [env.reset(seed=seed+i if seed else None) for i, env in enumerate(self.envs)]
        return np.array([r[0] for r in results]), [r[1] for r in results]
    
    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
            results.append((obs, reward, term, trunc, info))
        return (np.array([r[0] for r in results]), np.array([r[1] for r in results]),
                np.array([r[2] for r in results]), np.array([r[3] for r in results]),
                [r[4] for r in results])
    
    def close(self):
        for env in self.envs:
            env.close()


def make_obstacle_env(**kwargs):
    """创建避障环境"""
    return LowLevelAUV2DObstacleEnv(**kwargs)
