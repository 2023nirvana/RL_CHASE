"""
简化奖励的二维环境 - 实验不同奖励结构
==========================================

目的：解耦复杂奖励，测试最简单的奖励是否能训练成功

提供多种奖励模式：
1. sparse_only      - 纯稀疏：只有到达给奖励
2. distance_linear  - 线性距离：-k * distance
3. distance_exp     - 指数距离：-exp(distance) 或 exp(-distance)
4. heading_only     - 只有朝向奖励
5. minimal          - 最小化：距离 + 朝向 + 到达

动作空间（6个）：forward, backward, left, right, yaw_left, yaw_right
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV2DSimpleRewardEnv(gym.Env):
    """
    简化奖励的2D环境
    
    可通过 reward_type 参数选择不同奖励模式
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    
    # 奖励类型常量
    REWARD_SPARSE_ONLY = 'sparse_only'           # 纯稀疏
    REWARD_SPARSE_HEADING = 'sparse_heading'     # 稀疏+朝向（改进版稀疏）
    REWARD_DISTANCE_LINEAR = 'distance_linear'   # 线性距离
    REWARD_DISTANCE_EXP = 'distance_exp'         # 指数距离
    REWARD_HEADING_ONLY = 'heading_only'         # 只有朝向
    REWARD_MINIMAL = 'minimal'                   # 最小化组合
    REWARD_SHAPING = 'shaping'                   # 势能塑形
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 target_distance: float = 3.0,
                 reward_type: str = 'minimal',
                 # 奖励参数
                 goal_reward: float = 100.0,
                 distance_coef: float = 1.0,
                 heading_coef: float = 0.3,
                 time_penalty: float = 0.01,
                 success_threshold: float = 0.3,
                 max_steps: int = 2000):
        """
        Args:
            reward_type: 奖励类型，见 REWARD_* 常量
            goal_reward: 到达目标的奖励
            distance_coef: 距离奖励系数
            heading_coef: 朝向奖励系数
            time_penalty: 每步时间惩罚
            success_threshold: 成功阈值（米）
            max_steps: 最大步数
        """
        # 配置路径
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(config_path=config_path, render_mode=render_mode, seed=seed)
        
        self.fixed_z = fixed_z
        self.target_distance = target_distance
        
        # 奖励配置
        self.reward_type = reward_type
        self.goal_reward = goal_reward
        self.distance_coef = distance_coef
        self.heading_coef = heading_coef
        self.time_penalty = time_penalty
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        
        # 动作映射
        self.action_map = {
            0: 1,  # forward  -> surge_forward
            1: 2,  # backward -> surge_backward
            2: 3,  # left     -> sway_left
            3: 4,  # right    -> sway_right
            4: 7,  # yaw_left
            5: 8,  # yaw_right
        }
        
        self.action_space = spaces.Discrete(6)
        
        # 观测空间：8维
        # [pos_error_x, pos_error_y, heading_error, target_angle, vx, vy, yaw_rate, distance]
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=-10 * np.ones(self.obs_dim), 
            high=10 * np.ones(self.obs_dim), 
            dtype=np.float32
        )
        
        # 状态
        self.step_count = 0
        self.prev_distance = None  # 用于势能塑形
        self.episode_count = 0
        self.success_count = 0

    def _enforce_2d(self):
        """强制2D约束"""
        try:
            self.inner.dynamics.eta[2] = self.fixed_z
            self.inner.dynamics.eta[3:5] = 0.0
            self.inner.dynamics.nu[2:5] = 0.0
        except:
            pass

    def _sample_target(self):
        """采样目标位置"""
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(self.target_distance * 0.5, self.target_distance)
        
        init_pos = self.inner.dynamics.eta[:2]
        target_x = init_pos[0] + dist * np.cos(angle)
        target_y = init_pos[1] + dist * np.sin(angle)
        
        self.inner.target_state = np.zeros(12)
        self.inner.target_state[:2] = [target_x, target_y]
        self.inner.target_state[2] = self.fixed_z

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
        
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            nu[0] / 2.0,  # vx
            nu[1] / 2.0,  # vy
            nu[5] / 2.0,  # yaw_rate
            distance / 5.0
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self) -> Tuple[float, Dict]:
        """
        根据 reward_type 计算奖励
        """
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
        
        # 根据类型计算奖励
        if self.reward_type == self.REWARD_SPARSE_ONLY:
            # 纯稀疏：只有到达给奖励
            reward = goal_bonus
        
        elif self.reward_type == self.REWARD_SPARSE_HEADING:
            # 改进版稀疏：到达奖励 + 稀疏朝向奖励
            # 当朝向正确时给小额正奖励，鼓励面向目标移动
            heading_bonus = 0.0
            if heading_error < np.pi / 6:  # 朝向误差小于30度
                heading_bonus = 0.1  # 小额正向奖励
            elif heading_error < np.pi / 3:  # 朝向误差小于60度
                heading_bonus = 0.05
            # 如果朝向完全相反，给小惩罚
            elif heading_error > np.pi * 0.8:
                heading_bonus = -0.05
            
            reward = goal_bonus + heading_bonus
            reward_info['heading_bonus'] = heading_bonus
            
        elif self.reward_type == self.REWARD_DISTANCE_LINEAR:
            # 线性距离：-k * distance
            reward = -self.distance_coef * distance + goal_bonus
            reward_info['distance'] = -self.distance_coef * distance
            
        elif self.reward_type == self.REWARD_DISTANCE_EXP:
            # 指数距离：exp(-distance) - 1 (接近时趋近0，远离时趋近-1)
            exp_reward = np.exp(-distance) - 1.0
            reward = self.distance_coef * exp_reward + goal_bonus
            reward_info['distance'] = self.distance_coef * exp_reward
            
        elif self.reward_type == self.REWARD_HEADING_ONLY:
            # 只有朝向：鼓励面朝目标
            heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
            reward = heading_reward + goal_bonus
            reward_info['heading'] = heading_reward
            
        elif self.reward_type == self.REWARD_MINIMAL:
            # 最小化组合：距离 + 朝向 + 到达
            distance_reward = -self.distance_coef * distance
            heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
            reward = distance_reward + heading_reward + goal_bonus - self.time_penalty
            reward_info['distance'] = distance_reward
            reward_info['heading'] = heading_reward
            reward_info['time'] = -self.time_penalty
            
        elif self.reward_type == self.REWARD_SHAPING:
            # 势能塑形：基于距离变化
            if self.prev_distance is not None:
                shaping = 10.0 * (self.prev_distance - distance)  # 靠近得正奖励
            else:
                shaping = 0.0
            self.prev_distance = distance
            
            heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
            reward = shaping + heading_reward + goal_bonus - self.time_penalty
            reward_info['shaping'] = shaping
            reward_info['heading'] = heading_reward
            reward_info['time'] = -self.time_penalty
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
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
        
        # 超出边界
        if abs(eta[0]) > 20 or abs(eta[1]) > 20:
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
        self._enforce_2d()
        
        self.step_count = 0
        self.prev_distance = np.linalg.norm(self.inner.dynamics.eta[:2] - self.inner.target_state[:2])
        self.episode_count += 1
        
        obs = self._get_obs()
        info = {
            'target_pos': self.inner.target_state[:2].tolist(),
            'init_distance': self.prev_distance,
            'reward_type': self.reward_type
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
            'success_rate': self.success_count / max(1, self.episode_count)
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()


class VecLowLevelAUV2DSimpleRewardEnv:
    """向量化简化奖励环境"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DSimpleRewardEnv(**kwargs) for _ in range(num_envs)]
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


# ================== 便捷函数 ==================
def make_sparse_env(**kwargs):
    """创建纯稀疏奖励环境"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='sparse_only', **kwargs)

def make_sparse_heading_env(**kwargs):
    """创建改进版稀疏奖励环境（稀疏+朝向）"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='sparse_heading', **kwargs)

def make_linear_env(**kwargs):
    """创建线性距离奖励环境"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='distance_linear', **kwargs)

def make_exp_env(**kwargs):
    """创建指数距离奖励环境"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='distance_exp', **kwargs)

def make_minimal_env(**kwargs):
    """创建最小化组合奖励环境"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='minimal', **kwargs)

def make_shaping_env(**kwargs):
    """创建势能塑形奖励环境"""
    return LowLevelAUV2DSimpleRewardEnv(reward_type='shaping', **kwargs)


# ================== 课程学习环境 ==================
class CurriculumLowLevelAUV2DEnv(LowLevelAUV2DSimpleRewardEnv):
    """
    课程学习环境 - 逐步增加目标距离
    
    训练策略：
    1. 先从短距离（如1m）开始训练
    2. 当成功率达到阈值后，自动增加目标距离
    3. 逐步扩展到更远距离
    """
    
    def __init__(self,
                 initial_distance: float = 1.0,
                 max_distance: float = 10.0,
                 distance_increment: float = 0.5,
                 success_threshold_to_advance: float = 0.8,
                 window_size: int = 100,
                 **kwargs):
        """
        Args:
            initial_distance: 初始目标距离
            max_distance: 最大目标距离
            distance_increment: 每次增加的距离
            success_threshold_to_advance: 进阶到下一阶段所需的成功率
            window_size: 计算成功率的窗口大小
        """
        self.current_target_distance = initial_distance
        self.max_distance = max_distance
        self.distance_increment = distance_increment
        self.success_threshold_to_advance = success_threshold_to_advance
        self.window_size = window_size
        
        # 成功率追踪
        self.recent_successes = []
        self.curriculum_level = 0
        self.total_advances = 0
        
        # 使用当前距离初始化
        super().__init__(target_distance=initial_distance, **kwargs)
    
    def _check_curriculum_advance(self):
        """检查是否应该进阶到下一阶段"""
        if len(self.recent_successes) < self.window_size:
            return False
        
        success_rate = sum(self.recent_successes[-self.window_size:]) / self.window_size
        
        if success_rate >= self.success_threshold_to_advance:
            if self.current_target_distance < self.max_distance:
                self.current_target_distance = min(
                    self.current_target_distance + self.distance_increment,
                    self.max_distance
                )
                self.target_distance = self.current_target_distance
                self.curriculum_level += 1
                self.total_advances += 1
                self.recent_successes = []  # 重置成功率窗口
                print(f"\n>>> Curriculum Advanced! Level {self.curriculum_level}, "
                      f"Target Distance: {self.current_target_distance:.1f}m")
                return True
        return False
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['curriculum_level'] = self.curriculum_level
        info['current_target_distance'] = self.current_target_distance
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 记录成功/失败
        if terminated or truncated:
            success = info.get('termination_reason') == 'success'
            self.recent_successes.append(1 if success else 0)
            
            # 检查是否进阶
            self._check_curriculum_advance()
        
        info['curriculum_level'] = self.curriculum_level
        info['current_target_distance'] = self.current_target_distance
        if len(self.recent_successes) >= 10:
            info['recent_success_rate'] = sum(self.recent_successes[-100:]) / min(100, len(self.recent_successes))
        
        return obs, reward, terminated, truncated, info
    
    def get_curriculum_info(self):
        """获取课程学习信息"""
        return {
            'level': self.curriculum_level,
            'current_distance': self.current_target_distance,
            'max_distance': self.max_distance,
            'total_advances': self.total_advances,
            'recent_success_rate': sum(self.recent_successes[-100:]) / max(1, min(100, len(self.recent_successes)))
        }


def make_curriculum_env(**kwargs):
    """创建课程学习环境"""
    return CurriculumLowLevelAUV2DEnv(**kwargs)
