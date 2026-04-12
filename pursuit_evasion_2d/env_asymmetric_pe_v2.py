"""
二维非对称追逃博弈环境 V2 - 完整重新设计
===========================================

核心改进：
1. 更平衡的动力学参数设计
2. 丰富的稠密奖励函数
3. 增强的观测空间
4. 课程学习支持
5. 多种逃脱者策略模式

研究思路：追击者最大速度 < 逃脱者，但最大加速度 > 逃脱者；
借助边界、预测拦截策略，利用机动性优势完成追捕。

关键设计原则：
- 追击者速度劣势要适度，不能太大（建议 0.85-0.95 倍）
- 追击者加速度优势要显著（建议 2-4 倍）
- 逃脱者转向能力受限
- 边界可以成为追击者的战术工具
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============== 动力学参数（经过仔细平衡） ==============
# 方案：追击者速度略低但加速度显著更高，且逃脱者转向受限
DEFAULT_PURSUER_V_MAX = 1.2      # 追击者最大速度
DEFAULT_PURSUER_A_MAX = 2.5     # 追击者最大加速度（高机动性）
DEFAULT_EVADER_V_MAX = 1.4      # 逃脱者最大速度（略高）
DEFAULT_EVADER_A_MAX = 0.6      # 逃脱者最大加速度（低机动性）
DEFAULT_EVADER_TURN_RATE = 2.0  # 逃脱者最大转向率 (rad/s)

# 追捕参数
DEFAULT_CATCH_RADIUS = 0.5
DEFAULT_WORLD_SIZE = 10.0
DEFAULT_MAX_STEPS = 500  # 缩短回合长度

# 离散动作
NUM_ACCEL_ACTIONS = 9  # 0=不加速, 1~8=8方向


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    """裁剪向量到最大范数"""
    n = np.linalg.norm(x)
    if n <= max_norm or max_norm <= 0:
        return x
    return x * (max_norm / n)


def _normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def _action_to_acceleration(action: int, a_max: float) -> np.ndarray:
    """离散动作 -> 2D 加速度向量 (世界坐标系)"""
    if action == 0:
        return np.zeros(2, dtype=np.float32)
    # 1~8: 8个方向，从0°开始（右/东），逆时针
    angles = np.arange(8) * (2 * np.pi / 8)
    idx = action - 1
    ax = a_max * np.cos(angles[idx])
    ay = a_max * np.sin(angles[idx])
    return np.array([ax, ay], dtype=np.float32)


class PursuitEvasion2DAsymmetricEnvV2(gym.Env):
    """
    二维非对称追逃环境 V2 - 完整重新设计
    
    主要改进：
    1. 更平衡的动力学：追击者速度劣势减小，加速度优势增大
    2. 逃脱者转向受限：无法立即改变方向
    3. 丰富的观测空间：包含相对量、边界距离等
    4. 稠密奖励函数：距离改进、方向对齐、接近速度等
    5. 课程学习支持：可调节难度
    6. 多种逃脱者模式：从简单到困难
    
    观测空间 (20维):
        [0:2]   自身位置 (归一化)
        [2:4]   自身速度 (归一化)
        [4:6]   相对位置 (自身到目标的向量, 归一化)
        [6:8]   相对速度 (归一化)
        [8]     距离 (归一化)
        [9]     接近速度 (正=接近, 负=远离)
        [10:12] 目标速度方向 (单位向量)
        [12]    到上边界距离 (自身)
        [13]    到下边界距离 (自身)
        [14]    到左边界距离 (自身)
        [15]    到右边界距离 (自身)
        [16]    目标到最近边界距离
        [17]    自身速度大小 (归一化)
        [18]    目标速度大小 (归一化)
        [19]    角度差 (自身速度方向 vs 相对位置方向, -1~1)
        
    动作空间 (离散9):
        0: 不加速（惯性滑行）
        1-8: 8个方向加速（以 a_max 大小）
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        # 动力学参数
        pursuer_v_max: float = DEFAULT_PURSUER_V_MAX,
        pursuer_a_max: float = DEFAULT_PURSUER_A_MAX,
        evader_v_max: float = DEFAULT_EVADER_V_MAX,
        evader_a_max: float = DEFAULT_EVADER_A_MAX,
        evader_turn_rate: float = DEFAULT_EVADER_TURN_RATE,
        # 环境参数
        catch_radius: float = DEFAULT_CATCH_RADIUS,
        world_size: float = DEFAULT_WORLD_SIZE,
        dt: float = 0.05,
        max_steps: int = DEFAULT_MAX_STEPS,
        # 课程学习参数
        difficulty: float = 1.0,  # 0.0=最简单, 1.0=正常
        # 奖励权重
        reward_distance_improve: float = 5.0,     # 距离改进奖励权重
        reward_closing_speed: float = 1.0,        # 接近速度奖励权重
        reward_direction_align: float = 0.5,      # 方向对齐奖励权重
        reward_boundary_trap: float = 0.3,        # 边界围堵奖励权重
        reward_catch: float = 100.0,              # 成功捕获奖励
        penalty_time: float = 0.05,               # 时间惩罚
        penalty_boundary: float = 0.1,            # 撞墙惩罚
        # 逃脱者模式
        evader_mode: str = "smart",  # "random", "simple", "smart", "expert"
        evader_reaction_delay: int = 0,  # 反应延迟步数
        evader_noise: float = 0.0,  # 动作噪声
        # 初始化参数
        init_distance_range: Tuple[float, float] = (2.0, 8.0),
        # 其他
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # 动力学参数
        self.pursuer_v_max = pursuer_v_max
        self.pursuer_a_max = pursuer_a_max
        self.evader_v_max = evader_v_max
        self.evader_a_max = evader_a_max
        self.evader_turn_rate = evader_turn_rate
        
        # 环境参数
        self.catch_radius = catch_radius
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        
        # 课程学习
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self._apply_difficulty()
        
        # 奖励权重
        self.reward_distance_improve = reward_distance_improve
        self.reward_closing_speed = reward_closing_speed
        self.reward_direction_align = reward_direction_align
        self.reward_boundary_trap = reward_boundary_trap
        self.reward_catch = reward_catch
        self.penalty_time = penalty_time
        self.penalty_boundary = penalty_boundary
        
        # 逃脱者设置
        self.evader_mode = evader_mode
        self.evader_reaction_delay = evader_reaction_delay
        self.evader_noise = evader_noise
        self.init_distance_range = init_distance_range
        
        # 状态变量
        self._p_pos = np.zeros(2, dtype=np.float32)
        self._p_vel = np.zeros(2, dtype=np.float32)
        self._e_pos = np.zeros(2, dtype=np.float32)
        self._e_vel = np.zeros(2, dtype=np.float32)
        self._e_heading = 0.0  # 逃脱者朝向角
        
        self._step_count = 0
        self._prev_distance = 0.0
        self._evader_action_buffer = []  # 反应延迟缓冲
        
        self.np_random = np.random.default_rng(seed)
        
        # 观测空间：20维
        obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim) * 10.0,
            high=np.ones(obs_dim) * 10.0,
            dtype=np.float32
        )
        
        # 动作空间
        self.action_space = spaces.Discrete(NUM_ACCEL_ACTIONS)
        
        self.render_mode = render_mode
        
        # 统计信息
        self._episode_rewards = []
        self._episode_distance_traveled = 0.0
        
    def _apply_difficulty(self):
        """应用难度调整"""
        # 难度影响逃脱者的能力
        # difficulty=0: 逃脱者能力降低50%
        # difficulty=1: 正常能力
        factor = 0.5 + 0.5 * self.difficulty
        self._effective_evader_v_max = self.evader_v_max * factor
        self._effective_evader_a_max = self.evader_a_max * factor
        
    def set_difficulty(self, difficulty: float):
        """动态设置难度（用于课程学习）"""
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self._apply_difficulty()
        
    def _step_pursuer(self, acc: np.ndarray):
        """更新追击者状态"""
        # 裁剪加速度
        acc = _clip_norm(acc, self.pursuer_a_max)
        
        # 更新速度
        self._p_vel = self._p_vel + acc * self.dt
        self._p_vel = _clip_norm(self._p_vel, self.pursuer_v_max)
        
        # 更新位置
        new_pos = self._p_pos + self._p_vel * self.dt
        
        # 边界处理（反弹）
        hit_boundary = False
        for i in range(2):
            if new_pos[i] < -self.world_size:
                new_pos[i] = -self.world_size
                self._p_vel[i] = abs(self._p_vel[i]) * 0.3  # 反弹损失
                hit_boundary = True
            elif new_pos[i] > self.world_size:
                new_pos[i] = self.world_size
                self._p_vel[i] = -abs(self._p_vel[i]) * 0.3
                hit_boundary = True
                
        self._p_pos = new_pos
        return hit_boundary
        
    def _step_evader(self, desired_acc: np.ndarray):
        """更新逃脱者状态（带转向限制）"""
        # 逃脱者转向受限：不能立即改变方向
        if np.linalg.norm(desired_acc) > 1e-6:
            desired_heading = np.arctan2(desired_acc[1], desired_acc[0])
            heading_diff = _normalize_angle(desired_heading - self._e_heading)
            
            # 限制转向率
            max_turn = self.evader_turn_rate * self.dt
            heading_diff = np.clip(heading_diff, -max_turn, max_turn)
            self._e_heading = _normalize_angle(self._e_heading + heading_diff)
            
            # 实际加速度方向受限于当前朝向
            acc_mag = min(np.linalg.norm(desired_acc), self._effective_evader_a_max)
            acc = acc_mag * np.array([np.cos(self._e_heading), np.sin(self._e_heading)])
        else:
            acc = np.zeros(2)
            
        # 更新速度
        self._e_vel = self._e_vel + acc * self.dt
        self._e_vel = _clip_norm(self._e_vel, self._effective_evader_v_max)
        
        # 更新位置
        new_pos = self._e_pos + self._e_vel * self.dt
        
        # 边界处理
        for i in range(2):
            if new_pos[i] < -self.world_size:
                new_pos[i] = -self.world_size
                self._e_vel[i] = abs(self._e_vel[i]) * 0.3
            elif new_pos[i] > self.world_size:
                new_pos[i] = self.world_size
                self._e_vel[i] = -abs(self._e_vel[i]) * 0.3
                
        self._e_pos = new_pos
        
        # 更新朝向（基于速度方向）
        if np.linalg.norm(self._e_vel) > 0.1:
            self._e_heading = np.arctan2(self._e_vel[1], self._e_vel[0])
            
    def _get_evader_action(self) -> np.ndarray:
        """获取逃脱者的动作（基于模式）"""
        if self.evader_mode == "random":
            # 完全随机
            action = self.np_random.integers(0, NUM_ACCEL_ACTIONS)
            return _action_to_acceleration(action, self._effective_evader_a_max)
            
        elif self.evader_mode == "simple":
            # 简单逃跑：直接远离追击者
            diff = self._e_pos - self._p_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                return np.zeros(2)
            direction = diff / dist
            return direction * self._effective_evader_a_max
            
        elif self.evader_mode == "smart":
            # 智能逃跑：考虑边界、预测追击者
            return self._smart_evader_policy()
            
        elif self.evader_mode == "expert":
            # 专家逃跑：最优策略（用于测试）
            return self._expert_evader_policy()
            
        else:
            return np.zeros(2)
            
    def _smart_evader_policy(self) -> np.ndarray:
        """智能逃脱者策略"""
        diff = self._e_pos - self._p_pos
        dist = np.linalg.norm(diff)
        
        if dist < 1e-6:
            # 太近了，随机逃
            angle = self.np_random.uniform(0, 2 * np.pi)
            return self._effective_evader_a_max * np.array([np.cos(angle), np.sin(angle)])
            
        # 基本逃跑方向
        escape_dir = diff / dist
        
        # 考虑边界：避免逃向边界
        boundary_penalty = np.zeros(2)
        margin = 2.0
        
        if self._e_pos[0] > self.world_size - margin:
            boundary_penalty[0] -= 1.0
        if self._e_pos[0] < -self.world_size + margin:
            boundary_penalty[0] += 1.0
        if self._e_pos[1] > self.world_size - margin:
            boundary_penalty[1] -= 1.0
        if self._e_pos[1] < -self.world_size + margin:
            boundary_penalty[1] += 1.0
            
        # 合并方向
        combined = escape_dir + 0.5 * boundary_penalty
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
            
        # 添加噪声
        if self.evader_noise > 0:
            noise = self.np_random.normal(0, self.evader_noise, 2)
            combined = combined + noise
            combined = combined / (np.linalg.norm(combined) + 1e-6)
            
        return combined * self._effective_evader_a_max
        
    def _expert_evader_policy(self) -> np.ndarray:
        """专家逃脱者策略（用于测试上限）"""
        diff = self._e_pos - self._p_pos
        dist = np.linalg.norm(diff)
        
        if dist < 1e-6:
            angle = self.np_random.uniform(0, 2 * np.pi)
            return self._effective_evader_a_max * np.array([np.cos(angle), np.sin(angle)])
            
        # 基本逃跑方向
        escape_dir = diff / dist
        
        # 预测追击者位置
        predicted_p_pos = self._p_pos + self._p_vel * 0.5  # 预测0.5秒后
        predicted_diff = self._e_pos - predicted_p_pos
        predicted_dist = np.linalg.norm(predicted_diff)
        if predicted_dist > 1e-6:
            predicted_escape = predicted_diff / predicted_dist
        else:
            predicted_escape = escape_dir
            
        # 考虑边界
        boundary_penalty = np.zeros(2)
        margin = 2.5
        
        if self._e_pos[0] > self.world_size - margin:
            boundary_penalty[0] -= (self._e_pos[0] - (self.world_size - margin)) / margin
        if self._e_pos[0] < -self.world_size + margin:
            boundary_penalty[0] += ((-self.world_size + margin) - self._e_pos[0]) / margin
        if self._e_pos[1] > self.world_size - margin:
            boundary_penalty[1] -= (self._e_pos[1] - (self.world_size - margin)) / margin
        if self._e_pos[1] < -self.world_size + margin:
            boundary_penalty[1] += ((-self.world_size + margin) - self._e_pos[1]) / margin
            
        # 合并：基本逃跑 + 预测逃跑 + 边界避免
        combined = 0.4 * escape_dir + 0.4 * predicted_escape + 0.2 * boundary_penalty
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
            
        return combined * self._effective_evader_a_max
        
    def _get_observation(self) -> np.ndarray:
        """获取追击者的观测"""
        # 归一化因子
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.evader_v_max)
        
        # 计算各种特征
        rel_pos = self._e_pos - self._p_pos  # 相对位置（指向目标）
        rel_vel = self._e_vel - self._p_vel  # 相对速度
        distance = np.linalg.norm(rel_pos)
        
        # 接近速度（正=接近，负=远离）
        if distance > 1e-6:
            closing_speed = -np.dot(rel_vel, rel_pos) / distance
        else:
            closing_speed = 0.0
            
        # 目标速度方向
        evader_speed = np.linalg.norm(self._e_vel)
        if evader_speed > 1e-6:
            evader_vel_dir = self._e_vel / evader_speed
        else:
            evader_vel_dir = np.zeros(2)
            
        # 边界距离（自身）
        dist_to_top = self.world_size - self._p_pos[1]
        dist_to_bottom = self._p_pos[1] + self.world_size
        dist_to_right = self.world_size - self._p_pos[0]
        dist_to_left = self._p_pos[0] + self.world_size
        
        # 目标到最近边界距离
        evader_to_boundary = min(
            self.world_size - abs(self._e_pos[0]),
            self.world_size - abs(self._e_pos[1])
        )
        
        # 自身速度大小
        pursuer_speed = np.linalg.norm(self._p_vel)
        
        # 角度差：自身速度方向 vs 相对位置方向
        if pursuer_speed > 1e-6 and distance > 1e-6:
            p_vel_dir = self._p_vel / pursuer_speed
            target_dir = rel_pos / distance
            angle_alignment = np.dot(p_vel_dir, target_dir)  # -1 到 1
        else:
            angle_alignment = 0.0
            
        # 组装观测向量 (20维)
        obs = np.array([
            # 自身位置 (2)
            self._p_pos[0] * pos_scale,
            self._p_pos[1] * pos_scale,
            # 自身速度 (2)
            self._p_vel[0] * vel_scale,
            self._p_vel[1] * vel_scale,
            # 相对位置 (2)
            rel_pos[0] * pos_scale,
            rel_pos[1] * pos_scale,
            # 相对速度 (2)
            rel_vel[0] * vel_scale,
            rel_vel[1] * vel_scale,
            # 距离 (1)
            distance * pos_scale,
            # 接近速度 (1)
            closing_speed * vel_scale,
            # 目标速度方向 (2)
            evader_vel_dir[0],
            evader_vel_dir[1],
            # 边界距离 - 自身 (4)
            dist_to_top * pos_scale,
            dist_to_bottom * pos_scale,
            dist_to_left * pos_scale,
            dist_to_right * pos_scale,
            # 目标到边界距离 (1)
            evader_to_boundary * pos_scale,
            # 速度大小 (2)
            pursuer_speed * vel_scale,
            evader_speed * vel_scale,
            # 角度对齐 (1)
            angle_alignment,
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)
        
    def _compute_reward(self, hit_boundary: bool) -> Tuple[float, Dict[str, float]]:
        """计算奖励（稠密信号）"""
        reward = 0.0
        reward_info = {}
        
        # 当前距离
        distance = np.linalg.norm(self._e_pos - self._p_pos)
        
        # 1. 距离改进奖励（最重要的稠密信号）
        distance_improvement = self._prev_distance - distance
        r_improve = self.reward_distance_improve * distance_improvement
        reward += r_improve
        reward_info["r_improve"] = r_improve
        
        # 2. 接近速度奖励
        rel_pos = self._e_pos - self._p_pos
        rel_vel = self._e_vel - self._p_vel
        if distance > 1e-6:
            closing_speed = -np.dot(rel_vel, rel_pos) / distance
            r_closing = self.reward_closing_speed * np.clip(closing_speed, -1.0, 1.0)
        else:
            r_closing = 0.0
        reward += r_closing
        reward_info["r_closing"] = r_closing
        
        # 3. 方向对齐奖励：追击者速度指向目标
        pursuer_speed = np.linalg.norm(self._p_vel)
        if pursuer_speed > 0.1 and distance > 1e-6:
            vel_dir = self._p_vel / pursuer_speed
            target_dir = rel_pos / distance
            alignment = np.dot(vel_dir, target_dir)  # -1 到 1
            r_align = self.reward_direction_align * alignment
        else:
            r_align = 0.0
        reward += r_align
        reward_info["r_align"] = r_align
        
        # 4. 边界围堵奖励：逃脱者靠近边界时给奖励（刻意塑形：鼓励追击者把 E 逼向墙角/边，成功案例中 E 触边是策略学出来的）
        evader_to_boundary = min(
            self.world_size - abs(self._e_pos[0]),
            self.world_size - abs(self._e_pos[1])
        )
        if evader_to_boundary < 2.0:
            r_boundary_trap = self.reward_boundary_trap * (2.0 - evader_to_boundary) / 2.0
        else:
            r_boundary_trap = 0.0
        reward += r_boundary_trap
        reward_info["r_boundary_trap"] = r_boundary_trap
        
        # 5. 时间惩罚（鼓励快速追捕）
        r_time = -self.penalty_time
        reward += r_time
        reward_info["r_time"] = r_time
        
        # 6. 撞墙惩罚
        if hit_boundary:
            r_boundary_hit = -self.penalty_boundary
            reward += r_boundary_hit
            reward_info["r_boundary_hit"] = r_boundary_hit
        
        # 7. 接近奖励（近距离额外奖励）
        if distance < 2.0:
            r_proximity = 0.2 * (2.0 - distance)
            reward += r_proximity
            reward_info["r_proximity"] = r_proximity
            
        # 8. 成功捕获奖励
        if distance < self.catch_radius:
            r_catch = self.reward_catch
            reward += r_catch
            reward_info["r_catch"] = r_catch
            
        # 更新历史
        self._prev_distance = distance
        
        return reward, reward_info
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)
        
        # 处理课程学习选项
        if options is not None:
            if "difficulty" in options:
                self.set_difficulty(options["difficulty"])
            if "evader_mode" in options:
                self.evader_mode = options["evader_mode"]
                
        # 随机初始化位置
        min_dist, max_dist = self.init_distance_range
        
        # 追击者位置
        self._p_pos = self.np_random.uniform(
            -self.world_size * 0.7, self.world_size * 0.7, size=2
        ).astype(np.float32)
        
        # 逃脱者位置（保证初始距离在范围内）
        for _ in range(100):
            self._e_pos = self.np_random.uniform(
                -self.world_size * 0.7, self.world_size * 0.7, size=2
            ).astype(np.float32)
            dist = np.linalg.norm(self._e_pos - self._p_pos)
            if min_dist <= dist <= max_dist:
                break
                
        # 初始速度（可以设为零或小随机值）
        self._p_vel = np.zeros(2, dtype=np.float32)
        self._e_vel = np.zeros(2, dtype=np.float32)
        
        # 逃脱者初始朝向：远离追击者
        diff = self._e_pos - self._p_pos
        self._e_heading = np.arctan2(diff[1], diff[0])
        
        # 重置状态
        self._step_count = 0
        self._prev_distance = np.linalg.norm(self._e_pos - self._p_pos)
        self._evader_action_buffer = []
        self._episode_rewards = []
        self._episode_distance_traveled = 0.0
        
        obs = self._get_observation()
        info = {
            "distance": float(self._prev_distance),
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "difficulty": self.difficulty,
            "evader_mode": self.evader_mode,
        }
        
        return obs, info
        
    def step(
        self, action: int, evader_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        self._step_count += 1
        
        # 追击者动作
        p_acc = _action_to_acceleration(int(action), self.pursuer_a_max)
        
        # 逃脱者动作
        if evader_action is None:
            evader_action = self._get_evader_action()
            
            # 反应延迟处理
            if self.evader_reaction_delay > 0:
                self._evader_action_buffer.append(evader_action)
                if len(self._evader_action_buffer) > self.evader_reaction_delay:
                    evader_action = self._evader_action_buffer.pop(0)
                else:
                    evader_action = np.zeros(2)
                    
        # 记录移动距离（统计用）
        old_p_pos = self._p_pos.copy()
        
        # 更新动力学
        hit_boundary = self._step_pursuer(p_acc)
        self._step_evader(evader_action)
        
        # 统计
        self._episode_distance_traveled += np.linalg.norm(self._p_pos - old_p_pos)
        
        # 计算奖励
        reward, reward_info = self._compute_reward(hit_boundary)
        self._episode_rewards.append(reward)
        
        # 终止条件
        distance = np.linalg.norm(self._e_pos - self._p_pos)
        caught = distance < self.catch_radius
        terminated = caught
        truncated = self._step_count >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "distance": float(distance),
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "evader_vel": self._e_vel.copy(),
            "caught": caught,
            "step": self._step_count,
            "reward_info": reward_info,
        }
        
        if terminated or truncated:
            info["episode_reward"] = sum(self._episode_rewards)
            info["episode_length"] = self._step_count
            info["distance_traveled"] = self._episode_distance_traveled
            info["success"] = caught
            
        return obs, float(reward), terminated, truncated, info
        
    def get_evader_obs(self) -> np.ndarray:
        """获取逃脱者视角的观测（用于双智能体训练）"""
        # 与追击者观测类似，但角色互换
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.evader_v_max)
        
        rel_pos = self._p_pos - self._e_pos  # 相对位置（指向追击者，是危险方向）
        rel_vel = self._p_vel - self._e_vel
        distance = np.linalg.norm(rel_pos)
        
        if distance > 1e-6:
            closing_speed = -np.dot(rel_vel, rel_pos) / distance
        else:
            closing_speed = 0.0
            
        pursuer_speed = np.linalg.norm(self._p_vel)
        if pursuer_speed > 1e-6:
            pursuer_vel_dir = self._p_vel / pursuer_speed
        else:
            pursuer_vel_dir = np.zeros(2)
            
        dist_to_top = self.world_size - self._e_pos[1]
        dist_to_bottom = self._e_pos[1] + self.world_size
        dist_to_right = self.world_size - self._e_pos[0]
        dist_to_left = self._e_pos[0] + self.world_size
        
        pursuer_to_boundary = min(
            self.world_size - abs(self._p_pos[0]),
            self.world_size - abs(self._p_pos[1])
        )
        
        evader_speed = np.linalg.norm(self._e_vel)
        
        if evader_speed > 1e-6 and distance > 1e-6:
            e_vel_dir = self._e_vel / evader_speed
            danger_dir = rel_pos / distance
            angle_alignment = np.dot(e_vel_dir, danger_dir)
        else:
            angle_alignment = 0.0
            
        obs = np.array([
            self._e_pos[0] * pos_scale,
            self._e_pos[1] * pos_scale,
            self._e_vel[0] * vel_scale,
            self._e_vel[1] * vel_scale,
            rel_pos[0] * pos_scale,
            rel_pos[1] * pos_scale,
            rel_vel[0] * vel_scale,
            rel_vel[1] * vel_scale,
            distance * pos_scale,
            closing_speed * vel_scale,
            pursuer_vel_dir[0],
            pursuer_vel_dir[1],
            dist_to_top * pos_scale,
            dist_to_bottom * pos_scale,
            dist_to_left * pos_scale,
            dist_to_right * pos_scale,
            pursuer_to_boundary * pos_scale,
            evader_speed * vel_scale,
            pursuer_speed * vel_scale,
            angle_alignment,
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)
        
    def get_state(self) -> Dict[str, np.ndarray]:
        """获取完整状态"""
        return {
            "pursuer_pos": self._p_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "evader_pos": self._e_pos.copy(),
            "evader_vel": self._e_vel.copy(),
            "evader_heading": self._e_heading,
        }
        
    def render(self):
        if self.render_mode is None:
            return None
        return None
        
    def close(self):
        pass


# ============== 便捷函数 ==============

def make_easy_env(**kwargs):
    """创建简单难度环境（用于初始训练）"""
    default_kwargs = {
        "difficulty": 0.3,
        "evader_mode": "simple",
        "evader_reaction_delay": 2,
        "init_distance_range": (1.5, 4.0),
    }
    default_kwargs.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2(**default_kwargs)


def make_medium_env(**kwargs):
    """创建中等难度环境"""
    default_kwargs = {
        "difficulty": 0.6,
        "evader_mode": "smart",
        "evader_reaction_delay": 1,
        "evader_noise": 0.1,
        "init_distance_range": (2.0, 6.0),
    }
    default_kwargs.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2(**default_kwargs)


def make_hard_env(**kwargs):
    """创建困难环境"""
    default_kwargs = {
        "difficulty": 1.0,
        "evader_mode": "smart",
        "evader_reaction_delay": 0,
        "evader_noise": 0.05,
        "init_distance_range": (3.0, 8.0),
    }
    default_kwargs.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2(**default_kwargs)


def make_expert_env(**kwargs):
    """创建专家难度环境（用于测试上限）"""
    default_kwargs = {
        "difficulty": 1.0,
        "evader_mode": "expert",
        "evader_reaction_delay": 0,
        "init_distance_range": (4.0, 8.0),
    }
    default_kwargs.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2(**default_kwargs)
