"""
三维环境 V2 - 解决"倒着开"问题（简化版）
========================================

核心改进：在原有 3d_habl_h07_strict 基础上，只添加一个简单惩罚：
- 当朝向目标（heading_error < 90°）却使用后退动作时，给予惩罚

动作空间（8个）：
  0: forward      - 前进
  1: backward     - 后退
  2: left         - 左平移
  3: right        - 右平移
  4: yaw_left     - 左转
  5: yaw_right    - 右转
  6: heave_up     - 上升
  7: heave_down   - 下潜
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


class LowLevelAUV3DEnvV2(gym.Env):
    """
    三维八动作环境 V2 - 简化版
    
    基于 3d_habl_h07_strict 配置，只添加一个惩罚：
    - backward_facing_target_penalty: 朝向目标却后退的惩罚
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right", "heave_up", "heave_down"]
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 # === 目标距离配置 ===
                 target_distance: float = 12.0,           # 增大默认距离
                 z_range: Tuple[float, float] = (2.0, 4.0),
                 # === 原有奖励参数（来自 3d_habl_h07_strict）===
                 heading_coef: float = 0.7,
                 direction_coef: float = 0.6,
                 strict_direction_heading: bool = True,   # 默认开启严格模式
                 heading_penalty_wrong: float = 0.0,
                 # === 新增：反向移动惩罚 ===
                 backward_facing_target_penalty: float = 0.5,  # 朝向目标却后退的惩罚
                 # === Z 轴改进（缓解深度控制不足）===
                 z_coef: float = 1.0,              # Z 方向奖励系数（原 0.5，提高以加强 heave）
                 z_boost_when_xy_close: float = 1.5,  # XY 接近时 Z 奖励放大倍数
                 xy_close_threshold: float = 1.0,     # 判定 "XY 接近" 的阈值 (m)
                 relaxed_heading_when_xy_close: bool = True,  # XY 接近时放宽 heading 要求到 60°
                 # === 其他 ===
                 max_steps: int = 4000):
        """
        Args:
            target_distance: 水平面目标距离范围
            z_range: 深度范围 (min_z, max_z)
            heading_coef: 朝向目标奖励系数
            direction_coef: 对准且前进时的速度奖励系数
            strict_direction_heading: True 时仅当朝向误差<30°才给前进奖励
            heading_penalty_wrong: 朝向误差>90°且仍在前进时的惩罚
            backward_facing_target_penalty: 【新增】朝向目标（<90°）却后退时的惩罚
            max_steps: 最大步数
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
        self.backward_facing_target_penalty = backward_facing_target_penalty
        self.z_coef = z_coef
        self.z_boost_when_xy_close = z_boost_when_xy_close
        self.xy_close_threshold = xy_close_threshold
        self.relaxed_heading_when_xy_close = relaxed_heading_when_xy_close
        self.max_steps = max_steps

        # 8 动作映射
        self.action_map = {
            0: 1,   # forward  -> surge_forward
            1: 2,   # backward -> surge_backward
            2: 3,   # left     -> sway_left
            3: 4,   # right    -> sway_right
            4: 7,   # yaw_left  -> yaw_left
            5: 8,   # yaw_right -> yaw_right
            6: 5,   # heave_up   -> heave_up
            7: 6,   # heave_down -> heave_down
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
        self.current_action = None

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_3d_constraints(self):
        """强制 3D 约束（保持 roll/pitch 稳定）"""
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            
            eta[3] = 0.0  # roll
            eta[4] = 0.0  # pitch
            nu[3] = 0.0   # p (roll rate)
            nu[4] = 0.0   # q (pitch rate)
            
            eta[2] = np.clip(eta[2], self.z_range[0], self.z_range[1])
        except Exception:
            pass

    def _sample_3d_target(self):
        """采样 3D 目标点"""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.target_distance * 0.5, self.target_distance)
        
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        target_z = np.random.uniform(self.z_range[0], self.z_range[1])
        
        return np.array([target_x, target_y, target_z])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        obs, info = self.inner.reset(seed=seed, options=options)
        
        self.inner.dynamics.eta[:3] = np.array([0.0, 0.0, np.mean(self.z_range)])
        self.inner.dynamics.eta[3:] = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
        self.inner.dynamics.nu[:] = 0.0
        
        target_pos = self._sample_3d_target()
        self.inner.target_state = np.zeros(6)
        self.inner.target_state[:3] = target_pos
        
        self._enforce_3d_constraints()
        
        self.episode_count += 1
        self.prev_pos_error = None
        self.step_count = 0
        self.current_action = None
        
        obs = self._get_obs()
        info = {
            'target_pos': target_pos.tolist(),
            'init_pos': self.inner.dynamics.eta[:3].tolist(),
            'init_yaw': self.inner.dynamics.eta[5],
            'init_distance': np.linalg.norm(target_pos - self.inner.dynamics.eta[:3])
        }
        
        return obs, info

    def step(self, action: int):
        """执行动作"""
        self.current_action = action
        mapped_action = self.action_map[action]
        _, _, _, _, info = self.inner.step(mapped_action)
        
        self._enforce_3d_constraints()
        self.step_count += 1
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
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
        
        vx = nu[0]
        vy = nu[1]
        vz = nu[2]
        yaw_rate = nu[5]
        
        # 归一化因子增大以适应更远目标
        obs = np.array([
            pos_error_body_x / 10.0,
            pos_error_body_y / 10.0,
            pos_error_body_z / 5.0,
            heading_error / np.pi,
            target_angle_xy / np.pi,
            vx / 2.0,
            vy / 2.0,
            vz / 2.0,
            yaw_rate / 2.0,
            distance / 15.0
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self, action: int) -> Tuple[float, Dict]:
        """
        计算奖励 - 基于原 3d_habl_h07_strict，只添加一个惩罚
        """
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:3] - target[:3])
        
        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))
        
        reward_info = {}
        
        # 1. 位置误差惩罚（与原版相同）
        position_reward = -0.3 * pos_error
        reward_info['position'] = position_reward
        
        # 2. 位置改进奖励（与原版相同）
        improvement_reward = 0.0
        if self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward
        
        # 3. 朝向奖励（与原版相同）
        heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
        if heading_error < np.pi / 6:
            heading_reward += 0.15
        elif heading_error < np.pi / 3:
            heading_reward += 0.05
        reward_info['heading'] = heading_reward

        # 4. 方向引导（方案4: XY 接近时放宽 heading 要求）
        xy_error = np.linalg.norm(eta[:2] - target[:2])
        direction_reward = 0.0
        if pos_error > 0.1:
            forward_vel = nu[0]
            if self.strict_direction_heading:
                # XY 接近 (<0.5m) 时放宽到 60°，减少 yaw 微调、便于使用 heave
                if self.relaxed_heading_when_xy_close and xy_error < 0.5:
                    align_ok = heading_error < np.pi / 3  # 60°
                else:
                    align_ok = heading_error < np.pi / 6  # 30°
            else:
                align_ok = heading_error < 0.5
            if align_ok and forward_vel > 0:
                direction_reward = self.direction_coef * forward_vel
        reward_info['direction'] = direction_reward

        # 4.2 反朝向惩罚（与原版相同）
        wrong_way_penalty = 0.0
        if self.heading_penalty_wrong > 0 and pos_error > 0.1 and nu[0] > 0 and heading_error > np.pi / 2:
            wrong_way_penalty = -self.heading_penalty_wrong
        reward_info['wrong_way_penalty'] = wrong_way_penalty
        
        # 【新增】4.3 朝向目标却后退的惩罚
        backward_penalty = 0.0
        if self.backward_facing_target_penalty > 0 and pos_error > 0.5:
            # 朝向目标（<90°）却使用后退动作
            if action == 1 and heading_error < np.pi / 2:
                backward_penalty = -self.backward_facing_target_penalty
                reward_info['backward_type'] = 'facing_target_but_backward'
        reward_info['backward_penalty'] = backward_penalty
        
        # 4.5 Z 轴方向引导（方案1+2: z_coef 提高 + XY 接近时放大）
        z_error = target[2] - eta[2]
        z_vel = nu[2]
        z_direction_reward = 0.0
        if abs(z_error) > 0.1:
            if z_error > 0 and z_vel > 0:
                z_direction_reward = self.z_coef * z_vel
            elif z_error < 0 and z_vel < 0:
                z_direction_reward = self.z_coef * abs(z_vel)
        # 方案2: XY 接近时 Z 奖励放大，引导策略在 XY 到位后重点调 Z
        if xy_error < self.xy_close_threshold and z_direction_reward > 0:
            z_direction_reward *= self.z_boost_when_xy_close
        reward_info['z_direction'] = z_direction_reward
        
        # 5. 接近目标奖励（与原版相同）
        if pos_error < 0.5:
            proximity_reward = 3.0 * (0.5 - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward
        
        # 6. 到达目标大奖励（与原版相同）
        goal_bonus = 0.0
        if pos_error < 0.5:
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus
        
        # 7. 时间惩罚（与原版相同）
        time_penalty = -0.02
        reward_info['time'] = time_penalty
        
        total_reward = (position_reward + improvement_reward + heading_reward +
                       direction_reward + wrong_way_penalty + backward_penalty +
                       z_direction_reward + proximity_reward + goal_bonus + time_penalty)
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
        
        if pos_error < 0.5:
            info['termination_reason'] = 'success'
            self.success_count += 1
            return True, False, info
        
        # 扩大边界以适应更远目标
        if abs(eta[0]) > 25 or abs(eta[1]) > 25 or eta[2] < 0 or eta[2] > 10:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info
        
        if self.step_count >= self.max_steps:
            info['termination_reason'] = 'timeout'
            return False, True, info
        
        info['termination_reason'] = None
        info['pos_error'] = pos_error
        return False, False, info

    def render(self):
        return self.inner.render()

    def close(self):
        return self.inner.close()


class VecLowLevelAUV3DEnvV2:
    """向量化 3D 环境 V2"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV3DEnvV2(**kwargs) for _ in range(num_envs)]
        
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


if __name__ == "__main__":
    print("=" * 60)
    print("测试 LowLevelAUV3DEnvV2（简化版）")
    print("=" * 60)
    
    env = LowLevelAUV3DEnvV2(target_distance=12.0)
    
    obs, info = env.reset(seed=42)
    print(f"\n初始信息:")
    print(f"  目标位置: {info['target_pos']}")
    print(f"  初始距离: {info['init_distance']:.2f}m")
    
    # 测试后退惩罚
    print("\n测试后退惩罚（朝向目标却后退）:")
    for _ in range(5):
        obs, reward, _, _, info = env.step(1)  # 后退
        bp = info.get('backward_penalty', 0)
        print(f"  后退动作 | 奖励: {reward:.3f} | backward_penalty: {bp:.3f}")
    
    env.close()
    print("\n测试完成!")
