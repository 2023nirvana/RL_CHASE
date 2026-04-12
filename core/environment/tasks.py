"""  
特定任务环境  
============  
  
定义不同的训练任务，如悬停、航点跟踪、轨迹跟踪等。  
"""  
  
import numpy as np  
from typing import Dict, Tuple, Optional  
from .low_level_env import LowLevelAUVEnv  
  
  
class HoverTask(LowLevelAUVEnv):  
    """  
    悬停任务  
      
    目标：在指定位置保持稳定悬停  
    """  
      
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)  
        self.task_name = "hover"  
          
    def _sample_target_state(self) -> np.ndarray:  
        """悬停任务的目标是原点"""  
        target_pos = np.zeros(3)  
        target_ori = np.zeros(3)  
        target_vel = np.zeros(6)  
        return np.concatenate([target_pos, target_ori, target_vel])  
      
    def _sample_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:  
        """从随机位置开始"""  
        stage = self.curriculum_config  
        position_range = stage['position_range']  
          
        # 随机位置  
        position = self.np_random.uniform(-position_range, position_range, 3)  
          
        # 小范围随机姿态  
        orientation = self.np_random.uniform(-0.2, 0.2, 3)  
          
        eta_init = np.concatenate([position, orientation])  
          
        # 根据课程阶段决定初始速度  
        vel_type = stage.get('velocity_init', 'zero')  
        if vel_type == 'zero':  
            nu_init = np.zeros(6)  
        elif vel_type == 'small':  
            nu_init = self.np_random.uniform(-0.3, 0.3, 6)  
        else:  
            nu_init = self.np_random.uniform(-0.5, 0.5, 6)  
              
        return eta_init, nu_init  
  
  
class WaypointTask(LowLevelAUVEnv):  
    """  
    航点跟踪任务  
      
    目标：依次访问多个航点  
    """  
      
    def __init__(self, num_waypoints: int = 5, **kwargs):  
        super().__init__(**kwargs)  
        self.task_name = "waypoint"  
        self.num_waypoints = num_waypoints  
        self.current_waypoint_idx = 0  
        self.waypoints = []  
          
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):  
        """重置并生成新的航点序列"""  
        obs, info = super().reset(seed=seed, options=options)  
          
        # 生成航点  
        self._generate_waypoints()  
        self.current_waypoint_idx = 0  
          
        # 设置第一个航点为目标  
        self._set_current_waypoint_as_target()  
          
        info['waypoints'] = [wp.tolist() for wp in self.waypoints]  
        info['current_waypoint'] = 0  
          
        return self._get_observation(), info  
      
    def _generate_waypoints(self):  
        """生成航点序列"""  
        stage = self.curriculum_config  
        range_val = stage['position_range']  
          
        self.waypoints = []  
        for _ in range(self.num_waypoints):  
            waypoint = self.np_random.uniform(-range_val, range_val, 3)  
            self.waypoints.append(waypoint)  
              
    def _set_current_waypoint_as_target(self):  
        """设置当前航点为目标"""  
        if self.current_waypoint_idx < len(self.waypoints):  
            target_pos = self.waypoints[self.current_waypoint_idx]  
            target_ori = np.zeros(3)  
            target_vel = np.zeros(6)  
            self.target_state = np.concatenate([target_pos, target_ori, target_vel])  
              
    def step(self, action: int):  
        """执行一步，检查是否到达当前航点"""  
        obs, reward, terminated, truncated, info = super().step(action)  
          
        # 检查是否到达当前航点  
        eta = self.dynamics.eta  
        target_pos = self.target_state[:3]  
        pos_error = np.linalg.norm(eta[:3] - target_pos)  
          
        if pos_error < self.reward_thresholds['position_tolerance']:  
            # 到达航点，额外奖励  
            reward += 5.0  
            info['waypoint_reached'] = self.current_waypoint_idx  
              
            # 前进到下一个航点  
            self.current_waypoint_idx += 1  
              
            if self.current_waypoint_idx >= len(self.waypoints):  
                # 所有航点访问完毕  
                terminated = True  
                info['termination_reason'] = 'all_waypoints_reached'  
                reward += 20.0  
            else:  
                # 设置下一个航点  
                self._set_current_waypoint_as_target()  
                  
        info['current_waypoint'] = self.current_waypoint_idx  
          
        return obs, reward, terminated, truncated, info  
  
  
class TrajectoryTrackingTask(LowLevelAUVEnv):  
    """  
    轨迹跟踪任务  
      
    目标：跟踪预定义的轨迹  
    """  
      
    def __init__(self, trajectory_type: str = "circle", **kwargs):  
        super().__init__(**kwargs)  
        self.task_name = "trajectory_tracking"  
        self.trajectory_type = trajectory_type  
        self.trajectory_time = 0.0  
          
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):  
        """重置"""  
        obs, info = super().reset(seed=seed, options=options)  
        self.trajectory_time = 0.0  
          
        # 初始目标  
        self._update_trajectory_target()  
          
        info['trajectory_type'] = self.trajectory_type  
          
        return self._get_observation(), info  
      
    def _update_trajectory_target(self):  
        """更新轨迹目标点"""  
        t = self.trajectory_time  
          
        if self.trajectory_type == "circle":  
            # 圆形轨迹  
            radius = 3.0  
            omega = 0.3  # 角频率  
            target_pos = np.array([  
                radius * np.cos(omega * t),  
                radius * np.sin(omega * t),  
                0.0  
            ])  
            # 切向方向  
            target_yaw = omega * t + np.pi/2  
            target_ori = np.array([0, 0, target_yaw])  
              
        elif self.trajectory_type == "figure8":  
            # 8字形轨迹  
            scale = 3.0  
            omega = 0.2  
            target_pos = np.array([  
                scale * np.sin(omega * t),  
                scale * np.sin(2 * omega * t) / 2,  
                0.0  
            ])  
            target_ori = np.zeros(3)  
              
        elif self.trajectory_type == "helix":  
            # 螺旋轨迹  
            radius = 2.0  
            omega = 0.3  
            z_rate = 0.1  
            target_pos = np.array([  
                radius * np.cos(omega * t),  
                radius * np.sin(omega * t),  
                z_rate * t  
            ])  
            target_ori = np.array([0, 0, omega * t + np.pi/2])  
              
        else:  
            # 默认静止  
            target_pos = np.zeros(3)  
            target_ori = np.zeros(3)  
              
        target_vel = np.zeros(6)  
        self.target_state = np.concatenate([target_pos, target_ori, target_vel])  
          
    def step(self, action: int):  
        """执行一步，更新轨迹目标"""  
        obs, reward, terminated, truncated, info = super().step(action)  
          
        # 更新轨迹时间和目标  
        self.trajectory_time += self.dt  
        self._update_trajectory_target()  
          
        info['trajectory_time'] = self.trajectory_time  
          
        return obs, reward, terminated, truncated, info  
  
  
class PositionControlTask(LowLevelAUVEnv):  
    """  
    位置控制任务  
      
    目标：精确控制到达指定位置  
    """  
      
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)  
        self.task_name = "position_control"  
          
    def _compute_reward(self, action: int) -> Tuple[float, Dict]:  
        """增强的位置控制奖励"""  
        reward, reward_info = super()._compute_reward(action)  
          
        # 额外的位置精度奖励  
        pos_error = reward_info['pos_error']  
          
        # 精度奖励（指数衰减）  
        precision_bonus = 5.0 * np.exp(-pos_error * 2)  
        reward += precision_bonus  
        reward_info['precision_bonus'] = precision_bonus  
          
        # 稳定奖励（如果在目标附近保持稳定）  
        if hasattr(self, 'prev_positions'):  
            self.prev_positions.append(self.dynamics.eta[:3].copy())  
            if len(self.prev_positions) > 50:  
                self.prev_positions.pop(0)  
                # 计算位置方差  
                positions = np.array(self.prev_positions)  
                variance = np.var(positions, axis=0).sum()  
                if variance < 0.01 and pos_error < 0.5:  
                    stability_bonus = 2.0  
                    reward += stability_bonus  
                    reward_info['stability_bonus'] = stability_bonus  
        else:  
            self.prev_positions = []  
              
        return reward, reward_info  
  
  
class AttitudeControlTask(LowLevelAUVEnv):  
    """  
    姿态控制任务  
      
    目标：控制到达并保持指定姿态  
    """  
      
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)  
        self.task_name = "attitude_control"  
          
    def _sample_target_state(self) -> np.ndarray:  
        """采样目标姿态"""  
        # 位置保持在原点附近  
        target_pos = self.np_random.uniform(-1, 1, 3)  
          
        # 随机目标姿态  
        target_ori = self.np_random.uniform(-0.5, 0.5, 3)  
          
        target_vel = np.zeros(6)  
        return np.concatenate([target_pos, target_ori, target_vel])  
      
    def _compute_reward(self, action: int) -> Tuple[float, Dict]:  
        """姿态控制专用奖励"""  
        reward, reward_info = super()._compute_reward(action)  
          
        # 增加姿态权重  
        ori_error = reward_info['ori_error']  
        extra_ori_penalty = -2.0 * ori_error  
        reward += extra_ori_penalty  
        reward_info['extra_ori_penalty'] = extra_ori_penalty  
          
        return reward, reward_info  
