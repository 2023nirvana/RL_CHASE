"""  
底层控制环境  
============  
  
用于训练AUV基础运动技能的强化学习环境。  
  
技能包括：  
- 悬停（hover）  
- 平移（surge, sway, heave）  
- 旋转（roll, pitch, yaw）  
- 组合运动  
  
环境特点：  
1. 完整的6DOF非线性动力学  
2. 离散动作空间  
3. 支持课程学习  
4. 支持域随机化  
5. 详细的奖励设计  
"""  
  
import numpy as np  
import gymnasium as gym  
from gymnasium import spaces  
from typing import Dict, Tuple, Optional, List, Any  
import yaml  
import os  
import sys  
  
# 添加项目根目录到路径  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
sys.path.insert(0, PROJECT_ROOT)  
  
from core.dynamics.auv_dynamics import AUVDynamics  
from core.dynamics.hydrodynamics import WaterCurrentModel 
from core.control.dual_loop_pid import IntegratedController   
  
  
class LowLevelAUVEnv(gym.Env):  
    """  
    底层AUV控制环境  
      
    目标：学习基础运动技能，使AUV能够稳定地移动到目标状态  
    """  
      
    # 动作定义  
    ACTION_NAMES = [  
        "hover",           # 0: 悬停  
        "surge_forward",   # 1: 前进  
        "surge_backward",  # 2: 后退  
        "sway_left",       # 3: 左平移  
        "sway_right",      # 4: 右平移  
        "heave_up",        # 5: 上浮  
        "heave_down",      # 6: 下潜  
        "yaw_left",        # 7: 左偏航  
        "yaw_right",       # 8: 右偏航  
        "pitch_up",        # 9: 抬头  
        "pitch_down",      # 10: 低头  
        "roll_left",       # 11: 左横滚  
        "roll_right",      # 12: 右横滚  
    ]  
      
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  
      
    def __init__(self,  
                 config_path: str = None,  
                 render_mode: Optional[str] = None,  
                 curriculum_stage: int = 0,  
                 enable_domain_randomization: bool = True,  
                 seed: Optional[int] = None):  
        """  
        初始化环境  
          
        Args:  
            config_path: 配置文件路径  
            render_mode: 渲染模式  
            curriculum_stage: 课程学习阶段 (0-3)  
            enable_domain_randomization: 是否启用域随机化  
            seed: 随机种子  
        """  
        super().__init__()  
          
        # 加载配置  
        if config_path is None:  
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")  
          
        with open(config_path, 'r') as f:  
            self.config = yaml.safe_load(f)  
              
        # 加载机器人配置  
        robot_config_path = os.path.join(PROJECT_ROOT, "configs/robot_config.yaml")  
        with open(robot_config_path, 'r') as f:  
            self.robot_config = yaml.safe_load(f)  
              
        # 加载环境配置  
        env_config_path = os.path.join(PROJECT_ROOT, "configs/env_config.yaml")  
        with open(env_config_path, 'r') as f:  
            self.env_config = yaml.safe_load(f)  
          
        # 设置随机种子  
        self.seed(seed)  
          
        # 渲染模式  
        self.render_mode = render_mode  
          
        # 课程学习  
        self.curriculum_stage = curriculum_stage  
        self.curriculum_config = self.config['curriculum']['stages'][curriculum_stage]  
          
        # 域随机化  
        self.enable_domain_randomization = enable_domain_randomization  
          
        # 时间配置  
        self.dt = self.env_config['simulation']['dt']  
        self.max_episode_time = self.env_config['simulation']['max_episode_time']  
        self.max_steps = int(self.max_episode_time / self.dt)  
          
        # 初始化动力学模型  
        self.dynamics = AUVDynamics(  
            config_path=robot_config_path,  
            randomize=self.enable_domain_randomization  
        )  
        self.dynamics.set_dt(self.dt)  
          
        # 初始化水流模型  
        self.water_current = WaterCurrentModel(self.env_config['world']['current'])  
          
        # 构建动作映射  
        self._build_action_mapping()  
          
        # 定义动作空间（离散）  
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))  
          
        # 定义观测空间  
        self._build_observation_space()  
          
        # 奖励配置  
        self.reward_weights = self.config['reward']['weights']  
        self.reward_thresholds = self.config['reward']['thresholds']  
  
        # 终止条件配置（避免长时间无进展）  
        self.no_progress_cfg = self.config.get('termination', {}).get('no_progress', {})  
          
        # Episode状态  
        self.current_step = 0  
        self.episode_reward = 0.0  
        self.target_state = np.zeros(12)  # [position, orientation, velocity]  
          
        # 用于记录和调试  
        self.episode_info = {  
            'position_errors': [],  
            'orientation_errors': [],  
            'rewards': [],  
            'actions': [],
            'reward_components': []  
        }  

                  
        # 渲染相关  
        self.trajectory = []  
        self.fig = None  
        self.ax = None
        self.integrated_controller = IntegratedController(config=self.config)   
        self.last_thrust_command = np.zeros(self.robot_config['thruster']['num_thrusters'])  
        self.prev_heading_target = None  
          
    def seed(self, seed: Optional[int] = None):  
        """设置随机种子"""  
        self.np_random, seed = gym.utils.seeding.np_random(seed)  
        return [seed]  
      
    def _build_action_mapping(self):  
        """构建动作到推力的映射"""  
        action_config = self.config['action_space']['action_thrust_mapping']  
        max_thrust = self.robot_config['thruster']['max_thrust']  
          
        self.action_to_thrust = {}  
        for i, action_name in enumerate(self.ACTION_NAMES):  
            # 获取归一化推力  
            normalized_thrust = action_config.get(action_name, [0]*8)  
            # 转换为实际推力  
            actual_thrust = np.array(normalized_thrust) * max_thrust  
            self.action_to_thrust[i] = actual_thrust  
              
    def _build_observation_space(self):  
        """构建观测空间"""  
        obs_config = self.config['observation_space']  
          
        # 计算观测维度  
        obs_dim = 0  
          
        # 自身状态  
        if obs_config['include_position']:  
            obs_dim += 3  # x, y, z  
        if obs_config['include_orientation']:  
            obs_dim += 3  # roll, pitch, yaw  
        if obs_config['include_velocity']:  
            obs_dim += 6  # u, v, w, p, q, r  
              
        # 目标状态  
        if obs_config['include_target_position']:  
            obs_dim += 3  
        if obs_config['include_target_orientation']:  
            obs_dim += 3  
              
        # 误差  
        if obs_config['include_error']:  
            obs_dim += 6  # position error (3) + orientation error (3)  
              
        self.obs_dim = obs_dim  
          
        # 观测空间范围  
        obs_high = np.ones(obs_dim) * obs_config['clip_obs']  
        self.observation_space = spaces.Box(  
            low=-obs_high,  
            high=obs_high,  
            dtype=np.float32  
        )  
          
    def _get_observation(self) -> np.ndarray:  
        """  
        获取当前观测 - 体坐标系版本  
        """  
        obs_config = self.config['observation_space']  
        obs_parts = []  
        
        eta = self.dynamics.eta  # [x, y, z, roll, pitch, yaw]  
        nu = self.dynamics.nu    # [u, v, w, p, q, r]  
        
        target_pos = self.target_state[:3]  
        target_ori = self.target_state[3:6]  
        
        # ===== 计算体坐标系下的位置误差 =====  
        pos_error_world = target_pos - eta[:3]  # 世界坐标系下指向目标的向量  
        R = self.dynamics._rotation_matrix(eta[3], eta[4], eta[5])  
        pos_error_body = R.T @ pos_error_world  # 转换到体坐标系  
        
        # ===== 计算姿态误差 =====  
        ori_error = self._angle_difference(eta[3:6], target_ori)  
        
        # 1. 位置误差（体坐标系）: 3维  
        if obs_config['include_position']:  
            obs_parts.append(pos_error_body / 10.0)  
        
        # 2. 姿态误差: 3维  
        if obs_config['include_orientation']:  
            obs_parts.append(ori_error / np.pi)  
        
        # 3. 速度: 6维  
        if obs_config['include_velocity']:  
            lin_vel_normalized = nu[:3] / 2.0  # 体坐标系线速度  
            ang_vel_normalized = nu[3:] / 2.0  # 体坐标系角速度  
            obs_parts.append(lin_vel_normalized)  
            obs_parts.append(ang_vel_normalized)  
        
        # 4. 目标位置（体坐标系）: 3维  
        if obs_config['include_target_position']:  
            obs_parts.append(pos_error_body / 10.0)  
        
        # 5. 目标姿态: 3维  
        if obs_config['include_target_orientation']:  
            obs_parts.append(target_ori / np.pi)  
        
        # 6. 误差: 6维  
        if obs_config['include_error']:  
            obs_parts.append(pos_error_body / 10.0)  
            obs_parts.append(ori_error / np.pi)  
        
        observation = np.concatenate(obs_parts)  
        
        # 验证维度  
        assert len(observation) == self.obs_dim, ( 
            f"Observation dim mismatch: got {len(observation)}, expected {self.obs_dim}"  
        )
        clip_val = obs_config['clip_obs']  
        observation = np.clip(observation, -clip_val, clip_val)  
        
        return observation.astype(np.float32)  
      
    @staticmethod  
    def _angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:  
        """  
        计算角度差（处理周期性）  
          
        结果在[-π, π]范围内  
        """  
        diff = angle1 - angle2  
        return ((diff + np.pi) % (2 * np.pi)) - np.pi  
      
    # def _compute_reward(self, action: int) -> Tuple[float, Dict]:  
    #     """  
    #     计算奖励  
          
    #     Args:  
    #         action: 执行的动作  
              
    #     Returns:  
    #         reward: 总奖励  
    #         reward_info: 奖励分解  

    def _compute_reward(self, action: int) -> Tuple[float, Dict]:    
        """计算奖励"""    
        weights = self.reward_weights    
        thresholds = self.reward_thresholds    
        
        eta = self.dynamics.eta    
        nu = self.dynamics.nu    
        
        target_pos = self.target_state[:3]    
        target_ori = self.target_state[3:6]    
        
        pos_error = np.linalg.norm(eta[:3] - target_pos)    
        ori_error = np.linalg.norm(self._angle_difference(eta[3:6], target_ori))    
        
        lin_vel_mag = np.linalg.norm(nu[:3])    
        ang_vel_mag = np.linalg.norm(nu[3:])    
        
        thrust = self.last_thrust_command if hasattr(self, 'last_thrust_command') else self.action_to_thrust[action]    
        thruster_count = self.robot_config['thruster']['num_thrusters']  
        energy_cost = np.sum(np.abs(thrust)) / (self.robot_config['thruster']['max_thrust'] * thruster_count)    
        
        reward_info = {}    
        
        # 1. 位置误差奖励（修复：确保奖励方向正确）
        # 基础惩罚：误差越大，惩罚越大
        position_reward_base = weights['position_error'] * pos_error  # weights['position_error'] = -0.3，所以是负的
        
        # 位置改进奖励（关键：必须确保方向正确）- 这是最关键的稠密奖励信号
        pos_improvement_reward = 0.0
        if hasattr(self, 'prev_pos_error') and self.prev_pos_error > 0:
            pos_improvement = self.prev_pos_error - pos_error  # 如果误差减少（靠近目标），这个值为正
            if pos_improvement > 0:
                # 改进奖励：每减少0.1m给0.8奖励（大幅增强，确保信号足够强）
                pos_improvement_reward = 0.8 * pos_improvement
            elif pos_improvement < 0:
                # 如果远离目标，给予惩罚（确保远离是负奖励）
                pos_improvement_reward = 0.5 * pos_improvement  # 远离惩罚增强
        
        # 当接近目标时，给予额外奖励（稠密奖励）
        proximity_reward = 0.0
        if pos_error < 1.0:  # 距离小于1m时
            proximity_reward = 1.0 * (1.0 - pos_error)  # 越近奖励越大，最多1.0（增加）
        
        # 方向引导奖励（新增加）：根据体坐标系误差给予方向性奖励
        # 这是关键修复：帮助智能体理解正确的动作方向
        direction_reward = 0.0
        eta = self.dynamics.eta
        nu = self.dynamics.nu
        target_pos = self.target_state[:3]
        pos_error_world = target_pos - eta[:3]  # 世界坐标系下指向目标的向量
        R = self.dynamics._rotation_matrix(eta[3], eta[4], eta[5])
        pos_error_body = R.T @ pos_error_world  # 体坐标系下的误差
        
        # 当前速度在体坐标系下
        current_vel_body = nu[:3]
        
        # 方向引导：如果速度方向与误差方向一致，给奖励
        # pos_error_body表示目标在体坐标系下的方向
        # 如果current_vel_body与pos_error_body方向一致（点积>0），说明在向目标移动
        if np.linalg.norm(pos_error_body) > 0.05 and np.linalg.norm(current_vel_body) > 0.05:
            # 归一化
            error_dir = pos_error_body / (np.linalg.norm(pos_error_body) + 1e-6)
            vel_dir = current_vel_body / (np.linalg.norm(current_vel_body) + 1e-6)
            # 方向一致性（点积，范围[-1, 1]）
            direction_alignment = np.dot(error_dir, vel_dir)
            # 如果方向一致（向目标移动），给奖励；不一致（远离目标），给惩罚
            # 增强奖励权重，确保信号足够强
            direction_reward = 0.4 * direction_alignment  # 最多0.4奖励（从0.2增加）
        elif np.linalg.norm(pos_error_body) > 0.05:
            # 即使速度很小，如果误差存在，也根据误差方向给予小的方向提示
            error_dir = pos_error_body / (np.linalg.norm(pos_error_body) + 1e-6)
            # 假设应该产生的速度方向（与误差方向一致）
            desired_vel_dir = error_dir
            # 当前速度方向（归一化，即使很小）
            if np.linalg.norm(current_vel_body) > 0.01:
                vel_dir = current_vel_body / np.linalg.norm(current_vel_body)
            else:
                vel_dir = np.zeros(3)
            direction_alignment = np.dot(error_dir, desired_vel_dir) if np.linalg.norm(vel_dir) > 0 else 0
            direction_reward = 0.1 * direction_alignment  # 小的方向提示
        
        position_reward = position_reward_base + pos_improvement_reward + proximity_reward + direction_reward
        
        reward_info['position'] = position_reward_base
        reward_info['position_improvement'] = pos_improvement_reward
        reward_info['proximity'] = proximity_reward
        reward_info['direction'] = direction_reward    
        
        # 2. 姿态误差惩罚    
        orientation_reward = weights['orientation_error'] * ori_error    
        reward_info['orientation'] = orientation_reward    
        
        # 3. 速度惩罚    
        velocity_penalty = 0.0    
        if lin_vel_mag > thresholds['velocity_limit']:    
            velocity_penalty = weights['velocity_penalty'] * (lin_vel_mag - thresholds['velocity_limit'])    
        reward_info['velocity'] = velocity_penalty    
        
        # 4. 角速度惩罚    
        angular_penalty = 0.0    
        if ang_vel_mag > thresholds['angular_velocity_limit']:    
            angular_penalty = weights['angular_velocity_penalty'] * (ang_vel_mag - thresholds['angular_velocity_limit'])    
        reward_info['angular'] = angular_penalty    
        
        # 5. 能耗惩罚    
        energy_reward = weights['energy_cost'] * energy_cost    
        reward_info['energy'] = energy_reward    
        
        # ===== 改进：姿态稳定性奖励（进一步减少惩罚，避免震荡）=====
        # 对于Stage 0，几乎不惩罚姿态，让智能体专注于位置控制
        reward_cfg = self.config.get('reward', {})
        
        if self.curriculum_stage == 0:
            # Stage 0：姿态惩罚极低，只在严重倾斜时轻微惩罚
            roll_coeff = -0.1
            pitch_coeff = -0.1
            angular_coeff = -0.01
        else:
            roll_coeff = reward_cfg.get('attitude_roll_coeff', -0.5)
            pitch_coeff = reward_cfg.get('attitude_pitch_coeff', -0.5)
            angular_coeff = reward_cfg.get('attitude_angular_coeff', -0.05)

        # 使用更温和的惩罚形式（平滑函数）
        # 只在角度较大时才惩罚，小角度几乎不惩罚
        roll_penalty = roll_coeff * max(0, np.abs(eta[3]) - 0.2)  # 超过0.2rad（约11度）才开始惩罚
        pitch_penalty = pitch_coeff * max(0, np.abs(eta[4]) - 0.2)
        angular_rate_penalty = angular_coeff * (np.abs(nu[3]) + np.abs(nu[4]))

        attitude_penalty_raw = roll_penalty + pitch_penalty + angular_rate_penalty

        # 裁剪阈值（宽松）
        clip_limit = reward_cfg.get('attitude_clip_limit', 0.5) if self.curriculum_stage == 0 else 1.0
        attitude_penalty = max(attitude_penalty_raw, -abs(clip_limit))

        # 如果姿态稳定（接近水平），给予奖励
        if abs(eta[3]) < 0.15 and abs(eta[4]) < 0.15:  # 约8.6度内
            attitude_penalty += 0.1  # 增加奖励，鼓励稳定

        reward_info['attitude_stability_raw'] = attitude_penalty_raw
        reward_info['attitude_stability'] = attitude_penalty
        reward_info['attitude_clip_limit'] = clip_limit
        # ===== 改进结束 =====
        
        # 6. 稳定性奖励    
        stability_bonus = 0.0    
        if pos_error < thresholds['position_tolerance'] * 2:    
            stability = np.exp(-lin_vel_mag * 2)    
            stability_bonus = weights['stability_bonus'] * stability    
        reward_info['stability'] = stability_bonus    
        
        # 7. 到达目标奖励（使用与成功条件一致的判断）
        goal_bonus = 0.0
        # 使用与成功条件一致的阈值
        if self.curriculum_stage == 0:
            goal_pos_threshold = thresholds['position_tolerance']  # 0.6m（与成功条件一致）
            goal_ori_threshold = thresholds['orientation_tolerance'] * 1.33  # 0.4rad（与成功条件一致）
        else:
            goal_pos_threshold = thresholds['position_tolerance']
            goal_ori_threshold = thresholds['orientation_tolerance'] * 2.0
            
        if (pos_error < goal_pos_threshold and ori_error < goal_ori_threshold):
            goal_bonus = weights['goal_reached']
            # 距离越近，奖励越大（稠密奖励）
            if pos_error < thresholds['position_tolerance'] * 0.5:  # 非常接近时
                goal_bonus += weights['goal_reached'] * 0.5  # 额外50%奖励
            reward_info['goal_reached'] = True
        else:
            reward_info['goal_reached'] = False
        reward_info['goal'] = goal_bonus
        
        # 8. 时间惩罚    
        time_penalty = weights['time_penalty']    
        reward_info['time'] = time_penalty    
        
        # 总奖励（position_reward已经包含了位置改进奖励）
        total_reward = (    
            position_reward +     # 已包含改进奖励
            orientation_reward +     
            velocity_penalty +     
            angular_penalty +     
            energy_reward +     
            attitude_penalty +    
            stability_bonus +     
            goal_bonus +     
            time_penalty    
        )
        
        # 更新误差记录（用于下次计算改进奖励）
        self.prev_pos_error = pos_error    
        self.prev_ori_error = ori_error
        
        reward_info['total'] = total_reward    
        reward_info['pos_error'] = pos_error    
        reward_info['ori_error'] = ori_error    
        
        return total_reward, reward_info    

  
      
    def _check_termination(self) -> Tuple[bool, bool, Dict]:  
        """检查终止条件"""  
        info = {}  
        
        eta = self.dynamics.eta  
        nu = self.dynamics.nu  
        
        target_pos = self.target_state[:3]  
        target_ori = self.target_state[3:6]  
        
        pos_error = np.linalg.norm(eta[:3] - target_pos)  
        ori_error = np.linalg.norm(self._angle_difference(eta[3:6], target_ori))  
        
        # 成功条件（合理设置，确保需要努力才能达到）
        # Stage 0：相对宽松但仍需要一些步骤
        if self.curriculum_stage == 0:
            # 位置要求：0.5m（从0.9m收紧，确保不是一开始就满足）
            position_reached = pos_error < self.reward_thresholds['position_tolerance']  # 0.5m
            # 姿态要求：0.4rad（约23度，从0.9rad收紧）
            orientation_reached = ori_error < self.reward_thresholds['orientation_tolerance'] * 1.33  # 0.4rad
        else:
            position_reached = pos_error < self.reward_thresholds['position_tolerance']  
            orientation_reached = ori_error < self.reward_thresholds['orientation_tolerance'] * 2.0
        
        success = position_reached and orientation_reached  
        if success:  
            info['termination_reason'] = 'success'  
            return True, False, info  
        
        # 检查边界  
        world_bounds = self.env_config['world']  
        margin = 5.0  
        if (eta[0] < world_bounds['x_range'][0] + margin or   
            eta[0] > world_bounds['x_range'][1] - margin or  
            eta[1] < world_bounds['y_range'][0] + margin or   
            eta[1] > world_bounds['y_range'][1] - margin or  
            eta[2] < world_bounds['z_range'][0] + 0.5 or  
            eta[2] > world_bounds['z_range'][1] - margin):  
            info['termination_reason'] = 'out_of_bounds'  
            return True, False, info  
  
        # 无进展终止：长时间未接近目标  
        if self.no_progress_cfg.get('enabled', True):  
            patience = int(self.no_progress_cfg.get('patience_steps', 250))  
            min_improve = float(self.no_progress_cfg.get('min_improvement', 0.02))  
            min_distance = float(self.no_progress_cfg.get('min_distance', 0.0))  
            if pos_error > min_distance:  
                if (self.best_pos_error - pos_error) > min_improve:  
                    self.best_pos_error = pos_error  
                    self.no_progress_steps = 0  
                else:  
                    self.no_progress_steps += 1  
                if self.no_progress_steps >= patience:  
                    info['termination_reason'] = 'no_progress'  
                    return False, True, info  
            else:  
                self.no_progress_steps = 0  
        
        # 翻转检测：放宽阈值，给智能体恢复机会
        # 使用更宽松的阈值（约120度），只有在严重翻转时才终止
        flip_threshold = self.config.get('reward', {}).get('flip_threshold', np.pi * 2.0)  # 约120度
        # 只检查pitch，roll可以允许更大范围（因为是水平推进器，roll影响较小）
        if abs(eta[4]) > flip_threshold:  # 只检查pitch，更宽松
            info['termination_reason'] = 'flip'
            return True, False, info
        # Roll超过90度才终止（更宽松）
        if abs(eta[3]) > np.pi * 1.5:  # 约90度
            info['termination_reason'] = 'flip'
            return True, False, info

        # flip_threshold = np.pi * 0.8  # 约144度  
        # if abs(eta[4]) > flip_threshold:  # 只检查pitch  
        #     info['termination_reason'] = 'flip'  
        #     return True, False, info  
        
        # 检查超时  
        if self.current_step >= self.max_steps:  
            info['termination_reason'] = 'timeout'  
            # 超时时也检查是否满足成功条件（可能在最后一步达到目标）
            pos_error = np.linalg.norm(eta[:3] - target_pos)
            ori_error = np.linalg.norm(self._angle_difference(eta[3:6], target_ori))
            if self.curriculum_stage == 0:
                success_pos = pos_error < self.reward_thresholds['position_tolerance']
                success_ori = ori_error < self.reward_thresholds['orientation_tolerance'] * 1.33
            else:
                success_pos = pos_error < self.reward_thresholds['position_tolerance']
                success_ori = ori_error < self.reward_thresholds['orientation_tolerance'] * 2.0
            if success_pos and success_ori:
                info['termination_reason'] = 'success'  # 即使是超时，如果满足条件也算成功
                return True, False, info
            return False, True, info  
        
        return False, False, info  

      
    def _sample_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        根据课程学习阶段采样初始状态  
        """  
        stage = self.curriculum_config  
        world_bounds = self.env_config['world']  
        z_range = world_bounds['z_range']  
        position_range = stage['position_range']  
        
        if self.curriculum_stage == 0:  
            # ===== Stage 0: 简化任务，但确保初始距离大于成功阈值 =====  
            # 获取成功阈值（Stage 0使用position_tolerance，不加倍）
            success_pos_threshold = self.reward_thresholds['position_tolerance']  # 0.5m
            
            # 初始距离必须大于成功阈值，确保需要一些步骤才能成功
            # 至少比成功阈值大0.4m，最小0.8m
            min_distance = max(0.8, success_pos_threshold + 0.4)  # 至少0.9m
            max_distance = min(position_range, 2.0)  # 最大距离限制为2.0m
            
            # 确保min_distance < max_distance
            if min_distance >= max_distance:
                min_distance = max_distance * 0.7
            
            # 随机方向（均匀分布在球面上）
            angle_xy = self.np_random.uniform(0, 2 * np.pi)  # 水平角度
            angle_z = self.np_random.uniform(-np.pi/6, np.pi/6)  # 垂直角度（±30度）
            distance = self.np_random.uniform(min_distance, max_distance)
            
            # 转换为笛卡尔坐标
            x = distance * np.cos(angle_z) * np.cos(angle_xy)
            y = distance * np.cos(angle_z) * np.sin(angle_xy)
            z = 2.0 + distance * np.sin(angle_z)  # 基准深度2m
            
            # 确保 z 在有效范围内  
            z = np.clip(z, z_range[0] + 0.5, z_range[1] - 1.0)  
            
            position = np.array([x, y, z])  
            # 初始姿态接近水平（允许小偏差）
            orientation = self.np_random.uniform(-0.1, 0.1, 3)  # 约±5.7度
            # 初始速度为零或很小
            nu_init = np.zeros(6)  
            
        else:  
            # 其他 Stage 保持不变  
            x = self.np_random.uniform(  
                max(world_bounds['x_range'][0] + 1, -position_range),  
                min(world_bounds['x_range'][1] - 1, position_range)  
            )  
            y = self.np_random.uniform(  
                max(world_bounds['y_range'][0] + 1, -position_range),  
                min(world_bounds['y_range'][1] - 1, position_range)  
            )  
            z_min = max(z_range[0] + 1, 1.0)  
            z_max = min(z_range[1] - 1, z_min + position_range)  
            z = self.np_random.uniform(z_min, z_max)  
            
            position = np.array([x, y, z])  
            
            if stage.get('orientation_required', False):  
                orientation = self.np_random.uniform(-0.1, 0.1, 3)  
            else:  
                orientation = np.zeros(3)  
            
            vel_type = stage.get('velocity_init', 'zero')  
            if vel_type == 'zero':  
                nu_init = np.zeros(6)  
            elif vel_type == 'small':  
                nu_init = self.np_random.uniform(-0.1, 0.1, 6)  
            elif vel_type == 'random':  
                nu_init = self.np_random.uniform(-0.3, 0.3, 6)  
            else:  
                nu_init = np.zeros(6)  
        
        eta_init = np.concatenate([position, orientation])  
        return eta_init, nu_init  

    
    
    def _sample_target_state(self) -> np.ndarray:    
        """    
        采样目标状态    
        
        Returns:    
            target_state: 目标状态 [x, y, z, roll, pitch, yaw, 0, 0, 0, 0, 0, 0]    
        """    
        stage = self.curriculum_config    
        
        # 获取世界边界    
        world_bounds = self.env_config['world']    
        z_range = world_bounds['z_range']    
        
        position_range = stage['position_range']    
        
        # ===== Stage 0: 固定目标在原点，简化任务 =====  
        if self.curriculum_stage == 0:  
            # 目标固定在原点，高度固定（让任务明确简单）
            target_x = 0.0  # 固定为原点
            target_y = 0.0  
            target_z = 2.0  # 固定深度2m
            target_ori = np.zeros(3)  # 目标姿态为水平  
            
        else:  
            # ===== 其他Stage: 随机目标 =====  
            if self.np_random.random() < 0.5:    
                # 目标为原点附近    
                target_x = self.np_random.uniform(-1, 1)    
                target_y = self.np_random.uniform(-1, 1)    
                target_z = self.np_random.uniform(max(z_range[0] + 1, 1.0), max(z_range[0] + 3, 3.0))    
            else:    
                # 随机目标点    
                range_val = position_range * 0.5    
                target_x = self.np_random.uniform(-range_val, range_val)    
                target_y = self.np_random.uniform(-range_val, range_val)    
                target_z = self.np_random.uniform(max(z_range[0] + 1, 1.0), max(z_range[0] + 1 + range_val, 1 + range_val))    
            
            # 目标姿态    
            if stage.get('orientation_required', False):    
                target_ori = self.np_random.uniform(-0.1, 0.1, 3)    
            else:    
                target_ori = np.zeros(3)    
        
        target_pos = np.array([target_x, target_y, target_z])    
        target_vel = np.zeros(6)    
        
        return np.concatenate([target_pos, target_ori, target_vel])    
 

      
    def reset(self,   
              seed: Optional[int] = None,   
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:  
        """  
        重置环境  
          
        Args:  
            seed: 随机种子  
            options: 重置选项  
              
        Returns:  
            observation: 初始观测  
            info: 初始信息  
        """  
        super().reset(seed=seed)  
          
        if seed is not None:  
            self.seed(seed)  
              
        # 重置动力学模型（如果启用域随机化，会重新采样参数）  
        eta_init, nu_init = self._sample_initial_state()  
        self.dynamics.reset(eta=eta_init, nu=nu_init)  
          
        # 采样目标状态  
        self.target_state = self._sample_target_state()  
          
        # 重置水流  
        self.water_current.reset()  
          
        # 重置Episode状态  
        self.current_step = 0  
        self.episode_reward = 0.0  
        self.prev_pos_error = np.linalg.norm(eta_init[:3] - self.target_state[:3])  
        self.prev_ori_error = np.linalg.norm(self._angle_difference(eta_init[3:6], self.target_state[3:6]))  
        self.no_progress_steps = 0  
        self.best_pos_error = self.prev_pos_error  
          
        # 重置记录  
        self.episode_info = {  
            'position_errors': [],  
            'orientation_errors': [],  
            'rewards': [],  
            'actions': [],
            'reward_components': []
        }  
        self.trajectory = []  
          
        # 获取初始观测  
        observation = self._get_observation()  
          
        info = {  
            'initial_state': eta_init.tolist(),  
            'target_state': self.target_state[:6].tolist(),  
            'curriculum_stage': self.curriculum_stage  
        }  
        self.integrated_controller.reset()  
        self.last_thrust_command = np.zeros(self.robot_config['thruster']['num_thrusters'])  
        self.prev_heading_target = None  
        return observation, info  
      
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:  
        """  
        执行一步 - 带姿态稳定  
        """  
        # ===== 调试信息 =====  
       # print(f"DEBUG: action={action}, type={type(action)}, shape={getattr(action, 'shape', 'N/A')}")  
        
        # 转换为 int  
        if isinstance(action, np.ndarray):  
            action = action.item() if action.ndim == 0 else int(action[0])  
        elif hasattr(action, 'item'):  
            action = action.item()  
        else:  
            action = int(action)  
        
        #print(f"DEBUG: after conversion action={action}, type={type(action)}")  
        
        assert self.action_space.contains(action), f"Invalid action {action}"    
        # assert self.action_space.contains(action), f"Invalid action {action}"  
        
        # 获取基础推力命令  
        # thrust_command = self.action_to_thrust[action].copy()  
        
        # # ===== 添加姿态稳定 =====  
        # thrust_command = self._add_attitude_stabilization(thrust_command)  
            # ===== 修改：使用集成控制器计算推力 =====  
        eta = self.dynamics.eta  
        nu = self.dynamics.nu  
  
        # 位置动作时可对准目标方向，减少绕圈  
        action_cfg = self.config.get('action_space', {})  
        heading_mode = action_cfg.get('translation_heading', 'lock')  
        target_orientation = None  
        if heading_mode == 'target' and action in [0, 1, 2, 3, 4, 5, 6]:  
            target_pos = self.target_state[:3]  
            desired_yaw = np.arctan2(target_pos[1] - eta[1], target_pos[0] - eta[0])  
            prev_yaw = eta[5] if self.prev_heading_target is None else self.prev_heading_target  
            yaw_delta = np.arctan2(np.sin(desired_yaw - prev_yaw), np.cos(desired_yaw - prev_yaw))  
            max_delta = float(action_cfg.get('heading_max_delta', 0.12))  
            yaw_delta = float(np.clip(yaw_delta, -max_delta, max_delta))  
            target_yaw = prev_yaw + yaw_delta  
            target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))  
            self.prev_heading_target = target_yaw  
            target_orientation = np.array([0.0, 0.0, target_yaw], dtype=float)  
        
        thrust_command, ctrl_debug = self.integrated_controller.compute_thrust(  
            action=action,  
            eta=eta,  
            nu=nu,  
            Ts=self.dt,  
            target_orientation=target_orientation,  
            target_position=self.target_state[:3]  
        )  
        self.last_thrust_command = thrust_command.copy()  

        # 计算水流力（外部扰动）  
        external_force = self._compute_external_force()  
        
        # 执行动力学仿真  
        eta, nu = self.dynamics.step(thrust_command, external_force)  
        
        # 更新水流时间  
        self.water_current.step(self.dt)  
        
        # 计算奖励  
        reward, reward_info = self._compute_reward(action)  
        
        # 检查终止  
        terminated, truncated, term_info = self._check_termination()  
        
        # 更新Episode状态  
        self.current_step += 1  
        self.episode_reward += reward  
        
        # 记录  
        self.episode_info['position_errors'].append(reward_info['pos_error'])  
        self.episode_info['orientation_errors'].append(reward_info['ori_error'])  
        self.episode_info['rewards'].append(reward)  
        self.episode_info['actions'].append(action)  
        # 保存每步奖励分解（浅拷贝），便于训练时统计与可视化
        try:
            self.episode_info['reward_components'].append(dict(reward_info))
        except Exception:
            # 保持健壮性：若 reward_info 不是可转 dict 的结构则忽略
            pass
        
        # 轨迹记录  
        self.trajectory.append({  
            'position': eta[:3].copy(),  
            'orientation': eta[3:6].copy(),  
            'velocity': nu.copy(),  
            'action': action,  
            'reward': reward  
        })  
        
        # 获取新观测  
        observation = self._get_observation()  
        
        # 构建info  
        info = {  
            **reward_info,  
            **term_info,  
            'step': self.current_step,  
            'episode_reward': self.episode_reward,  
            'action_name': self.ACTION_NAMES[action]  
        }  
        
        # Episode统计  
        if terminated or truncated:  
            # 判断是否成功（使用与成功条件完全一致的判断）
            termination_reason = term_info.get('termination_reason', 'unknown')
            final_pos_err = reward_info.get('pos_error', 999)
            final_ori_err = reward_info.get('ori_error', 999)
            
            # 判断成功（与_check_termination中的逻辑完全一致）
            if termination_reason == 'success':
                is_success = True
            else:
                # 如果是因为其他原因终止，但误差足够小，也算成功
                if self.curriculum_stage == 0:
                    pos_threshold = self.reward_thresholds['position_tolerance']  # 0.5m
                    ori_threshold = self.reward_thresholds['orientation_tolerance'] * 1.33  # 0.4rad
                else:
                    pos_threshold = self.reward_thresholds['position_tolerance']
                    ori_threshold = self.reward_thresholds['orientation_tolerance'] * 2.0
                
                is_success = (final_pos_err < pos_threshold and final_ori_err < ori_threshold)
            
            info['episode'] = {  
                'r': self.episode_reward,  
                'l': self.current_step,  
                'success': is_success,  # 确保success字段存在且正确
                'mean_pos_error': np.mean(self.episode_info['position_errors']) if self.episode_info['position_errors'] else 999,
                'mean_ori_error': np.mean(self.episode_info['orientation_errors']) if self.episode_info['orientation_errors'] else 999,
                'final_pos_error': final_pos_err,  
                'final_ori_error': final_ori_err,
                'termination_reason': termination_reason,
                'reward_components_mean': {}
            }  
            # 计算并填充每个奖励分量的均值，方便训练监控（忽略非数值字段）
            rc_list = self.episode_info.get('reward_components', [])
            if len(rc_list) > 0:
                keys = set()
                for d in rc_list:
                    if isinstance(d, dict):
                        keys.update(d.keys())
                for k in sorted(keys):
                    vals = []
                    for d in rc_list:
                        if isinstance(d, dict) and k in d:
                            v = d[k]
                            # 支持 numpy 标量与 Python 原生数字
                            if np.isscalar(v) or isinstance(v, (int, float, bool)):
                                try:
                                    vals.append(float(v))
                                except Exception:
                                    continue
                    if len(vals) > 0:
                        info['episode']['reward_components_mean'][k] = float(np.mean(vals))
        
        return observation, reward, terminated, truncated, info  
    
    
    # def _add_attitude_stabilization(self, thrust_command: np.ndarray) -> np.ndarray:  
    #     """  
    #     添加姿态稳定化推力  
        
    #     使用PD控制器自动稳定roll和pitch，让智能体专注于位置控制  
    #     """  
    #     eta = self.dynamics.eta  
    #     nu = self.dynamics.nu  
        
    #     roll = eta[3]  
    #     pitch = eta[4]  
    #     roll_rate = nu[3]  
    #     pitch_rate = nu[4]  
        
    #     # PD增益（较强的稳定控制）  
    #     Kp_roll = 20.0  
    #     Kd_roll = 10.0  
    #     Kp_pitch = 30.0  
    #     Kd_pitch = 15.0  
        
    #     # 计算修正量  
    #     roll_correction = -Kp_roll * roll - Kd_roll * roll_rate  
    #     pitch_correction = -Kp_pitch * pitch - Kd_pitch * pitch_rate  
        
    #     # 限制最大修正量  
    #     max_correction = 20.0  
    #     roll_correction = np.clip(roll_correction, -max_correction, max_correction)  
    #     pitch_correction = np.clip(pitch_correction, -max_correction, max_correction)  
        
    #     # Pitch修正：使用垂直推进器 T5(前) 和 T6(后)  
    #     thrust_command[4] += -pitch_correction * 0.8  # T5  
    #     thrust_command[5] += pitch_correction * 0.8  # T6  
        
    #     # ===== 新增：尝试用水平推进器修正 roll =====  
    #     # 虽然效果有限，但聊胜于无  
    #     # T1/T4 在右侧，T2/T3 在左侧  
    #     # roll > 0 时（向右倾），需要右侧推进器减小推力  
    #     roll_thrust_correction = roll_correction * 0.1  
    #     thrust_command[0] += -roll_thrust_correction  # T1 右前  
    #     thrust_command[3] += -roll_thrust_correction  # T4 右后  
    #     thrust_command[1] += roll_thrust_correction   # T2 左前  
    #     thrust_command[2] += roll_thrust_correction   # T3 左后

    #     return thrust_command  
      
    def _compute_external_force(self) -> np.ndarray:  
        """  
        计算外部力（水流等）  
          
        Returns:  
            external_force: 6维外部力/力矩向量  
        """  
        # 获取当前状态  
        eta = self.dynamics.eta  
        nu = self.dynamics.nu  
          
        # 获取水流速度（惯性系）  
        current_vel_inertial = self.water_current.get_current_velocity(eta[:3])  
          
        # 转换到体坐标系  
        R = self.dynamics._rotation_matrix(eta[3], eta[4], eta[5])  
        current_vel_body = R.T @ current_vel_inertial  
          
        # 相对速度  
        nu_r = nu.copy()  
        nu_r[:3] = nu[:3] - current_vel_body  
          
        # 水流产生的额外力（简化模型）  
        # 这里使用阻尼力模型估计水流效应  
        drag_coeff = 10.0  # 简化的阻力系数  
        external_force = np.zeros(6)  
        external_force[:3] = -drag_coeff * (nu[:3] - current_vel_body)  
          
        # 添加课程学习的扰动  
        if self.curriculum_config.get('add_disturbance', False):  
            # 随机扰动  
            noise = self.np_random.normal(0, 1.0, 6)  
            external_force += noise  
              
        return external_force  
      
    def set_curriculum_stage(self, stage: int):  
        """  
        设置课程学习阶段  
          
        Args:  
            stage: 阶段索引 (0-3)  
        """  
        assert 0 <= stage < len(self.config['curriculum']['stages'])  
        self.curriculum_stage = stage  
        self.curriculum_config = self.config['curriculum']['stages'][stage]  
          
    def render(self):  
        """渲染环境"""  
        if self.render_mode == "human":  
            self._render_human()  
        elif self.render_mode == "rgb_array":  
            return self._render_rgb_array()  
              
    def _render_human(self):  
        """人类可视化模式"""  
        import matplotlib.pyplot as plt  
          
        if self.fig is None:  
            plt.ion()  
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 6))  
              
        # 清除之前的内容  
        for ax in self.ax:  
            ax.clear()  
              
        # 绘制2D轨迹  
        if len(self.trajectory) > 0:  
            positions = np.array([t['position'] for t in self.trajectory])  
              
            # XY视图  
            self.ax[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)  
            self.ax[0].plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')  
            self.ax[0].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='Current')  
              
            # 目标  
            target = self.target_state[:3]  
            self.ax[0].plot(target[0], target[1], 'r*', markersize=15, label='Target')  
              
            self.ax[0].set_xlabel('X (m)')  
            self.ax[0].set_ylabel('Y (m)')  
            self.ax[0].set_title('XY Trajectory')  
            self.ax[0].legend()  
            self.ax[0].grid(True)  
            self.ax[0].axis('equal')  
              
            # 误差曲线  
            pos_errors = self.episode_info['position_errors']  
            ori_errors = self.episode_info['orientation_errors']  
              
            steps = range(len(pos_errors))  
            self.ax[1].plot(steps, pos_errors, 'b-', label='Position Error')  
            self.ax[1].plot(steps, ori_errors, 'r-', label='Orientation Error')  
            self.ax[1].axhline(y=self.reward_thresholds['position_tolerance'],   
                               color='b', linestyle='--', alpha=0.5)  
            self.ax[1].axhline(y=self.reward_thresholds['orientation_tolerance'],  
                               color='r', linestyle='--', alpha=0.5)  
            self.ax[1].set_xlabel('Step')  
            self.ax[1].set_ylabel('Error')  
            self.ax[1].set_title('Error Over Time')  
            self.ax[1].legend()  
            self.ax[1].grid(True)  
              
        plt.tight_layout()  
        plt.pause(0.01)  
          
    def _render_rgb_array(self) -> np.ndarray:  
        """RGB数组模式"""  
        import matplotlib.pyplot as plt  
        from matplotlib.backends.backend_agg import FigureCanvasAgg  
          
        fig, ax = plt.subplots(figsize=(8, 8))  
          
        if len(self.trajectory) > 0:  
            positions = np.array([t['position'] for t in self.trajectory])  
              
            ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)  
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10)  
            ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10)  
              
            target = self.target_state[:3]  
            ax.plot(target[0], target[1], 'r*', markersize=15)  
              
        ax.set_xlabel('X (m)')  
        ax.set_ylabel('Y (m)')  
        ax.grid(True)  
        ax.axis('equal')  
          
        canvas = FigureCanvasAgg(fig)  
        canvas.draw()  
          
        width, height = fig.get_size_inches() * fig.get_dpi()  
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  
        image = image.reshape(int(height), int(width), 3)  
          
        plt.close(fig)  
          
        return image  
      
    def close(self):  
        """关闭环境"""  
        if self.fig is not None:  
            import matplotlib.pyplot as plt  
            plt.close(self.fig)  
            self.fig = None  
            self.ax = None  
              
    def get_episode_statistics(self) -> Dict:  
        """获取Episode统计信息"""  
        if not self.episode_info['rewards']:  
            return {}  
              
        return {  
            'total_reward': self.episode_reward,  
            'episode_length': self.current_step,  
            'mean_position_error': np.mean(self.episode_info['position_errors']),  
            'mean_orientation_error': np.mean(self.episode_info['orientation_errors']),  
            'final_position_error': self.episode_info['position_errors'][-1] if self.episode_info['position_errors'] else 0,  
            'final_orientation_error': self.episode_info['orientation_errors'][-1] if self.episode_info['orientation_errors'] else 0,  
            'action_distribution': {  
                name: self.episode_info['actions'].count(i)   
                for i, name in enumerate(self.ACTION_NAMES)  
            }  
        }  
  
  
class VecLowLevelAUVEnv:  
    """  
    向量化环境包装器  
      
    用于并行训练多个环境  
    """  
      
    def __init__(self,   
                 num_envs: int,  
                 config_path: str = None,  
                 curriculum_stage: int = 0,  
                 enable_domain_randomization: bool = True,  
                 seeds: Optional[List[int]] = None):  
        """  
        初始化  
          
        Args:  
            num_envs: 并行环境数量  
            config_path: 配置文件路径  
            curriculum_stage: 课程学习阶段  
            enable_domain_randomization: 是否启用域随机化  
            seeds: 随机种子列表  
        """  
        self.num_envs = num_envs  
          
        # 创建环境  
        if seeds is None:  
            seeds = [None] * num_envs  
              
        self.envs = [  
            LowLevelAUVEnv(  
                config_path=config_path,  
                curriculum_stage=curriculum_stage,  
                enable_domain_randomization=enable_domain_randomization,  
                seed=seeds[i]  
            )  
            for i in range(num_envs)  
        ]  
          
        # 获取空间信息  
        self.observation_space = self.envs[0].observation_space  
        self.action_space = self.envs[0].action_space  
        self.single_observation_space = self.observation_space  
        self.single_action_space = self.action_space  
          
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:  
        """重置所有环境"""  
        observations = []  
        infos = []  
          
        for i, env in enumerate(self.envs):  
            env_seed = seed + i if seed is not None else None  
            obs, info = env.reset(seed=env_seed)  
            observations.append(obs)  
            infos.append(info)  
              
        return np.array(observations), infos  
      
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:  
        """执行一步（向量化版本）"""  
        observations = []  
        rewards = []  
        terminateds = []  
        truncateds = []  
        infos = []  
          
             # 确保 actions 是可迭代的  
        if isinstance(actions, np.ndarray):  
            actions = actions.flatten()   
        for i, (env, action) in enumerate(zip(self.envs, actions)):  
            action = actions[i]  
            
            # 转换为 int  
            if isinstance(action, np.ndarray):  
                action = action.item() if action.ndim == 0 else int(action[0])  
            elif hasattr(action, 'item'):  
                action = action.item()  
            else:  
                action = int(action)  
            
            obs, reward, terminated, truncated, info = env.step(action)  
                         
            # 自动重置  
            if terminated or truncated:  
                final_info = info.copy()  
                obs, reset_info = env.reset()  
                info = reset_info  
                info['final_info'] = final_info  
              
            observations.append(obs)  
            rewards.append(reward)  
            terminateds.append(terminated)  
            truncateds.append(truncated)  
            infos.append(info)  
          
        return (  
            np.array(observations),  
            np.array(rewards),  
            np.array(terminateds),  
            np.array(truncateds),  
            infos  
        )  
      
      
    def set_curriculum_stage(self, stage: int):  
        """设置所有环境的课程阶段"""  
        for env in self.envs:  
            env.set_curriculum_stage(stage)  
              
    def close(self):  
        """关闭所有环境"""  
        for env in self.envs:  
            env.close()  
