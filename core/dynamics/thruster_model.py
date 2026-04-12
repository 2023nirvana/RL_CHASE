"""  
推进器模型  
==========  
  
详细的水下推进器模型，包括：  
1. 推力曲线（非线性）  
2. 动态响应  
3. 效率损失  
4. 饱和与死区  
"""  
  
import numpy as np  
from typing import Optional, List, Dict  
  
  
class ThrusterModel:  
    """  
    单个推进器的详细模型  
    """  
      
    def __init__(self, config: dict):  
        """  
        初始化推进器  
          
        Args:  
            config: 推进器配置  
        """  
        self.max_thrust = config.get('max_thrust', 20.0)  
        self.min_thrust = config.get('min_thrust', -15.0)  # 反推通常弱于正推  
        self.time_constant = config.get('time_constant', 0.1)  
        self.deadband = config.get('deadband', 0.5)  
          
        # 推进器特性曲线参数（非线性）  
        self.thrust_coefficient = config.get('thrust_coefficient', 0.01)  
        self.torque_coefficient = config.get('torque_coefficient', 0.001)  
          
        # 状态  
        self.current_thrust = 0.0  
        self.current_rpm = 0.0  
          
    def compute_thrust(self, command: float,   
                       inflow_velocity: float = 0.0) -> float:  
        """  
        计算实际推力  
          
        考虑：  
        1. 来流速度影响（推力随来流减小）  
        2. 非线性推力曲线  
          
        Args:  
            command: 推力命令  
            inflow_velocity: 来流速度（正值表示顺流）  
              
        Returns:  
            actual_thrust: 实际推力  
        """  
        # 饱和  
        command = np.clip(command, self.min_thrust, self.max_thrust)  
          
        # 死区  
        if abs(command) < self.deadband:  
            command = 0.0  
              
        # 来流速度影响  
        # 当来流与推力方向相同时，推力效率降低  
        if inflow_velocity > 0 and command > 0:  
            efficiency = max(0.5, 1.0 - 0.1 * inflow_velocity)  
        elif inflow_velocity < 0 and command < 0:  
            efficiency = max(0.5, 1.0 + 0.1 * inflow_velocity)  
        else:  
            efficiency = 1.0  
              
        return command * efficiency  
      
    def step(self, command: float, dt: float,   
             inflow_velocity: float = 0.0) -> float:  
        """  
        执行一步推进器动态  
          
        Args:  
            command: 推力命令  
            dt: 时间步长  
            inflow_velocity: 来流速度  
              
        Returns:  
            thrust: 实际产生的推力  
        """  
        # 计算目标推力  
        target_thrust = self.compute_thrust(command, inflow_velocity)  
          
        # 一阶动态响应  
        alpha = dt / (self.time_constant + dt)  
        self.current_thrust += alpha * (target_thrust - self.current_thrust)  
          
        return self.current_thrust  
      
    def reset(self):  
        """重置状态"""  
        self.current_thrust = 0.0  
        self.current_rpm = 0.0  
  
  
class ThrusterArray:  
    """  
    推进器阵列管理  
      
    管理多个推进器，计算合力/合力矩  
    """  
      
    def __init__(self, config: dict):  
        """  
        初始化推进器阵列  
          
        Args:  
            config: 推进器配置  
        """  
        self.num_thrusters = config['num_thrusters']  
        self.positions = np.array(config['positions'])  
        self.directions = np.array(config['directions'])  
          
        # 归一化方向向量  
        for i in range(self.num_thrusters):  
            norm = np.linalg.norm(self.directions[i])  
            if norm > 0:  
                self.directions[i] /= norm  
          
        # 创建各推进器  
        self.thrusters = [ThrusterModel(config) for _ in range(self.num_thrusters)]  
          
        # 构建配置矩阵  
        self.T = self._build_configuration_matrix()  
          
        # 计算伪逆（用于推力分配）  
        self.T_pinv = np.linalg.pinv(self.T)  
          
    def _build_configuration_matrix(self) -> np.ndarray:  
        """构建推进器配置矩阵"""  
        T = np.zeros((6, self.num_thrusters))  
          
        for i in range(self.num_thrusters):  
            # 力分量  
            T[0:3, i] = self.directions[i]  
            # 力矩分量  
            T[3:6, i] = np.cross(self.positions[i], self.directions[i])  
              
        return T  
      
    def allocate_thrust(self, tau_desired: np.ndarray) -> np.ndarray:  
        """  
        推力分配：从期望力/力矩计算各推进器推力  
          
        使用伪逆进行最优分配（最小二范数）  
          
        Args:  
            tau_desired: 期望的力/力矩 [Fx, Fy, Fz, Mx, My, Mz]  
              
        Returns:  
            thrust_commands: 各推进器推力命令  
        """  
        # 伪逆分配  
        thrust_commands = self.T_pinv @ tau_desired  
          
        # 检查是否超出推力限制  
        max_thrust = self.thrusters[0].max_thrust  
        min_thrust = self.thrusters[0].min_thrust  
          
        # 如果超限，进行缩放  
        max_cmd = np.max(thrust_commands)  
        min_cmd = np.min(thrust_commands)  
          
        if max_cmd > max_thrust:  
            thrust_commands *= max_thrust / max_cmd  
        if min_cmd < min_thrust:  
            thrust_commands *= min_thrust / min_cmd  
              
        return thrust_commands  
      
    def step(self, commands: np.ndarray, dt: float,  
             body_velocity: Optional[np.ndarray] = None) -> np.ndarray:  
        """  
        执行一步  
          
        Args:  
            commands: 各推进器推力命令  
            dt: 时间步长  
            body_velocity: 体坐标系速度（用于计算来流）  
              
        Returns:  
            tau: 合力/合力矩  
        """  
        actual_thrusts = np.zeros(self.num_thrusters)  
          
        for i, thruster in enumerate(self.thrusters):  
            # 计算来流速度（简化：只考虑沿推力方向的分量）  
            inflow = 0.0  
            if body_velocity is not None:  
                inflow = np.dot(body_velocity[:3], self.directions[i])  
                  
            actual_thrusts[i] = thruster.step(commands[i], dt, inflow)  
              
        # 计算合力/合力矩  
        tau = self.T @ actual_thrusts  
          
        return tau  
      
    def reset(self):  
        """重置所有推进器"""  
        for thruster in self.thrusters:  
            thruster.reset()  
      
    def get_state(self) -> Dict:  
        """获取推进器状态"""  
        return {  
            'thrusts': np.array([t.current_thrust for t in self.thrusters]),  
            'rpms': np.array([t.current_rpm for t in self.thrusters])  
        }  
  
  
class ThrustMapper:  
    """  
    推力映射器  
      
    将离散动作映射为推进器命令  
    """  
      
    def __init__(self, action_mapping: dict, num_thrusters: int, max_thrust: float):  
        """  
        初始化  
          
        Args:  
            action_mapping: 动作到推力的映射配置  
            num_thrusters: 推进器数量  
            max_thrust: 最大推力  
        """  
        self.action_mapping = action_mapping  
        self.num_thrusters = num_thrusters  
        self.max_thrust = max_thrust  
          
        # 构建映射表  
        self._build_mapping_table()  
          
    def _build_mapping_table(self):  
        """构建动作到推力的映射表"""  
        self.mapping_table = {}  
          
        for action_name, normalized_thrusts in self.action_mapping.items():  
            # 将归一化推力转换为实际推力  
            actual_thrusts = np.array(normalized_thrusts) * self.max_thrust  
            self.mapping_table[action_name] = actual_thrusts  
      
    def map_action_to_thrust(self, action_id: int, action_names: List[str]) -> np.ndarray:  
        """  
        将动作ID映射为推力命令  
          
        Args:  
            action_id: 动作ID  
            action_names: 动作名称列表  
              
        Returns:  
            thrust_commands: 推力命令数组  
        """  
        action_name = action_names[action_id]  
        return self.mapping_table.get(action_name, np.zeros(self.num_thrusters))  
