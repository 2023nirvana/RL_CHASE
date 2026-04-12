"""  
水动力学辅助模块  
================  
  
提供水流模型、额外的水动力效应计算等。  
"""  
  
import numpy as np  
from typing import Optional, Tuple  
  
  
class WaterCurrentModel:  
    """  
    水流模型  
      
    模拟真实水下环境中的水流扰动  
    """  
      
    def __init__(self, config: dict):  
        """  
        初始化水流模型  
          
        Args:  
            config: 水流配置字典  
        """  
        self.enabled = config.get('enabled', False)  
        self.type = config.get('type', 'constant')  
        self.base_velocity = np.array(config.get('base_velocity', [0, 0, 0]))  
        self.variation_amplitude = config.get('variation_amplitude', 0.0)  
        self.variation_frequency = config.get('variation_frequency', 0.0)  
          
        self.time = 0.0  
          
    def get_current_velocity(self, position: np.ndarray,   
                            time: Optional[float] = None) -> np.ndarray:  
        """  
        获取指定位置和时间的水流速度（惯性系）  
          
        Args:  
            position: 位置 [x, y, z]  
            time: 时间（如果None则使用内部时间）  
              
        Returns:  
            current_velocity: 水流速度 [vx, vy, vz]  
        """  
        if not self.enabled:  
            return np.zeros(3)  
              
        if time is None:  
            time = self.time  
              
        if self.type == 'constant':  
            return self.base_velocity.copy()  
              
        elif self.type == 'variable':  
            # 时变水流  
            variation = self.variation_amplitude * np.sin(  
                2 * np.pi * self.variation_frequency * time  
            )  
            return self.base_velocity + variation * np.array([1, 0.5, 0])  
              
        elif self.type == 'tidal':  
            # 潮汐模型（更复杂的周期性变化）  
            omega1 = 2 * np.pi * self.variation_frequency  
            omega2 = 2 * np.pi * self.variation_frequency * 0.5  
              
            vx = self.base_velocity[0] + self.variation_amplitude * (  
                np.sin(omega1 * time) + 0.3 * np.sin(omega2 * time)  
            )  
            vy = self.base_velocity[1] + self.variation_amplitude * 0.5 * (  
                np.cos(omega1 * time)  
            )  
            vz = self.base_velocity[2]  
              
            return np.array([vx, vy, vz])  
              
        elif self.type == 'spatial':  
            # 空间变化的水流（如接近海床时减小）  
            z = position[2]  
            depth_factor = 1.0 - 0.5 * np.exp(-z / 10.0)  # 越深水流越弱  
            return self.base_velocity * depth_factor  
              
        else:  
            return self.base_velocity.copy()  
      
    def compute_current_force(self, nu_r: np.ndarray,   
                              D_l: np.ndarray,   
                              D_q_coeffs: np.ndarray) -> np.ndarray:  
        """  
        计算水流产生的额外力  
          
        使用相对速度计算阻尼力  
          
        Args:  
            nu_r: 相对速度 ν_r = ν - ν_c  
            D_l: 线性阻尼矩阵  
            D_q_coeffs: 二次阻尼系数  
              
        Returns:  
            force: 水流力  
        """  
        # 线性项  
        F_linear = D_l @ nu_r  
          
        # 二次项  
        F_quadratic = D_q_coeffs * np.abs(nu_r) * nu_r  
          
        return F_linear + F_quadratic  
      
    def step(self, dt: float):  
        """更新时间"""  
        self.time += dt  
          
    def reset(self):  
        """重置时间"""  
        self.time = 0.0  
  
  
class AddedMassEffect:  
    """  
    附加质量效应的详细计算  
      
    附加质量实际上随运动状态变化，这里提供更精确的模型  
    """  
      
    def __init__(self, robot_config: dict):  
        """  
        初始化  
          
        Args:  
            robot_config: 机器人配置  
        """  
        self.length = robot_config.get('length', 1.0)  
        self.width = robot_config.get('width', 0.3)  
        self.height = robot_config.get('height', 0.3)  
          
        # 基础附加质量系数  
        self.base_coeffs = np.ones(6)  
          
    def compute_frequency_dependent_added_mass(self,   
                                               frequency: float,  
                                               base_added_mass: np.ndarray) -> np.ndarray:  
        """  
        计算频率相关的附加质量  
          
        在高频运动时，附加质量可能减小  
          
        Args:  
            frequency: 运动频率  
            base_added_mass: 基础附加质量矩阵  
              
        Returns:  
            修正后的附加质量矩阵  
        """  
        # 简化的频率修正因子  
        # 实际应用中需要通过CFD或实验获得  
        cutoff_freq = 2.0  # Hz  
        if frequency > cutoff_freq:  
            factor = cutoff_freq / frequency  
        else:  
            factor = 1.0  
              
        return base_added_mass * factor  
  
  
class VortexSheddingModel:  
    """  
    涡脱落模型  
      
    当AUV以一定速度运动时，尾部会产生涡脱落，  
    导致周期性的侧向力。这是非线性效应的一个来源。  
    """  
      
    def __init__(self, characteristic_length: float = 1.0):  
        """  
        初始化  
          
        Args:  
            characteristic_length: 特征长度（通常是AUV直径）  
        """  
        self.L = characteristic_length  
        self.strouhal_number = 0.2  # 典型值  
        self.phase = 0.0  
          
    def compute_vortex_force(self, velocity: float,   
                             rho: float = 1025.0) -> Tuple[float, float]:  
        """  
        计算涡脱落力  
          
        Args:  
            velocity: 流速  
            rho: 水密度  
              
        Returns:  
            lift_force: 升力（侧向力）  
            frequency: 涡脱落频率  
        """  
        if abs(velocity) < 0.1:  
            return 0.0, 0.0  
              
        # 涡脱落频率: f = St * V / L  
        frequency = self.strouhal_number * abs(velocity) / self.L  
          
        # 涡脱落引起的侧向力（简化模型）  
        # 实际应该是周期性的，这里给出幅值  
        Cl = 0.3  # 升力系数  
        A = self.L * self.L * 0.25  # 参考面积  
        lift_amplitude = 0.5 * rho * velocity**2 * Cl * A  
          
        # 周期性力  
        lift_force = lift_amplitude * np.sin(self.phase)  
          
        return lift_force, frequency  
      
    def step(self, velocity: float, dt: float):  
        """更新相位"""  
        _, freq = self.compute_vortex_force(velocity)  
        self.phase += 2 * np.pi * freq * dt  
        self.phase = self.phase % (2 * np.pi)  
          
    def reset(self):  
        """重置"""  
        self.phase = 0.0  
