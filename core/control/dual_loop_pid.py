"""  
双环PID姿态控制器  
================  
  
基于控制架构：  
- 外环PI：角度误差 → 参考角速度  
- 内环PD：角速度误差 → 反馈力矩  
- 前馈控制：已知模型补偿  
- ESO观测器：扰动估计与补偿  
  
动力学方程：τ + τ_ext = J·ω̇ + B·ω + D|ω|ω  
  
真机参数：  
- 尺寸：52cm × 35cm × 25cm  
- 重量：15kg  
- 功率：60W  
- 8推进器布局  
  
日期: 2024  
"""  
  
import numpy as np  
from typing import Tuple, Dict, Optional  
  
  
# =============================================================================  
# 动力学参数  
# =============================================================================  
  
class DynamicsParams:  
    """  
    动力学参数  
      
    简化动力学方程：τ + τ_ext = J·ω̇ + B·ω + D|ω|ω  
      
    参数顺序: [yaw, pitch, roll]  
    """  
      
    # 转动惯量 J (kg·m²)  
    J = np.array([0.35, 0.27, 0.16])  
      
    # 线性阻尼系数 B (N·m·s/rad)  
    B = np.array([0.06, 0.05, 0.04])  
      
    # 二次阻尼系数 D (N·m·s²/rad²)  
    D = np.array([0.01, 0.008, 0.06])  
  
  
# =============================================================================  
# PID增益参数  
# =============================================================================  
  
class PIDGains:  
    """PID控制器增益参数"""  
      
    # # 外环PI参数（角度环）: [yaw, pitch, roll]  
    # Kp_outer = np.array([2.5, 2.5, 3.0])  
    # Ki_outer = np.array([0.05, 0.05, 0.05])  
      
    # # 内环PD参数（角速度环）: [yaw, pitch, roll]  
    # Kp_inner = np.array([2.0, 2.0, 1.5])  
    # Kd_inner = np.array([0.02, 0.02, 0.01])  
      
    # # ESO观测器带宽: [yaw, pitch, roll]  
    # w0 = np.array([8.0, 8.0, 8.0])  
      
    # # 限幅参数  
    # integral_limit = np.array([1.0, 1.0, 1.0])  
    # omega_ref_limit = np.array([2.0, 2.0, 2.0])  
    # tau_limit = np.array([10.0, 10.0, 10.0])  
    
    # 外环PI参数: [yaw, pitch, roll]  
    Kp_outer = np.array([2.5, 5.0, 5.0])     # pitch和roll增大  
    Ki_outer = np.array([0.05, 0.2, 0.2])    # 增大积分  
      
    # 内环PD参数: [yaw, pitch, roll]  
    Kp_inner = np.array([2.0, 5.0, 4.0])     # pitch和roll增大  
    Kd_inner = np.array([0.02, 0.1, 0.08])   # 增大微分阻尼  
      
    # ESO观测器带宽: [yaw, pitch, roll]  
    w0 = np.array([8.0, 12.0, 12.0])         # pitch和roll带宽增大  
      
    # 限幅参数  
    integral_limit = np.array([1.0, 2.0, 2.0])  
    omega_ref_limit = np.array([2.0, 3.0, 3.0])  
    tau_limit = np.array([10.0, 15.0, 15.0])   # 增大力矩限制  
  
  
# =============================================================================  
# ESO扩张状态观测器  
# =============================================================================  
  
class ESOObserver:  
    """  
    扩张状态观测器 (ESO)  
      
    观测器方程：  
        x̂̇₁ = x̂₂ + (1/J)τ + β₁(y - x̂₁)  
        x̂̇₂ = β₂(y - x̂₁)  
      
    观测器增益：  
        β₁ = 2ω₀  
        β₂ = ω₀²  
      
    离散化实现：  
        e = ω_real - x̂₁  
        u_ESO = τ_act - B·ω_real - D|ω_real|·ω_real  
        x̂₁ = x̂₁ + (u_ESO/J + β₁·e)·Ts  
        x̂₂ = x̂₂ + β₂·e·Ts  
        τ_ESO = J·x̂₂  
    """  
      
    def __init__(self):  
        """初始化ESO观测器"""  
        # 动力学参数  
        self.J = DynamicsParams.J  
        self.B = DynamicsParams.B  
        self.D = DynamicsParams.D  
          
        # 观测器增益  
        w0 = PIDGains.w0  
        self.beta1 = 2.0 * w0  
        self.beta2 = w0 ** 2  
          
        # 观测器状态: [yaw, pitch, roll]  
        self.x1_hat = np.zeros(3)  # 估计角速度  
        self.x2_hat = np.zeros(3)  # 估计扰动  
          
    def reset(self):  
        """重置观测器状态"""  
        self.x1_hat = np.zeros(3)  
        self.x2_hat = np.zeros(3)  
          
    def update(self, omega_real: np.ndarray, tau_act: np.ndarray, Ts: float) -> np.ndarray:  
        """  
        更新ESO观测器  
          
        Args:  
            omega_real: 实际角速度 [yaw, pitch, roll] (rad/s)  
            tau_act: 上一时刻的控制力矩 [yaw, pitch, roll]  
            Ts: 采样周期 (s)  
              
        Returns:  
            tau_ESO: ESO估计的扰动力矩 [yaw, pitch, roll]  
        """  
        # Step 1: 观测误差  
        e = omega_real - self.x1_hat  
          
        # Step 2: 预处理（去掉已知模型部分）  
        u_ESO = tau_act - self.B * omega_real - self.D * np.abs(omega_real) * omega_real  
          
        # Step 3: 状态更新（离散化）  
        self.x1_hat = self.x1_hat + (u_ESO / self.J + self.beta1 * e) * Ts  
        self.x2_hat = self.x2_hat + self.beta2 * e * Ts  
          
        # Step 4: ESO估计的扰动力矩  
        tau_ESO = self.J * self.x2_hat  
          
        return tau_ESO  
  
  
# =============================================================================  
# 双环PID控制器  
# =============================================================================  
  
class DualLoopPIDController:  
    """  
    双环PID控制器  
      
    控制结构：  
        1. 外环PI：角度误差 → 参考角速度  
           ω_ref = Kp_outer·e_θ + Ki_outer·∫e_θ dt  
             
        2. 内环PD：角速度误差 → 反馈力矩  
           τ_PID = Kp_inner·e_ω + Kd_inner·ė_ω  
             
        3. 前馈控制：已知模型补偿  
           τ_FF = J·ω̇_ref + B·ω + D|ω|ω  
             
        4. ESO补偿：扰动估计  
           τ_ESO = J·x̂₂  
             
        5. 总输出：  
           τ_cmd = τ_PID + τ_FF - τ_ESO  
    """  
      
    def __init__(self):  
        """初始化双环PID控制器"""  
        # 动力学参数  
        self.J = DynamicsParams.J  
        self.B = DynamicsParams.B  
        self.D = DynamicsParams.D  
          
        # PID增益  
        self.Kp_outer = PIDGains.Kp_outer  
        self.Ki_outer = PIDGains.Ki_outer  
        self.Kp_inner = PIDGains.Kp_inner  
        self.Kd_inner = PIDGains.Kd_inner  
          
        # 限幅  
        self.integral_limit = PIDGains.integral_limit  
        self.omega_ref_limit = PIDGains.omega_ref_limit  
        self.tau_limit = PIDGains.tau_limit  
          
        # ESO观测器  
        self.eso = ESOObserver()  
          
        # 控制器状态  
        self.angle_integral = np.zeros(3)      # 外环积分项  
        self.omega_error_prev = np.zeros(3)    # 内环微分项  
        self.omega_ref_prev = np.zeros(3)      # 前馈用  
        self.tau_cmd_prev = np.zeros(3)        # ESO用  
          
    def reset(self):  
        """重置控制器所有状态"""  
        self.eso.reset()  
        self.angle_integral = np.zeros(3)  
        self.omega_error_prev = np.zeros(3)  
        self.omega_ref_prev = np.zeros(3)  
        self.tau_cmd_prev = np.zeros(3)  
          
    def compute(self,  
                angle_target: np.ndarray,  
                angle_current: np.ndarray,  
                omega_current: np.ndarray,  
                Ts: float) -> Tuple[np.ndarray, Dict]:  
        """  
        计算控制力矩  
          
        Args:  
            angle_target: 目标姿态角 [yaw, pitch, roll] (rad)  
            angle_current: 当前姿态角 [yaw, pitch, roll] (rad)  
            omega_current: 当前角速度 [yaw, pitch, roll] (rad/s)  
            Ts: 采样周期 (s)  
              
        Returns:  
            tau_cmd: 控制力矩 [yaw, pitch, roll]  
            debug_info: 调试信息  
        """  
        # ==================== Step 1: 外环PI ====================  
        # 角度误差（归一化到[-π, π]）  
        angle_error = self._normalize_angle(angle_target - angle_current)  
          
        # 积分（带anti-windup）  
        self.angle_integral += angle_error * Ts  
        self.angle_integral = np.clip(  
            self.angle_integral,  
            -self.integral_limit,  
            self.integral_limit  
        )  
          
        # PI输出 → 参考角速度  
        omega_ref = self.Kp_outer * angle_error + self.Ki_outer * self.angle_integral  
        omega_ref = np.clip(omega_ref, -self.omega_ref_limit, self.omega_ref_limit)  
          
        # ==================== Step 2: 内环PD ====================  
        # 角速度误差  
        omega_error = omega_ref - omega_current  
          
        # 误差微分  
        if Ts > 0:  
            omega_error_dot = (omega_error - self.omega_error_prev) / Ts  
        else:  
            omega_error_dot = np.zeros(3)  
        self.omega_error_prev = omega_error.copy()  
          
        # PD输出  
        tau_PID = self.Kp_inner * omega_error + self.Kd_inner * omega_error_dot  
          
        # ==================== Step 3: 前馈控制 ====================  
        # 参考角加速度  
        if Ts > 0:  
            omega_dot_ref = (omega_ref - self.omega_ref_prev) / Ts  
        else:  
            omega_dot_ref = np.zeros(3)  
        self.omega_ref_prev = omega_ref.copy()  
          
        # 前馈力矩: τ_FF = J·ω̇_ref + B·ω + D|ω|ω  
        tau_FF = (self.J * omega_dot_ref +  
                  self.B * omega_current +  
                  self.D * np.abs(omega_current) * omega_current)  
          
        # ==================== Step 4: ESO扰动估计 ====================  
        tau_ESO = self.eso.update(omega_current, self.tau_cmd_prev, Ts)  
          
        # ==================== Step 5: 综合输出 ====================  
        # τ_cmd = τ_PID + τ_FF - τ_ESO  
        tau_cmd = tau_PID + tau_FF - tau_ESO  
          
        # 力矩限幅  
        tau_cmd = np.clip(tau_cmd, -self.tau_limit, self.tau_limit)  
          
        # 保存用于下次ESO更新  
        self.tau_cmd_prev = tau_cmd.copy()  
          
        # 调试信息  
        debug_info = {  
            'angle_error': angle_error.copy(),  
            'angle_integral': self.angle_integral.copy(),  
            'omega_ref': omega_ref.copy(),  
            'omega_error': omega_error.copy(),  
            'tau_PID': tau_PID.copy(),  
            'tau_FF': tau_FF.copy(),  
            'tau_ESO': tau_ESO.copy(),  
            'tau_cmd': tau_cmd.copy()  
        }  
          
        return tau_cmd, debug_info  
      
    @staticmethod  
    def _normalize_angle(angle: np.ndarray) -> np.ndarray:  
        """将角度归一化到[-π, π]"""  
        return np.arctan2(np.sin(angle), np.cos(angle))  
  
  
# =============================================================================  
# 八推进器推力分配  
# =============================================================================  
  
class ThrustAllocator:  
    """  
    八推进器推力分配  
      
    推进器布局（俯视图）：  
      
           前(x+)  
       T1 ↗    ↖ T4      水平推进器(45度)  
         T2  T6          垂直推进器  
           [ROV]  
         T3  T7          垂直推进器    
       T5 ↖    ↗ T8      水平推进器(45度)  
           后(x-)  
      
    分组功能：  
        T1, T4, T5, T8: 水平推进器 → Fx, Fy, Mz  
        T2, T3, T6, T7: 垂直推进器 → Fz, Mx, My  
      
    矩阵关系：  
        τ = B·u    (6×8) × (8×1) = (6×1)  
        u = A·τ    (8×6) × (6×1) = (8×1)  
    """  
      
    def __init__(self):  
        """初始化推力分配器"""  
        self._build_matrices()  
          
        # 推力限制（基于60W功率估算）  
        self.thrust_max = 40.0   # N  
        self.thrust_min = -35.0  # N  
      
    def _build_matrices(self):  
        """构建分配矩阵"""  
          
        sqrt2_2 = np.sqrt(2) / 2  # ≈ 0.707  
          
        # ==================== B矩阵 (6×8) ====================  
        # τ = B·u  
        # τ = [Fx, Fy, Fz, Mx, My, Mz]^T  
        # u = [T1, T2, T3, T4, T5, T6, T7, T8]^T  
          
        self.B = np.array([  
            #  T1       T2       T3       T4       T5       T6       T7       T8  
            [sqrt2_2,   0,       0,       sqrt2_2, sqrt2_2,  0,       0,      sqrt2_2],   # Fx  
            [sqrt2_2,   0,       0,      -sqrt2_2, sqrt2_2,  0,       0,     -sqrt2_2],   # Fy  
            [0,         1,       1,       0,       0,        1,       1,      0      ],   # Fz  
            [0,        -0.138,  -0.138,   0,       0,        0.138,   0.138,  0      ],   # Mx (roll)  
            [0,        -0.11,    0.11,    0,       0,        0.11,   -0.11,   0      ],   # My (pitch)  
            [0.253,     0,       0,       0.253,  -0.253,    0,       0,     -0.253  ]    # Mz (yaw)  
        ])  
          
        # ==================== A矩阵 (8×6) ====================  
        # u = A·τ  
        # 使用正则化伪逆: A = (B^T·B + λI)^(-1)·B^T  
        self.A = np.linalg.pinv(self.B)    
        # self.A = np.array([  
        #     #  Fx       Fy       Fz       Mx       My       Mz  
        #     [ 1.4149,  1.4149,  0,       0,       0,       0.0650],   # T1  
        #     [ 0,       0,      -4.0010,  0.0107,  0.0054,  0      ],   # T2  
        #     [ 0,       0,       4.0010, -0.0107,  0.0054,  0      ],   # T3  
        #     [ 1.4149, -1.4149,  0,       0,       0,       0.0650],   # T4  
        #     [-1.4149, -1.4149,  0,       0,       0,       0.0650],   # T5  
        #     [ 0,       0,      -4.0010, -0.0107, -0.0054,  0      ],   # T6  
        #     [ 0,       0,       4.0010,  0.0107, -0.0054,  0      ],   # T7  
        #     [-1.4149,  1.4149,  0,       0,       0,       0.0650]    # T8  
        # ])  
          
    def allocate(self, tau_cmd: np.ndarray) -> np.ndarray:  
        """  
        推力分配  
          
        Args:  
            tau_cmd: 期望力和力矩 [Fx, Fy, Fz, Mx, My, Mz]  
              
        Returns:  
            thrust: 八个推进器推力 [T1, T2, T3, T4, T5, T6, T7, T8]  
        """  
        # 推力分配: u = A·τ  
        thrust = self.A @ tau_cmd  
          
        # 推力限幅  
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)  
          
        return thrust  
      
    def allocate_with_attitude(self,   
                                force_cmd: np.ndarray,  
                                tau_attitude: np.ndarray) -> np.ndarray:  
        """  
        分配力命令和姿态力矩  
          
        Args:  
            force_cmd: 力命令 [Fx, Fy, Fz]  
            tau_attitude: 姿态力矩 [Mz, My, Mx] (yaw, pitch, roll顺序)  
              
        Returns:  
            thrust: 八推进器推力 [T1, ..., T8]  
        """  
        # 组合成完整向量 [Fx, Fy, Fz, Mx, My, Mz]  
        # 注意顺序转换: tau_attitude是[yaw, pitch, roll] = [Mz, My, Mx]  
        tau_cmd = np.array([  
            force_cmd[0],       # Fx  
            force_cmd[1],       # Fy  
            force_cmd[2],       # Fz  
            tau_attitude[2],    # Mx (roll)  
            tau_attitude[1],    # My (pitch)  
            tau_attitude[0]     # Mz (yaw)  
        ])  
          
        return self.allocate(tau_cmd)  
  
  
# =============================================================================  
# 集成控制器  
# =============================================================================  
  
class IntegratedController:  
    """  
    集成控制器  
      
    功能：  
        1. 将离散动作转换为期望力/姿态变化  
        2. 使用双环PID计算姿态控制力矩  
        3. 推力分配到8个推进器  
      
    动作定义 (13个离散动作):  
        0: hover         - 悬停  
        1: surge_forward - 前进  
        2: surge_backward- 后退  
        3: sway_left     - 左平移  
        4: sway_right    - 右平移  
        5: heave_up      - 上浮  
        6: heave_down    - 下潜  
        7: yaw_left      - 左偏航  
        8: yaw_right     - 右偏航  
        9: pitch_up      - 抬头  
        10: pitch_down   - 低头  
        11: roll_left    - 左横滚  
        12: roll_right   - 右横滚  
    """  
      
    # 动作名称  
    ACTION_NAMES = [  
        "hover", "surge_forward", "surge_backward",  
        "sway_left", "sway_right", "heave_up", "heave_down",  
        "yaw_left", "yaw_right", "pitch_up", "pitch_down",  
        "roll_left", "roll_right"  
    ]  
      
    def __init__(self, config: Optional[Dict] = None):  
        """初始化集成控制器"""  
        # 保存配置  
        self.config = config or {}  
  
        # 双环PID控制器  
        self.pid_controller = DualLoopPIDController()  
          
        # 推力分配器  
        self.thrust_allocator = ThrustAllocator()  
  
        # 读取动作空间与速度控制配置  
        action_cfg = self.config.get('action_space', {})  
        self.translation_mode = action_cfg.get('translation_mode', 'distance')  
  
        distance_cfg = action_cfg.get('action_distance', {})  
        self.action_distance = {  
            'surge': float(distance_cfg.get('surge', 0.05)),  
            'sway': float(distance_cfg.get('sway', 0.05)),  
            'heave': float(distance_cfg.get('heave', 0.04))  
        }  
  
        vel_cfg = action_cfg.get('velocity_controller', {})  
        self.vel_kp = np.array(vel_cfg.get('kp', [20.0, 20.0, 25.0]), dtype=float)  
        self.vel_kd = np.array(vel_cfg.get('kd', [6.0, 6.0, 8.0]), dtype=float)  
        self.max_force = np.array(vel_cfg.get('max_force', [30.0, 30.0, 30.0]), dtype=float)  
  
        # 构建动作命令映射  
        self._build_action_commands()  
        
        # 记忆目标yaw（避免yaw持续旋转的关键修复）
        self.target_yaw_memory = None  # 记住上次的目标yaw，用于锁定  
          
    def _build_action_commands(self):  
        """  
        构建离散动作到期望力/姿态变化的映射  
          
        格式: [Fx_or_dx, Fy_or_dy, Fz_or_dz, delta_yaw, delta_pitch, delta_roll]  
        """  
        action_cfg = self.config.get('action_space', {})  
        dual_cfg = self.config.get('dual_loop_pid', {})  
  
        # 姿态变化幅值（弧度）  
        angle_cfg = action_cfg.get('action_angle_delta', dual_cfg.get('action_angle_delta', {}))  
        delta_yaw = float(angle_cfg.get('yaw', 0.1))  
        delta_pitch = float(angle_cfg.get('pitch', 0.1))  
        delta_roll = float(angle_cfg.get('roll', 0.1))  
  
        # translation_mode = distance 时，线性项为位移（m）  
        if self.translation_mode == 'distance':  
            dx = self.action_distance['surge']  
            dy = self.action_distance['sway']  
            dz = self.action_distance['heave']  
            linear_cmds = {  
                'surge_forward':  [dx, 0, 0],  
                'surge_backward': [-dx, 0, 0],  
                'sway_left':      [0, dy, 0],  
                'sway_right':     [0, -dy, 0],  
                'heave_up':       [0, 0, dz],  
                'heave_down':     [0, 0, -dz]  
            }  
        else:  
            # translation_mode = force 时，线性项为期望力 (N)  
            force_cfg = dual_cfg.get('action_force', {})  
            F_surge = float(force_cfg.get('surge', 20.0))  
            F_sway = float(force_cfg.get('sway', 15.0))  
            F_heave = float(force_cfg.get('heave', 20.0))  
            linear_cmds = {  
                'surge_forward':  [F_surge, 0, 0],  
                'surge_backward': [-F_surge, 0, 0],  
                'sway_left':      [0, F_sway, 0],  
                'sway_right':     [0, -F_sway, 0],  
                'heave_up':       [0, 0, F_heave],  
                'heave_down':     [0, 0, -F_heave]  
            }  
  
        # action_id -> [Fx_or_dx, Fy_or_dy, Fz_or_dz, delta_yaw, delta_pitch, delta_roll]  
        self.action_commands = {  
            0:  [0, 0, 0, 0, 0, 0],                                   # hover  
            1:  [*linear_cmds['surge_forward'], 0, 0, 0],             # surge_forward  
            2:  [*linear_cmds['surge_backward'], 0, 0, 0],            # surge_backward  
            3:  [*linear_cmds['sway_left'], 0, 0, 0],                 # sway_left  
            4:  [*linear_cmds['sway_right'], 0, 0, 0],                # sway_right  
            5:  [*linear_cmds['heave_up'], 0, 0, 0],                  # heave_up  
            6:  [*linear_cmds['heave_down'], 0, 0, 0],                # heave_down  
            7:  [0, 0, 0, delta_yaw, 0, 0],                           # yaw_left  
            8:  [0, 0, 0, -delta_yaw, 0, 0],                          # yaw_right  
            9:  [0, 0, 0, 0, delta_pitch, 0],                         # pitch_up  
            10: [0, 0, 0, 0, -delta_pitch, 0],                        # pitch_down  
            11: [0, 0, 0, 0, 0, delta_roll],                          # roll_left  
            12: [0, 0, 0, 0, 0, -delta_roll],                         # roll_right  
        }  
          
    def reset(self):  
        """重置控制器状态"""  
        self.pid_controller.reset()
        self.target_yaw_memory = None  # 重置yaw记忆  
          
    def compute_thrust(self,  
                       action: int,  
                       eta: np.ndarray,  
                       nu: np.ndarray,  
                       Ts: float,  
                       target_orientation: np.ndarray = None,  
                       target_position: np.ndarray = None) -> Tuple[np.ndarray, Dict]:  
        """  
        计算推力命令  
          
        Args:  
            action: 离散动作ID (0-12)  
            eta: 当前位姿 [x, y, z, roll, pitch, yaw]  
            nu: 当前速度 [u, v, w, p, q, r]  
            Ts: 采样周期 (s)  
            target_orientation: 基准目标姿态 [roll, pitch, yaw]，None则保持水平  
              
        Returns:  
            thrust: 八推进器推力 [T1, T2, T3, T4, T5, T6, T7, T8]  
            debug_info: 调试信息  
        """  
        # 获取动作命令  
        cmd = np.array(self.action_commands.get(action, [0] * 6), dtype=float)  
        linear_cmd = cmd[:3]         # [Fx_or_dx, Fy_or_dy, Fz_or_dz]  
        angle_delta = cmd[3:6]       # [delta_yaw, delta_pitch, delta_roll]  
  
        # 线速度控制（distance -> desired velocity）  
        current_vel_body = nu[:3]  
        if self.translation_mode == 'distance':  
            desired_vel_body = linear_cmd / max(Ts, 1e-6)  
            # 目标附近减速，避免冲过目标后绕圈  
            slowdown_scale = 1.0  
            action_cfg = self.config.get('action_space', {})  
            slowdown_cfg = action_cfg.get('goal_slowdown', {})  
            if target_position is not None and slowdown_cfg.get('enabled', True):  
                radius = float(slowdown_cfg.get('radius', 1.0))  
                min_scale = float(slowdown_cfg.get('min_scale', 0.2))  
                pos_error = np.linalg.norm(target_position - eta[:3])  
                if radius > 1e-6:  
                    slowdown_scale = max(min_scale, min(1.0, pos_error / radius))  
                    desired_vel_body = desired_vel_body * slowdown_scale  
            vel_error = desired_vel_body - current_vel_body  
            force_cmd = self.vel_kp * vel_error - self.vel_kd * current_vel_body  
            force_cmd = np.clip(force_cmd, -self.max_force, self.max_force)  
        else:  
            desired_vel_body = np.zeros(3)  
            vel_error = np.zeros(3)  
            force_cmd = linear_cmd  
          
        # 当前姿态（从eta提取，注意顺序）  
        roll_current = eta[3]  
        pitch_current = eta[4]  
        yaw_current = eta[5]  
          
        # 当前角速度（从nu提取）  
        p = nu[3]  # roll rate  
        q = nu[4]  # pitch rate  
        r = nu[5]  # yaw rate  
          
        # 确定目标姿态（修复：避免yaw持续旋转的关键修复）
        if target_orientation is None:  
            # 根据动作类型确定目标姿态
            # 关键修复：对于位置控制动作，锁定yaw到初始值或记忆值，避免持续旋转
            
            if action in [7, 8]:  # yaw_left, yaw_right - 允许yaw变化
                # 执行yaw动作时，更新目标yaw（累加）
                target_yaw = yaw_current + angle_delta[0]
                # 归一化到[-π, π]范围
                target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))
                # 记住新的目标yaw
                self.target_yaw_memory = target_yaw
                target_pitch = 0.0  # 保持水平
                target_roll = 0.0
            elif action in [9, 10]:  # pitch_up, pitch_down - 允许pitch变化（但限制范围）
                # 限制pitch在合理范围内（约±30度）
                max_pitch = np.pi / 6.0  # 30度
                target_pitch = np.clip(angle_delta[1], -max_pitch, max_pitch)
                target_roll = 0.0
                # 使用记忆的yaw目标，如果没有则使用当前yaw
                if self.target_yaw_memory is not None:
                    target_yaw = self.target_yaw_memory
                else:
                    target_yaw = yaw_current
                    self.target_yaw_memory = yaw_current
            elif action in [11, 12]:  # roll_left, roll_right - 允许roll变化（但限制范围）
                # 限制roll在合理范围内（约±30度）
                max_roll = np.pi / 6.0  # 30度
                target_roll = np.clip(angle_delta[2], -max_roll, max_roll)
                target_pitch = 0.0
                # 使用记忆的yaw目标，如果没有则使用当前yaw
                if self.target_yaw_memory is not None:
                    target_yaw = self.target_yaw_memory
                else:
                    target_yaw = yaw_current
                    self.target_yaw_memory = yaw_current
            else:  # 位置控制动作（hover, surge, sway, heave）
                # 保持水平姿态
                target_roll = 0.0
                target_pitch = 0.0
                # 关键修复：使用记忆的yaw目标，如果没有则初始化为当前yaw并锁定
                if self.target_yaw_memory is not None:
                    target_yaw = self.target_yaw_memory  # 锁定到上次的目标yaw
                else:
                    # 第一次执行位置动作时，锁定当前yaw
                    target_yaw = yaw_current
                    self.target_yaw_memory = yaw_current
        else:  
            # 使用外部指定的目标姿态，叠加动作增量
            target_roll = target_orientation[0] + angle_delta[2]  
            target_pitch = target_orientation[1] + angle_delta[1]  
            target_yaw = target_orientation[2] + angle_delta[0]  
          
        # 转换为PID控制器的输入格式: [yaw, pitch, roll]  
        angle_target = np.array([target_yaw, target_pitch, target_roll])  
        angle_current = np.array([yaw_current, pitch_current, roll_current])  
        omega_current = np.array([r, q, p])  # [yaw_rate, pitch_rate, roll_rate]  
          
        # 双环PID计算姿态力矩  
        tau_attitude, pid_debug = self.pid_controller.compute(  
            angle_target, angle_current, omega_current, Ts  
        )  
        # tau_attitude = [Mz, My, Mx] 对应 [yaw, pitch, roll]  
          
        # 推力分配  
        thrust = self.thrust_allocator.allocate_with_attitude(force_cmd, tau_attitude)  
          
        # 汇总调试信息  
        debug_info = {  
            'action': action,  
            'action_name': self.ACTION_NAMES[action] if action < len(self.ACTION_NAMES) else 'unknown',  
            'force_cmd': force_cmd.copy(),  
            'translation_mode': self.translation_mode,  
            'desired_vel_body': desired_vel_body.copy(),  
            'vel_error': vel_error.copy(),  
            'slowdown_scale': float(slowdown_scale) if self.translation_mode == 'distance' else 1.0,  
            'angle_delta': angle_delta.copy(),  
            'angle_target': angle_target.copy(),  
            'angle_current': angle_current.copy(),  
            'omega_current': omega_current.copy(),  
            'tau_attitude': tau_attitude.copy(),  
            'thrust': thrust.copy(),  
            **pid_debug  
        }  
          
        return thrust, debug_info  
  
  
# =============================================================================  
# 便捷函数  
# =============================================================================  
  
def thrust_to_pwm(thrust: np.ndarray,  
                  thrust_max: float = 40.0,  
                  pwm_neutral: int = 1500,  
                  pwm_range: int = 400) -> np.ndarray:  
    """  
    将推力转换为PWM信号（用于实际硬件）  
      
    Args:  
        thrust: 推力数组 [T1, ..., T8]  
        thrust_max: 最大推力 (N)  
        pwm_neutral: 中位PWM值  
        pwm_range: PWM范围  
          
    Returns:  
        pwm: PWM信号数组  
    """  
    thrust_normalized = np.clip(thrust / thrust_max, -1.0, 1.0)  
    pwm = pwm_neutral + thrust_normalized * pwm_range  
    return pwm.astype(int)  
  
  
# =============================================================================  
# 测试代码  
# =============================================================================  
  
if __name__ == "__main__":  
    print("=" * 70)  
    print("双环PID控制器测试")  
    print("=" * 70)  
      
    # 打印动力学参数  
    print("\n【动力学参数】")  
    print(f"  J (转动惯量)  : yaw={DynamicsParams.J[0]}, pitch={DynamicsParams.J[1]}, roll={DynamicsParams.J[2]}")  
    print(f"  B (线性阻尼)  : yaw={DynamicsParams.B[0]}, pitch={DynamicsParams.B[1]}, roll={DynamicsParams.B[2]}")  
    print(f"  D (二次阻尼)  : yaw={DynamicsParams.D[0]}, pitch={DynamicsParams.D[1]}, roll={DynamicsParams.D[2]}")  
      
    # 创建控制器  
    controller = IntegratedController()  
      
    # 模拟状态  
    eta = np.array([0, 0, 5, 0.1, 0.05, 0.2])  # x,y,z,roll,pitch,yaw  
    nu = np.array([0.5, 0.1, 0, 0.02, 0.01, 0.05])  # u,v,w,p,q,r  
    Ts = 0.02  # 20ms  
      
    # 测试各个动作  
    print("\n【动作 → 推力映射】")  
    print("-" * 70)  
    print(f"{'ID':<3} {'动作名':<16} {'推力 [T1, T2, T3, T4, T5, T6, T7, T8]'}")  
    print("-" * 70)  
      
    for action_id, action_name in enumerate(IntegratedController.ACTION_NAMES):  
        controller.reset()  
        thrust, debug = controller.compute_thrust(action_id, eta, nu, Ts, target_position=None)  
        thrust_str = ', '.join([f'{t:6.2f}' for t in thrust])  
        print(f"{action_id:<3} {action_name:<16} [{thrust_str}]")  
      
    # 控制循环模拟  
    print("\n" + "=" * 70)  
    print("控制循环模拟 - 悬停稳定测试")  
    print("=" * 70)  
      
    controller.reset()  
      
    # 初始状态：有姿态偏差  
    eta = np.array([0, 0, 5, 0.15, 0.1, 0.2])  # roll=8.6°, pitch=5.7°, yaw=11.5°  
    nu = np.array([0, 0, 0, 0, 0, 0])  
      
    print(f"\n初始姿态 (deg): roll={np.rad2deg(eta[3]):.1f}, pitch={np.rad2deg(eta[4]):.1f}, yaw={np.rad2deg(eta[5]):.1f}")  
    print("目标: 保持水平 (roll=0, pitch=0)")  
    print("-" * 70)  
      
    for step in range(5):  
        thrust, debug = controller.compute_thrust(  
            action=0,  # hover  
            eta=eta,  
            nu=nu,  
            Ts=Ts,  
            target_position=None  
        )  
          
        print(f"\nStep {step}:")  
        print(f"  角度误差 (deg): yaw={np.rad2deg(debug['angle_error'][0]):+6.2f}, "  
              f"pitch={np.rad2deg(debug['angle_error'][1]):+6.2f}, "  
              f"roll={np.rad2deg(debug['angle_error'][2]):+6.2f}")  
        print(f"  τ_PID: [{debug['tau_PID'][0]:+.4f}, {debug['tau_PID'][1]:+.4f}, {debug['tau_PID'][2]:+.4f}]")  
        print(f"  τ_FF:  [{debug['tau_FF'][0]:+.4f}, {debug['tau_FF'][1]:+.4f}, {debug['tau_FF'][2]:+.4f}]")  
        print(f"  τ_ESO: [{debug['tau_ESO'][0]:+.4f}, {debug['tau_ESO'][1]:+.4f}, {debug['tau_ESO'][2]:+.4f}]")  
        print(f"  τ_cmd: [{debug['tau_cmd'][0]:+.4f}, {debug['tau_cmd'][1]:+.4f}, {debug['tau_cmd'][2]:+.4f}]")  
        print(f"  推力:  [{', '.join([f'{t:+6.2f}' for t in thrust])}]")  
          
        # 简单模拟状态更新  
        eta[3:6] *= 0.85  # 姿态逐渐恢复  
        nu[3:6] = debug['tau_cmd'][::-1] / DynamicsParams.J[::-1] * 0.1  
      
    print("\n" + "=" * 70)  
    print("测试完成")  
    print("=" * 70)  
