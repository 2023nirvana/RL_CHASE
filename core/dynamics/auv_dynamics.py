"""  
AUV 6DOF 非线性动力学模型  
==========================  
  
基于Fossen的水下机器人动力学理论，实现完整的6自由度非线性模型。  
  
动力学方程:  
    M * ν̇ + C(ν) * ν + D(ν) * ν + g(η) = τ  
  
其中:  
    - η = [x, y, z, φ, θ, ψ]ᵀ : 位置和姿态（惯性坐标系/NED）  
    - ν = [u, v, w, p, q, r]ᵀ : 线速度和角速度（体坐标系）  
    - M : 惯性矩阵（含附加质量）  
    - C(ν) : 科里奥利和向心力矩阵  
    - D(ν) : 阻尼矩阵（含非线性二次阻尼）  
    - g(η) : 重力和浮力恢复力  
    - τ : 推进器力/力矩  
  
作者: [您的名字]  
日期: 2024  
"""  
  
import numpy as np  
from typing import Dict, Tuple, Optional  
import yaml  
  
  
class AUVDynamics:  
    """  
    AUV 6自由度非线性动力学模型  
      
    特点:  
    1. 完整的非线性建模，不做简化  
    2. 支持参数不确定性（域随机化）  
    3. 包含推进器动态响应  
    4. 支持外部扰动（水流）  
    """  
      
    def __init__(self, config_path: str = "configs/robot_config.yaml",   
                 randomize: bool = False,  
                 random_seed: Optional[int] = None):  
        """  
        初始化AUV动力学模型  
          
        Args:  
            config_path: 配置文件路径  
            randomize: 是否启用参数随机化（域随机化）  
            random_seed: 随机种子  
        """  
        # 加载配置  
        with open(config_path, 'r', encoding='utf-8') as f:  
            self.config = yaml.safe_load(f)  
          
        self.randomize = randomize  
        if random_seed is not None:  
            np.random.seed(random_seed)  
          
        # 调试/记录变量  
        self.debug_info = {}  
          
        # 初始化参数  
        self._init_parameters()  
          
    def _init_parameters(self):  
        """初始化所有动力学参数"""  
        robot_cfg = self.config['robot']  
        hydro_cfg = self.config['hydrodynamics']  
        thruster_cfg = self.config['thruster']  
          
        # ==================== 刚体参数 ====================  
        self.mass = robot_cfg['mass']  
        self.volume = robot_cfg['volume']  
        self.rho = robot_cfg['water_density']  
          
        # 重心和浮心位置  
        self.r_g = np.array(robot_cfg['center_of_gravity'])  
        self.r_b = np.array(robot_cfg['center_of_buoyancy'])  
          
        # 惯性矩阵  
        inertia = robot_cfg['inertia']  
        self.I_b = np.array([  
            [inertia['Ixx'], -inertia['Ixy'], -inertia['Ixz']],  
            [-inertia['Ixy'], inertia['Iyy'], -inertia['Iyz']],  
            [-inertia['Ixz'], -inertia['Iyz'], inertia['Izz']]  
        ])  
          
        # 重力和浮力  
        self.W = self.mass * 9.81  # 重力  
        self.B = self.rho * self.volume * 9.81  # 浮力  
          
        # ==================== 水动力学参数 ====================  
        # 附加质量  
        added_mass = hydro_cfg['added_mass']  
        self.M_A = -np.diag([  
            added_mass['X_udot'],  
            added_mass['Y_vdot'],  
            added_mass['Z_wdot'],  
            added_mass['K_pdot'],  
            added_mass['M_qdot'],  
            added_mass['N_rdot']  
        ])  
          
        # 线性阻尼  
        lin_damp = hydro_cfg['linear_damping']  
        self.D_l = -np.diag([  
            lin_damp['X_u'],  
            lin_damp['Y_v'],  
            lin_damp['Z_w'],  
            lin_damp['K_p'],  
            lin_damp['M_q'],  
            lin_damp['N_r']  
        ])  
          
        # 二次阻尼（关键非线性项！）  
        quad_damp = hydro_cfg['quadratic_damping']  
        self.D_q_coeffs = -np.array([  
            quad_damp['X_uu'],  
            quad_damp['Y_vv'],  
            quad_damp['Z_ww'],  
            quad_damp['K_pp'],  
            quad_damp['M_qq'],  
            quad_damp['N_rr']  
        ])  
          
        # ==================== 域随机化 ====================  
        if self.randomize:  
            self._apply_domain_randomization(hydro_cfg['uncertainty'])  
          
        # ==================== 构建系统矩阵 ====================  
        self._build_system_matrices()  
          
        # ==================== 推进器参数 ====================  
        self.num_thrusters = thruster_cfg['num_thrusters']  
        self.thruster_positions = np.array(thruster_cfg['positions'])  
        self.thruster_directions = np.array(thruster_cfg['directions'])  
        self.max_thrust = thruster_cfg['max_thrust']  
        self.min_thrust = thruster_cfg['min_thrust']  
        self.thruster_time_constant = thruster_cfg['time_constant']  
        self.thruster_deadband = thruster_cfg['deadband']  
          
        # 构建推进器配置矩阵  
        self._build_thruster_configuration_matrix()  
          
    def _apply_domain_randomization(self, uncertainty_cfg: Dict):  
        """  
        应用域随机化，增加模型鲁棒性  
          
        这是应对非线性项不确定性的关键策略！  
        通过在训练时随机采样参数，让网络学会适应参数变化。  
        """  
        # 附加质量随机化  
        am_range = uncertainty_cfg['added_mass_range']  
        am_scale = np.random.uniform(am_range[0], am_range[1], 6)  
        self.M_A = self.M_A * np.diag(am_scale)  
          
        # 线性阻尼随机化  
        ld_range = uncertainty_cfg['linear_damping_range']  
        ld_scale = np.random.uniform(ld_range[0], ld_range[1], 6)  
        self.D_l = self.D_l * np.diag(ld_scale)  
          
        # 二次阻尼随机化（非线性项的不确定性通常更大）  
        qd_range = uncertainty_cfg['quadratic_damping_range']  
        qd_scale = np.random.uniform(qd_range[0], qd_range[1], 6)  
        self.D_q_coeffs = self.D_q_coeffs * qd_scale  
          
        # 记录随机化参数（用于调试）  
        self.debug_info['domain_randomization'] = {  
            'added_mass_scale': am_scale,  
            'linear_damping_scale': ld_scale,  
            'quadratic_damping_scale': qd_scale  
        }  
          
    def _build_system_matrices(self):  
        """构建系统惯性矩阵"""  
        # 刚体惯性矩阵 M_RB  
        # 6x6矩阵，左上3x3是质量，右下3x3是惯性矩  
        self.M_RB = np.zeros((6, 6))  
        self.M_RB[0:3, 0:3] = self.mass * np.eye(3)  
        self.M_RB[0:3, 3:6] = -self.mass * self._skew(self.r_g)  
        self.M_RB[3:6, 0:3] = self.mass * self._skew(self.r_g)  
        self.M_RB[3:6, 3:6] = self.I_b  
          
        # 总惯性矩阵 M = M_RB + M_A  
        self.M = self.M_RB + self.M_A  
          
        # 预计算M的逆（用于加速计算）  
        self.M_inv = np.linalg.inv(self.M)  
          
    def _build_thruster_configuration_matrix(self):  
        """  
        构建推进器配置矩阵 T  
          
        τ = T * f，其中f是推进器力向量  
        """  
        sqrt2_2 = np.sqrt(2) / 2  
      
    # 直接使用图片4的B矩阵  
        self.T = np.array([  
            #  T1       T2       T3       T4       T5       T6       T7       T8  
            [sqrt2_2,   0,       0,       sqrt2_2, sqrt2_2,  0,       0,      sqrt2_2],   # Fx  
            [sqrt2_2,   0,       0,      -sqrt2_2, sqrt2_2,  0,       0,     -sqrt2_2],   # Fy  
            [0,         1,       1,       0,       0,        1,       1,      0      ],   # Fz  
            [0,        -0.138,  -0.138,   0,       0,        0.138,   0.138,  0      ],   # Mx (roll)  
            [0,        -0.11,    0.11,    0,       0,        0.11,   -0.11,   0      ],   # My (pitch)  
            [0.253,     0,       0,       0.253,  -0.253,    0,       0,     -0.253  ]    # Mz (yaw)  
        ])  
          
        # for i in range(self.num_thrusters):  
        #     pos = self.thruster_positions[i]  
        #     dir = self.thruster_directions[i]  
              
        #     # 力分量  
        #     self.T[0:3, i] = dir  
              
        #     # 力矩分量 (r × F)  
        #     self.T[3:6, i] = np.cross(pos, dir)  
              
    @staticmethod  
    def _skew(v: np.ndarray) -> np.ndarray:  
        """  
        计算向量的反对称矩阵（叉乘矩阵）  
          
        skew(v) * u = v × u  
        """  
        return np.array([  
            [0, -v[2], v[1]],  
            [v[2], 0, -v[0]],  
            [-v[1], v[0], 0]  
        ])  
      
    def _rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:  
        """  
        计算从体坐标系到惯性坐标系的旋转矩阵 R  
          
        使用ZYX欧拉角顺序（偏航-俯仰-横滚）  
          
        Args:  
            phi: 横滚角 (roll)  
            theta: 俯仰角 (pitch)    
            psi: 偏航角 (yaw)  
        """  
        cphi, sphi = np.cos(phi), np.sin(phi)  
        cth, sth = np.cos(theta), np.sin(theta)  
        cpsi, spsi = np.cos(psi), np.sin(psi)  
          
        R = np.array([  
            [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],  
            [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],  
            [-sth, cth*sphi, cth*cphi]  
        ])  
          
        return R  
      
    def _transformation_matrix(self, phi: float, theta: float) -> np.ndarray:  
        """  
        计算角速度变换矩阵 T_Θ  
          
        Θ̇ = T_Θ * ω  
          
        其中 Θ = [φ, θ, ψ]ᵀ, ω = [p, q, r]ᵀ  
          
        注意：当theta接近±90°时会出现奇异性（万向锁）  
        """  
        cphi, sphi = np.cos(phi), np.sin(phi)  
        cth, sth = np.cos(theta), np.sin(theta)  
          
        # 避免除零  
        if abs(cth) < 1e-10:  
            cth = 1e-10 * np.sign(cth) if cth != 0 else 1e-10  
              
        tth = sth / cth  
          
        T_theta = np.array([  
            [1, sphi*tth, cphi*tth],  
            [0, cphi, -sphi],  
            [0, sphi/cth, cphi/cth]  
        ])  
          
        return T_theta  
      
    def _jacobian(self, eta: np.ndarray) -> np.ndarray:  
        """  
        计算完整的雅可比矩阵 J(η)  
          
        η̇ = J(η) * ν  
          
        J = | R    0   |  
            | 0    T_Θ |  
        """  
        phi, theta, psi = eta[3], eta[4], eta[5]  
          
        R = self._rotation_matrix(phi, theta, psi)  
        T_theta = self._transformation_matrix(phi, theta)  
          
        J = np.zeros((6, 6))  
        J[0:3, 0:3] = R  
        J[3:6, 3:6] = T_theta  
          
        return J  
      
    def _coriolis_matrix(self, nu: np.ndarray) -> np.ndarray:  
        """  
        计算科里奥利和向心力矩阵 C(ν)  
          
        C(ν) = C_RB(ν) + C_A(ν)  
          
        这是非线性项！矩阵元素依赖于速度。  
        """  
        u, v, w, p, q, r = nu  
          
        # 刚体科里奥利矩阵 C_RB  
        C_RB = np.zeros((6, 6))  
          
        # 使用Fossen的参数化形式  
        m = self.mass  
        xg, yg, zg = self.r_g  
        Ixx, Iyy, Izz = self.I_b[0, 0], self.I_b[1, 1], self.I_b[2, 2]  
        Ixy, Ixz, Iyz = -self.I_b[0, 1], -self.I_b[0, 2], -self.I_b[1, 2]  
          
        # C_RB矩阵 (基于Fossen 2011)  
        C_RB[0, 3] = m * (yg*q + zg*r)  
        C_RB[0, 4] = -m * (xg*q - w)  
        C_RB[0, 5] = -m * (xg*r + v)  
          
        C_RB[1, 3] = -m * (yg*p + w)  
        C_RB[1, 4] = m * (zg*r + xg*p)  
        C_RB[1, 5] = -m * (yg*r - u)  
          
        C_RB[2, 3] = -m * (zg*p - v)  
        C_RB[2, 4] = -m * (zg*q + u)  
        C_RB[2, 5] = m * (xg*p + yg*q)  
          
        C_RB[3, 0] = -C_RB[0, 3]  
        C_RB[3, 1] = -C_RB[1, 3]  
        C_RB[3, 2] = -C_RB[2, 3]  
        C_RB[3, 4] = Izz*r - Iyz*q - Ixz*p  
        C_RB[3, 5] = -Iyy*q + Iyz*r + Ixy*p  
          
        C_RB[4, 0] = -C_RB[0, 4]  
        C_RB[4, 1] = -C_RB[1, 4]  
        C_RB[4, 2] = -C_RB[2, 4]  
        C_RB[4, 3] = -Izz*r + Iyz*q + Ixz*p  
        C_RB[4, 5] = Ixx*p - Ixz*r - Ixy*q  
          
        C_RB[5, 0] = -C_RB[0, 5]  
        C_RB[5, 1] = -C_RB[1, 5]  
        C_RB[5, 2] = -C_RB[2, 5]  
        C_RB[5, 3] = Iyy*q - Iyz*r - Ixy*p  
        C_RB[5, 4] = -Ixx*p + Ixz*r + Ixy*q  
          
        # 附加质量科里奥利矩阵 C_A (简化的对角形式)  
        C_A = np.zeros((6, 6))  
          
        # 使用附加质量对角元素  
        Xu, Yv, Zw = self.M_A[0, 0], self.M_A[1, 1], self.M_A[2, 2]  
        Kp, Mq, Nr = self.M_A[3, 3], self.M_A[4, 4], self.M_A[5, 5]  
          
        C_A[0, 4] = Zw * w  
        C_A[0, 5] = -Yv * v  
        C_A[1, 3] = -Zw * w  
        C_A[1, 5] = Xu * u  
        C_A[2, 3] = Yv * v  
        C_A[2, 4] = -Xu * u  
          
        C_A[3, 1] = Zw * w  
        C_A[3, 2] = -Yv * v  
        C_A[3, 4] = Nr * r  
        C_A[3, 5] = -Mq * q  
          
        C_A[4, 0] = -Zw * w  
        C_A[4, 2] = Xu * u  
        C_A[4, 3] = -Nr * r  
        C_A[4, 5] = Kp * p  
          
        C_A[5, 0] = Yv * v  
        C_A[5, 1] = -Xu * u  
        C_A[5, 3] = Mq * q  
        C_A[5, 4] = -Kp * p  
          
        return C_RB + C_A  
      
    def _damping_force(self, nu: np.ndarray) -> np.ndarray:  
        """  
        计算阻尼力/力矩  
          
        D(ν) * ν = D_l * ν + D_q(ν) * ν  
          
        其中D_q(ν)是二次阻尼，是关键的非线性项！  
          
        二次阻尼公式: F_d = D_q * |v| * v  
        """  
        # 线性阻尼  
        F_linear = self.D_l @ nu  
          
        # 二次阻尼（非线性！）  
        # 对于每个自由度: F_q_i = D_q_i * |nu_i| * nu_i  
        F_quadratic = self.D_q_coeffs * np.abs(nu) * nu  
          
        # 总阻尼力  
        F_damping = F_linear + F_quadratic  
          
        # 记录调试信息  
        self.debug_info['damping'] = {  
            'linear': F_linear.copy(),  
            'quadratic': F_quadratic.copy(),  
            'total': F_damping.copy()  
        }  
          
        return F_damping  
      
    def _restoring_force(self, eta: np.ndarray) -> np.ndarray:  
        """  
        计算重力和浮力恢复力  
          
        g(η) 取决于姿态角  
        """  
        phi, theta = eta[3], eta[4]  
          
        cphi, sphi = np.cos(phi), np.sin(phi)  
        cth, sth = np.cos(theta), np.sin(theta)  
          
        # 重力和浮力在体坐标系下的分量  
        g = np.zeros(6)  
          
        # 力分量  
        g[0] = (self.W - self.B) * sth  
        g[1] = -(self.W - self.B) * cth * sphi  
        g[2] = -(self.W - self.B) * cth * cphi  
          
        # 力矩分量 (考虑重心和浮心不重合)  
        xg, yg, zg = self.r_g  
        xb, yb, zb = self.r_b  
          
        g[3] = (yg*self.W - yb*self.B) * cth * cphi - (zg*self.W - zb*self.B) * cth * sphi  
        g[4] = -(zg*self.W - zb*self.B) * sth - (xg*self.W - xb*self.B) * cth * cphi  
        g[5] = (xg*self.W - xb*self.B) * cth * sphi + (yg*self.W - yb*self.B) * sth  
          
        return g  
      
    def _thruster_dynamics(self, thrust_command: np.ndarray) -> np.ndarray:  
        """  
        推进器动态响应模型  
          
        考虑:  
        1. 推力饱和  
        2. 死区  
        3. 一阶延迟响应  
          
        这使得控制更加真实，不是理想的瞬时响应。  
        """  
        # 应用推力限制（饱和）  
        thrust_command = np.clip(thrust_command, self.min_thrust, self.max_thrust)  
          
        # 应用死区  
        thrust_command = np.where(  
            np.abs(thrust_command) < self.thruster_deadband,  
            0.0,  
            thrust_command  
        )  
          
        # 一阶延迟动态: τ̇ = (τ_cmd - τ) / T  
        # 离散化: τ_new = τ_old + dt/T * (τ_cmd - τ_old)  
        alpha = self.dt / self.thruster_time_constant  
        alpha = np.clip(alpha, 0, 1)  # 确保稳定  
          
        self.thruster_state = self.thruster_state + alpha * (thrust_command - self.thruster_state)  
          
        return self.thruster_state.copy()  
      
    def compute_derivatives(self, eta: np.ndarray, nu: np.ndarray,   
                           tau: np.ndarray,   
                           external_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        计算状态导数  
          
        动力学方程:  
            M * ν̇ + C(ν) * ν + D(ν) * ν + g(η) = τ + τ_ext  
              
        运动学方程:  
            η̇ = J(η) * ν  
              
        Args:  
            eta: 位置和姿态 [x, y, z, φ, θ, ψ]  
            nu: 速度 [u, v, w, p, q, r]  
            tau: 推进器力/力矩  
            external_force: 外部力（如水流）  
              
        Returns:  
            eta_dot: 位置导数  
            nu_dot: 速度导数  
        """  
        # 外部力  
        if external_force is None:  
            external_force = np.zeros(6)  
              
        # 计算各项力  
        C_nu = self._coriolis_matrix(nu) @ nu  
        D_nu = self._damping_force(nu)  
        g_eta = self._restoring_force(eta)  
          
        # 速度导数: ν̇ = M⁻¹ * (τ + τ_ext - C(ν)ν - D(ν)ν - g(η))  
        nu_dot = self.M_inv @ (tau + external_force - C_nu - D_nu - g_eta)  
          
        # 位置导数: η̇ = J(η) * ν  
        J = self._jacobian(eta)  
        eta_dot = J @ nu  
          
        # 记录调试信息  
        self.debug_info['forces'] = {  
            'coriolis': C_nu.copy(),  
            'damping': D_nu.copy(),  
            'restoring': g_eta.copy(),  
            'thrust': tau.copy(),  
            'external': external_force.copy()  
        }  
          
        return eta_dot, nu_dot  
      
    def step(self, action: np.ndarray,   
             external_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        执行一个仿真步  
          
        使用4阶Runge-Kutta积分保证精度  
          
        Args:  
            action: 推进器力命令 [f1, f2, ..., fn]  
            external_force: 外部力扰动  
              
        Returns:  
            eta: 更新后的位置姿态  
            nu: 更新后的速度  
        """  
        # 推进器动态  
        actual_thrust = self._thruster_dynamics(action)  
          
        # 计算作用在机体上的合力/合力矩  
        tau = self.T @ actual_thrust  
          
        # RK4积分  
        eta, nu = self._rk4_step(self.eta, self.nu, tau, external_force)  
          
        # 角度归一化到[-π, π]  
        eta[3:6] = self._normalize_angles(eta[3:6])  
          
        # 更新状态  
        self.eta = eta  
        self.nu = nu  
          
        # 记录推进器信息  
        self.debug_info['thruster'] = {  
            'command': action.copy(),  
            'actual': actual_thrust.copy(),  
            'tau': tau.copy()  
        }  
          
        return self.eta.copy(), self.nu.copy()  
      
    def _rk4_step(self, eta: np.ndarray, nu: np.ndarray,   
                  tau: np.ndarray, external_force: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        4阶Runge-Kutta积分  
          
        相比欧拉法更精确，对于非线性系统尤其重要  
        """  
        dt = self.dt  
          
        # k1  
        eta_dot1, nu_dot1 = self.compute_derivatives(eta, nu, tau, external_force)  
          
        # k2  
        eta2 = eta + 0.5 * dt * eta_dot1  
        nu2 = nu + 0.5 * dt * nu_dot1  
        eta_dot2, nu_dot2 = self.compute_derivatives(eta2, nu2, tau, external_force)  
          
        # k3  
        eta3 = eta + 0.5 * dt * eta_dot2  
        nu3 = nu + 0.5 * dt * nu_dot2  
        eta_dot3, nu_dot3 = self.compute_derivatives(eta3, nu3, tau, external_force)  
          
        # k4  
        eta4 = eta + dt * eta_dot3  
        nu4 = nu + dt * nu_dot3  
        eta_dot4, nu_dot4 = self.compute_derivatives(eta4, nu4, tau, external_force)  
          
        # 更新  
        eta_new = eta + (dt / 6.0) * (eta_dot1 + 2*eta_dot2 + 2*eta_dot3 + eta_dot4)  
        nu_new = nu + (dt / 6.0) * (nu_dot1 + 2*nu_dot2 + 2*nu_dot3 + nu_dot4)  
          
        return eta_new, nu_new  
      
    @staticmethod  
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:  
        """将角度归一化到[-π, π]"""  
        return ((angles + np.pi) % (2 * np.pi)) - np.pi  
      
    def reset(self, eta: Optional[np.ndarray] = None,   
              nu: Optional[np.ndarray] = None):  
        """  
        重置状态  
          
        Args:  
            eta: 初始位置姿态，None则为零  
            nu: 初始速度，None则为零  
        """  
        self.eta = eta if eta is not None else np.zeros(6)  
        self.nu = nu if nu is not None else np.zeros(6)  
        self.thruster_state = np.zeros(self.num_thrusters)  
        self.debug_info = {}  
          
        # 如果启用域随机化，重新随机化参数  
        if self.randomize:  
            self._init_parameters()  
      
    def set_dt(self, dt: float):  
        """设置仿真时间步长"""  
        self.dt = dt  
      
    def get_state(self) -> Dict[str, np.ndarray]:  
        """获取当前完整状态"""  
        return {  
            'eta': self.eta.copy(),  
            'nu': self.nu.copy(),  
            'position': self.eta[:3].copy(),  
            'orientation': self.eta[3:].copy(),  
            'linear_velocity': self.nu[:3].copy(),  
            'angular_velocity': self.nu[3:].copy(),  
            'thruster_state': self.thruster_state.copy()  
        }  
      
    def get_debug_info(self) -> Dict:  
        """获取调试信息"""  
        return self.debug_info.copy()  
