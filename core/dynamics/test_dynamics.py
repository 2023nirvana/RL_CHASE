"""  
动力学模块测试脚本  
==================  
  
用于验证动力学模型的正确性  
"""  
  
import numpy as np  
import matplotlib.pyplot as plt  
from auv_dynamics import AUVDynamics  
from hydrodynamics import WaterCurrentModel  
import sys  
import os  
  
# 添加项目根目录到路径  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))  
  
  
def test_basic_motion():  
    """测试基本运动"""  
    print("=" * 60)  
    print("测试1: 基本运动响应")  
    print("=" * 60)  
      
    # 创建动力学模型  
    dynamics = AUVDynamics(config_path="../../configs/robot_config.yaml",   
                           randomize=False)  
    dynamics.set_dt(0.02)  
      
    # 初始化  
    dynamics.reset()  
      
    # 记录数据  
    t_log = []  
    eta_log = []  
    nu_log = []  
      
    # 施加前向推力  
    thrust_command = np.array([10, 10, 10, 10, 0, 0])  # 前进  
      
    # 仿真10秒  
    for i in range(500):  
        t = i * dynamics.dt  
        eta, nu = dynamics.step(thrust_command)  
          
        t_log.append(t)  
        eta_log.append(eta.copy())  
        nu_log.append(nu.copy())  
      
    # 转换为数组  
    t_log = np.array(t_log)  
    eta_log = np.array(eta_log)  
    nu_log = np.array(nu_log)  
      
    # 绘图  
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  
      
    # 位置  
    axes[0, 0].plot(t_log, eta_log[:, 0], label='x')  
    axes[0, 0].plot(t_log, eta_log[:, 1], label='y')  
    axes[0, 0].plot(t_log, eta_log[:, 2], label='z')  
    axes[0, 0].set_xlabel('Time (s)')  
    axes[0, 0].set_ylabel('Position (m)')  
    axes[0, 0].legend()  
    axes[0, 0].set_title('Position')  
    axes[0, 0].grid(True)  
      
    # 姿态  
    axes[0, 1].plot(t_log, np.degrees(eta_log[:, 3]), label='roll')  
    axes[0, 1].plot(t_log, np.degrees(eta_log[:, 4]), label='pitch')  
    axes[0, 1].plot(t_log, np.degrees(eta_log[:, 5]), label='yaw')  
    axes[0, 1].set_xlabel('Time (s)')  
    axes[0, 1].set_ylabel('Angle (deg)')  
    axes[0, 1].legend()  
    axes[0, 1].set_title('Orientation')  
    axes[0, 1].grid(True)  
      
    # 线速度  
    axes[1, 0].plot(t_log, nu_log[:, 0], label='u (surge)')  
    axes[1, 0].plot(t_log, nu_log[:, 1], label='v (sway)')  
    axes[1, 0].plot(t_log, nu_log[:, 2], label='w (heave)')  
    axes[1, 0].set_xlabel('Time (s)')  
    axes[1, 0].set_ylabel('Velocity (m/s)')  
    axes[1, 0].legend()  
    axes[1, 0].set_title('Linear Velocity')  
    axes[1, 0].grid(True)  
      
    # 角速度  
    axes[1, 1].plot(t_log, np.degrees(nu_log[:, 3]), label='p (roll rate)')  
    axes[1, 1].plot(t_log, np.degrees(nu_log[:, 4]), label='q (pitch rate)')  
    axes[1, 1].plot(t_log, np.degrees(nu_log[:, 5]), label='r (yaw rate)')  
    axes[1, 1].set_xlabel('Time (s)')  
    axes[1, 1].set_ylabel('Angular Velocity (deg/s)')  
    axes[1, 1].legend()  
    axes[1, 1].set_title('Angular Velocity')  
    axes[1, 1].grid(True)  
      
    # XY轨迹  
    axes[2, 0].plot(eta_log[:, 0], eta_log[:, 1])  
    axes[2, 0].plot(eta_log[0, 0], eta_log[0, 1], 'go', markersize=10, label='Start')  
    axes[2, 0].plot(eta_log[-1, 0], eta_log[-1, 1], 'ro', markersize=10, label='End')  
    axes[2, 0].set_xlabel('X (m)')  
    axes[2, 0].set_ylabel('Y (m)')  
    axes[2, 0].legend()  
    axes[2, 0].set_title('XY Trajectory')  
    axes[2, 0].grid(True)  
    axes[2, 0].axis('equal')  
      
    # 3D轨迹  
    ax3d = fig.add_subplot(3, 2, 6, projection='3d')  
    ax3d.plot(eta_log[:, 0], eta_log[:, 1], eta_log[:, 2])  
    ax3d.scatter(eta_log[0, 0], eta_log[0, 1], eta_log[0, 2], c='g', s=100, label='Start')  
    ax3d.scatter(eta_log[-1, 0], eta_log[-1, 1], eta_log[-1, 2], c='r', s=100, label='End')  
    ax3d.set_xlabel('X (m)')  
    ax3d.set_ylabel('Y (m)')  
    ax3d.set_zlabel('Z (m)')  
    ax3d.legend()  
    ax3d.set_title('3D Trajectory')  
      
    plt.tight_layout()  
    plt.savefig('../../visualization/plots/test_basic_motion.png', dpi=150)  
    plt.show()  
      
    print(f"最终位置: x={eta_log[-1, 0]:.2f}, y={eta_log[-1, 1]:.2f}, z={eta_log[-1, 2]:.2f}")  
    print(f"最终速度: u={nu_log[-1, 0]:.2f} m/s")  
    print("基本运动测试完成!\n")  
  
  
def test_damping_effect():  
    """测试阻尼效果（非线性特性）"""  
    print("=" * 60)  
    print("测试2: 阻尼效果（线性 vs 二次）")  
    print("=" * 60)  
      
    dynamics = AUVDynamics(config_path="../../configs/robot_config.yaml",   
                           randomize=False)  
    dynamics.set_dt(0.02)  
      
    # 给一个初始速度，让其自由减速  
    dynamics.reset(  
        eta=np.zeros(6),  
        nu=np.array([2.0, 0, 0, 0, 0, 0])  # 初始前向速度2m/s  
    )  
      
    t_log = []  
    velocity_log = []  
    damping_linear_log = []  
    damping_quadratic_log = []  
      
    # 无推力情况下的减速  
    thrust_command = np.zeros(6)  
      
    for i in range(500):  
        t = i * dynamics.dt  
          
        # 记录阻尼力（在step之前）  
        debug = dynamics.get_debug_info()  
        if 'damping' in debug:  
            damping_linear_log.append(debug['damping'].get('linear', np.zeros(6))[0])  
            damping_quadratic_log.append(debug['damping'].get('quadratic', np.zeros(6))[0])  
          
        eta, nu = dynamics.step(thrust_command)  
          
        t_log.append(t)  
        velocity_log.append(nu[0])  # surge速度  
      
    t_log = np.array(t_log)  
    velocity_log = np.array(velocity_log)  
      
    # 绘图  
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  
      
    # 速度衰减曲线  
    axes[0].plot(t_log, velocity_log, 'b-', linewidth=2)  
    axes[0].set_xlabel('Time (s)')  
    axes[0].set_ylabel('Surge Velocity (m/s)')  
    axes[0].set_title('Velocity Decay (Free Deceleration)')  
    axes[0].grid(True)  
      
    # 指数衰减对比（线性阻尼理论曲线）  
    # 纯线性阻尼: v(t) = v0 * exp(-D_l/M * t)  
    v0 = 2.0  
    D_l = 5.0  # 从配置文件  
    M = 30.0 + 5.0  # 质量 + 附加质量  
    t_theory = np.linspace(0, 10, 100)  
    v_linear_theory = v0 * np.exp(-D_l / M * t_theory)  
    axes[0].plot(t_theory, v_linear_theory, 'r--', label='Pure Linear Damping (theory)')  
    axes[0].legend()  
      
    # 阻尼力对比  
    if len(damping_linear_log) > 0:  
        axes[1].plot(t_log[:len(damping_linear_log)],   
                     np.abs(damping_linear_log), label='Linear Damping Force')  
        axes[1].plot(t_log[:len(damping_quadratic_log)],   
                     np.abs(damping_quadratic_log), label='Quadratic Damping Force')  
        axes[1].set_xlabel('Time (s)')  
        axes[1].set_ylabel('Damping Force (N)')  
        axes[1].set_title('Damping Force Components')  
        axes[1].legend()  
        axes[1].grid(True)  
      
    plt.tight_layout()  
    plt.savefig('../../visualization/plots/test_damping.png', dpi=150)  
    plt.show()  
      
    print("可以看到，实际衰减比纯线性阻尼快，这是二次阻尼的贡献!")  
    print("高速时二次阻尼占主导，低速时线性阻尼占主导。\n")  
  
  
def test_rotation():  
    """测试旋转运动"""  
    print("=" * 60)  
    print("测试3: 旋转运动（偏航）")  
    print("=" * 60)  
      
    dynamics = AUVDynamics(config_path="../../configs/robot_config.yaml",   
                           randomize=False)  
    dynamics.set_dt(0.02)  
    dynamics.reset()  
      
    t_log = []  
    yaw_log = []  
    yaw_rate_log = []  
      
    # 施加偏航力矩（左转）  
    # 使用差动推力实现转向  
    thrust_command = np.array([-5, 5, 5, -5, 0, 0])  
      
    for i in range(500):  
        t = i * dynamics.dt  
        eta, nu = dynamics.step(thrust_command)  
          
        t_log.append(t)  
        yaw_log.append(np.degrees(eta[5]))  # 偏航角  
        yaw_rate_log.append(np.degrees(nu[5]))  # 偏航角速度  
      
    t_log = np.array(t_log)  
    yaw_log = np.array(yaw_log)  
    yaw_rate_log = np.array(yaw_rate_log)  
      
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  
      
    axes[0].plot(t_log, yaw_log)  
    axes[0].set_xlabel('Time (s)')  
    axes[0].set_ylabel('Yaw Angle (deg)')  
    axes[0].set_title('Yaw Angle vs Time')  
    axes[0].grid(True)  
      
    axes[1].plot(t_log, yaw_rate_log)  
    axes[1].set_xlabel('Time (s)')  
    axes[1].set_ylabel('Yaw Rate (deg/s)')  
    axes[1].set_title('Yaw Rate vs Time')  
    axes[1].grid(True)  
      
    plt.tight_layout()  
    plt.savefig('../../visualization/plots/test_rotation.png', dpi=150)  
    plt.show()  
      
    print(f"最终偏航角: {yaw_log[-1]:.1f} 度")  
    print(f"稳态偏航角速度: {yaw_rate_log[-1]:.1f} deg/s\n")  
  
  
def test_domain_randomization():  
    """测试域随机化效果"""  
    print("=" * 60)  
    print("测试4: 域随机化效果")  
    print("=" * 60)  
      
    fig, ax = plt.subplots(figsize=(10, 6))  
      
    # 多次仿真，每次参数不同  
    for trial in range(10):  
        dynamics = AUVDynamics(  
            config_path="../../configs/robot_config.yaml",   
            randomize=True,  
            random_seed=trial  
        )  
        dynamics.set_dt(0.02)  
        dynamics.reset(nu=np.array([2.0, 0, 0, 0, 0, 0]))  
          
        t_log = []  
        velocity_log = []  
          
        for i in range(300):  
            t = i * dynamics.dt  
            eta, nu = dynamics.step(np.zeros(6))  
            t_log.append(t)  
            velocity_log.append(nu[0])  
          
        ax.plot(t_log, velocity_log, alpha=0.5, label=f'Trial {trial+1}')  
      
    ax.set_xlabel('Time (s)')  
    ax.set_ylabel('Surge Velocity (m/s)')  
    ax.set_title('Velocity Decay with Domain Randomization\n(10 Different Parameter Sets)')  
    ax.grid(True)  
    ax.legend(loc='upper right', fontsize=8)  
      
    plt.tight_layout()  
    plt.savefig('../../visualization/plots/test_domain_randomization.png', dpi=150)  
    plt.show()  
      
    print("每条曲线代表不同的参数配置，展示了参数不确定性的影响。")  
    print("训练时使用域随机化可以提高策略的鲁棒性。\n")  
  
  
def test_thruster_dynamics():  
    """测试推进器动态响应"""  
    print("=" * 60)  
    print("测试5: 推进器动态响应")  
    print("=" * 60)  
      
    dynamics = AUVDynamics(config_path="../../configs/robot_config.yaml",   
                           randomize=False)  
    dynamics.set_dt(0.02)  
    dynamics.reset()  
      
    t_log = []  
    command_log = []  
    actual_log = []  
      
    # 阶跃推力命令  
    for i in range(200):  
        t = i * dynamics.dt  
          
        # 在t=1s时施加阶跃  
        if t < 1.0:  
            command = np.zeros(6)  
        else:  
            command = np.array([15, 15, 15, 15, 0, 0])  
          
        dynamics.step(command)  
          
        t_log.append(t)  
        command_log.append(command[0])  
        actual_log.append(dynamics.thruster_state[0])  
      
    t_log = np.array(t_log)  
    command_log = np.array(command_log)  
    actual_log = np.array(actual_log)  
      
    plt.figure(figsize=(10, 4))  
    plt.plot(t_log, command_log, 'b--', label='Command', linewidth=2)  
    plt.plot(t_log, actual_log, 'r-', label='Actual', linewidth=2)  
    plt.xlabel('Time (s)')  
    plt.ylabel('Thrust (N)')  
    plt.title('Thruster Dynamic Response (Step Input)')  
    plt.legend()  
    plt.grid(True)  
      
    plt.tight_layout()  
    plt.savefig('../../visualization/plots/test_thruster_dynamics.png', dpi=150)  
    plt.show()  
      
    print("推进器响应存在延迟，不是理想的瞬时响应。")  
    print("这增加了控制的难度，但更接近真实情况。\n")  
  
  
def run_all_tests():  
    """运行所有测试"""  
    print("\n" + "=" * 60)  
    print("AUV 动力学模块测试")  
    print("=" * 60 + "\n")  
      
    # 确保输出目录存在  
    os.makedirs('../../visualization/plots', exist_ok=True)  
      
    test_basic_motion()  
    test_damping_effect()  
    test_rotation()  
    test_domain_randomization()  
    test_thruster_dynamics()  
      
    print("=" * 60)  
    print("所有测试完成!")  
    print("测试图像保存在 visualization/plots/ 目录")  
    print("=" * 60)  
  
  
if __name__ == "__main__":  
    run_all_tests()  
