"""  
环境测试脚本  
============  
  
测试底层控制环境的功能  
"""  
  
import numpy as np  
import matplotlib.pyplot as plt  
import sys  
import os  
  
# 添加项目根目录到路径  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
sys.path.insert(0, PROJECT_ROOT)  
  
from core.environment.low_level_env import LowLevelAUVEnv, VecLowLevelAUVEnv  
from core.environment.tasks import HoverTask, WaypointTask, TrajectoryTrackingTask  
  
  
def test_basic_env():  
    """测试基本环境功能"""  
    print("=" * 60)  
    print("测试1: 基本环境功能")  
    print("=" * 60)  
      
    # 创建环境  
    env = LowLevelAUVEnv(  
        curriculum_stage=0,  
        enable_domain_randomization=False,  
        seed=42  
    )  
      
    print(f"观测空间: {env.observation_space}")  
    print(f"动作空间: {env.action_space}")  
    print(f"动作名称: {env.ACTION_NAMES}")  
      
    # 重置  
    obs, info = env.reset()  
    print(f"\n初始观测形状: {obs.shape}")  
    print(f"初始观测范围: [{obs.min():.3f}, {obs.max():.3f}]")  
    print(f"初始信息: {info}")  
      
    # 运行几步  
    total_reward = 0  
    for step in range(100):  
        action = env.action_space.sample()  # 随机动作  
        obs, reward, terminated, truncated, info = env.step(action)  
        total_reward += reward  
          
        if terminated or truncated:  
            print(f"Episode结束于step {step}, 原因: {info.get('termination_reason', 'truncated')}")  
            break  
              
    print(f"累计奖励: {total_reward:.2f}")  
      
    env.close()  
    print("基本环境测试完成!\n")  
  
  
def test_action_effects():  
    """测试各个动作的效果"""  
    print("=" * 60)  
    print("测试2: 动作效果")  
    print("=" * 60)  
      
    env = LowLevelAUVEnv(  
        curriculum_stage=0,  
        enable_domain_randomization=False,  
        seed=42  
    )  
      
    results = {}  
      
    for action_id, action_name in enumerate(env.ACTION_NAMES):  
        # 重置  
        obs, _ = env.reset()  
        initial_state = env.dynamics.get_state()  
          
        # 执行同一动作100步  
        for _ in range(100):  
            obs, reward, terminated, truncated, info = env.step(action_id)  
            if terminated or truncated:  
                break  
                  
        final_state = env.dynamics.get_state()  
          
        # 计算状态变化  
        delta_pos = final_state['position'] - initial_state['position']  
        delta_ori = final_state['orientation'] - initial_state['orientation']  
          
        results[action_name] = {  
            'delta_position': delta_pos,  
            'delta_orientation': np.degrees(delta_ori),  
            'final_velocity': final_state['linear_velocity']  
        }  
          
        print(f"\n动作: {action_name}")  
        print(f"  位置变化: x={delta_pos[0]:.2f}, y={delta_pos[1]:.2f}, z={delta_pos[2]:.2f}")  
        print(f"  姿态变化(度): roll={np.degrees(delta_ori[0]):.1f}, pitch={np.degrees(delta_ori[1]):.1f}, yaw={np.degrees(delta_ori[2]):.1f}")  
          
    env.close()  
      
    # 绘制动作效果图  
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
      
    action_names = list(results.keys())  
    x_pos = [results[a]['delta_position'][0] for a in action_names]  
    y_pos = [results[a]['delta_position'][1] for a in action_names]  
    z_pos = [results[a]['delta_position'][2] for a in action_names]  
      
    # 位置变化  
    x = np.arange(len(action_names))  
    width = 0.25  
      
    axes[0, 0].bar(x - width, x_pos, width, label='X')  
    axes[0, 0].bar(x, y_pos, width, label='Y')  
    axes[0, 0].bar(x + width, z_pos, width, label='Z')  
    axes[0, 0].set_xticks(x)  
    axes[0, 0].set_xticklabels(action_names, rotation=45, ha='right')  
    axes[0, 0].set_ylabel('Position Change (m)')  
    axes[0, 0].set_title('Position Change by Action')  
    axes[0, 0].legend()  
    axes[0, 0].grid(True, alpha=0.3)  
      
    # 姿态变化  
    roll_ori = [results[a]['delta_orientation'][0] for a in action_names]  
    pitch_ori = [results[a]['delta_orientation'][1] for a in action_names]  
    yaw_ori = [results[a]['delta_orientation'][2] for a in action_names]  
      
    axes[0, 1].bar(x - width, roll_ori, width, label='Roll')  
    axes[0, 1].bar(x, pitch_ori, width, label='Pitch')  
    axes[0, 1].bar(x + width, yaw_ori, width, label='Yaw')  
    axes[0, 1].set_xticks(x)  
    axes[0, 1].set_xticklabels(action_names, rotation=45, ha='right')  
    axes[0, 1].set_ylabel('Orientation Change (deg)')  
    axes[0, 1].set_title('Orientation Change by Action')  
    axes[0, 1].legend()  
    axes[0, 1].grid(True, alpha=0.3)  
      
    # 清空不需要的子图  
    axes[1, 0].axis('off')  
    axes[1, 1].axis('off')  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(PROJECT_ROOT, 'visualization/plots/action_effects.png'), dpi=150)  
    plt.show()  
      
    print("\n动作效果测试完成!")  
  
  
def test_curriculum_learning():  
    """测试课程学习"""  
    print("=" * 60)  
    print("测试3: 课程学习")  
    print("=" * 60)  
      
    for stage in range(4):  
        env = LowLevelAUVEnv(  
            curriculum_stage=stage,  
            enable_domain_randomization=False,  
            seed=42  
        )  
          
        obs, info = env.reset()  
          
        print(f"\n课程阶段 {stage}:")  
        print(f"  阶段名称: {env.curriculum_config.get('name', 'unknown')}")  
        print(f"  位置范围: {env.curriculum_config.get('position_range', 'N/A')}")  
        print(f"  需要姿态控制: {env.curriculum_config.get('orientation_required', False)}")  
        print(f"  初始速度类型: {env.curriculum_config.get('velocity_init', 'zero')}")  
        print(f"  初始位置: {env.dynamics.eta[:3]}")  
        print(f"  初始速度: {env.dynamics.nu[:3]}")  
          
        env.close()  
          
    print("\n课程学习测试完成!")  
  
  
def test_domain_randomization():  
    """测试域随机化"""  
    print("=" * 60)  
    print("测试4: 域随机化")  
    print("=" * 60)  
      
    # 多次运行，观察不同参数下的行为差异  
    results = []  
      
    for trial in range(5):  
        env = LowLevelAUVEnv(  
            curriculum_stage=0,  
            enable_domain_randomization=True,  
            seed=trial  
        )  
          
        obs, _ = env.reset()  
          
        positions = []  
        velocities = []  
          
        # 持续前进  
        for _ in range(200):  
            action = 1  # surge_forward  
            obs, reward, terminated, truncated, info = env.step(action)  
            positions.append(env.dynamics.eta[:3].copy())  
            velocities.append(env.dynamics.nu[0])  # 前向速度  
              
            if terminated or truncated:  
                break  
                  
        results.append({  
            'positions': np.array(positions),  
            'velocities': np.array(velocities)  
        })  
          
        # 获取域随机化参数  
        debug = env.dynamics.get_debug_info()  
        if 'domain_randomization' in debug:  
            dr = debug['domain_randomization']  
            print(f"Trial {trial}: 阻尼缩放 = {dr['quadratic_damping_scale'][:3]}")  
              
        env.close()  
      
    # 绘图  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  
      
    for i, res in enumerate(results):  
        t = np.arange(len(res['velocities'])) * 0.02  
        axes[0].plot(t, res['velocities'], label=f'Trial {i+1}', alpha=0.7)  
        axes[1].plot(res['positions'][:, 0], res['positions'][:, 1],   
                    label=f'Trial {i+1}', alpha=0.7)  
      
    axes[0].set_xlabel('Time (s)')  
    axes[0].set_ylabel('Forward Velocity (m/s)')  
    axes[0].set_title('Velocity with Domain Randomization')  
    axes[0].legend()  
    axes[0].grid(True)  
      
    axes[1].set_xlabel('X (m)')  
    axes[1].set_ylabel('Y (m)')  
    axes[1].set_title('Trajectory with Domain Randomization')  
    axes[1].legend()  
    axes[1].grid(True)  
    axes[1].axis('equal')  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(PROJECT_ROOT, 'visualization/plots/domain_randomization.png'), dpi=150)  
    plt.show()  
      
    print("\n域随机化测试完成!")  
  
  
def test_reward_function():  
    """测试奖励函数"""  
    print("=" * 60)  
    print("测试5: 奖励函数")  
    print("=" * 60)  
      
    env = LowLevelAUVEnv(  
        curriculum_stage=1,  
        enable_domain_randomization=False,  
        seed=42  
    )  
      
    obs, _ = env.reset()  
      
    rewards = []  
    pos_errors = []  
    reward_components = {  
        'position': [],  
        'orientation': [],  
        'stability': [],  
        'goal': [],  
        'shaping': []  
    }  
      
    # 使用简单策略：向目标移动  
    for step in range(500):  
        # 获取当前误差  
        current_pos = env.dynamics.eta[:3]  
        target_pos = env.target_state[:3]  
        error = target_pos - current_pos  
          
        # 简单策略：根据误差选择动作  
        if abs(error[0]) > abs(error[1]) and abs(error[0]) > abs(error[2]):  
            action = 1 if error[0] > 0 else 2  # surge forward/backward  
        elif abs(error[1]) > abs(error[2]):  
            action = 4 if error[1] > 0 else 3  # sway right/left  
        else:  
            action = 5 if error[2] > 0 else 6  # heave up/down  
              
        obs, reward, terminated, truncated, info = env.step(action)  
          
        rewards.append(reward)  
        pos_errors.append(info['pos_error'])  
          
        for key in reward_components:  
            if key in info:  
                reward_components[key].append(info[key])  
            else:  
                reward_components[key].append(0)  
                  
        if terminated or truncated:  
            print(f"Episode结束于step {step}, 原因: {info.get('termination_reason', 'truncated')}")  
            break  
      
    env.close()  
      
    # 确保有足够的数据点  
    if len(rewards) < 10:  
        print("警告: Episode太短，跳过绘图")  
        print(f"总奖励: {sum(rewards):.2f}")  
        print("奖励函数测试完成!\n")  
        return  
      
    # 绘图  
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
      
    steps = list(range(len(rewards)))  
      
    # 总奖励  
    axes[0, 0].plot(steps, rewards, 'b-', alpha=0.5, label='Raw')  
      
    # 平滑曲线 - 修复：确保窗口大小不超过数据长度  
    window_size = min(20, len(rewards) // 2)  # 动态调整窗口大小  
    if window_size >= 2:  
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')  
        # 修复：正确对齐x轴  
        smooth_steps = list(range(window_size - 1, len(rewards)))  
        axes[0, 0].plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (w={window_size})')  
      
    axes[0, 0].set_xlabel('Step')  
    axes[0, 0].set_ylabel('Reward')  
    axes[0, 0].set_title('Step Reward')  
    axes[0, 0].legend()  
    axes[0, 0].grid(True)  
      
    # 累计奖励  
    axes[0, 1].plot(steps, np.cumsum(rewards))  
    axes[0, 1].set_xlabel('Step')  
    axes[0, 1].set_ylabel('Cumulative Reward')  
    axes[0, 1].set_title('Cumulative Reward')  
    axes[0, 1].grid(True)  
      
    # 位置误差  
    axes[1, 0].plot(steps, pos_errors)  
    axes[1, 0].axhline(y=env.reward_thresholds['position_tolerance'],   
                       color='r', linestyle='--', label='Tolerance')  
    axes[1, 0].set_xlabel('Step')  
    axes[1, 0].set_ylabel('Position Error (m)')  
    axes[1, 0].set_title('Position Error')  
    axes[1, 0].legend()  
    axes[1, 0].grid(True)  
      
    # 奖励分解  
    has_data = False  
    for key, values in reward_components.items():  
        if any(v != 0 for v in values):  
            axes[1, 1].plot(steps, values, label=key, alpha=0.7)  
            has_data = True  
      
    if has_data:  
        axes[1, 1].legend()  
    axes[1, 1].set_xlabel('Step')  
    axes[1, 1].set_ylabel('Reward Component')  
    axes[1, 1].set_title('Reward Components')  
    axes[1, 1].grid(True)  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(PROJECT_ROOT, 'visualization/plots/reward_function.png'), dpi=150)  
    plt.show()  
      
    print(f"\n总奖励: {sum(rewards):.2f}")  
    print(f"最终位置误差: {pos_errors[-1]:.3f} m")  
    print("\n奖励函数测试完成!")  

  
  
def test_vectorized_env():  
    """测试向量化环境"""  
    print("=" * 60)  
    print("测试6: 向量化环境")  
    print("=" * 60)  
      
    num_envs = 4  
    vec_env = VecLowLevelAUVEnv(  
        num_envs=num_envs,  
        curriculum_stage=0,  
        enable_domain_randomization=True  
    )  
      
    print(f"环境数量: {num_envs}")  
    print(f"观测空间: {vec_env.observation_space}")  
    print(f"动作空间: {vec_env.action_space}")  
      
    # 重置  
    observations, infos = vec_env.reset(seed=42)  
    print(f"\n观测形状: {observations.shape}")  
      
    # 运行  
    total_rewards = np.zeros(num_envs)  
    episode_counts = np.zeros(num_envs)  
      
    for step in range(1000):  
        # 随机动作  
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])  
        observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)  
          
        total_rewards += rewards  
          
        for i, info in enumerate(infos):  
            if 'final_info' in info:  
                episode_counts[i] += 1  
                print(f"  环境{i}完成episode, 奖励: {info['final_info']['episode']['r']:.2f}")  
                  
    vec_env.close()  
      
    print(f"\n各环境完成的episode数: {episode_counts}")  
    print("\n向量化环境测试完成!")  
  
  
def test_tasks():  
    """测试特定任务环境"""  
    print("=" * 60)  
    print("测试7: 特定任务环境")  
    print("=" * 60)  
      
    # 测试悬停任务  
    print("\n1. 悬停任务:")  
    hover_env = HoverTask(curriculum_stage=0, seed=42)  
    obs, info = hover_env.reset()  
    print(f"   目标位置: {hover_env.target_state[:3]}")  
    hover_env.close()  
      
    # 测试航点任务  
    print("\n2. 航点任务:")  
    waypoint_env = WaypointTask(num_waypoints=3, curriculum_stage=0, seed=42)  
    obs, info = waypoint_env.reset()  
    print(f"   航点: {info.get('waypoints', [])}")  
    waypoint_env.close()  
      
    # 测试轨迹跟踪任务  
    print("\n3. 轨迹跟踪任务:")  
    trajectory_env = TrajectoryTrackingTask(trajectory_type="circle", curriculum_stage=0, seed=42)  
    obs, info = trajectory_env.reset()  
      
    # 运行并记录目标轨迹  
    target_positions = []  
    for _ in range(500):  
        action = 0  # hover  
        obs, reward, terminated, truncated, info = trajectory_env.step(action)  
        target_positions.append(trajectory_env.target_state[:3].copy())  
          
    trajectory_env.close()  
      
    # 绘制目标轨迹  
    target_positions = np.array(target_positions)  
    plt.figure(figsize=(8, 8))  
    plt.plot(target_positions[:, 0], target_positions[:, 1], 'b-')  
    plt.plot(target_positions[0, 0], target_positions[0, 1], 'go', markersize=10)  
    plt.xlabel('X (m)')  
    plt.ylabel('Y (m)')  
    plt.title('Circle Trajectory Target')  
    plt.axis('equal')  
    plt.grid(True)  
    plt.savefig(os.path.join(PROJECT_ROOT, 'visualization/plots/trajectory_task.png'), dpi=150)  
    plt.show()  
      
    print("\n任务环境测试完成!")  
  
  
def test_rendering():  
    """测试渲染功能"""  
    print("=" * 60)  
    print("测试8: 渲染功能")  
    print("=" * 60)  
      
    env = LowLevelAUVEnv(  
        curriculum_stage=1,  
        enable_domain_randomization=False,  
        render_mode="human",  
        seed=42  
    )  
      
    obs, _ = env.reset()  
      
    print("运行100步并渲染...")  
    for step in range(100):  
        # 简单策略  
        action = 1  # 前进  
        obs, reward, terminated, truncated, info = env.step(action)  
          
        # 渲染  
        if step % 10 == 0:  
            env.render()  
              
        if terminated or truncated:  
            break  
              
    env.close()  
    print("渲染测试完成!")  
  
  
def run_all_tests():  
    """运行所有测试"""  
    print("\n" + "=" * 60)  
    print("底层控制环境测试")  
    print("=" * 60 + "\n")  
      
    # 确保目录存在  
    os.makedirs(os.path.join(PROJECT_ROOT, "visualization/plots"), exist_ok=True)  
      
    test_basic_env()  
    test_action_effects()  
    test_curriculum_learning()  
    test_domain_randomization()  
    test_reward_function()  
    test_vectorized_env()  
    test_tasks()  
    # test_rendering()  # 需要图形界面，可选  
      
    print("\n" + "=" * 60)  
    print("所有环境测试完成!")  
    print("=" * 60)  
  
  
if __name__ == "__main__":  
    run_all_tests()  
