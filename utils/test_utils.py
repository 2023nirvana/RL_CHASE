"""  
工具模块测试脚本  
"""  
  
import numpy as np  
import os  
import sys  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
  
from utils.logger import Logger, MetricsTracker, EpisodeTracker  
from utils.visualizer import TrainingVisualizer, TrajectoryVisualizer  
from utils.video_recorder import VideoRecorder  
from utils.metrics import RunningMeanStd, SuccessRateTracker, PerformanceMetrics, GAECalculator  
  
  
def test_logger():  
    """测试日志记录器"""  
    print("=" * 60)  
    print("测试日志记录器")  
    print("=" * 60)  
      
    # 创建日志记录器  
    logger = Logger(  
        log_dir="visualization/logs",  
        experiment_name="test_experiment",  
        use_tensorboard=True,  
        use_csv=True  
    )  
      
    # 模拟训练过程  
    for step in range(100):  
        logger.step = step  
          
        # 记录标量  
        logger.log_scalar("train/loss", np.random.random() * (1 - step/100))  
        logger.log_scalar("train/reward", np.random.random() * step/10)  
          
    # 记录Episode信息  
    for episode in range(10):  
        episode_info = {  
            'total_reward': np.random.random() * 100,  
            'episode_length': np.random.randint(50, 200),  
            'success': np.random.random() > 0.5  
        }  
        logger.log_episode(episode_info)  
          
    logger.close()  
    print("日志记录器测试完成！")  
    print(f"日志保存在: {logger.log_dir}\n")  
  
  
def test_visualizer():  
    """测试可视化器"""  
    print("=" * 60)  
    print("测试可视化器")  
    print("=" * 60)  
      
    vis = TrainingVisualizer(save_dir="visualization/plots/test")  
      
    # 添加模拟数据  
    for step in range(1000):  
        vis.add_scalar("episode/total_reward",   
                       np.sin(step/100) * 50 + np.random.randn() * 10 + step/10,  
                       step)  
        vis.add_scalar("train/actor_loss",   
                       np.exp(-step/500) + np.random.random() * 0.1,  
                       step)  
        vis.add_scalar("train/critic_loss",  
                       np.exp(-step/300) + np.random.random() * 0.2,  
                       step)  
                         
    # 绘制曲线  
    vis.plot_reward_curve()  
    vis.plot_loss_curves()  
    vis.plot_training_curves()  
      
    print("可视化测试完成！")  
    print(f"图像保存在: {vis.save_dir}\n")  
  
  
def test_trajectory_visualizer():  
    """测试轨迹可视化器"""  
    print("=" * 60)  
    print("测试轨迹可视化器")  
    print("=" * 60)  
      
    vis = TrajectoryVisualizer(save_dir="visualization/plots/test")  
      
    # 生成模拟轨迹  
    t = np.linspace(0, 10, 500)  
    positions = np.column_stack([  
        t * np.cos(t * 0.5),  
        t * np.sin(t * 0.5),  
        t * 0.1  
    ])  
    orientations = np.column_stack([  
        np.zeros_like(t),  
        np.zeros_like(t),  
        t * 0.5  
    ])  
    velocities = np.column_stack([  
        np.cos(t * 0.5),  
        np.sin(t * 0.5),  
        np.ones_like(t) * 0.1,  
        np.zeros_like(t),  
        np.zeros_like(t),  
        np.ones_like(t) * 0.5  
    ])  
      
    # 绘制  
    vis.plot_2d_trajectory(positions, save_name="test_traj_2d.png")  
    vis.plot_3d_trajectory(positions, orientations, save_name="test_traj_3d.png")  
    vis.plot_state_evolution(t, positions, orientations, velocities,   
                             save_name="test_state.png")  
      
    print("轨迹可视化测试完成！\n")  
  
  
def test_video_recorder():  
    """测试视频录制器"""  
    print("=" * 60)  
    print("测试视频录制器")  
    print("=" * 60)  
      
    recorder = VideoRecorder(save_dir="visualization/videos/test", fps=30)  
      
    # 生成模拟数据  
    recorder.start_recording()  
      
    for i in range(300):  # 10秒视频  
        t = i * 0.033  
        frame_data = {  
            'position': np.array([t * np.cos(t), t * np.sin(t), t * 0.05]),  
            'orientation': np.array([0, 0, t]),  
            'time': t  
        }  
        recorder.add_frame(frame_data)  
          
    recorder.stop_recording()  
      
    # 保存视频  
    try:  
        recorder.save_video_2d("test_2d.mp4")  
        recorder.save_video_3d("test_3d.mp4")  
        print("视频录制测试完成！")  
    except Exception as e:  
        print(f"视频保存失败（可能缺少FFmpeg）: {e}")  
        print("提示: 安装FFmpeg后可生成视频")  
    print()  
  
  
def test_metrics():  
    """测试指标计算"""  
    print("=" * 60)  
    print("测试指标计算")  
    print("=" * 60)  
      
    # 测试RunningMeanStd  
    rms = RunningMeanStd(shape=(3,))  
    for _ in range(100):  
        data = np.random.randn(32, 3)  
        rms.update(data)  
    print(f"RunningMeanStd: mean={rms.mean}, std={rms.std}")  
      
    # 测试成功率追踪  
    tracker = SuccessRateTracker(window_size=100)  
    for i in range(150):  
        tracker.add(i > 50)  # 前50次失败，后100次成功  
    print(f"SuccessRateTracker: rate={tracker.rate:.2f}")  
      
    # 测试GAE计算  
    gae = GAECalculator(gamma=0.99, lam=0.95)  
    rewards = np.random.randn(100)  
    values = np.random.randn(100)  
    dones = np.zeros(100)  
    dones[-1] = 1  
    advantages, returns = gae.compute_gae(rewards, values, dones, 0.0)  
    print(f"GAE: advantages shape={advantages.shape}, returns shape={returns.shape}")  
      
    print("指标计算测试完成！\n")  
  
  
def run_all_tests():  
    """运行所有测试"""  
    print("\n" + "=" * 60)  
    print("工具模块测试")  
    print("=" * 60 + "\n")  
      
    # 确保目录存在  
    os.makedirs("visualization/logs", exist_ok=True)  
    os.makedirs("visualization/plots/test", exist_ok=True)  
    os.makedirs("visualization/videos/test", exist_ok=True)  
      
    test_logger()  
    test_visualizer()  
    test_trajectory_visualizer()  
    test_video_recorder()  
    test_metrics()  
      
    print("=" * 60)  
    print("所有工具测试完成！")  
    print("=" * 60)  
      
  
if __name__ == "__main__":  
    run_all_tests()  
