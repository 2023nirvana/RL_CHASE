"""  
训练监控脚本  
============  
  
实时监控训练进度  
  
使用方法:  
    python training/monitor.py --log-dir visualization/logs/experiment_name  
"""  
  
import os  
import sys  
import argparse  
import time  
import json  
from datetime import datetime  
import numpy as np  
  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
try:  
    import matplotlib.pyplot as plt  
    from matplotlib.animation import FuncAnimation  
    MATPLOTLIB_AVAILABLE = True  
except ImportError:  
    MATPLOTLIB_AVAILABLE = False  
      
  
class TrainingMonitor:  
    """  
    训练监控器  
      
    实时显示训练进度  
    """  
      
    def __init__(self, log_dir: str, refresh_interval: int = 5):  
        """  
        初始化监控器  
          
        Args:  
            log_dir: 日志目录  
            refresh_interval: 刷新间隔（秒）  
        """  
        self.log_dir = log_dir  
        self.refresh_interval = refresh_interval  
          
        # 数据存储  
        self.rewards = []  
        self.lengths = []  
        self.actor_losses = []  
        self.critic_losses = []  
        self.entropies = []  
        self.timesteps = []  
          
        # 检查TensorBoard日志  
        self.tb_dir = os.path.join(log_dir, "tensorboard")  
          
    def load_csv_data(self):  
        """从CSV加载数据"""  
        csv_dir = os.path.join(self.log_dir, "csv")  
          
        # 尝试加载episode日志  
        episode_log = os.path.join(csv_dir, "episode_log.csv")  
        if os.path.exists(episode_log):  
            try:  
                import pandas as pd  
                df = pd.read_csv(episode_log)  
                self.rewards = df['total_reward'].tolist() if 'total_reward' in df else []  
                self.lengths = df['episode_length'].tolist() if 'episode_length' in df else []  
            except Exception as e:  
                print(f"Error loading CSV: {e}")  
                  
    def load_tensorboard_data(self):  
        """从TensorBoard加载数据"""  
        try:  
            from tensorboard.backend.event_processing import event_accumulator  
              
            ea = event_accumulator.EventAccumulator(self.tb_dir)  
            ea.Reload()  
              
            # 获取标量数据  
            tags = ea.Tags()['scalars']  
              
            if 'rollout/mean_reward' in tags:  
                events = ea.Scalars('rollout/mean_reward')  
                self.rewards = [e.value for e in events]  
                self.timesteps = [e.step for e in events]  
                  
            if 'train/actor_loss' in tags:  
                events = ea.Scalars('train/actor_loss')  
                self.actor_losses = [e.value for e in events]  
                  
            if 'train/critic_loss' in tags:  
                events = ea.Scalars('train/critic_loss')  
                self.critic_losses = [e.value for e in events]  
                  
            if 'train/entropy' in tags:  
                events = ea.Scalars('train/entropy')  
                self.entropies = [e.value for e in events]  
                  
        except Exception as e:  
            print(f"Error loading TensorBoard data: {e}")  
            self.load_csv_data()  
              
    def print_status(self):  
        """打印训练状态"""  
        self.load_tensorboard_data()  
          
        os.system('clear' if os.name == 'posix' else 'cls')  
          
        print("=" * 60)  
        print("TRAINING MONITOR")  
        print(f"Log directory: {self.log_dir}")  
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
        print("=" * 60)  
          
        if self.rewards:  
            print(f"\nTraining Progress:")  
            print(f"  Timesteps: {self.timesteps[-1] if self.timesteps else 'N/A':,}")  
            print(f"  Updates: {len(self.rewards)}")  
              
            print(f"\nReward Statistics:")  
            print(f"  Current: {self.rewards[-1]:.2f}")  
            print(f"  Mean (last 100): {np.mean(self.rewards[-100:]):.2f}")  
            print(f"  Max: {max(self.rewards):.2f}")  
            print(f"  Min: {min(self.rewards):.2f}")  
              
            if self.actor_losses:  
                print(f"\nLoss Statistics:")  
                print(f"  Actor Loss: {self.actor_losses[-1]:.4f}")  
                print(f"  Critic Loss: {self.critic_losses[-1]:.4f}" if self.critic_losses else "")  
                print(f"  Entropy: {self.entropies[-1]:.4f}" if self.entropies else "")  
                  
            # 简单的进度条  
            if self.timesteps:  
                # 假设总步数为1M  
                total_steps = 1000000  
                progress = min(self.timesteps[-1] / total_steps, 1.0)  
                bar_length = 40  
                filled = int(bar_length * progress)  
                bar = '█' * filled + '░' * (bar_length - filled)  
                print(f"\nProgress: [{bar}] {progress*100:.1f}%")  
        else:  
            print("\nNo training data available yet...")  
              
        print("\n" + "=" * 60)  
        print("Press Ctrl+C to exit")  
          
    def plot_live(self):  
        """实时绘图"""  
        if not MATPLOTLIB_AVAILABLE:  
            print("Matplotlib not available. Using text mode.")  
            self.run_text_mode()  
            return  
              
        plt.ion()  
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
          
        while True:  
            try:  
                self.load_tensorboard_data()  
                  
                for ax in axes.flatten():  
                    ax.clear()  
                      
                # 奖励曲线  
                if self.rewards:  
                    axes[0, 0].plot(self.rewards, 'b-', alpha=0.5)  
                    if len(self.rewards) > 20:  
                        smoothed = np.convolve(self.rewards, np.ones(20)/20, mode='valid')  
                        axes[0, 0].plot(range(19, len(self.rewards)), smoothed, 'r-', linewidth=2)  
                    axes[0, 0].set_xlabel('Update')  
                    axes[0, 0].set_ylabel('Mean Reward')  
                    axes[0, 0].set_title('Training Reward')  
                    axes[0, 0].grid(True)  
                      
                # Actor损失  
                if self.actor_losses:  
                    axes[0, 1].plot(self.actor_losses, 'b-')  
                    axes[0, 1].set_xlabel('Update')  
                    axes[0, 1].set_ylabel('Loss')  
                    axes[0, 1].set_title('Actor Loss')  
                    axes[0, 1].grid(True)  
                      
                # Critic损失  
                if self.critic_losses:  
                    axes[1, 0].plot(self.critic_losses, 'r-')  
                    axes[1, 0].set_xlabel('Update')  
                    axes[1, 0].set_ylabel('Loss')  
                    axes[1, 0].set_title('Critic Loss')  
                    axes[1, 0].grid(True)  
                      
                # 熵  
                if self.entropies:  
                    axes[1, 1].plot(self.entropies, 'g-')  
                    axes[1, 1].set_xlabel('Update')  
                    axes[1, 1].set_ylabel('Entropy')  
                    axes[1, 1].set_title('Policy Entropy')  
                    axes[1, 1].grid(True)  
                      
                plt.tight_layout()  
                plt.pause(self.refresh_interval)  
                  
            except KeyboardInterrupt:  
                break  
                  
        plt.ioff()  
        plt.close()  
          
    def run_text_mode(self):  
        """文本模式监控"""  
        while True:  
            try:  
                self.print_status()  
                time.sleep(self.refresh_interval)  
            except KeyboardInterrupt:  
                break  
                  
    def run(self, mode: str = 'auto'):  
        """  
        运行监控  
          
        Args:  
            mode: 'text', 'plot', 或 'auto'  
        """  
        if mode == 'auto':  
            mode = 'plot' if MATPLOTLIB_AVAILABLE else 'text'  
              
        if mode == 'plot':  
            self.plot_live()  
        else:  
            self.run_text_mode()  
  
  
def parse_args():  
    parser = argparse.ArgumentParser(description="Monitor training progress")  
    parser.add_argument('--log-dir', type=str, required=True,  
                       help='Log directory to monitor')  
    parser.add_argument('--refresh', type=int, default=5,  
                       help='Refresh interval in seconds')  
    parser.add_argument('--mode', type=str, default='auto',  
                       choices=['text', 'plot', 'auto'],  
                       help='Display mode')  
    return parser.parse_args()  
  
  
def main():  
    args = parse_args()  
      
    if not os.path.exists(args.log_dir):  
        print(f"Log directory not found: {args.log_dir}")  
        return  
          
    monitor = TrainingMonitor(args.log_dir, args.refresh)  
    monitor.run(args.mode)  
  
  
if __name__ == "__main__":  
    main()  
