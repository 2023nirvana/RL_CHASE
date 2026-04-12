"""  
日志记录模块  
============  
  
提供训练过程的全面日志记录功能：  
1. TensorBoard集成  
2. CSV日志  
3. 控制台输出  
4. 检查点管理  
"""  
  
import os  
import json  
import csv  
import time  
from datetime import datetime  
from typing import Dict, List, Optional, Any, Union  
from collections import defaultdict  
import numpy as np  
  
try:  
    from torch.utils.tensorboard import SummaryWriter  
    TENSORBOARD_AVAILABLE = True  
except ImportError:  
    TENSORBOARD_AVAILABLE = False  
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")  
  
  
class Logger:  
    """  
    统一的日志记录器  
      
    支持多种输出格式：  
    - TensorBoard  
    - CSV文件  
    - JSON文件  
    - 控制台  
    """  
      
    def __init__(self,   
                 log_dir: str,  
                 experiment_name: Optional[str] = None,  
                 use_tensorboard: bool = True,  
                 use_csv: bool = True,  
                 console_log_freq: int = 100,  
                 config: Optional[Dict] = None):  
        """  
        初始化日志记录器  
          
        Args:  
            log_dir: 日志根目录  
            experiment_name: 实验名称（默认使用时间戳）  
            use_tensorboard: 是否使用TensorBoard  
            use_csv: 是否记录CSV  
            console_log_freq: 控制台打印频率（每多少步打印一次）  
            config: 实验配置（会被保存）  
        """  
        # 创建实验名称  
        if experiment_name is None:  
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")  
        self.experiment_name = experiment_name  
          
        # 创建目录结构  
        self.log_dir = os.path.join(log_dir, experiment_name)  
        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")  
        self.csv_dir = os.path.join(self.log_dir, "csv")  
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")  
          
        os.makedirs(self.log_dir, exist_ok=True)  
        os.makedirs(self.tensorboard_dir, exist_ok=True)  
        os.makedirs(self.csv_dir, exist_ok=True)  
        os.makedirs(self.checkpoint_dir, exist_ok=True)  
          
        # TensorBoard  
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE  
        if self.use_tensorboard:  
            self.writer = SummaryWriter(self.tensorboard_dir)  
        else:  
            self.writer = None  
              
        # CSV记录  
        self.use_csv = use_csv  
        self.csv_files = {}  
        self.csv_writers = {}  
          
        # 控制台设置  
        self.console_log_freq = console_log_freq  
          
        # 内部计数器  
        self.step = 0  
        self.episode = 0  
        self.start_time = time.time()  
          
        # 数据缓存（用于计算平均值）  
        self.episode_buffer = defaultdict(list)  
        self.step_buffer = defaultdict(list)  
          
        # 历史记录（用于绘图）  
        self.history = defaultdict(list)  
          
        # 保存配置  
        if config is not None:  
            self.save_config(config)  
              
        print(f"Logger initialized. Log directory: {self.log_dir}")  
          
    def save_config(self, config: Dict):  
        """保存实验配置"""  
        config_path = os.path.join(self.log_dir, "config.json")  
        with open(config_path, 'w') as f:  
            json.dump(config, f, indent=2, default=str)  
        print(f"Config saved to {config_path}")  
          
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):  
        """  
        记录标量值  
          
        Args:  
            tag: 指标名称  
            value: 值  
            step: 步数（默认使用内部计数器）  
        """  
        if step is None:  
            step = self.step  
              
        # TensorBoard  
        if self.use_tensorboard and self.writer is not None:  
            self.writer.add_scalar(tag, value, step)  
              
        # 历史记录  
        self.history[tag].append((step, value))  
          
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],   
                    step: Optional[int] = None):  
        """  
        记录多个相关标量（会在同一图中显示）  
          
        Args:  
            main_tag: 主标签  
            tag_scalar_dict: 标量字典  
            step: 步数  
        """  
        if step is None:  
            step = self.step  
              
        if self.use_tensorboard and self.writer is not None:  
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)  
              
        for tag, value in tag_scalar_dict.items():  
            full_tag = f"{main_tag}/{tag}"  
            self.history[full_tag].append((step, value))  
              
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):  
        """记录直方图"""  
        if step is None:  
            step = self.step  
              
        if self.use_tensorboard and self.writer is not None:  
            self.writer.add_histogram(tag, values, step)  
              
    def log_episode(self, episode_info: Dict[str, float]):  
        """  
        记录Episode级别的信息  
          
        Args:  
            episode_info: Episode信息字典  
        """  
        self.episode += 1  
          
        # 记录到TensorBoard  
        for key, value in episode_info.items():  
            self.log_scalar(f"episode/{key}", value, self.episode)  
              
        # 记录到CSV  
        if self.use_csv:  
            self._write_csv("episode_log", episode_info, self.episode)  
              
        # 控制台输出  
        if self.episode % self.console_log_freq == 0:  
            self._print_episode_summary(episode_info)  
              
    def log_step(self, step_info: Dict[str, float]):  
        """  
        记录Step级别的信息  
          
        Args:  
            step_info: Step信息字典  
        """  
        self.step += 1  
          
        # 累积到缓冲区  
        for key, value in step_info.items():  
            self.step_buffer[key].append(value)  

    def log_reward_components_step(self, env_id: int, rollout_step: int, global_timestep: int, components: Dict[str, Any]):
        """
        记录每步的 reward components 到 CSV（逐步记录，便于后续分析）

        Args:
            env_id: 环境索引（向量化环境的子环境 id）
            rollout_step: 本次 rollout 中的 step 索引
            global_timestep: 全局时间步（训练器计数）
            components: reward components 字典（可能包含 numpy types）
        """
        if not self.use_csv:
            return

        # 扁平化并规范化值为可写入 CSV 的标量
        flat = {
            'global_timestep': int(global_timestep),
            'env_id': int(env_id),
            'rollout_step': int(rollout_step)
        }

        for k, v in components.items():
            # 跳过非标量或嵌套结构
            if isinstance(v, (dict, list, tuple)):
                continue
            try:
                if np.isscalar(v):
                    flat[k] = float(v)
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)
                else:
                    # 尝试转为字符串（保留信息）
                    flat[k] = str(v)
            except Exception:
                flat[k] = str(v)

        # 写入 CSV（文件名固定为 reward_components_steps.csv）
        try:
            self._write_csv('reward_components_steps', flat, global_timestep)
        except Exception:
            # 保守处理，不影响训练主流程
            return
              
    def log_training_step(self,   
                          actor_loss: float,  
                          critic_loss: float,  
                          entropy: float,  
                          learning_rate: float,  
                          **kwargs):  
        """  
        记录训练步信息  
          
        Args:  
            actor_loss: Actor损失  
            critic_loss: Critic损失  
            entropy: 策略熵  
            learning_rate: 当前学习率  
            **kwargs: 其他指标  
        """  
        self.log_scalar("train/actor_loss", actor_loss)  
        self.log_scalar("train/critic_loss", critic_loss)  
        self.log_scalar("train/entropy", entropy)  
        self.log_scalar("train/learning_rate", learning_rate)  
          
        for key, value in kwargs.items():  
            self.log_scalar(f"train/{key}", value)  
              
    def log_evaluation(self, eval_results: Dict[str, float], step: Optional[int] = None):  
        """  
        记录评估结果  
          
        Args:  
            eval_results: 评估结果字典  
            step: 步数  
        """  
        if step is None:  
            step = self.step  
              
        for key, value in eval_results.items():  
            self.log_scalar(f"eval/{key}", value, step)  
              
        # 保存到CSV  
        if self.use_csv:  
            self._write_csv("eval_log", eval_results, step)  
              
        # 打印  
        print("\n" + "=" * 50)  
        print(f"Evaluation at step {step}:")  
        for key, value in eval_results.items():  
            print(f"  {key}: {value:.4f}")  
        print("=" * 50 + "\n")  
          
    def _write_csv(self, filename: str, data: Dict[str, float], step: int):  
        """写入CSV文件"""  
        filepath = os.path.join(self.csv_dir, f"{filename}.csv")  
          
        # 首次写入时创建文件和表头  
        if filename not in self.csv_files:  
            self.csv_files[filename] = open(filepath, 'w', newline='')  
            self.csv_writers[filename] = csv.writer(self.csv_files[filename])  
            # 写表头  
            header = ['step'] + list(data.keys())  
            self.csv_writers[filename].writerow(header)  
              
        # 写入数据  
        row = [step] + list(data.values())  
        self.csv_writers[filename].writerow(row)  
        self.csv_files[filename].flush()  
          
    def _print_episode_summary(self, episode_info: Dict[str, float]):  
        """打印Episode摘要"""  
        elapsed_time = time.time() - self.start_time  
        fps = self.step / elapsed_time if elapsed_time > 0 else 0  
          
        print(f"\n[Episode {self.episode}] Step: {self.step}, Time: {elapsed_time:.1f}s, FPS: {fps:.1f}")  
        for key, value in episode_info.items():  
            if isinstance(value, float):  
                print(f"  {key}: {value:.4f}")  
            else:  
                print(f"  {key}: {value}")  
                  
    def get_history(self, tag: str) -> List[tuple]:  
        """获取指定指标的历史记录"""  
        return self.history.get(tag, [])  
      
    def flush(self):  
        """刷新所有缓冲区"""  
        if self.writer is not None:  
            self.writer.flush()  
        for f in self.csv_files.values():  
            f.flush()  
              
    def close(self):  
        """关闭日志记录器"""  
        if self.writer is not None:  
            self.writer.close()  
        for f in self.csv_files.values():  
            f.close()  
        print(f"Logger closed. Logs saved to {self.log_dir}")  
  
  
class MetricsTracker:  
    """  
    指标追踪器  
      
    用于追踪和计算运行时指标的统计信息  
    """  
      
    def __init__(self, window_size: int = 100):  
        """  
        初始化  
          
        Args:  
            window_size: 滑动窗口大小  
        """  
        self.window_size = window_size  
        self.metrics = defaultdict(list)  
        self.episode_metrics = defaultdict(list)  
          
    def add(self, name: str, value: float):  
        """添加一个值"""  
        self.metrics[name].append(value)  
        # 保持窗口大小  
        if len(self.metrics[name]) > self.window_size * 2:  
            self.metrics[name] = self.metrics[name][-self.window_size:]  
              
    def add_episode_metric(self, name: str, value: float):  
        """添加Episode级别的指标"""  
        self.episode_metrics[name].append(value)  
          
    def get_mean(self, name: str) -> float:  
        """获取平均值"""  
        values = self.metrics.get(name, [])  
        if not values:  
            return 0.0  
        return np.mean(values[-self.window_size:])  
      
    def get_std(self, name: str) -> float:  
        """获取标准差"""  
        values = self.metrics.get(name, [])  
        if not values:  
            return 0.0  
        return np.std(values[-self.window_size:])  
      
    def get_min(self, name: str) -> float:  
        """获取最小值"""  
        values = self.metrics.get(name, [])  
        if not values:  
            return 0.0  
        return np.min(values[-self.window_size:])  
      
    def get_max(self, name: str) -> float:  
        """获取最大值"""  
        values = self.metrics.get(name, [])  
        if not values:  
            return 0.0  
        return np.max(values[-self.window_size:])  
      
    def get_stats(self, name: str) -> Dict[str, float]:  
        """获取完整统计信息"""  
        return {  
            'mean': self.get_mean(name),  
            'std': self.get_std(name),  
            'min': self.get_min(name),  
            'max': self.get_max(name)  
        }  
      
    def get_all_means(self) -> Dict[str, float]:  
        """获取所有指标的平均值"""  
        return {name: self.get_mean(name) for name in self.metrics.keys()}  
      
    def reset(self):  
        """重置所有指标"""  
        self.metrics.clear()  
        self.episode_metrics.clear()  
  
  
class EpisodeTracker:  
    """  
    Episode追踪器  
      
    追踪单个Episode内的信息  
    """  
      
    def __init__(self):  
        self.reset()  
          
    def reset(self):  
        """重置追踪器"""  
        self.rewards = []  
        self.actions = []  
        self.states = []  
        self.infos = []  
        self.step_count = 0  
        self.start_time = time.time()  
          
    def add_step(self, reward: float, action: Any, state: Any = None, info: Dict = None):  
        """记录一步"""  
        self.rewards.append(reward)  
        self.actions.append(action)  
        if state is not None:  
            self.states.append(state)  
        if info is not None:  
            self.infos.append(info)  
        self.step_count += 1  
          
    def get_episode_summary(self) -> Dict[str, float]:  
        """获取Episode摘要"""  
        elapsed_time = time.time() - self.start_time  
          
        summary = {  
            'total_reward': sum(self.rewards),  
            'mean_reward': np.mean(self.rewards) if self.rewards else 0,  
            'episode_length': self.step_count,  
            'elapsed_time': elapsed_time  
        }  
          
        # 从info中提取额外信息  
        if self.infos:  
            # 收集所有info中的数值型字段  
            info_keys = set()  
            for info in self.infos:  
                info_keys.update(info.keys())  
              
            for key in info_keys:  
                values = [info.get(key, 0) for info in self.infos if isinstance(info.get(key, None), (int, float))]  
                if values:  
                    summary[f'mean_{key}'] = np.mean(values)  
                      
        return summary  
      
    def get_action_distribution(self) -> Dict[int, int]:  
        """获取动作分布"""  
        if not self.actions:  
            return {}  
        action_counts = defaultdict(int)  
        for action in self.actions:  
            action_counts[action] += 1  
        return dict(action_counts)  
