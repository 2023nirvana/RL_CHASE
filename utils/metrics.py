"""  
指标计算模块  
============  
  
提供训练和评估所需的各种指标计算功能  
"""  
  
import numpy as np  
from typing import Dict, List, Tuple, Optional  
from collections import deque  
  
  
class RunningMeanStd:  
    """  
    在线计算均值和标准差  
      
    使用Welford算法，数值稳定  
    """  
      
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):  
        """  
        初始化  
          
        Args:  
            shape: 数据形状  
            epsilon: 防止除零的小常数  
        """  
        self.mean = np.zeros(shape, dtype=np.float64)  
        self.var = np.ones(shape, dtype=np.float64)  
        self.count = epsilon  
        self.epsilon = epsilon  
          
    def update(self, x: np.ndarray):  
        """  
        更新统计量  
          
        Args:  
            x: 新的数据批次  
        """  
        batch_mean = np.mean(x, axis=0)  
        batch_var = np.var(x, axis=0)  
        batch_count = x.shape[0]  
          
        self._update_from_moments(batch_mean, batch_var, batch_count)  
          
    def _update_from_moments(self, batch_mean: np.ndarray,   
                             batch_var: np.ndarray,   
                             batch_count: int):  
        """从批次统计量更新"""  
        delta = batch_mean - self.mean  
        tot_count = self.count + batch_count  
          
        new_mean = self.mean + delta * batch_count / tot_count  
        m_a = self.var * self.count  
        m_b = batch_var * batch_count  
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count  
        new_var = M2 / tot_count  
          
        self.mean = new_mean  
        self.var = new_var  
        self.count = tot_count  
          
    @property  
    def std(self) -> np.ndarray:  
        """获取标准差"""  
        return np.sqrt(self.var + self.epsilon)  
      
    def normalize(self, x: np.ndarray) -> np.ndarray:  
        """归一化数据"""  
        return (x - self.mean) / self.std  
      
    def denormalize(self, x: np.ndarray) -> np.ndarray:  
        """反归一化"""  
        return x * self.std + self.mean  
  
  
class RewardNormalizer:  
    """  
    奖励归一化器  
      
    用于稳定训练  
    """  
      
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):  
        """  
        初始化  
          
        Args:  
            gamma: 折扣因子  
            epsilon: 防止除零  
        """  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.running_ms = RunningMeanStd(shape=())  
        self.returns = 0.0  
          
    def normalize(self, reward: float, done: bool) -> float:  
        """  
        归一化奖励  
          
        Args:  
            reward: 原始奖励  
            done: 是否终止  
              
        Returns:  
            归一化后的奖励  
        """  
        self.returns = self.returns * self.gamma + reward  
        self.running_ms.update(np.array([self.returns]))  
          
        normalized = reward / (self.running_ms.std + self.epsilon)  
          
        if done:  
            self.returns = 0.0  
              
        return normalized  
      
    def reset(self):  
        """重置"""  
        self.returns = 0.0  
  
  
class SuccessRateTracker:  
    """  
    成功率追踪器  
    """  
      
    def __init__(self, window_size: int = 100):  
        """  
        初始化  
          
        Args:  
            window_size: 滑动窗口大小  
        """  
        self.window_size = window_size  
        self.successes = deque(maxlen=window_size)  
          
    def add(self, success: bool):  
        """添加一个结果"""  
        self.successes.append(1 if success else 0)  
          
    @property  
    def rate(self) -> float:  
        """计算成功率"""  
        if not self.successes:  
            return 0.0  
        return sum(self.successes) / len(self.successes)  
      
    @property  
    def count(self) -> int:  
        """成功次数"""  
        return sum(self.successes)  
      
    def reset(self):  
        """重置"""  
        self.successes.clear()  
  
  
class PerformanceMetrics:  
    """  
    性能指标计算器  
      
    用于评估AUV控制性能  
    """  
      
    def __init__(self):  
        """初始化"""  
        self.reset()  
          
    def reset(self):  
        """重置所有记录"""  
        self.position_errors = []  
        self.orientation_errors = []  
        self.velocities = []  
        self.control_efforts = []  
        self.settling_times = []  
          
    def add_step(self,   
                 position_error: float,  
                 orientation_error: float,  
                 velocity: np.ndarray,  
                 control_effort: float):  
        """  
        添加一步数据  
          
        Args:  
            position_error: 位置误差  
            orientation_error: 姿态误差  
            velocity: 速度向量  
            control_effort: 控制努力（推力大小）  
        """  
        self.position_errors.append(position_error)  
        self.orientation_errors.append(orientation_error)  
        self.velocities.append(velocity)  
        self.control_efforts.append(control_effort)  
          
    def compute_metrics(self) -> Dict[str, float]:  
        """  
        计算所有指标  
          
        Returns:  
            指标字典  
        """  
        if not self.position_errors:  
            return {}  
              
        position_errors = np.array(self.position_errors)  
        orientation_errors = np.array(self.orientation_errors)  
        velocities = np.array(self.velocities)  
        control_efforts = np.array(self.control_efforts)  
          
        metrics = {  
            # 位置误差指标  
            'mean_position_error': np.mean(position_errors),  
            'max_position_error': np.max(position_errors),  
            'final_position_error': position_errors[-1],  
            'position_rmse': np.sqrt(np.mean(position_errors**2)),  
              
            # 姿态误差指标  
            'mean_orientation_error': np.mean(orientation_errors),  
            'max_orientation_error': np.max(orientation_errors),  
            'final_orientation_error': orientation_errors[-1],  
              
            # 速度指标  
            'mean_speed': np.mean(np.linalg.norm(velocities[:, :3], axis=1)),  
            'max_speed': np.max(np.linalg.norm(velocities[:, :3], axis=1)),  
              
            # 控制指标  
            'mean_control_effort': np.mean(control_efforts),  
            'total_control_effort': np.sum(control_efforts),  
              
            # 稳定性指标  
            'position_variance': np.var(position_errors[-50:]) if len(position_errors) > 50 else np.var(position_errors),  
        }  
          
        # 计算settling time（如果误差收敛）  
        threshold = 0.5  # 0.5米  
        settled_indices = np.where(position_errors < threshold)[0]  
        if len(settled_indices) > 0:  
            metrics['settling_time_idx'] = settled_indices[0]  
        else:  
            metrics['settling_time_idx'] = len(position_errors)  
              
        return metrics  
  
  
class PursuitMetrics:  
    """  
    追逃博弈专用指标  
    """  
      
    def __init__(self, capture_radius: float = 1.0):  
        """  
        初始化  
          
        Args:  
            capture_radius: 捕获半径  
        """  
        self.capture_radius = capture_radius  
        self.reset()  
          
    def reset(self):  
        """重置"""  
        self.pursuer_positions = []  # List of List[np.ndarray]  
        self.evader_positions = []  
        self.capture_time = None  
        self.captured = False  
          
    def add_step(self, pursuer_positions: List[np.ndarray], evader_position: np.ndarray):  
        """  
        添加一步数据  
          
        Args:  
            pursuer_positions: 所有追击者位置  
            evader_position: 逃逸者位置  
        """  
        self.pursuer_positions.append([p.copy() for p in pursuer_positions])  
        self.evader_positions.append(evader_position.copy())  
          
        # 检查是否捕获  
        if not self.captured:  
            for p_pos in pursuer_positions:  
                distance = np.linalg.norm(p_pos - evader_position)  
                if distance < self.capture_radius:  
                    self.captured = True  
                    self.capture_time = len(self.evader_positions)  
                    break  
                      
    def compute_metrics(self) -> Dict[str, float]:  
        """  
        计算追逃指标  
          
        Returns:  
            指标字典  
        """  
        if not self.evader_positions:  
            return {}  
              
        evader_positions = np.array(self.evader_positions)  
        n_steps = len(evader_positions)  
        n_pursuers = len(self.pursuer_positions[0])  
          
        # 计算距离历史  
        distances = []  
        for t in range(n_steps):  
            step_distances = []  
            for i in range(n_pursuers):  
                d = np.linalg.norm(  
                    self.pursuer_positions[t][i] - self.evader_positions[t]  
                )  
                step_distances.append(d)  
            distances.append(step_distances)  
        distances = np.array(distances)  # [n_steps, n_pursuers]  
          
        # 最小距离（任意追击者）  
        min_distances = np.min(distances, axis=1)  
          
        metrics = {  
            'captured': float(self.captured),  
            'capture_time': self.capture_time if self.captured else n_steps,  
            'min_distance': np.min(min_distances),  
            'mean_min_distance': np.mean(min_distances),  
            'final_min_distance': min_distances[-1],  
              
            # 逃逸者行进距离  
            'evader_distance_traveled': self._compute_distance_traveled(evader_positions),  
        }  
          
        # 每个追击者的指标  
        for i in range(n_pursuers):  
            pursuer_positions = np.array([self.pursuer_positions[t][i] for t in range(n_steps)])  
            metrics[f'pursuer_{i}_distance_traveled'] = self._compute_distance_traveled(pursuer_positions)  
            metrics[f'pursuer_{i}_mean_distance'] = np.mean(distances[:, i])  
              
        # 包围程度（追击者的空间分布）  
        if n_pursuers > 1:  
            final_pursuer_positions = [self.pursuer_positions[-1][i] for i in range(n_pursuers)]  
            metrics['encirclement_score'] = self._compute_encirclement(  
                final_pursuer_positions,   
                self.evader_positions[-1]  
            )  
              
        return metrics  
      
    def _compute_distance_traveled(self, positions: np.ndarray) -> float:  
        """计算行进距离"""  
        if len(positions) < 2:  
            return 0.0  
        deltas = np.diff(positions, axis=0)  
        return np.sum(np.linalg.norm(deltas, axis=1))  
      
    def _compute_encirclement(self, pursuer_positions: List[np.ndarray],   
                               evader_position: np.ndarray) -> float:  
        """  
        计算包围程度  
          
        基于追击者相对于逃逸者的角度分布  
        角度分布越均匀，包围程度越高  
        """  
        if len(pursuer_positions) < 2:  
            return 0.0  
              
        # 计算每个追击者相对于逃逸者的角度  
        angles = []  
        for p_pos in pursuer_positions:  
            delta = p_pos[:2] - evader_position[:2]  # 只考虑XY平面  
            angle = np.arctan2(delta[1], delta[0])  
            angles.append(angle)  
              
        angles = np.sort(angles)  
          
        # 计算角度间隔  
        gaps = np.diff(angles)  
        gaps = np.append(gaps, 2*np.pi + angles[0] - angles[-1])  # 首尾间隔  
          
        # 理想情况下，间隔应该均匀 = 2π/n  
        ideal_gap = 2 * np.pi / len(pursuer_positions)  
        gap_variance = np.var(gaps)  
          
        # 归一化得分（方差越小越好）  
        max_variance = (2 * np.pi) ** 2  # 最大可能方差  
        score = 1.0 - min(gap_variance / max_variance, 1.0)  
          
        return score  
  
  
class GAECalculator:  
    """  
    广义优势估计 (GAE) 计算器  
    """  
      
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):  
        """  
        初始化  
          
        Args:  
            gamma: 折扣因子  
            lam: GAE参数  
        """  
        self.gamma = gamma  
        self.lam = lam  
          
    def compute_gae(self,   
                    rewards: np.ndarray,  
                    values: np.ndarray,  
                    dones: np.ndarray,  
                    next_value: float) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        计算GAE和returns  
          
        Args:  
            rewards: 奖励序列 [T]  
            values: 价值估计序列 [T]  
            dones: 终止标志序列 [T]  
            next_value: 最后状态的价值估计  
              
        Returns:  
            advantages: 优势估计 [T]  
            returns: 回报估计 [T]  
        """  
        T = len(rewards)  
        advantages = np.zeros(T)  
        lastgaelam = 0  
          
        for t in reversed(range(T)):  
            if t == T - 1:  
                nextnonterminal = 1.0 - dones[t]  
                nextvalue = next_value  
            else:  
                nextnonterminal = 1.0 - dones[t]  
                nextvalue = values[t + 1]  
                  
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]  
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam  
              
        returns = advantages + values  
          
        return advantages, returns  
      
    def compute_returns(self,   
                        rewards: np.ndarray,  
                        dones: np.ndarray,  
                        next_value: float) -> np.ndarray:  
        """  
        计算折扣回报  
          
        Args:  
            rewards: 奖励序列  
            dones: 终止标志  
            next_value: 最后状态的价值  
              
        Returns:  
            returns: 折扣回报  
        """  
        T = len(rewards)  
        returns = np.zeros(T)  
          
        running_return = next_value  
        for t in reversed(range(T)):  
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])  
            returns[t] = running_return  
              
        return returns  
  
  
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:  
    """  
    计算解释方差  
      
    Args:  
        y_pred: 预测值  
        y_true: 真实值  
          
    Returns:  
        解释方差比例  
    """  
    var_y = np.var(y_true)  
    if var_y == 0:  
        return np.nan  
    return 1 - np.var(y_true - y_pred) / var_y  
