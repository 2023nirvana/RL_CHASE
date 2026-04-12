"""  
经验回放缓冲区  
==============  
  
用于存储和采样训练数据，支持LSTM的序列化存储。  
"""  
  
import numpy as np  
import torch  
from typing import Dict, Tuple, Optional, List, Generator  
from dataclasses import dataclass  
  
  
@dataclass  
class Transition:  
    """单步转换数据"""  
    observation: np.ndarray  
    action: int  
    reward: float  
    next_observation: np.ndarray  
    done: bool  
    value: float  
    log_prob: float  
    hidden_state: Optional[Tuple[np.ndarray, np.ndarray]] = None  
  
  
class RolloutBuffer:  
    """  
    Rollout缓冲区  
      
    用于存储on-policy数据，支持GAE计算  
    """  
      
    def __init__(self,  
                 buffer_size: int,  
                 observation_dim: int,  
                 num_envs: int = 1,  
                 gamma: float = 0.99,  
                 gae_lambda: float = 0.95,  
                 device: torch.device = torch.device("cpu")):  
        """  
        初始化缓冲区  
          
        Args:  
            buffer_size: 缓冲区大小（每个环境的步数）  
            observation_dim: 观测维度  
            num_envs: 并行环境数量  
            gamma: 折扣因子  
            gae_lambda: GAE参数  
            device: 计算设备  
        """  
        self.buffer_size = buffer_size  
        self.observation_dim = observation_dim  
        self.num_envs = num_envs  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.device = device  
          
        # 初始化存储  
        self.reset()  
          
    def reset(self):  
        """重置缓冲区"""  
        self.observations = np.zeros(  
            (self.buffer_size, self.num_envs, self.observation_dim),   
            dtype=np.float32  
        )  
        self.actions = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.int64  
        )  
        self.rewards = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
        self.dones = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
        self.values = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
        self.log_probs = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
          
        # GAE计算结果  
        self.advantages = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
        self.returns = np.zeros(  
            (self.buffer_size, self.num_envs),   
            dtype=np.float32  
        )  
          
        self.pos = 0  
        self.full = False  
          
    def add(self,  
            observation: np.ndarray,  
            action: np.ndarray,  
            reward: np.ndarray,  
            done: np.ndarray,  
            value: np.ndarray,  
            log_prob: np.ndarray):  
        """  
        添加一步数据  
          
        Args:  
            observation: 观测 [num_envs, obs_dim]  
            action: 动作 [num_envs]  
            reward: 奖励 [num_envs]  
            done: 终止标志 [num_envs]  
            value: 价值估计 [num_envs]  
            log_prob: 对数概率 [num_envs]  
        """  
        self.observations[self.pos] = observation  
        self.actions[self.pos] = action  
        self.rewards[self.pos] = reward  
        self.dones[self.pos] = done  
        self.values[self.pos] = value  
        self.log_probs[self.pos] = log_prob  
          
        self.pos += 1  
        if self.pos == self.buffer_size:  
            self.full = True  
              
    def compute_returns_and_advantages(self, last_value, last_done):  
        """  
        计算GAE和returns  
        
        Args:  
            last_value: 最后状态的价值估计 [num_envs]  
            last_done: 最后状态的终止标志 [num_envs]  
        """  
        # 确保输入是 numpy array，避免类型问题  
        if isinstance(last_value, torch.Tensor):  
            last_value = last_value.detach().cpu().numpy()  
        if hasattr(last_value, '__array__'):  
            last_value = np.asarray(last_value, dtype=np.float32)  
        else:  
            last_value = np.array(last_value, dtype=np.float32)  
        
        if isinstance(last_done, torch.Tensor):  
            last_done = last_done.detach().cpu().numpy()  
        if hasattr(last_done, '__array__'):  
            last_done = np.asarray(last_done, dtype=np.float32)  
        else:  
            last_done = np.array(last_done, dtype=np.float32)  
        
        # 确保形状正确  
        last_value = last_value.flatten()  
        last_done = last_done.flatten()  
        
        # GAE计算  
        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)  
        
        for step in reversed(range(self.buffer_size)):  
            if step == self.buffer_size - 1:  
                next_non_terminal = 1.0 - last_done  
                next_value = last_value.copy()  
            else:  
                next_non_terminal = 1.0 - self.dones[step + 1]  
                next_value = self.values[step + 1].copy()  
            
            delta = (self.rewards[step] +   
                    float(self.gamma) * next_value * next_non_terminal -   
                    self.values[step])  
            
            last_gae_lam = (delta +   
                        float(self.gamma) * float(self.gae_lambda) * next_non_terminal * last_gae_lam)  
            
            self.advantages[step] = last_gae_lam  
            
        self.returns = self.advantages + self.values  
          
    def get(self, batch_size: Optional[int] = None) -> Generator[Dict[str, torch.Tensor], None, None]:  
        """  
        获取训练数据的生成器  
          
        Args:  
            batch_size: 批次大小，None表示全部数据  
              
        Yields:  
            batch: 数据字典  
        """  
        # 展平数据  
        total_size = self.buffer_size * self.num_envs  
          
        observations = self.observations.reshape(total_size, -1)  
        actions = self.actions.reshape(total_size)  
        values = self.values.reshape(total_size)  
        log_probs = self.log_probs.reshape(total_size)  
        advantages = self.advantages.reshape(total_size)  
        returns = self.returns.reshape(total_size)  
          
        # 归一化优势  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  
          
        if batch_size is None:  
            batch_size = total_size  
              
        # 随机打乱索引  
        indices = np.random.permutation(total_size)  
          
        # 生成批次  
        start_idx = 0  
        while start_idx < total_size:  
            batch_indices = indices[start_idx:start_idx + batch_size]  
              
            yield {  
                'observations': torch.tensor(observations[batch_indices], device=self.device),  
                'actions': torch.tensor(actions[batch_indices], device=self.device),  
                'old_values': torch.tensor(values[batch_indices], device=self.device),  
                'old_log_probs': torch.tensor(log_probs[batch_indices], device=self.device),  
                'advantages': torch.tensor(advantages[batch_indices], device=self.device),  
                'returns': torch.tensor(returns[batch_indices], device=self.device),  
            }  
              
            start_idx += batch_size  
  
  
class RolloutBufferWithLSTM(RolloutBuffer):  
    """  
    支持LSTM的Rollout缓冲区  
      
    存储隐藏状态，支持序列化训练  
    """  
      
    def __init__(self,  
                 buffer_size: int,  
                 observation_dim: int,  
                 num_envs: int = 1,  
                 lstm_hidden_size: int = 256,  
                 lstm_num_layers: int = 2,  
                 gamma: float = 0.99,  
                 gae_lambda: float = 0.95,  
                 device: torch.device = torch.device("cpu")):  
        """  
        初始化  
          
        Args:  
            buffer_size: 缓冲区大小  
            observation_dim: 观测维度  
            num_envs: 环境数量  
            lstm_hidden_size: LSTM隐藏层大小  
            lstm_num_layers: LSTM层数  
            gamma: 折扣因子  
            gae_lambda: GAE参数  
            device: 设备  
        """  
        self.lstm_hidden_size = lstm_hidden_size  
        self.lstm_num_layers = lstm_num_layers  
          
        super().__init__(  
            buffer_size=buffer_size,  
            observation_dim=observation_dim,  
            num_envs=num_envs,  
            gamma=gamma,  
            gae_lambda=gae_lambda,  
            device=device  
        )  
          
    def reset(self):  
        """重置缓冲区"""  
        super().reset()  
          
        # LSTM隐藏状态存储  
        self.hidden_states_h = np.zeros(  
            (self.buffer_size, self.lstm_num_layers, self.num_envs, self.lstm_hidden_size),  
            dtype=np.float32  
        )  
        self.hidden_states_c = np.zeros(  
            (self.buffer_size, self.lstm_num_layers, self.num_envs, self.lstm_hidden_size),  
            dtype=np.float32  
        )  
          
        # 记录每个episode的起始位置  
        self.episode_starts = np.ones((self.buffer_size, self.num_envs), dtype=np.float32)  
          
    def add(self,  
            observation: np.ndarray,  
            action: np.ndarray,  
            reward: np.ndarray,  
            done: np.ndarray,  
            value: np.ndarray,  
            log_prob: np.ndarray,  
            hidden_state: Tuple[np.ndarray, np.ndarray],  
            episode_start: np.ndarray):  
        """  
        添加一步数据  
          
        Args:  
            observation: 观测  
            action: 动作  
            reward: 奖励  
            done: 终止标志  
            value: 价值估计  
            log_prob: 对数概率  
            hidden_state: LSTM隐藏状态 (h, c)  
            episode_start: episode开始标志  
        """  
        self.observations[self.pos] = observation  
        self.actions[self.pos] = action  
        self.rewards[self.pos] = reward  
        self.dones[self.pos] = done  
        self.values[self.pos] = value  
        self.log_probs[self.pos] = log_prob  
        self.hidden_states_h[self.pos] = hidden_state[0]  
        self.hidden_states_c[self.pos] = hidden_state[1]  
        self.episode_starts[self.pos] = episode_start  
          
        self.pos += 1  
        if self.pos == self.buffer_size:  
            self.full = True  
              
    def get_sequential(self,   
                       seq_length: int = 16,  
                       batch_size: int = 32) -> Generator[Dict[str, torch.Tensor], None, None]:  
        """  
        获取序列化的训练数据  
          
        用于LSTM训练，保持时序结构  
          
        Args:  
            seq_length: 序列长度  
            batch_size: 批次大小（序列数量）  
              
        Yields:  
            batch: 序列化数据字典  
        """  
        # 找到有效的序列起始点  
        # 避免跨越episode边界  
        valid_starts = []  
          
        for env_idx in range(self.num_envs):  
            for start in range(self.buffer_size - seq_length + 1):  
                # 检查这个序列是否跨越episode边界  
                # 如果中间有done=True，则序列不连续  
                # 但我们允许，因为会在评估时处理done标志  
                valid_starts.append((start, env_idx))  
                  
        np.random.shuffle(valid_starts)  
          
        # 生成批次  
        for batch_start in range(0, len(valid_starts), batch_size):  
            batch_indices = valid_starts[batch_start:batch_start + batch_size]  
              
            if len(batch_indices) < batch_size // 2:  # 跳过太小的批次  
                continue  
                  
            actual_batch_size = len(batch_indices)  
              
            # 收集序列数据  
            obs_batch = np.zeros((seq_length, actual_batch_size, self.observation_dim), dtype=np.float32)  
            action_batch = np.zeros((seq_length, actual_batch_size), dtype=np.int64)  
            value_batch = np.zeros((seq_length, actual_batch_size), dtype=np.float32)  
            log_prob_batch = np.zeros((seq_length, actual_batch_size), dtype=np.float32)  
            advantage_batch = np.zeros((seq_length, actual_batch_size), dtype=np.float32)  
            return_batch = np.zeros((seq_length, actual_batch_size), dtype=np.float32)  
            done_batch = np.zeros((seq_length, actual_batch_size), dtype=np.float32)  
              
            # 初始隐藏状态  
            initial_h = np.zeros((self.lstm_num_layers, actual_batch_size, self.lstm_hidden_size), dtype=np.float32)  
            initial_c = np.zeros((self.lstm_num_layers, actual_batch_size, self.lstm_hidden_size), dtype=np.float32)  
              
            for batch_idx, (start, env_idx) in enumerate(batch_indices):  
                end = start + seq_length  
                  
                obs_batch[:, batch_idx] = self.observations[start:end, env_idx]  
                action_batch[:, batch_idx] = self.actions[start:end, env_idx]  
                value_batch[:, batch_idx] = self.values[start:end, env_idx]  
                log_prob_batch[:, batch_idx] = self.log_probs[start:end, env_idx]  
                advantage_batch[:, batch_idx] = self.advantages[start:end, env_idx]  
                return_batch[:, batch_idx] = self.returns[start:end, env_idx]  
                done_batch[:, batch_idx] = self.dones[start:end, env_idx]  
                  
                # 使用序列开始时的隐藏状态  
                initial_h[:, batch_idx] = self.hidden_states_h[start, :, env_idx]  
                initial_c[:, batch_idx] = self.hidden_states_c[start, :, env_idx]  
                  
            # 归一化优势  
            advantages_flat = advantage_batch.flatten()  
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)  
            advantage_batch = advantages_flat.reshape(seq_length, actual_batch_size)  
              
            yield {  
                'observations': torch.tensor(obs_batch, device=self.device),  
                'actions': torch.tensor(action_batch, device=self.device),  
                'old_values': torch.tensor(value_batch, device=self.device),  
                'old_log_probs': torch.tensor(log_prob_batch, device=self.device),  
                'advantages': torch.tensor(advantage_batch, device=self.device),  
                'returns': torch.tensor(return_batch, device=self.device),  
                'dones': torch.tensor(done_batch, device=self.device),  
                'initial_hidden': (  
                    torch.tensor(initial_h, device=self.device),  
                    torch.tensor(initial_c, device=self.device)  
                )  
            }  
  
  
class EpisodeBuffer:  
    """  
    Episode缓冲区  
      
    按episode存储完整轨迹  
    """  
      
    def __init__(self, max_episodes: int = 100):  
        """  
        初始化  
          
        Args:  
            max_episodes: 最大存储episode数  
        """  
        self.max_episodes = max_episodes  
        self.episodes = []  
          
    def add_episode(self, episode: Dict[str, np.ndarray]):  
        """  
        添加一个完整的episode  
          
        Args:  
            episode: episode数据字典  
        """  
        self.episodes.append(episode)  
          
        # 如果超出容量，移除最旧的  
        if len(self.episodes) > self.max_episodes:  
            self.episodes.pop(0)  
              
    def sample_episodes(self, num_episodes: int) -> List[Dict[str, np.ndarray]]:  
        """  
        随机采样episodes  
          
        Args:  
            num_episodes: 采样数量  
              
        Returns:  
            episodes: episode列表  
        """  
        indices = np.random.choice(  
            len(self.episodes),   
            size=min(num_episodes, len(self.episodes)),  
            replace=False  
        )  
        return [self.episodes[i] for i in indices]  
      
    def get_statistics(self) -> Dict[str, float]:  
        """获取统计信息"""  
        if not self.episodes:  
            return {}  
              
        episode_lengths = [len(ep['rewards']) for ep in self.episodes]  
        episode_rewards = [ep['rewards'].sum() for ep in self.episodes]  
          
        return {  
            'mean_episode_length': np.mean(episode_lengths),  
            'mean_episode_reward': np.mean(episode_rewards),  
            'max_episode_reward': np.max(episode_rewards),  
            'min_episode_reward': np.min(episode_rewards),  
            'num_episodes': len(self.episodes)  
        }  
      
    def clear(self):  
        """清空缓冲区"""  
        self.episodes = []  
