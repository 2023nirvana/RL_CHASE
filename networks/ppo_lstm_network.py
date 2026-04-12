"""  
PPO+LSTM 网络架构  
=================  
  
用于底层AUV控制的Actor-Critic网络。  
  
特点：  
1. 使用LSTM处理时序信息（隐式学习动力学）  
2. 共享特征提取器 + 分离的Actor/Critic头  
3. 支持离散动作空间和连续动作空间  
4. 包含正交初始化等训练技巧  
"""  
  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.distributions import Categorical, Normal  
from typing import Tuple, Optional, Dict, List  
  
  
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:  
    """  
    正交初始化  
      
    对于RL网络特别有效，可以加速训练  
      
    Args:  
        layer: 要初始化的层  
        std: 标准差  
        bias_const: 偏置常数  
          
    Returns:  
        初始化后的层  
    """  
    torch.nn.init.orthogonal_(layer.weight, std)  
    torch.nn.init.constant_(layer.bias, bias_const)  
    return layer  
  
  
class FeatureExtractor(nn.Module):  
    """  
    特征提取器  
      
    将原始观测转换为高级特征  
    """  
      
    def __init__(self,   
                 input_dim: int,  
                 hidden_sizes: List[int] = [256, 256],  
                 activation: str = "relu",  
                 use_layer_norm: bool = True):  
        """  
        初始化特征提取器  
          
        Args:  
            input_dim: 输入维度  
            hidden_sizes: 隐藏层尺寸列表  
            activation: 激活函数  
            use_layer_norm: 是否使用LayerNorm  
        """  
        super().__init__()  
          
        self.input_dim = input_dim  
        self.hidden_sizes = hidden_sizes  
        self.use_layer_norm = use_layer_norm  
          
        # 选择激活函数  
        if activation == "relu":  
            self.activation = nn.ReLU()  
        elif activation == "tanh":  
            self.activation = nn.Tanh()  
        elif activation == "elu":  
            self.activation = nn.ELU()  
        elif activation == "leaky_relu":  
            self.activation = nn.LeakyReLU()  
        else:  
            self.activation = nn.ReLU()  
              
        # 构建网络层  
        layers = []  
        prev_dim = input_dim  
          
        for hidden_size in hidden_sizes:  
            layers.append(layer_init(nn.Linear(prev_dim, hidden_size)))  
            if use_layer_norm:  
                layers.append(nn.LayerNorm(hidden_size))  
            layers.append(self.activation)  
            prev_dim = hidden_size  
              
        self.network = nn.Sequential(*layers)  
        self.output_dim = hidden_sizes[-1] if hidden_sizes else input_dim  
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            x: 输入张量 [batch, input_dim] 或 [seq, batch, input_dim]  
              
        Returns:  
            features: 特征张量  
        """  
        return self.network(x)  
  
  
class LSTMModule(nn.Module):  
    """  
    LSTM模块  
      
    处理时序信息，学习动力学的隐式表示  
    """  
      
    def __init__(self,  
                 input_dim: int,  
                 hidden_size: int = 256,  
                 num_layers: int = 2,  
                 dropout: float = 0.0):  
        """  
        初始化LSTM模块  
          
        Args:  
            input_dim: 输入维度  
            hidden_size: LSTM隐藏层大小  
            num_layers: LSTM层数  
            dropout: Dropout率（仅在num_layers > 1时有效）  
        """  
        super().__init__()  
          
        self.input_dim = input_dim  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
          
        self.lstm = nn.LSTM(  
            input_size=input_dim,  
            hidden_size=hidden_size,  
            num_layers=num_layers,  
            batch_first=False,  # 使用 [seq, batch, feature] 格式  
            dropout=dropout if num_layers > 1 else 0.0  
        )  
          
        # 初始化LSTM权重  
        for name, param in self.lstm.named_parameters():  
            if 'weight_ih' in name:  
                torch.nn.init.orthogonal_(param.data)  
            elif 'weight_hh' in name:  
                torch.nn.init.orthogonal_(param.data)  
            elif 'bias' in name:  
                param.data.fill_(0)  
                # 设置遗忘门偏置为1（有助于长期记忆）  
                n = param.size(0)  
                param.data[n//4:n//2].fill_(1.0)  
                  
    def forward(self,   
                x: torch.Tensor,   
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  
        """  
        前向传播  
          
        Args:  
            x: 输入张量 [seq_len, batch, input_dim]  
            hidden_state: 隐藏状态元组 (h, c)  
              
        Returns:  
            output: LSTM输出 [seq_len, batch, hidden_size]  
            hidden_state: 新的隐藏状态  
        """  
        if hidden_state is None:  
            hidden_state = self.get_initial_state(x.size(1), x.device)  
              
        output, hidden_state = self.lstm(x, hidden_state)  
        return output, hidden_state  
      
    def get_initial_state(self, batch_size: int, device: torch.device  
                          ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """  
        获取初始隐藏状态  
          
        Args:  
            batch_size: 批次大小  
            device: 设备  
              
        Returns:  
            initial_state: (h_0, c_0)  
        """  
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)  
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)  
        return (h_0, c_0)  
  
  
class ActorHead(nn.Module):  
    """  
    Actor头（策略网络）  
      
    输出动作的概率分布  
    """  
      
    def __init__(self,  
                 input_dim: int,  
                 num_actions: int,  
                 hidden_sizes: List[int] = [128]):  
        """  
        初始化Actor头  
          
        Args:  
            input_dim: 输入维度  
            num_actions: 动作数量（离散）  
            hidden_sizes: 隐藏层尺寸  
        """  
        super().__init__()  
          
        self.num_actions = num_actions  
          
        layers = []  
        prev_dim = input_dim  
          
        for hidden_size in hidden_sizes:  
            layers.append(layer_init(nn.Linear(prev_dim, hidden_size)))  
            layers.append(nn.ReLU())  
            prev_dim = hidden_size  
              
        # 输出层（更小的初始化标准差，使初始策略接近均匀）  
        layers.append(layer_init(nn.Linear(prev_dim, num_actions), std=0.01))  
          
        self.network = nn.Sequential(*layers)  
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            x: 输入特征  
              
        Returns:  
            logits: 动作logits  
        """  
        return self.network(x)  
      
    def get_distribution(self, x: torch.Tensor) -> Categorical:  
        """  
        获取动作分布  
          
        Args:  
            x: 输入特征  
              
        Returns:  
            distribution: Categorical分布  
        """  
        logits = self.forward(x)  
        return Categorical(logits=logits)  
  
  
class CriticHead(nn.Module):  
    """  
    Critic头（价值网络）  
      
    输出状态价值估计  
    """  
      
    def __init__(self,  
                 input_dim: int,  
                 hidden_sizes: List[int] = [128]):  
        """  
        初始化Critic头  
          
        Args:  
            input_dim: 输入维度  
            hidden_sizes: 隐藏层尺寸  
        """  
        super().__init__()  
          
        layers = []  
        prev_dim = input_dim  
          
        for hidden_size in hidden_sizes:  
            layers.append(layer_init(nn.Linear(prev_dim, hidden_size)))  
            layers.append(nn.ReLU())  
            prev_dim = hidden_size  
              
        # 输出层（输出单个值）  
        layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))  
          
        self.network = nn.Sequential(*layers)  
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            x: 输入特征  
              
        Returns:  
            value: 状态价值 [batch, 1]  
        """  
        return self.network(x)  


class GaussianActorHead(nn.Module):
    """
    连续动作Actor头（高斯策略）
    
    输出高斯分布的均值和标准差
    """
    
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0
    
    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int] = [128],
                 init_log_std: float = -0.5,
                 state_dependent_std: bool = False):
        """
        初始化连续动作Actor头
        
        Args:
            input_dim: 输入维度
            action_dim: 动作维度
            hidden_sizes: 隐藏层尺寸
            init_log_std: 初始log标准差
            state_dependent_std: 是否使用状态依赖的标准差
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_size)))
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # 均值输出（使用较小的初始化标准差）
        self.mu_head = layer_init(nn.Linear(prev_dim, action_dim), std=0.01)
        
        # 标准差
        if state_dependent_std:
            self.log_std_head = layer_init(nn.Linear(prev_dim, action_dim), std=0.01)
        else:
            # 可学习的独立log_std参数
            self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
        
        Returns:
            mu: 动作均值
            log_std: 动作log标准差
        """
        hidden = self.hidden_layers(x)
        mu = self.mu_head(hidden)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(hidden)
        else:
            log_std = self.log_std.expand_as(mu)
        
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std
    
    def get_distribution(self, x: torch.Tensor) -> Normal:
        """
        获取动作分布
        
        Args:
            x: 输入特征
        
        Returns:
            distribution: Normal分布
        """
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mu, std)
    
    def sample_action(self, x: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作（使用tanh squashing）
        
        Args:
            x: 输入特征
            deterministic: 是否确定性
        
        Returns:
            action: 动作 [-1, 1]
            log_prob: 对数概率
        """
        dist = self.get_distribution(x)
        
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        
        # Tanh squashing
        action = torch.tanh(z)
        
        # 计算log_prob（需要考虑tanh变换的雅可比行列式）
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_action(self, x: torch.Tensor, action: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Args:
            x: 输入特征
            action: 动作 (已经经过tanh squashing)
        
        Returns:
            log_prob: 对数概率
            entropy: 熵估计
        """
        dist = self.get_distribution(x)
        
        # 反tanh（atanh）
        action_clipped = torch.clamp(action, -0.999, 0.999)
        z = 0.5 * (torch.log1p(action_clipped) - torch.log1p(-action_clipped))
        
        # 计算log_prob
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        # 熵估计（使用高斯分布的熵，忽略tanh变换）
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class PPOLSTMNetwork(nn.Module):
    """  
    完整的PPO+LSTM网络  
      
    结构：  
    观测 -> 特征提取器 -> LSTM -> Actor头 -> 动作分布  
                              -> Critic头 -> 状态价值  
    """  
      
    def __init__(self,  
                 obs_dim: int,  
                 num_actions: int,  
                 feature_hidden_sizes: List[int] = [256, 256],  
                 lstm_hidden_size: int = 256,  
                 lstm_num_layers: int = 2,  
                 actor_hidden_sizes: List[int] = [128],  
                 critic_hidden_sizes: List[int] = [128],  
                 encoder_hidden_sizes: Optional[List[int]] = None,  
                 decoder_hidden_sizes: Optional[List[int]] = None,  
                 use_layer_norm: bool = True,  
                 activation: str = "relu"):  
        """  
        初始化PPO+LSTM网络  
          
        Args:  
            obs_dim: 观测维度  
            num_actions: 动作数量  
            feature_hidden_sizes: 特征提取器隐藏层  
            lstm_hidden_size: LSTM隐藏层大小  
            lstm_num_layers: LSTM层数  
            actor_hidden_sizes: Actor头隐藏层  
            critic_hidden_sizes: Critic头隐藏层  
            encoder_hidden_sizes: 可选Encoder隐藏层  
            decoder_hidden_sizes: 可选Decoder隐藏层  
            use_layer_norm: 是否使用LayerNorm  
            activation: 激活函数  
        """  
        super().__init__()  
          
        self.obs_dim = obs_dim  
        self.num_actions = num_actions  
        self.lstm_hidden_size = lstm_hidden_size  
        self.lstm_num_layers = lstm_num_layers  
          
        # 可选Encoder  
        self.encoder = None  
        encoder_input_dim = obs_dim  
        if encoder_hidden_sizes:  
            self.encoder = FeatureExtractor(  
                input_dim=obs_dim,  
                hidden_sizes=encoder_hidden_sizes,  
                activation=activation,  
                use_layer_norm=use_layer_norm  
            )  
            encoder_input_dim = self.encoder.output_dim  
  
        # 特征提取器  
        self.feature_extractor = FeatureExtractor(  
            input_dim=encoder_input_dim,  
            hidden_sizes=feature_hidden_sizes,  
            activation=activation,  
            use_layer_norm=use_layer_norm  
        )  
          
        feature_dim = self.feature_extractor.output_dim  
          
        # LSTM  
        self.lstm = LSTMModule(  
            input_dim=feature_dim,  
            hidden_size=lstm_hidden_size,  
            num_layers=lstm_num_layers  
        )  
          
        # 可选Decoder  
        self.decoder = None  
        decoder_output_dim = lstm_hidden_size  
        if decoder_hidden_sizes:  
            self.decoder = FeatureExtractor(  
                input_dim=lstm_hidden_size,  
                hidden_sizes=decoder_hidden_sizes,  
                activation=activation,  
                use_layer_norm=use_layer_norm  
            )  
            decoder_output_dim = self.decoder.output_dim  
  
        # Actor头  
        self.actor = ActorHead(  
            input_dim=decoder_output_dim,  
            num_actions=num_actions,  
            hidden_sizes=actor_hidden_sizes  
        )  
          
        # Critic头  
        self.critic = CriticHead(  
            input_dim=decoder_output_dim,  
            hidden_sizes=critic_hidden_sizes  
        )  
          
    def forward(self,  
                obs: torch.Tensor,  
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
                done: Optional[torch.Tensor] = None  
                ) -> Tuple[Categorical, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  
        """  
        前向传播  
        
        Args:  
            obs: 观测 [seq_len, batch, obs_dim] 或 [batch, obs_dim] 或 [obs_dim]  
            hidden_state: LSTM隐藏状态  
            done: 终止标志（用于重置隐藏状态）  
            
        Returns:  
            action_dist: 动作分布  
            value: 状态价值  
            hidden_state: 新的隐藏状态  
        """  
        # 处理不同维度的输入  
        if obs.dim() == 1:  
            # [obs_dim] -> [1, 1, obs_dim]  
            obs = obs.unsqueeze(0).unsqueeze(0)  
            batch_size = 1  
        elif obs.dim() == 2:  
            # [batch, obs_dim] -> [1, batch, obs_dim]  
            obs = obs.unsqueeze(0)  
            batch_size = obs.shape[1]  
        else:  
            # [seq_len, batch, obs_dim]  
            batch_size = obs.shape[1]  
            
        seq_len = obs.shape[0]  
        
        # 可选Encoder  
        if self.encoder is not None:  
            obs = self.encoder(obs)  
  
        # 特征提取  
        features = self.feature_extractor(obs)  # [seq_len, batch, feature_dim]  
        
        # 处理done标志（重置已结束episode的隐藏状态）  
        if hidden_state is None:  
            hidden_state = self.lstm.get_initial_state(batch_size, obs.device)  
        elif done is not None:  
            # 确保done的形状正确  
            if done.dim() == 0:  
                done = done.unsqueeze(0)  
            if done.dim() == 1:  
                done = done.view(1, -1, 1)  
            # 将done扩展为正确的形状  
            done_mask = done.float()  
            hidden_state = (  
                hidden_state[0] * (1 - done_mask),  
                hidden_state[1] * (1 - done_mask)  
            )  
            
        # LSTM处理  
        lstm_out, new_hidden_state = self.lstm(features, hidden_state)  
        
        # 取最后一个时间步的输出  
        lstm_out_last = lstm_out[-1]  # [batch, lstm_hidden_size]  
        
        # 可选Decoder  
        if self.decoder is not None:  
            lstm_out_last = self.decoder(lstm_out_last)  
  
        # Actor和Critic  
        action_dist = self.actor.get_distribution(lstm_out_last)  
        value = self.critic(lstm_out_last)  
        
        return action_dist, value.squeeze(-1), new_hidden_state  

      
    def get_action_and_value(self,  
                            obs: torch.Tensor,  
                            hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
                            done: Optional[torch.Tensor] = None,  
                            action: Optional[torch.Tensor] = None  
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,   
                                    Tuple[torch.Tensor, torch.Tensor]]:  
        """  
        获取动作和价值（用于训练）  
        
        Args:  
            obs: 观测  
            hidden_state: 隐藏状态  
            done: 终止标志  
            action: 已知动作（用于计算log_prob）  
            
        Returns:  
            action: 采样的动作  
            log_prob: 动作的对数概率  
            entropy: 策略熵  
            value: 状态价值  
            hidden_state: 新的隐藏状态  
        """  
        action_dist, value, new_hidden_state = self.forward(obs, hidden_state, done)  
        
        if action is None:  
            action = action_dist.sample()  
            
        log_prob = action_dist.log_prob(action)  
        entropy = action_dist.entropy()  
        
        return action, log_prob, entropy, value, new_hidden_state  

      
    def get_value(self,  
                obs: torch.Tensor,  
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
                done: Optional[torch.Tensor] = None) -> torch.Tensor:  
        """  
        仅获取状态价值  
        
        Args:  
            obs: 观测  
            hidden_state: 隐藏状态  
            done: 终止标志  
            
        Returns:  
            value: 状态价值  
        """  
        _, value, _ = self.forward(obs, hidden_state, done)  
        return value  

      
    def evaluate_actions(self,  
                         obs: torch.Tensor,  
                         actions: torch.Tensor,  
                         hidden_states: Tuple[torch.Tensor, torch.Tensor],  
                         dones: torch.Tensor  
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        评估给定动作（用于PPO更新）  
          
        处理整个序列，适用于训练时的批量评估  
          
        Args:  
            obs: 观测序列 [seq_len, batch, obs_dim]  
            actions: 动作序列 [seq_len, batch]  
            hidden_states: 初始隐藏状态  
            dones: 终止标志 [seq_len, batch]  
              
        Returns:  
            log_probs: 动作对数概率 [seq_len, batch]  
            values: 状态价值 [seq_len, batch]  
            entropies: 策略熵 [seq_len, batch]  
        """  
        seq_len, batch_size, _ = obs.shape  
          
        # 可选Encoder  
        if self.encoder is not None:  
            obs = self.encoder(obs)  
  
        # 特征提取  
        features = self.feature_extractor(obs)  
          
        # 逐步处理LSTM（需要正确处理done标志）  
        lstm_outputs = []  
        h, c = hidden_states  
          
        for t in range(seq_len):  
            # 重置已结束episode的隐藏状态  
            if t > 0:  
                done_mask = dones[t-1].view(1, -1, 1)  
                h = h * (1 - done_mask)  
                c = c * (1 - done_mask)  
                  
            # 单步LSTM  
            lstm_out, (h, c) = self.lstm(features[t:t+1], (h, c))  
            lstm_outputs.append(lstm_out)  
              
        lstm_outputs = torch.cat(lstm_outputs, dim=0)  # [seq_len, batch, hidden_size]  
          
        # 计算所有时间步的log_prob、value和entropy  
        log_probs = []  
        values = []  
        entropies = []  
          
        for t in range(seq_len):  
            lstm_step = lstm_outputs[t]  
            if self.decoder is not None:  
                lstm_step = self.decoder(lstm_step)  
            action_dist = self.actor.get_distribution(lstm_step)  
            value = self.critic(lstm_step)  
              
            log_probs.append(action_dist.log_prob(actions[t]))  
            values.append(value.squeeze(-1))  
            entropies.append(action_dist.entropy())  
              
        log_probs = torch.stack(log_probs)  # [seq_len, batch]  
        values = torch.stack(values)  # [seq_len, batch]  
        entropies = torch.stack(entropies)  # [seq_len, batch]  
          
        return log_probs, values, entropies  
      
    def get_hidden_state_shape(self) -> Tuple[int, int]:  
        """获取隐藏状态形状"""  
        return (self.lstm_num_layers, self.lstm_hidden_size)  
      
    def save(self, path: str):  
        """保存模型"""  
        torch.save({  
            'state_dict': self.state_dict(),  
            'config': {  
                'obs_dim': self.obs_dim,  
                'num_actions': self.num_actions,  
                'lstm_hidden_size': self.lstm_hidden_size,  
                'lstm_num_layers': self.lstm_num_layers  
            }  
        }, path)  
          
    @classmethod  
    def load(cls, path: str, device: torch.device = None) -> 'PPOLSTMNetwork':  
        """加载模型"""  
        checkpoint = torch.load(path, map_location=device)  
        config = checkpoint['config']  
          
        model = cls(  
            obs_dim=config['obs_dim'],  
            num_actions=config['num_actions'],  
            lstm_hidden_size=config['lstm_hidden_size'],  
            lstm_num_layers=config['lstm_num_layers']  
        )  
        model.load_state_dict(checkpoint['state_dict'])  
          
        if device is not None:  
            model = model.to(device)  
              
        return model  
  
  
class PPONetwork(nn.Module):  
    """  
    无LSTM的PPO网络（用于对比和简单任务）  
      
    直接使用MLP，不处理时序信息  
    """  
      
    def __init__(self,  
                 obs_dim: int,  
                 num_actions: int,  
                 hidden_sizes: List[int] = [256, 256],  
                 use_layer_norm: bool = True,  
                 activation: str = "relu"):  
        """  
        初始化  
          
        Args:  
            obs_dim: 观测维度  
            num_actions: 动作数量  
            hidden_sizes: 隐藏层尺寸  
            use_layer_norm: 是否使用LayerNorm  
            activation: 激活函数  
        """  
        super().__init__()  
          
        self.obs_dim = obs_dim  
        self.num_actions = num_actions  
          
        # 共享特征提取器  
        self.feature_extractor = FeatureExtractor(  
            input_dim=obs_dim,  
            hidden_sizes=hidden_sizes,  
            activation=activation,  
            use_layer_norm=use_layer_norm  
        )  
          
        feature_dim = self.feature_extractor.output_dim  
          
        # Actor头  
        self.actor = ActorHead(  
            input_dim=feature_dim,  
            num_actions=num_actions,  
            hidden_sizes=[128]  
        )  
          
        # Critic头  
        self.critic = CriticHead(  
            input_dim=feature_dim,  
            hidden_sizes=[128]  
        )  
          
    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:  
        """  
        前向传播  
          
        Args:  
            obs: 观测 [batch, obs_dim]  
              
        Returns:  
            action_dist: 动作分布  
            value: 状态价值  
        """  
        features = self.feature_extractor(obs)  
        action_dist = self.actor.get_distribution(features)  
        value = self.critic(features)  
        return action_dist, value.squeeze(-1)  
      
    def get_action_and_value(self,  
                             obs: torch.Tensor,  
                             action: Optional[torch.Tensor] = None  
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        获取动作和价值  
          
        Args:  
            obs: 观测  
            action: 已知动作  
              
        Returns:  
            action: 动作  
            log_prob: 对数概率  
            entropy: 熵  
            value: 价值  
        """  
        action_dist, value = self.forward(obs)  
          
        if action is None:  
            action = action_dist.sample()  
              
        log_prob = action_dist.log_prob(action)  
        entropy = action_dist.entropy()  
          
        return action, log_prob, entropy, value  
      
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:  
        """获取价值"""  
        features = self.feature_extractor(obs)  
        return self.critic(features).squeeze(-1)  


class PPOLSTMNetworkContinuous(nn.Module):
    """
    连续动作版本的PPO+LSTM网络
    
    结构：
    观测 -> 特征提取器 -> LSTM -> GaussianActor头 -> 连续动作分布
                              -> Critic头 -> 状态价值
    
    适用于底层连续控制任务，可以学习时序动力学特征。
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 feature_hidden_sizes: List[int] = [128, 128],
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 1,
                 actor_hidden_sizes: List[int] = [64],
                 critic_hidden_sizes: List[int] = [64],
                 use_layer_norm: bool = False,
                 activation: str = "relu",
                 init_log_std: float = -0.5,
                 state_dependent_std: bool = False):
        """
        初始化连续动作PPO+LSTM网络
        
        Args:
            obs_dim: 观测维度
            action_dim: 连续动作维度
            feature_hidden_sizes: 特征提取器隐藏层
            lstm_hidden_size: LSTM隐藏层大小
            lstm_num_layers: LSTM层数
            actor_hidden_sizes: Actor头隐藏层
            critic_hidden_sizes: Critic头隐藏层
            use_layer_norm: 是否使用LayerNorm
            activation: 激活函数
            init_log_std: 初始log标准差
            state_dependent_std: 是否使用状态依赖的标准差
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=obs_dim,
            hidden_sizes=feature_hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # LSTM
        self.lstm = LSTMModule(
            input_dim=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers
        )
        
        # 连续动作Actor头
        self.actor = GaussianActorHead(
            input_dim=lstm_hidden_size,
            action_dim=action_dim,
            hidden_sizes=actor_hidden_sizes,
            init_log_std=init_log_std,
            state_dependent_std=state_dependent_std
        )
        
        # Critic头
        self.critic = CriticHead(
            input_dim=lstm_hidden_size,
            hidden_sizes=critic_hidden_sizes
        )
    
    def forward(self,
                obs: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                done: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            obs: 观测 [seq_len, batch, obs_dim] 或 [batch, obs_dim] 或 [obs_dim]
            hidden_state: LSTM隐藏状态
            done: 终止标志（用于重置隐藏状态）
        
        Returns:
            mu: 动作均值
            log_std: 动作log标准差
            value: 状态价值
            hidden_state: 新的隐藏状态
        """
        # 处理不同维度的输入
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
            batch_size = 1
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0)
            batch_size = obs.shape[1]
        else:
            batch_size = obs.shape[1]
        
        # 特征提取
        features = self.feature_extractor(obs)
        
        # 处理隐藏状态
        if hidden_state is None:
            hidden_state = self.lstm.get_initial_state(batch_size, obs.device)
        elif done is not None:
            if done.dim() == 0:
                done = done.unsqueeze(0)
            if done.dim() == 1:
                done = done.view(1, -1, 1)
            done_mask = done.float()
            hidden_state = (
                hidden_state[0] * (1 - done_mask),
                hidden_state[1] * (1 - done_mask)
            )
        
        # LSTM处理
        lstm_out, new_hidden_state = self.lstm(features, hidden_state)
        lstm_out_last = lstm_out[-1]
        
        # Actor和Critic
        mu, log_std = self.actor(lstm_out_last)
        value = self.critic(lstm_out_last)
        
        return mu, log_std, value.squeeze(-1), new_hidden_state
    
    def get_action(self,
                   obs: torch.Tensor,
                   hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   done: Optional[torch.Tensor] = None,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                              Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取动作
        
        Args:
            obs: 观测
            hidden_state: 隐藏状态
            done: 终止标志
            deterministic: 是否确定性
        
        Returns:
            action: 动作 [-1, 1]
            log_prob: 对数概率
            entropy: 熵估计
            value: 状态价值
            hidden_state: 新的隐藏状态
        """
        mu, log_std, value, new_hidden_state = self.forward(obs, hidden_state, done)
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        if deterministic:
            z = mu
        else:
            z = dist.rsample()
        
        # Tanh squashing
        action = torch.tanh(z)
        
        # 计算log_prob
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        # 熵估计
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value, new_hidden_state
    
    def get_value(self,
                  obs: torch.Tensor,
                  hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  done: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅获取状态价值"""
        _, _, value, _ = self.forward(obs, hidden_state, done)
        return value
    
    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         hidden_states: Tuple[torch.Tensor, torch.Tensor],
                         dones: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于PPO更新）
        
        处理整个序列，适用于训练时的批量评估
        
        Args:
            obs: 观测序列 [seq_len, batch, obs_dim]
            actions: 动作序列 [seq_len, batch, action_dim]
            hidden_states: 初始隐藏状态
            dones: 终止标志 [seq_len, batch]
        
        Returns:
            log_probs: 动作对数概率 [seq_len, batch]
            values: 状态价值 [seq_len, batch]
            entropies: 策略熵 [seq_len, batch]
        """
        seq_len, batch_size, _ = obs.shape
        
        # 特征提取
        features = self.feature_extractor(obs)
        
        # 逐步处理LSTM
        lstm_outputs = []
        h, c = hidden_states
        
        for t in range(seq_len):
            if t > 0:
                done_mask = dones[t-1].view(1, -1, 1)
                h = h * (1 - done_mask)
                c = c * (1 - done_mask)
            
            lstm_out, (h, c) = self.lstm(features[t:t+1], (h, c))
            lstm_outputs.append(lstm_out)
        
        lstm_outputs = torch.cat(lstm_outputs, dim=0)
        
        # 计算所有时间步的log_prob、value和entropy
        log_probs = []
        values = []
        entropies = []
        
        for t in range(seq_len):
            lstm_step = lstm_outputs[t]
            
            # Actor评估
            log_prob, entropy = self.actor.evaluate_action(lstm_step, actions[t])
            value = self.critic(lstm_step)
            
            log_probs.append(log_prob)
            values.append(value.squeeze(-1))
            entropies.append(entropy)
        
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)
        
        return log_probs, values, entropies
    
    def get_hidden_state_shape(self) -> Tuple[int, int]:
        """获取隐藏状态形状"""
        return (self.lstm_num_layers, self.lstm_hidden_size)
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取初始隐藏状态"""
        return self.lstm.get_initial_state(batch_size, device)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'lstm_hidden_size': self.lstm_hidden_size,
                'lstm_num_layers': self.lstm_num_layers
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'PPOLSTMNetworkContinuous':
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            lstm_hidden_size=config['lstm_hidden_size'],
            lstm_num_layers=config['lstm_num_layers']
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model
