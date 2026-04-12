"""  
Actor-Critic 辅助模块  
=====================  
  
提供额外的网络组件和工具  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from typing import Tuple, Optional, Dict, List  
  
  
class NoisyLinear(nn.Module):  
    """  
    带噪声的线性层  
      
    用于探索（NoisyNet方法）  
    """  
      
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):  
        super().__init__()  
          
        self.in_features = in_features  
        self.out_features = out_features  
        self.std_init = std_init  
          
        # 学习的参数  
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))  
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))  
        self.bias_mu = nn.Parameter(torch.empty(out_features))  
        self.bias_sigma = nn.Parameter(torch.empty(out_features))  
          
        # 噪声缓冲（不需要梯度）  
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))  
        self.register_buffer('bias_epsilon', torch.empty(out_features))  
          
        self.reset_parameters()  
        self.reset_noise()  
          
    def reset_parameters(self):  
        """重置参数"""  
        mu_range = 1 / np.sqrt(self.in_features)  
        self.weight_mu.data.uniform_(-mu_range, mu_range)  
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))  
        self.bias_mu.data.uniform_(-mu_range, mu_range)  
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))  
          
    def reset_noise(self):  
        """重置噪声"""  
        epsilon_in = self._scale_noise(self.in_features)  
        epsilon_out = self._scale_noise(self.out_features)  
          
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))  
        self.bias_epsilon.copy_(epsilon_out)  
          
    def _scale_noise(self, size: int) -> torch.Tensor:  
        """生成缩放后的噪声"""  
        x = torch.randn(size, device=self.weight_mu.device)  
        return x.sign().mul_(x.abs().sqrt_())  
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """前向传播"""  
        if self.training:  
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon  
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon  
        else:  
            weight = self.weight_mu  
            bias = self.bias_mu  
              
        return F.linear(x, weight, bias)  
  
  
class AttentionModule(nn.Module):  
    """  
    自注意力模块  
      
    可选的，用于处理多智能体观测  
    """  
      
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):  
        super().__init__()  
          
        self.attention = nn.MultiheadAttention(  
            embed_dim=embed_dim,  
            num_heads=num_heads,  
            dropout=dropout,  
            batch_first=True  
        )  
          
        self.norm = nn.LayerNorm(embed_dim)  
          
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            x: 输入 [batch, seq_len, embed_dim]  
            mask: 注意力掩码  
              
        Returns:  
            output: 输出  
        """  
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)  
        return self.norm(x + attn_out)  
  
  
class RunningMeanStdNorm(nn.Module):  
    """  
    运行时均值标准差归一化  
      
    用于观测归一化  
    """  
      
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):  
        super().__init__()  
          
        self.epsilon = epsilon  
          
        self.register_buffer('mean', torch.zeros(shape))  
        self.register_buffer('var', torch.ones(shape))  
        self.register_buffer('count', torch.tensor(epsilon))  
          
    def update(self, x: torch.Tensor):  
        """更新统计量"""  
        with torch.no_grad():  
            batch_mean = x.mean(dim=0)  
            batch_var = x.var(dim=0)  
            batch_count = x.shape[0]  
              
            delta = batch_mean - self.mean  
            tot_count = self.count + batch_count  
              
            new_mean = self.mean + delta * batch_count / tot_count  
            m_a = self.var * self.count  
            m_b = batch_var * batch_count  
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count  
            new_var = M2 / tot_count  
              
            self.mean = new_mean  
            self.var = new_var  
            self.count = tot_count  
              
    def normalize(self, x: torch.Tensor) -> torch.Tensor:  
        """归一化"""  
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)  
      
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:  
        """反归一化"""  
        return x * torch.sqrt(self.var + self.epsilon) + self.mean  
  
  
class GRUModule(nn.Module):  
    """  
    GRU模块  
      
    作为LSTM的替代选择，参数更少  
    """  
      
    def __init__(self,  
                 input_dim: int,  
                 hidden_size: int = 256,  
                 num_layers: int = 1):  
        super().__init__()  
          
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
          
        self.gru = nn.GRU(  
            input_size=input_dim,  
            hidden_size=hidden_size,  
            num_layers=num_layers,  
            batch_first=False  
        )  
          
    def forward(self,   
                x: torch.Tensor,   
                hidden_state: Optional[torch.Tensor] = None  
                ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """前向传播"""  
        if hidden_state is None:  
            hidden_state = self.get_initial_state(x.size(1), x.device)  
              
        output, hidden_state = self.gru(x, hidden_state)  
        return output, hidden_state  
      
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:  
        """获取初始状态"""  
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)  
  
  
class ICM(nn.Module):  
    """  
    内在好奇心模块 (Intrinsic Curiosity Module)  
      
    提供内在奖励，鼓励探索  
    """  
      
    def __init__(self, obs_dim: int, action_dim: int, feature_dim: int = 256):  
        super().__init__()  
          
        self.obs_dim = obs_dim  
        self.action_dim = action_dim  
        self.feature_dim = feature_dim  
          
        # 特征编码器  
        self.encoder = nn.Sequential(  
            nn.Linear(obs_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, feature_dim)  
        )  
          
        # 前向模型：预测下一个状态的特征  
        self.forward_model = nn.Sequential(  
            nn.Linear(feature_dim + action_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, feature_dim)  
        )  
          
        # 逆向模型：从状态特征预测动作  
        self.inverse_model = nn.Sequential(  
            nn.Linear(feature_dim * 2, 256),  
            nn.ReLU(),  
            nn.Linear(256, action_dim)  
        )  
          
    def forward(self,   
                obs: torch.Tensor,   
                next_obs: torch.Tensor,   
                action: torch.Tensor  
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        前向传播  
          
        Args:  
            obs: 当前观测  
            next_obs: 下一个观测  
            action: 执行的动作  
              
        Returns:  
            intrinsic_reward: 内在奖励  
            forward_loss: 前向模型损失  
            inverse_loss: 逆向模型损失  
        """  
        # 编码  
        phi_obs = self.encoder(obs)  
        phi_next_obs = self.encoder(next_obs)  
          
        # 动作独热编码  
        action_onehot = F.one_hot(action, self.action_dim).float()  
          
        # 前向预测  
        pred_next_phi = self.forward_model(torch.cat([phi_obs, action_onehot], dim=-1))  
          
        # 逆向预测  
        pred_action = self.inverse_model(torch.cat([phi_obs, phi_next_obs], dim=-1))  
          
        # 内在奖励（预测误差）  
        intrinsic_reward = 0.5 * ((pred_next_phi - phi_next_obs.detach()) ** 2).sum(dim=-1)  
          
        # 损失  
        forward_loss = 0.5 * ((pred_next_phi - phi_next_obs.detach()) ** 2).mean()  
        inverse_loss = F.cross_entropy(pred_action, action)  
          
        return intrinsic_reward, forward_loss, inverse_loss  
  
  
class ValueNorm(nn.Module):  
    """  
    价值归一化  
      
    用于稳定价值函数训练  
    """  
      
    def __init__(self, input_shape: int, epsilon: float = 1e-5, beta: float = 0.999):  
        super().__init__()  
          
        self.epsilon = epsilon  
        self.beta = beta  
          
        self.register_buffer('running_mean', torch.zeros(input_shape))  
        self.register_buffer('running_var', torch.ones(input_shape))  
        self.register_buffer('step', torch.tensor(0))  
          
    def update(self, x: torch.Tensor):  
        """更新运行统计量"""  
        with torch.no_grad():  
            self.step += 1  
            mean = x.mean()  
            var = x.var()  
              
            # 指数移动平均  
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * mean  
            self.running_var = self.beta * self.running_var + (1 - self.beta) * var  
              
    def normalize(self, x: torch.Tensor) -> torch.Tensor:  
        """归一化"""  
        # 偏差校正  
        mean = self.running_mean / (1 - self.beta ** self.step)  
        var = self.running_var / (1 - self.beta ** self.step)  
        return (x - mean) / torch.sqrt(var + self.epsilon)  
      
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:  
        """反归一化"""  
        mean = self.running_mean / (1 - self.beta ** self.step)  
        var = self.running_var / (1 - self.beta ** self.step)  
        return x * torch.sqrt(var + self.epsilon) + mean  
"""  
Actor-Critic 辅助模块  
=====================  
  
提供额外的网络组件和工具  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from typing import Tuple, Optional, Dict, List  
  
  
class NoisyLinear(nn.Module):  
    """  
    带噪声的线性层  
      
    用于探索（NoisyNet方法）  
    """  
      
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):  
        super().__init__()  
          
        self.in_features = in_features  
        self.out_features = out_features  
        self.std_init = std_init  
          
        # 学习的参数  
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))  
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))  
        self.bias_mu = nn.Parameter(torch.empty(out_features))  
        self.bias_sigma = nn.Parameter(torch.empty(out_features))  
          
        # 噪声缓冲（不需要梯度）  
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))  
        self.register_buffer('bias_epsilon', torch.empty(out_features))  
          
        self.reset_parameters()  
        self.reset_noise()  
          
    def reset_parameters(self):  
        """重置参数"""  
        mu_range = 1 / np.sqrt(self.in_features)  
        self.weight_mu.data.uniform_(-mu_range, mu_range)  
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))  
        self.bias_mu.data.uniform_(-mu_range, mu_range)  
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))  
          
    def reset_noise(self):  
        """重置噪声"""  
        epsilon_in = self._scale_noise(self.in_features)  
        epsilon_out = self._scale_noise(self.out_features)  
          
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))  
        self.bias_epsilon.copy_(epsilon_out)  
          
    def _scale_noise(self, size: int) -> torch.Tensor:  
        """生成缩放后的噪声"""  
        x = torch.randn(size, device=self.weight_mu.device)  
        return x.sign().mul_(x.abs().sqrt_())  
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """前向传播"""  
        if self.training:  
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon  
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon  
        else:  
            weight = self.weight_mu  
            bias = self.bias_mu  
              
        return F.linear(x, weight, bias)  
  
  
class AttentionModule(nn.Module):  
    """  
    自注意力模块  
      
    可选的，用于处理多智能体观测  
    """  
      
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):  
        super().__init__()  
          
        self.attention = nn.MultiheadAttention(  
            embed_dim=embed_dim,  
            num_heads=num_heads,  
            dropout=dropout,  
            batch_first=True  
        )  
          
        self.norm = nn.LayerNorm(embed_dim)  
          
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  
        """  
        前向传播  
          
        Args:  
            x: 输入 [batch, seq_len, embed_dim]  
            mask: 注意力掩码  
              
        Returns:  
            output: 输出  
        """  
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)  
        return self.norm(x + attn_out)  
  
  
class RunningMeanStdNorm(nn.Module):  
    """  
    运行时均值标准差归一化  
      
    用于观测归一化  
    """  
      
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):  
        super().__init__()  
          
        self.epsilon = epsilon  
          
        self.register_buffer('mean', torch.zeros(shape))  
        self.register_buffer('var', torch.ones(shape))  
        self.register_buffer('count', torch.tensor(epsilon))  
          
    def update(self, x: torch.Tensor):  
        """更新统计量"""  
        with torch.no_grad():  
            batch_mean = x.mean(dim=0)  
            batch_var = x.var(dim=0)  
            batch_count = x.shape[0]  
              
            delta = batch_mean - self.mean  
            tot_count = self.count + batch_count  
              
            new_mean = self.mean + delta * batch_count / tot_count  
            m_a = self.var * self.count  
            m_b = batch_var * batch_count  
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count  
            new_var = M2 / tot_count  
              
            self.mean = new_mean  
            self.var = new_var  
            self.count = tot_count  
              
    def normalize(self, x: torch.Tensor) -> torch.Tensor:  
        """归一化"""  
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)  
      
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:  
        """反归一化"""  
        return x * torch.sqrt(self.var + self.epsilon) + self.mean  
  
  
class GRUModule(nn.Module):  
    """  
    GRU模块  
      
    作为LSTM的替代选择，参数更少  
    """  
      
    def __init__(self,  
                 input_dim: int,  
                 hidden_size: int = 256,  
                 num_layers: int = 1):  
        super().__init__()  
          
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
          
        self.gru = nn.GRU(  
            input_size=input_dim,  
            hidden_size=hidden_size,  
            num_layers=num_layers,  
            batch_first=False  
        )  
          
    def forward(self,   
                x: torch.Tensor,   
                hidden_state: Optional[torch.Tensor] = None  
                ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """前向传播"""  
        if hidden_state is None:  
            hidden_state = self.get_initial_state(x.size(1), x.device)  
              
        output, hidden_state = self.gru(x, hidden_state)  
        return output, hidden_state  
      
    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:  
        """获取初始状态"""  
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)  
  
  
class ICM(nn.Module):  
    """  
    内在好奇心模块 (Intrinsic Curiosity Module)  
      
    提供内在奖励，鼓励探索  
    """  
      
    def __init__(self, obs_dim: int, action_dim: int, feature_dim: int = 256):  
        super().__init__()  
          
        self.obs_dim = obs_dim  
        self.action_dim = action_dim  
        self.feature_dim = feature_dim  
          
        # 特征编码器  
        self.encoder = nn.Sequential(  
            nn.Linear(obs_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, feature_dim)  
        )  
          
        # 前向模型：预测下一个状态的特征  
        self.forward_model = nn.Sequential(  
            nn.Linear(feature_dim + action_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, feature_dim)  
        )  
          
        # 逆向模型：从状态特征预测动作  
        self.inverse_model = nn.Sequential(  
            nn.Linear(feature_dim * 2, 256),  
            nn.ReLU(),  
            nn.Linear(256, action_dim)  
        )  
          
    def forward(self,   
                obs: torch.Tensor,   
                next_obs: torch.Tensor,   
                action: torch.Tensor  
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        前向传播  
          
        Args:  
            obs: 当前观测  
            next_obs: 下一个观测  
            action: 执行的动作  
              
        Returns:  
            intrinsic_reward: 内在奖励  
            forward_loss: 前向模型损失  
            inverse_loss: 逆向模型损失  
        """  
        # 编码  
        phi_obs = self.encoder(obs)  
        phi_next_obs = self.encoder(next_obs)  
          
        # 动作独热编码  
        action_onehot = F.one_hot(action, self.action_dim).float()  
          
        # 前向预测  
        pred_next_phi = self.forward_model(torch.cat([phi_obs, action_onehot], dim=-1))  
          
        # 逆向预测  
        pred_action = self.inverse_model(torch.cat([phi_obs, phi_next_obs], dim=-1))  
          
        # 内在奖励（预测误差）  
        intrinsic_reward = 0.5 * ((pred_next_phi - phi_next_obs.detach()) ** 2).sum(dim=-1)  
          
        # 损失  
        forward_loss = 0.5 * ((pred_next_phi - phi_next_obs.detach()) ** 2).mean()  
        inverse_loss = F.cross_entropy(pred_action, action)  
          
        return intrinsic_reward, forward_loss, inverse_loss  
  
  
class ValueNorm(nn.Module):  
    """  
    价值归一化  
      
    用于稳定价值函数训练  
    """  
      
    def __init__(self, input_shape: int, epsilon: float = 1e-5, beta: float = 0.999):  
        super().__init__()  
          
        self.epsilon = epsilon  
        self.beta = beta  
          
        self.register_buffer('running_mean', torch.zeros(input_shape))  
        self.register_buffer('running_var', torch.ones(input_shape))  
        self.register_buffer('step', torch.tensor(0))  
          
    def update(self, x: torch.Tensor):  
        """更新运行统计量"""  
        with torch.no_grad():  
            self.step += 1  
            mean = x.mean()  
            var = x.var()  
              
            # 指数移动平均  
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * mean  
            self.running_var = self.beta * self.running_var + (1 - self.beta) * var  
              
    def normalize(self, x: torch.Tensor) -> torch.Tensor:  
        """归一化"""  
        # 偏差校正  
        mean = self.running_mean / (1 - self.beta ** self.step)  
        var = self.running_var / (1 - self.beta ** self.step)  
        return (x - mean) / torch.sqrt(var + self.epsilon)  
      
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:  
        """反归一化"""  
        mean = self.running_mean / (1 - self.beta ** self.step)  
        var = self.running_var / (1 - self.beta ** self.step)  
        return x * torch.sqrt(var + self.epsilon) + mean  
