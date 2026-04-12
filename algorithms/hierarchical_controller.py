"""  
分层控制器  
==========  
  
将高层策略与底层控制器结合  
"""  
  
import numpy as np  
import torch  
from typing import Dict, Tuple, Optional, List  
import os  
import sys  
  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
  
class HierarchicalController:  
    """  
    分层控制器  
      
    高层决策 -> 底层控制  
    """  
      
    def __init__(self,  
                 low_level_model_path: str,  
                 high_level_model_path: Optional[str] = None,  
                 high_level_freq: int = 10,  
                 device: torch.device = None):  
        """  
        初始化分层控制器  
          
        Args:  
            low_level_model_path: 底层模型路径  
            high_level_model_path: 高层模型路径（可选）  
            high_level_freq: 高层决策频率（每多少步决策一次）  
            device: 计算设备  
        """  
        if device is None:  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        else:  
            self.device = device  
              
        self.high_level_freq = high_level_freq  
        self.step_count = 0  
          
        # 加载底层控制器  
        self._load_low_level(low_level_model_path)  
          
        # 加载高层策略（如果有）  
        if high_level_model_path is not None:  
            self._load_high_level(high_level_model_path)  
        else:  
            self.high_level_network = None  
              
        # 当前高层动作/目标  
        self.current_high_level_action = None  
        self.current_target = None  
          
        # LSTM状态  
        self.low_level_hidden = None  
        self.high_level_hidden = None  
          
    def _load_low_level(self, model_path: str):  
        """加载底层控制器"""  
        from networks.ppo_lstm_network import PPOLSTMNetwork  
          
        checkpoint = torch.load(model_path, map_location=self.device)  
        config = checkpoint.get('config', {})  
          
        # 重建网络（需要知道配置）  
        self.low_level_network = PPOLSTMNetwork.load(model_path, self.device)  
        self.low_level_network.eval()  
          
        print(f"Loaded low-level controller from {model_path}")  
          
    def _load_high_level(self, model_path: str):  
        """加载高层策略"""  
        from networks.ppo_lstm_network import PPOLSTMNetwork  
          
        self.high_level_network = PPOLSTMNetwork.load(model_path, self.device)  
        self.high_level_network.eval()  
          
        print(f"Loaded high-level policy from {model_path}")  
          
    def reset(self):  
        """重置控制器状态"""  
        self.step_count = 0  
        self.current_high_level_action = None  
        self.current_target = None  
        self.low_level_hidden = None  
        self.high_level_hidden = None  
          
    def get_action(self,  
                   observation: np.ndarray,  
                   high_level_observation: Optional[np.ndarray] = None) -> int:  
        """  
        获取底层动作  
          
        Args:  
            observation: 底层观测  
            high_level_observation: 高层观测（包含追逃信息等）  
              
        Returns:  
            action: 底层离散动作  
        """  
        # 检查是否需要高层决策  
        if self.step_count % self.high_level_freq == 0:  
            if self.high_level_network is not None and high_level_observation is not None:  
                self._make_high_level_decision(high_level_observation)  
                  
        # 修改底层观测（根据高层目标）  
        modified_obs = self._modify_observation(observation)  
          
        # 底层决策  
        action = self._get_low_level_action(modified_obs)  
          
        self.step_count += 1  
          
        return action  
      
    def _make_high_level_decision(self, observation: np.ndarray):  
        """高层决策"""  
        with torch.no_grad():  
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)  
              
            if obs_tensor.dim() == 1:  
                obs_tensor = obs_tensor.unsqueeze(0)  
                  
            if self.high_level_hidden is None:  
                self.high_level_hidden = self.high_level_network.lstm.get_initial_state(  
                    1, self.device  
                )  
                  
            action_dist, _, self.high_level_hidden = self.high_level_network(  
                obs_tensor, self.high_level_hidden  
            )  
              
            # 使用确定性策略  
            self.current_high_level_action = action_dist.probs.argmax(dim=-1).item()  
              
    def _modify_observation(self, observation: np.ndarray) -> np.ndarray:  
        """  
        根据高层决策修改底层观测  
          
        这是分层控制的关键接口  
        """  
        # 如果没有高层策略，直接返回原始观测  
        if self.current_high_level_action is None:  
            return observation  
              
        # 根据高层动作设置目标  
        # 这里需要根据具体的高层动作定义来实现  
        # 简化示例：高层动作直接对应目标方向  
          
        modified_obs = observation.copy()  
          
        # 示例：根据高层动作修改目标位置  
        # 这部分需要根据实际的高层动作空间定义  
          
        return modified_obs  
      
    def _get_low_level_action(self, observation: np.ndarray) -> int:  
        """获取底层动作"""  
        with torch.no_grad():  
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)  
              
            if obs_tensor.dim() == 1:  
                obs_tensor = obs_tensor.unsqueeze(0)  
                  
            if self.low_level_hidden is None:  
                self.low_level_hidden = self.low_level_network.lstm.get_initial_state(  
                    1, self.device  
                )  
                  
            action_dist, _, self.low_level_hidden = self.low_level_network(  
                obs_tensor, self.low_level_hidden  
            )  
              
            # 使用确定性策略  
            action = action_dist.probs.argmax(dim=-1).item()  
              
        return action  
      
    def set_target(self, target_position: np.ndarray, target_orientation: Optional[np.ndarray] = None):  
        """  
        直接设置目标（绕过高层策略）  
          
        Args:  
            target_position: 目标位置 [x, y, z]  
            target_orientation: 目标姿态 [roll, pitch, yaw]  
        """  
        self.current_target = {  
            'position': target_position,  
            'orientation': target_orientation if target_orientation is not None else np.zeros(3)  
        }  
  
  
class LowLevelSkillExecutor:  
    """  
    底层技能执行器  
      
    封装底层控制器，提供简单的技能接口  
    """  
      
    # 技能定义  
    SKILLS = {  
        'hover': {'action': 0, 'duration': 50},  
        'forward': {'action': 1, 'duration': 100},  
        'backward': {'action': 2, 'duration': 100},  
        'left': {'action': 3, 'duration': 100},  
        'right': {'action': 4, 'duration': 100},  
        'up': {'action': 5, 'duration': 100},  
        'down': {'action': 6, 'duration': 100},  
        'turn_left': {'action': 7, 'duration': 50},  
        'turn_right': {'action': 8, 'duration': 50},  
    }  
      
    def __init__(self, model_path: str, device: torch.device = None):  
        """  
        初始化  
          
        Args:  
            model_path: 底层模型路径  
            device: 计算设备  
        """  
        if device is None:  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        else:  
            self.device = device  
              
        # 加载模型  
        from networks.ppo_lstm_network import PPOLSTMNetwork  
        self.network = PPOLSTMNetwork.load(model_path, self.device)  
        self.network.eval()  
          
        # 状态  
        self.hidden_state = None  
        self.current_skill = None  
        self.skill_step = 0  
          
    def reset(self):  
        """重置"""  
        self.hidden_state = None  
        self.current_skill = None  
        self.skill_step = 0  
          
    def execute_skill(self, skill_name: str, observation: np.ndarray) -> Tuple[int, bool]:  
        """  
        执行技能  
          
        Args:  
            skill_name: 技能名称  
            observation: 当前观测  
              
        Returns:  
            action: 动作  
            done: 技能是否完成  
        """  
        if skill_name not in self.SKILLS:  
            raise ValueError(f"Unknown skill: {skill_name}")  
              
        # 开始新技能  
        if self.current_skill != skill_name:  
            self.current_skill = skill_name  
            self.skill_step = 0  
              
        skill_config = self.SKILLS[skill_name]  
          
        # 获取动作（这里简化为固定动作，实际应该使用学习的策略）  
        action = skill_config['action']  
          
        self.skill_step += 1  
        done = self.skill_step >= skill_config['duration']  
          
        if done:  
            self.current_skill = None  
            self.skill_step = 0  
              
        return action, done  
      
    def get_action(self, observation: np.ndarray, target: np.ndarray) -> int:  
        """  
        根据目标获取动作  
          
        Args:  
            observation: 当前观测  
            target: 目标状态  
              
        Returns:  
            action: 动作  
        """  
        with torch.no_grad():  
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)  
              
            if obs_tensor.dim() == 1:  
                obs_tensor = obs_tensor.unsqueeze(0)  
                  
            if self.hidden_state is None:  
                self.hidden_state = self.network.lstm.get_initial_state(1, self.device)  
                  
            action_dist, _, self.hidden_state = self.network(obs_tensor, self.hidden_state)  
            action = action_dist.probs.argmax(dim=-1).item()  
              
        return action  
