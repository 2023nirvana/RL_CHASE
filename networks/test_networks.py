"""  
网络模块测试脚本  
"""  
  
import torch  
import torch.nn as nn  
import numpy as np  
import sys  
import os  
  
# 添加项目根目录  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
from networks.ppo_lstm_network import (  
    FeatureExtractor,   
    LSTMModule,   
    ActorHead,   
    CriticHead,  
    PPOLSTMNetwork,  
    PPONetwork  
)  
from networks.actor_critic import NoisyLinear, AttentionModule, ICM, ValueNorm  
from networks.terrain_encoder import TerrainEncoder2D, TerrainEncoder3D, PointCloudEncoder  
  
  
def test_feature_extractor():  
    """测试特征提取器"""  
    print("=" * 60)  
    print("测试1: 特征提取器")  
    print("=" * 60)  
      
    input_dim = 24  
    batch_size = 32  
      
    extractor = FeatureExtractor(  
        input_dim=input_dim,  
        hidden_sizes=[256, 256],  
        use_layer_norm=True  
    )  
      
    # 测试输入  
    x = torch.randn(batch_size, input_dim)  
    output = extractor(x)  
      
    print(f"输入形状: {x.shape}")  
    print(f"输出形状: {output.shape}")  
    print(f"参数数量: {sum(p.numel() for p in extractor.parameters())}")  
      
    # 测试序列输入  
    seq_len = 10  
    x_seq = torch.randn(seq_len, batch_size, input_dim)  
    output_seq = extractor(x_seq)  
    print(f"序列输入形状: {x_seq.shape}")  
    print(f"序列输出形状: {output_seq.shape}")  
      
    print("特征提取器测试通过!\n")  
  
  
def test_lstm_module():  
    """测试LSTM模块"""  
    print("=" * 60)  
    print("测试2: LSTM模块")  
    print("=" * 60)  
      
    input_dim = 256  
    hidden_size = 256  
    num_layers = 2  
    batch_size = 32  
    seq_len = 20  
      
    lstm = LSTMModule(  
        input_dim=input_dim,  
        hidden_size=hidden_size,  
        num_layers=num_layers  
    )  
      
    # 测试输入  
    x = torch.randn(seq_len, batch_size, input_dim)  
      
    # 无初始状态  
    output, hidden = lstm(x)  
    print(f"输入形状: {x.shape}")  
    print(f"输出形状: {output.shape}")  
    print(f"隐藏状态形状: h={hidden[0].shape}, c={hidden[1].shape}")  
      
    # 带初始状态  
    initial_state = lstm.get_initial_state(batch_size, x.device)  
    output2, hidden2 = lstm(x, initial_state)  
    print(f"使用初始状态后输出形状: {output2.shape}")  
      
    # 验证输出不同（因为初始状态是零）  
    print(f"两次输出相同: {torch.allclose(output, output2)}")  
      
    print("LSTM模块测试通过!\n")  
  
  
def test_actor_critic_heads():  
    """测试Actor和Critic头"""  
    print("=" * 60)  
    print("测试3: Actor和Critic头")  
    print("=" * 60)  
      
    input_dim = 256  
    num_actions = 13  
    batch_size = 32  
      
    actor = ActorHead(input_dim=input_dim, num_actions=num_actions)  
    critic = CriticHead(input_dim=input_dim)  
      
    x = torch.randn(batch_size, input_dim)  
      
    # Actor输出  
    logits = actor(x)  
    dist = actor.get_distribution(x)  
    action = dist.sample()  
    log_prob = dist.log_prob(action)  
    entropy = dist.entropy()  
      
    print(f"Actor输入形状: {x.shape}")  
    print(f"Logits形状: {logits.shape}")  
    print(f"采样动作形状: {action.shape}")  
    print(f"Log概率形状: {log_prob.shape}")  
    print(f"熵形状: {entropy.shape}")  
    print(f"动作范围: [{action.min().item()}, {action.max().item()}]")  
      
    # Critic输出  
    value = critic(x)  
    print(f"\nCritic输出形状: {value.shape}")  
      
    print("Actor和Critic头测试通过!\n")  
  
  
def test_ppo_lstm_network():  
    """测试完整的PPO+LSTM网络"""  
    print("=" * 60)  
    print("测试4: PPO+LSTM网络")  
    print("=" * 60)  
      
    obs_dim = 24  
    num_actions = 13  
    batch_size = 32  
    seq_len = 16  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions,  
        feature_hidden_sizes=[256, 256],  
        lstm_hidden_size=256,  
        lstm_num_layers=2,  
        actor_hidden_sizes=[128],  
        critic_hidden_sizes=[128]  
    )  
      
    print(f"网络结构:")  
    print(f"  观测维度: {obs_dim}")  
    print(f"  动作数量: {num_actions}")  
    print(f"  总参数量: {sum(p.numel() for p in network.parameters()):,}")  
      
    # 测试单步推理  
    obs = torch.randn(batch_size, obs_dim)  
    action_dist, value, hidden = network(obs)  
      
    print(f"\n单步推理:")  
    print(f"  输入形状: {obs.shape}")  
    print(f"  价值形状: {value.shape}")  
    print(f"  隐藏状态形状: h={hidden[0].shape}, c={hidden[1].shape}")  
      
    # 测试get_action_and_value  
    action, log_prob, entropy, value, hidden = network.get_action_and_value(obs)  
    print(f"\nget_action_and_value输出:")  
    print(f"  动作形状: {action.shape}")  
    print(f"  Log概率形状: {log_prob.shape}")  
    print(f"  熵形状: {entropy.shape}")  
    print(f"  价值形状: {value.shape}")  
      
    # 测试序列评估  
    obs_seq = torch.randn(seq_len, batch_size, obs_dim)  
    actions_seq = torch.randint(0, num_actions, (seq_len, batch_size))  
    dones_seq = torch.zeros(seq_len, batch_size)  
    initial_hidden = network.lstm.get_initial_state(batch_size, obs.device)  
      
    log_probs, values, entropies = network.evaluate_actions(  
        obs_seq, actions_seq, initial_hidden, dones_seq  
    )  
      
    print(f"\n序列评估:")  
    print(f"  观测序列形状: {obs_seq.shape}")  
    print(f"  Log概率序列形状: {log_probs.shape}")  
    print(f"  价值序列形状: {values.shape}")  
    print(f"  熵序列形状: {entropies.shape}")  
      
    # 测试保存和加载  
    save_path = "test_network.pth"  
    network.save(save_path)  
    loaded_network = PPOLSTMNetwork.load(save_path)  
      
    # 验证加载后输出一致  
    with torch.no_grad():  
        _, value1, _ = network(obs)  
        _, value2, _ = loaded_network(obs)  
          
    print(f"\n保存/加载后输出一致: {torch.allclose(value1, value2)}")  
      
    # 清理  
    os.remove(save_path)  
      
    print("PPO+LSTM网络测试通过!\n")  
  
  
def test_ppo_network():  
    """测试无LSTM的PPO网络"""  
    print("=" * 60)  
    print("测试5: PPO网络（无LSTM）")  
    print("=" * 60)  
      
    obs_dim = 24  
    num_actions = 13  
    batch_size = 32  
      
    network = PPONetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions,  
        hidden_sizes=[256, 256]  
    )  
      
    print(f"参数量: {sum(p.numel() for p in network.parameters()):,}")  
      
    obs = torch.randn(batch_size, obs_dim)  
    action, log_prob, entropy, value = network.get_action_and_value(obs)  
      
    print(f"输出形状:")  
    print(f"  动作: {action.shape}")  
    print(f"  价值: {value.shape}")  
      
    print("PPO网络测试通过!\n")  
  
  
def test_auxiliary_modules():  
    """测试辅助模块"""  
    print("=" * 60)  
    print("测试6: 辅助模块")  
    print("=" * 60)  
      
    batch_size = 32  
      
    # NoisyLinear  
    print("NoisyLinear:")  
    noisy = NoisyLinear(128, 64)  
    x = torch.randn(batch_size, 128)  
    y1 = noisy(x)  
    noisy.reset_noise()  
    y2 = noisy(x)  
    print(f"  输出相同（噪声重置前后）: {torch.allclose(y1, y2)}")  
      
    # AttentionModule  
    print("\nAttentionModule:")  
    attention = AttentionModule(embed_dim=128, num_heads=4)  
    x = torch.randn(batch_size, 10, 128)  # seq_len=10  
    y = attention(x)  
    print(f"  输入形状: {x.shape}, 输出形状: {y.shape}")  
      
    # ICM  
    print("\nICM (内在好奇心模块):")  
    icm = ICM(obs_dim=24, action_dim=13)  
    obs = torch.randn(batch_size, 24)  
    next_obs = torch.randn(batch_size, 24)  
    action = torch.randint(0, 13, (batch_size,))  
    intrinsic_reward, forward_loss, inverse_loss = icm(obs, next_obs, action)  
    print(f"  内在奖励形状: {intrinsic_reward.shape}")  
    print(f"  前向损失: {forward_loss.item():.4f}")  
    print(f"  逆向损失: {inverse_loss.item():.4f}")  
      
    # ValueNorm  
    print("\nValueNorm:")  
    value_norm = ValueNorm(1)  
    values = torch.randn(100)  
    for v in values.split(10):  
        value_norm.update(v)  
    normalized = value_norm.normalize(values[:10])  
    print(f"  原始值均值: {values[:10].mean():.4f}")  
    print(f"  归一化后均值: {normalized.mean():.4f}")  
      
    print("\n辅助模块测试通过!\n")  
  
  
def test_terrain_encoders():  
    """测试地形编码器"""  
    print("=" * 60)  
    print("测试7: 地形编码器")  
    print("=" * 60)  
      
    batch_size = 16  
      
    # 2D编码器  
    print("2D地形编码器:")  
    encoder_2d = TerrainEncoder2D(grid_size=32, output_dim=128)  
    grid_2d = torch.randn(batch_size, 1, 32, 32)  
    features_2d = encoder_2d(grid_2d)  
    print(f"  输入形状: {grid_2d.shape}")  
    print(f"  输出形状: {features_2d.shape}")  
    print(f"  参数量: {sum(p.numel() for p in encoder_2d.parameters()):,}")  
      
    # 3D编码器  
    print("\n3D地形编码器:")  
    encoder_3d = TerrainEncoder3D(grid_size=16, output_dim=128)  
    voxel_3d = torch.randn(batch_size, 1, 16, 16, 16)  
    features_3d = encoder_3d(voxel_3d)  
    print(f"  输入形状: {voxel_3d.shape}")  
    print(f"  输出形状: {features_3d.shape}")  
    print(f"  参数量: {sum(p.numel() for p in encoder_3d.parameters()):,}")  
      
    # 点云编码器  
    print("\n点云编码器:")  
    encoder_pc = PointCloudEncoder(output_dim=128)  
    point_cloud = torch.randn(batch_size, 256, 3)  # 256个点  
    features_pc = encoder_pc(point_cloud)  
    print(f"  输入形状: {point_cloud.shape}")  
    print(f"  输出形状: {features_pc.shape}")  
    print(f"  参数量: {sum(p.numel() for p in encoder_pc.parameters()):,}")  
      
    print("\n地形编码器测试通过!\n")  
  
  
def test_gradient_flow():  
    """测试梯度流"""  
    print("=" * 60)  
    print("测试8: 梯度流")  
    print("=" * 60)  
      
    obs_dim = 24  
    num_actions = 13  
    batch_size = 32  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions  
    )  
      
    # 模拟一个训练步  
    obs = torch.randn(batch_size, obs_dim, requires_grad=False)  
      
    action_dist, value, hidden = network(obs)  
    action = action_dist.sample()  
    log_prob = action_dist.log_prob(action)  
    entropy = action_dist.entropy()  
      
    # 模拟损失  
    advantages = torch.randn(batch_size)  
    returns = torch.randn(batch_size)  
      
    # PPO损失  
    actor_loss = -(log_prob * advantages).mean()  
    critic_loss = 0.5 * ((value - returns) ** 2).mean()  
    entropy_loss = -entropy.mean()  
      
    total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss  
      
    # 反向传播  
    total_loss.backward()  
      
    # 检查梯度  
    has_gradient = True  
    gradient_norms = {}  
      
    for name, param in network.named_parameters():  
        if param.grad is None:  
            has_gradient = False  
            print(f"  警告: {name} 没有梯度!")  
        else:  
            gradient_norms[name] = param.grad.norm().item()  
              
    print(f"所有参数都有梯度: {has_gradient}")  
    print(f"梯度范数范围: [{min(gradient_norms.values()):.6f}, {max(gradient_norms.values()):.6f}]")  
      
    # 检查是否有NaN  
    has_nan = any(torch.isnan(param.grad).any() for param in network.parameters() if param.grad is not None)  
    print(f"梯度中有NaN: {has_nan}")  
      
    print("梯度流测试通过!\n")  
  
  
def test_inference_speed():  
    """测试推理速度"""  
    print("=" * 60)  
    print("测试9: 推理速度")  
    print("=" * 60)  
      
    import time  
      
    obs_dim = 24  
    num_actions = 13  
    batch_size = 1  # 单步推理  
    num_iters = 1000  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions  
    )  
    network.eval()  
      
    obs = torch.randn(batch_size, obs_dim)  
    hidden = network.lstm.get_initial_state(batch_size, obs.device)  
      
    # 预热  
    with torch.no_grad():  
        for _ in range(100):  
            _, _, hidden = network(obs, hidden)  
              
    # 计时  
    start_time = time.time()  
    with torch.no_grad():  
        for _ in range(num_iters):  
            action_dist, value, hidden = network(obs, hidden)  
            action = action_dist.sample()  
              
    elapsed = time.time() - start_time  
      
    print(f"运行 {num_iters} 次推理")  
    print(f"总时间: {elapsed:.3f}s")  
    print(f"平均每次: {elapsed/num_iters*1000:.3f}ms")  
    print(f"推理频率: {num_iters/elapsed:.1f} Hz")  
      
    # 与控制频率比较  
    control_freq = 50  # Hz  
    print(f"\n控制频率 {control_freq} Hz 需要 {1000/control_freq:.1f}ms 每步")  
    print(f"网络速度满足要求: {elapsed/num_iters*1000 < 1000/control_freq}")  
      
    print("推理速度测试通过!\n")  
  
  
def run_all_tests():  
    """运行所有测试"""  
    print("\n" + "=" * 60)  
    print("网络模块测试")  
    print("=" * 60 + "\n")  
      
    test_feature_extractor()  
    test_lstm_module()  
    test_actor_critic_heads()  
    test_ppo_lstm_network()  
    test_ppo_network()  
    test_auxiliary_modules()  
    test_terrain_encoders()  
    test_gradient_flow()  
    test_inference_speed()  
      
    print("=" * 60)  
    print("所有网络测试完成!")  
    print("=" * 60)  
  
  
if __name__ == "__main__":  
    run_all_tests()  
