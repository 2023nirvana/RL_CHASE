#!/usr/bin/env python3
"""用更大网络训练V2"""
import sys
sys.path.insert(0, "/root/autodl-tmp/pursue_20260302")
from high_tabular.hrl_continuous_v2 import *

class PolicyNetLarge(nn.Module):
    def __init__(self, obs_dim=9, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden // 2, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value = nn.Linear(hidden // 2, 1)
    
    def forward(self, obs):
        h = self.shared(obs)
        mu = torch.tanh(self.mu(h)) * math.pi
        std = torch.exp(self.log_std.clamp(-2, 0.5))
        v = self.value(h)
        return mu, std, v
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, std, v = self.forward(obs)
            if deterministic:
                return mu.item(), v.item()
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample().clamp(-math.pi, math.pi)
            return a.item(), v.item()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    np.random.seed(123)
    torch.manual_seed(123)
    
    env = PursuitEnvV2(seed=123)
    policy = PolicyNetLarge().to(device)
    
    print("=" * 60)
    print("Large Network (256-256-128)")
    print("=" * 60)
    
    cap_hist, ret_hist = train_ppo(env, policy, device, episodes=4000, lr=5e-5)
    
    save_dir = "high_tabular/results_v2_large"
    os.makedirs(save_dir, exist_ok=True)
    final_cap = np.mean(cap_hist[-100:])
    print(f"Final capture rate: {final_cap:.1%}")
    
    plot_curves(cap_hist, ret_hist, save_dir)
    visualize_episode(env, policy, device, save_dir)
    torch.save({"policy_state_dict": policy.state_dict(), "cap_hist": cap_hist}, 
               os.path.join(save_dir, "checkpoint.pth"))
