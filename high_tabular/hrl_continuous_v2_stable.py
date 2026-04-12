#!/usr/bin/env python3
"""
稳定版V2 - 减少训练波动
改进：增加更新轮次、减小clip、学习率衰减、多episode聚合
"""
import sys
sys.path.insert(0, "/root/autodl-tmp/pursue_20260302")
from high_tabular.hrl_continuous_v2 import *


def train_ppo_stable(env, policy, device, episodes=4000, gamma=0.99, lam=0.95, 
                     lr_start=1e-4, lr_end=1e-5, clip_eps=0.1, update_epochs=10,
                     batch_episodes=4):
    """稳定版PPO：多episode聚合、更多更新轮次、学习率衰减"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_start)
    
    cap_hist, ret_hist = [], []
    batch_states, batch_actions, batch_rewards, batch_values, batch_log_probs = [], [], [], [], []
    batch_dones = []
    
    for ep in range(1, episodes + 1):
        # 学习率衰减
        lr = lr_start - (lr_start - lr_end) * (ep / episodes)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        obs = env.reset()
        done = False
        ep_ret = 0.0
        ep_states, ep_actions, ep_rewards, ep_values, ep_log_probs = [], [], [], [], []
        
        while not done:
            obs_t = obs_to_tensor(obs, device)
            mu, std, v = policy(obs_t)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            log_p = dist.log_prob(a).sum()
            
            a_val = a.clamp(-math.pi, math.pi).item()
            obs_next, r, done, info = env.step(a_val)
            
            ep_states.append(obs_t)
            ep_actions.append(a)
            ep_rewards.append(r)
            ep_values.append(v.item())
            ep_log_probs.append(log_p)
            
            obs = obs_next
            ep_ret += r
        
        cap_hist.append(1.0 if info.get("captured") else 0.0)
        ret_hist.append(ep_ret)
        
        # 计算该episode的GAE
        ep_values.append(0.0)
        advs = []
        gae = 0.0
        for t in reversed(range(len(ep_rewards))):
            delta = ep_rewards[t] + gamma * ep_values[t+1] - ep_values[t]
            gae = delta + gamma * lam * gae
            advs.insert(0, gae)
        returns = [advs[t] + ep_values[t] for t in range(len(advs))]
        
        # 添加到batch
        batch_states.extend(ep_states)
        batch_actions.extend(ep_actions)
        batch_rewards.extend(returns)  # 存returns而非rewards
        batch_values.extend(advs)  # 存advs
        batch_log_probs.extend(ep_log_probs)
        
        # 每batch_episodes个episode更新一次
        if ep % batch_episodes == 0:
            states_t = torch.cat(batch_states, dim=0)
            actions_t = torch.cat(batch_actions, dim=0)
            returns_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            advs_t = torch.tensor(batch_values, dtype=torch.float32, device=device)
            old_log_probs_t = torch.stack(batch_log_probs)
            
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
            
            # 多轮更新
            for _ in range(update_epochs):
                mu, std, v = policy(states_t)
                dist = torch.distributions.Normal(mu.squeeze(), std)
                log_probs_new = dist.log_prob(actions_t.squeeze())
                
                ratio = torch.exp(log_probs_new - old_log_probs_t.detach())
                surr1 = ratio * advs_t
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_t
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((v.squeeze() - returns_t) ** 2).mean()
                entropy = dist.entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.005 * entropy  # 减小entropy bonus
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
            
            # 清空batch
            batch_states, batch_actions, batch_rewards = [], [], []
            batch_values, batch_log_probs = [], []
        
        if ep % 100 == 0:
            recent_cap = np.mean(cap_hist[-100:])
            recent_ret = np.mean(ret_hist[-100:])
            print(f"[Stable] ep={ep:4d} lr={lr:.1e} cap={recent_cap:.1%} ret={recent_ret:.1f}")
    
    return cap_hist, ret_hist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="high_tabular/results_v2_stable")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = PursuitEnvV2(seed=args.seed)
    policy = PolicyNet().to(device)

    print("=" * 60)
    print("Stable PPO V2")
    print("  - 多episode聚合更新 (batch=4)")
    print("  - 更多更新轮次 (epochs=10)")  
    print("  - 更小clip (0.1)")
    print("  - 学习率衰减 (1e-4 → 1e-5)")
    print("=" * 60)

    cap_hist, ret_hist = train_ppo_stable(env, policy, device, episodes=args.episodes)

    os.makedirs(args.save_dir, exist_ok=True)
    final_cap = np.mean(cap_hist[-100:])
    max_cap = max(np.mean(cap_hist[i:i+50]) for i in range(len(cap_hist)-50))
    
    # 计算波动
    stds = [np.mean(cap_hist[i:i+50]) for i in range(len(cap_hist)//2, len(cap_hist)-50, 10)]
    std = np.std(stds)
    
    print("-" * 60)
    print(f"Final 100ep: {final_cap:.1%}")
    print(f"Max 50ep: {max_cap:.1%}")
    print(f"波动(std): {std:.3f}")
    print("-" * 60)

    plot_curves(cap_hist, ret_hist, args.save_dir)
    visualize_episode(env, policy, device, args.save_dir)
    torch.save({"policy_state_dict": policy.state_dict(), "cap_hist": cap_hist, "ret_hist": ret_hist},
               os.path.join(args.save_dir, "checkpoint.pth"))
    print(f"Results saved to: {args.save_dir}")
