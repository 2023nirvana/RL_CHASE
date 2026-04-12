"""
行为克隆 (Behavioral Cloning) 预训练高层策略
============================================
Phase 1: 用有监督学习在专家数据上拟合动作，不与环境交互。
比 Offline PPO 更稳定，无分布漂移问题。

损失函数:
  L = α * MSE(a_pred, a_expert) + (1-α) * (-NLL)  [actor]
    + β * MSE(V(s), returns)                       [critic, 可选]

网络结构: 复用 HighLevelActorCritic (256,128 + LayerNorm)
输出格式: 与 offline_ppo_pretrain.py 完全兼容，Phase 2 可直接加载
"""

import os
import sys
import json
import math
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.offline_ppo_pretrain import HighLevelActorCritic


# ───────── 数据集 ─────────

class BCDataset:
    """加载专家数据，分 train / val"""

    def __init__(self, path: str, val_ratio: float = 0.1,
                 gamma: float = 0.99, seed: int = 42):
        raw = np.load(path, allow_pickle=True)
        states = raw["states"].astype(np.float32)
        actions = raw["actions"].astype(np.float32)
        rewards = raw["rewards"].astype(np.float32)
        dones = raw["dones"].astype(np.float32)

        n = len(states)
        meta = json.loads(str(raw["meta"]))
        print(f"Loaded {n} transitions from {path}")
        print(f"  {meta.get('total_episodes', '?')} episodes, "
              f"{len(meta.get('per_strategy', {}))} strategies")

        # 计算 discounted returns (用于训练 critic)
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            if dones[t] > 0.5:
                running = 0.0
            running = rewards[t] + gamma * running
            returns[t] = running

        # 打乱 & 切分 train / val
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val

        idx_train = perm[:n_train]
        idx_val = perm[n_train:]

        self.train_s = torch.from_numpy(states[idx_train])
        self.train_a = torch.from_numpy(actions[idx_train])
        self.train_r = torch.from_numpy(returns[idx_train])
        self.val_s = torch.from_numpy(states[idx_val])
        self.val_a = torch.from_numpy(actions[idx_val])
        self.val_r = torch.from_numpy(returns[idx_val])
        self.n_train = n_train
        self.n_val = n_val

        # 统计专家动作分布 (用于诊断)
        a_mean = actions.mean(axis=0)
        a_std = actions.std(axis=0)
        print(f"  Expert actions: mean={a_mean}, std={a_std}")
        print(f"  Returns: mean={returns.mean():.1f}, "
              f"std={returns.std():.1f}")
        print(f"  Train: {n_train}, Val: {n_val}")

    def train_batches(self, batch_size: int, device: str = "cpu"):
        """生成一个 epoch 的 mini-batches"""
        perm = torch.randperm(self.n_train)
        for start in range(0, self.n_train, batch_size):
            idx = perm[start:start + batch_size]
            yield (self.train_s[idx].to(device),
                   self.train_a[idx].to(device),
                   self.train_r[idx].to(device))

    def val_batch(self, device: str = "cpu"):
        """返回完整验证集"""
        return (self.val_s.to(device),
                self.val_a.to(device),
                self.val_r.to(device))


# ───────── BC 训练器 ─────────

class BCTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = HighLevelActorCritic(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {param_count:,}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)

        self.dataset = BCDataset(
            args.dataset, val_ratio=args.val_ratio,
            gamma=args.gamma, seed=args.seed)

        self.exp_name = (args.experiment_name
                         or f"bc_phase1_{datetime.now():%Y%m%d_%H%M}")
        self.log_dir = os.path.join(
            PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(
            PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.history: List[Dict[str, float]] = []

    def compute_loss(self, states, actions, returns):
        """
        BC loss:
         - actor: MSE(tanh(mu), a_expert) + NLL权重
         - critic: MSE(V(s), returns)
        """
        args = self.args

        # Actor: 前向得到分布
        dist = self.model.get_dist(states)
        mu = torch.tanh(dist.mean)  # squashed mean

        # MSE loss on actions (主损失)
        mse_loss = F.mse_loss(mu, actions)

        # NLL loss (辅助, 让 std 也有梯度信号)
        # 对已 tanh 的 expert actions 反变换到 pre-squash 空间
        a_clip = torch.clamp(actions, -0.999, 0.999)
        z = 0.5 * (torch.log1p(a_clip) - torch.log1p(-a_clip))  # atanh
        nll = -dist.log_prob(z).sum(-1).mean()

        actor_loss = (args.mse_weight * mse_loss
                      + (1.0 - args.mse_weight) * nll)

        # Critic
        critic_loss = torch.tensor(0.0, device=self.device)
        if args.train_critic:
            v = self.model.get_value(states)
            critic_loss = F.mse_loss(v, returns)

        total = actor_loss + args.vf_coef * critic_loss

        return {
            "total": total,
            "mse": mse_loss.item(),
            "nll": nll.item(),
            "actor": actor_loss.item(),
            "critic": critic_loss.item(),
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        vs, va, vr = self.dataset.val_batch(self.device)
        info = self.compute_loss(vs, va, vr)
        self.model.train()
        return info

    def train_one_epoch(self):
        self.model.train()
        total_mse = 0.0
        total_nll = 0.0
        total_actor = 0.0
        total_critic = 0.0
        n_batches = 0

        for s, a, r in self.dataset.train_batches(
                self.args.batch_size, self.device):
            info = self.compute_loss(s, a, r)
            self.optimizer.zero_grad()
            info["total"].backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            total_mse += info["mse"]
            total_nll += info["nll"]
            total_actor += info["actor"]
            total_critic += info["critic"]
            n_batches += 1

        return {
            "mse": total_mse / n_batches,
            "nll": total_nll / n_batches,
            "actor": total_actor / n_batches,
            "critic": total_critic / n_batches,
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.args.obs_dim,
            "act_dim": self.args.act_dim,
            "hidden": list(self.args.hidden),
            "config": vars(self.args),
        }, path)
        print(f"  Saved: {path}")

    def train(self):
        n_epochs = self.args.n_epochs
        print(f"\nBC Pretraining: {n_epochs} epochs, "
              f"{self.dataset.n_train} train / "
              f"{self.dataset.n_val} val transitions")
        print(f"  MSE weight: {self.args.mse_weight}, "
              f"train_critic: {self.args.train_critic}")
        t0 = time.time()
        best_val = float("inf")
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            train_info = self.train_one_epoch()
            val_info = self.validate()
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "train_mse": train_info["mse"],
                "train_nll": train_info["nll"],
                "train_actor": train_info["actor"],
                "train_critic": train_info["critic"],
                "val_mse": val_info["mse"],
                "val_nll": val_info["nll"],
                "val_actor": val_info["actor"],
                "val_critic": val_info["critic"],
                "lr": lr_now,
            }
            self.history.append(record)

            if epoch % self.args.log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"train_mse={train_info['mse']:.5f}  "
                      f"val_mse={val_info['mse']:.5f}  "
                      f"nll={val_info['nll']:.3f}  "
                      f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

            # Best (按 val mse)
            val_metric = val_info["mse"]
            if val_metric < best_val:
                best_val = val_metric
                patience_counter = 0
                self.save_checkpoint("best")
            else:
                patience_counter += 1

            # 定期保存
            if epoch % self.args.save_every == 0:
                self.save_checkpoint("latest")

            # Early stopping
            if (self.args.patience > 0
                    and patience_counter >= self.args.patience):
                print(f"  Early stopping at epoch {epoch} "
                      f"(patience={self.args.patience})")
                break

        self.save_checkpoint("final")
        self.plot_curves()
        print(f"\nBC pretraining done. Best val MSE: {best_val:.5f}")
        print(f"Checkpoints: {self.ckpt_dir}")

    def plot_curves(self):
        if not self.history:
            return
        epochs = [h["epoch"] for h in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # MSE
        ax = axes[0, 0]
        ax.plot(epochs, [h["train_mse"] for h in self.history],
                label="train", lw=1.5)
        ax.plot(epochs, [h["val_mse"] for h in self.history],
                label="val", lw=1.5, ls="--")
        ax.set_title("Action MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.25)

        # NLL
        ax = axes[0, 1]
        ax.plot(epochs, [h["train_nll"] for h in self.history],
                label="train", lw=1.5)
        ax.plot(epochs, [h["val_nll"] for h in self.history],
                label="val", lw=1.5, ls="--")
        ax.set_title("Negative Log-Likelihood")
        ax.legend()
        ax.grid(True, alpha=0.25)

        # Critic
        ax = axes[1, 0]
        ax.plot(epochs, [h["train_critic"] for h in self.history],
                label="train", lw=1.5)
        ax.plot(epochs, [h["val_critic"] for h in self.history],
                label="val", lw=1.5, ls="--")
        ax.set_title("Critic MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.25)

        # LR
        ax = axes[1, 1]
        ax.plot(epochs, [h["lr"] for h in self.history], lw=1.5,
                color="tab:orange")
        ax.set_title("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        plt.suptitle(f"BC Pretraining — {self.exp_name}", fontsize=14)
        plt.tight_layout()
        out = os.path.join(self.log_dir, "bc_pretrain_curves.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


# ───────── main ─────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Behavioral Cloning pretrain")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to expert_dataset.npz")
    parser.add_argument("--experiment-name", type=str, default=None)

    # 网络
    parser.add_argument("--obs-dim", type=int, default=21)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument("--hidden", type=int, nargs="+",
                        default=[256, 128])

    # 训练
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 损失
    parser.add_argument("--mse-weight", type=float, default=0.7,
                        help="MSE vs NLL: 1.0=纯MSE, 0.0=纯NLL")
    parser.add_argument("--train-critic", action="store_true",
                        default=True,
                        help="同时训练 critic (用 discounted returns)")
    parser.add_argument("--no-train-critic", dest="train_critic",
                        action="store_false")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)

    # 其他
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = BCTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
