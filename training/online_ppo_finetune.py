"""
Online PPO 微调 (Phase 2) — v2
=================================
加载 offline 预训练的高层策略权重，接入底层控制器,
与环境在线交互做 PPO 微调。

v2 改进:
- SubprocVecEnv 多进程并行采集 (~N 倍加速)
- 奖励分项跟踪 (r_event / r_shaping / r_sg_obs / r_time)
- 可调奖励权重 & 参数 (CLI)
- LR 线性衰减
- 从 latest checkpoint 恢复训练
- 更丰富的日志/曲线
"""

import os
import sys
import json
import math
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_fixed import (
    load_v4_nn, PursuitEnvRealAUVFixed,
)
from training.offline_ppo_pretrain import HighLevelActorCritic
from training.collect_expert_trajectories import compute_high_reward
from training.vec_env import SubprocVecEnv


# ─────────── Rollout Buffer ───────────

class RolloutBuffer:
    """存储单个环境的 on-policy rollout 数据 (支持奖励分项)"""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.logprobs: List[float] = []
        # 奖励分项
        self.r_events: List[float] = []
        self.r_shapings: List[float] = []
        self.r_sg_obs_list: List[float] = []
        self.r_times: List[float] = []

    def add(self, s, a, r, d, v, lp, r_info=None):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.values.append(v)
        self.logprobs.append(lp)
        if r_info is not None:
            self.r_events.append(r_info.get("r_event", 0.0))
            self.r_shapings.append(r_info.get("r_shaping", 0.0))
            self.r_sg_obs_list.append(r_info.get("r_sg_obs", 0.0))
            self.r_times.append(r_info.get("r_time", 0.0))
        else:
            self.r_events.append(0.0)
            self.r_shapings.append(0.0)
            self.r_sg_obs_list.append(0.0)
            self.r_times.append(0.0)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logprobs.clear()
        self.r_events.clear()
        self.r_shapings.clear()
        self.r_sg_obs_list.clear()
        self.r_times.clear()

    @property
    def size(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float,
                    gamma: float = 0.99,
                    lam: float = 0.95):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nt = 0.0 if self.dones[t] else 1.0
            nv = last_value if t == n - 1 else self.values[t + 1]
            delta = (self.rewards[t]
                     + gamma * nv * nt - self.values[t])
            last_gae = delta + gamma * lam * nt * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def as_tensors(self, advantages, returns, device):
        return {
            "states": torch.tensor(
                np.array(self.states), dtype=torch.float32,
                device=device),
            "actions": torch.tensor(
                np.array(self.actions), dtype=torch.float32,
                device=device),
            "logprobs": torch.tensor(
                np.array(self.logprobs), dtype=torch.float32,
                device=device),
            "advantages": torch.tensor(
                advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(
                returns, dtype=torch.float32, device=device),
        }


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x_array):
        batch = np.asarray(x_array, dtype=np.float64)
        b_mean, b_var, b_count = batch.mean(), batch.var(), len(batch)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b
                    + delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8


# ─────────── Online PPO Trainer ───────────

class OnlinePPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # ── 奖励参数 (可 CLI 调整) ──
        self.reward_kwargs = dict(
            R_catch=args.r_catch,
            R_fail=args.r_fail,
            alpha=args.alpha_shaping,
            c_sg=args.c_sg,
            d_sg_safe=args.d_sg_safe,
            c_path=args.c_path,
            lam_time=args.lam_time,
        )

        # ── 环境公共参数 ──
        env_kwargs = dict(
            world_size=args.world_size,
            catch_radius=args.catch_radius,
            max_steps=args.max_steps,
            subgoal_range=args.subgoal_range,
            low_steps=args.low_steps,
            seed=args.seed,
            evader_mode=args.evader_mode,
        )

        # ── 创建向量化训练环境 ──
        if args.num_envs > 1:
            print(f"Creating SubprocVecEnv with {args.num_envs} workers ...")
            self.vec_env = SubprocVecEnv(
                n_envs=args.num_envs,
                project_root=PROJECT_ROOT,
                low_ckpt=args.low_level_ckpt,
                env_kwargs=env_kwargs,
                reward_kwargs=self.reward_kwargs,
                nn_device="cpu",
            )
            self.n_envs = args.num_envs
            self.use_vec = True
        else:
            # 单环境模式 (无子进程开销)
            self.nn_policy = load_v4_nn(
                args.low_level_ckpt, self.device)
            self.env = PursuitEnvRealAUVFixed(
                nn_policy=self.nn_policy, device=self.device,
                **env_kwargs,
            )
            self.n_envs = 1
            self.use_vec = False

        # ── 评估环境 (始终单进程) ──
        self.nn_policy_eval = load_v4_nn(
            args.low_level_ckpt, self.device)
        eval_kw = dict(env_kwargs)
        eval_kw["seed"] = args.seed + 9999
        self.eval_env = PursuitEnvRealAUVFixed(
            nn_policy=self.nn_policy_eval, device=self.device,
            **eval_kw,
        )

        # ── 高层网络 ──
        self.model = HighLevelActorCritic(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(self.device)

        # 加载 offline 预训练权重
        if args.pretrain_ckpt:
            self._load_pretrained(args.pretrain_ckpt)

        # 冻结一份 BC 策略用于 KL 惩罚
        self.bc_model = HighLevelActorCritic(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden=tuple(args.hidden),
        ).to(self.device)
        self.bc_model.load_state_dict(self.model.state_dict())
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)

        self.exp_name = (args.experiment_name
                         or f"online_ppo_{datetime.now():%Y%m%d_%H%M}")
        self.log_dir = os.path.join(
            PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(
            PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.total_steps = 0
        self.episode_count = 0
        self.success_count = 0
        self.start_update = 1

        # reward normalization
        self.reward_rms = (RunningMeanStd()
                           if getattr(args, 'normalize_reward', False)
                           else None)

        # ── 从 checkpoint 恢复 ──
        if args.resume:
            self._resume_from_checkpoint(args.resume)

    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrained (model): {path}")
        elif "actor" in ckpt:
            actor_sd = ckpt["actor"]
            new_sd = self.model.state_dict()
            if any(k.startswith("backbone.") for k in actor_sd):
                mapping = {
                    "backbone.0.weight": "actor_body.0.weight",
                    "backbone.0.bias": "actor_body.0.bias",
                    "backbone.1.weight": "actor_body.1.weight",
                    "backbone.1.bias": "actor_body.1.bias",
                    "backbone.3.weight": "actor_body.3.weight",
                    "backbone.3.bias": "actor_body.3.bias",
                    "backbone.4.weight": "actor_body.4.weight",
                    "backbone.4.bias": "actor_body.4.bias",
                    "mean_head.weight": "mu_head.weight",
                    "mean_head.bias": "mu_head.bias",
                }
                for src, dst in mapping.items():
                    if src in actor_sd:
                        new_sd[dst] = actor_sd[src]
                self.model.load_state_dict(new_sd)
                print(f"Loaded pretrained (BPPO→AC): {path}")
            else:
                mapping = {
                    "net.0.weight": "actor_body.0.weight",
                    "net.0.bias": "actor_body.0.bias",
                    "net.1.weight": "actor_body.1.weight",
                    "net.1.bias": "actor_body.1.bias",
                    "net.3.weight": "actor_body.3.weight",
                    "net.3.bias": "actor_body.3.bias",
                    "net.4.weight": "actor_body.4.weight",
                    "net.4.bias": "actor_body.4.bias",
                    "net.6.weight": "mu_head.weight",
                    "net.6.bias": "mu_head.bias",
                }
                for src, dst in mapping.items():
                    if src in actor_sd:
                        new_sd[dst] = actor_sd[src]
                self.model.load_state_dict(new_sd)
                print(f"Loaded pretrained (TD3Actor→AC): {path}")
        else:
            raise ValueError(f"Unknown ckpt keys: {list(ckpt.keys())}")

    def _resume_from_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        # steps_per_update = n_steps * n_envs (与 train() 一致)
        steps_per_update = self.args.n_steps * self.n_envs
        self.start_update = self.total_steps // steps_per_update + 1
        # 尝试加载历史
        hist_path = os.path.join(self.ckpt_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                self.history = json.load(f)
        print(f"Resumed from {path} (step={self.total_steps}, "
              f"update={self.start_update})")

    def collect_rollouts(self, n_steps: int):
        """采集数据 (支持单环境 / VecEnv)"""
        if self.use_vec:
            return self._collect_vec(n_steps)
        else:
            return self._collect_single(n_steps)

    # ───── 单环境采集 (兼容旧逻辑) ─────

    def _collect_single(self, n_steps):
        buf = RolloutBuffer()
        obs = self.env.reset()
        eta = self.env.auv_env.inner.dynamics.eta
        prev_dist = float(np.linalg.norm(
            np.array([eta[0], eta[1]]) - self.env.e))
        ep_rews, ep_lens = [], []
        cur_ep_rew, cur_ep_len = 0.0, 0
        # 分项累计
        ep_r_event, ep_r_shaping, ep_r_sg_obs, ep_r_time = 0.0, 0.0, 0.0, 0.0
        all_ep_components: List[Dict[str, float]] = []

        for _ in range(n_steps):
            obs_t = torch.tensor(
                obs, dtype=torch.float32,
                device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, lp = self.model.get_action(obs_t)
                v = self.model.get_value(obs_t)
            a_np = a[0].cpu().numpy()
            lp_val = lp.item()
            v_val = v.item()

            next_obs, _env_r, done, info = self.env.step(a_np)

            eta = self.env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            curr_dist = float(np.linalg.norm(p - self.env.e))

            r, r_info = compute_high_reward(
                self.env, prev_dist, curr_dist,
                info.get("captured", False),
                info.get("out_of_bounds", False),
                info.get("collision", False),
                subgoal=info.get("corrected_subgoal", None),
                done=done,
                **self.reward_kwargs,
            )

            buf.add(obs, a_np, r, done, v_val, lp_val, r_info=r_info)
            self.total_steps += 1
            cur_ep_rew += r
            cur_ep_len += 1
            ep_r_event += r_info["r_event"]
            ep_r_shaping += r_info["r_shaping"]
            ep_r_sg_obs += r_info["r_sg_obs"]
            ep_r_time += r_info["r_time"]

            if done:
                self.episode_count += 1
                if info.get("captured", False):
                    self.success_count += 1
                ep_rews.append(cur_ep_rew)
                ep_lens.append(cur_ep_len)
                all_ep_components.append({
                    "r_event": ep_r_event,
                    "r_shaping": ep_r_shaping,
                    "r_sg_obs": ep_r_sg_obs,
                    "r_time": ep_r_time,
                })
                cur_ep_rew, cur_ep_len = 0.0, 0
                ep_r_event, ep_r_shaping, ep_r_sg_obs, ep_r_time = 0.0, 0.0, 0.0, 0.0
                obs = self.env.reset()
                eta = self.env.auv_env.inner.dynamics.eta
                prev_dist = float(np.linalg.norm(
                    np.array([eta[0], eta[1]]) - self.env.e))
            else:
                obs = next_obs
                prev_dist = curr_dist

        with torch.no_grad():
            obs_t = torch.tensor(
                obs, dtype=torch.float32,
                device=self.device).unsqueeze(0)
            last_v = self.model.get_value(obs_t).item()

        rollout_info = self._summarize_rollout(
            ep_rews, ep_lens, all_ep_components)
        return [buf], [last_v], rollout_info

    # ───── VecEnv 并行采集 ─────

    def _collect_vec(self, n_steps):
        n_envs = self.n_envs
        bufs = [RolloutBuffer() for _ in range(n_envs)]
        obs_all = self.vec_env.reset()  # (n_envs, obs_dim)

        ep_rews, ep_lens = [], []
        cur_ep_rew = np.zeros(n_envs)
        cur_ep_len = np.zeros(n_envs, dtype=int)
        # 分项
        cur_r_event = np.zeros(n_envs)
        cur_r_shaping = np.zeros(n_envs)
        cur_r_sg_obs = np.zeros(n_envs)
        cur_r_time = np.zeros(n_envs)
        all_ep_components: List[Dict[str, float]] = []

        for _ in range(n_steps):
            obs_t = torch.tensor(
                obs_all, dtype=torch.float32,
                device=self.device)
            with torch.no_grad():
                actions, logprobs = self.model.get_action(obs_t)
                values = self.model.get_value(obs_t)
            actions_np = actions.cpu().numpy()
            lp_np = logprobs.cpu().numpy()
            v_np = values.cpu().numpy()

            next_obs, rews, dones, infos = self.vec_env.step(actions_np)

            for i in range(n_envs):
                bufs[i].add(
                    obs_all[i], actions_np[i], rews[i],
                    dones[i], v_np[i], lp_np[i],
                    r_info=infos[i],
                )
                cur_ep_rew[i] += rews[i]
                cur_ep_len[i] += 1
                cur_r_event[i] += infos[i].get("r_event", 0.0)
                cur_r_shaping[i] += infos[i].get("r_shaping", 0.0)
                cur_r_sg_obs[i] += infos[i].get("r_sg_obs", 0.0)
                cur_r_time[i] += infos[i].get("r_time", 0.0)

                if dones[i]:
                    self.episode_count += 1
                    if infos[i].get("captured", False):
                        self.success_count += 1
                    ep_rews.append(cur_ep_rew[i])
                    ep_lens.append(cur_ep_len[i])
                    all_ep_components.append({
                        "r_event": cur_r_event[i],
                        "r_shaping": cur_r_shaping[i],
                        "r_sg_obs": cur_r_sg_obs[i],
                        "r_time": cur_r_time[i],
                    })
                    cur_ep_rew[i] = 0.0
                    cur_ep_len[i] = 0
                    cur_r_event[i] = 0.0
                    cur_r_shaping[i] = 0.0
                    cur_r_sg_obs[i] = 0.0
                    cur_r_time[i] = 0.0

            self.total_steps += n_envs
            obs_all = next_obs

        # last values
        with torch.no_grad():
            obs_t = torch.tensor(
                obs_all, dtype=torch.float32,
                device=self.device)
            last_vs = self.model.get_value(obs_t).cpu().numpy()

        rollout_info = self._summarize_rollout(
            ep_rews, ep_lens, all_ep_components)
        return bufs, list(last_vs), rollout_info

    # ───── 汇总 rollout 统计 ─────

    @staticmethod
    def _summarize_rollout(ep_rews, ep_lens, all_ep_components):
        n = max(1, len(ep_rews))
        comp_means = {}
        for key in ["r_event", "r_shaping", "r_sg_obs", "r_time"]:
            vals = [c[key] for c in all_ep_components] if all_ep_components else [0.0]
            comp_means[key] = float(np.mean(vals))
        return {
            "ep_rews": list(ep_rews),
            "ep_lens": list(ep_lens),
            "mean_rew": float(np.mean(ep_rews)) if ep_rews else 0.0,
            "mean_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
            "n_eps": len(ep_rews),
            **comp_means,
        }

    def _get_beta_kl(self) -> float:
        """KL 惩罚系数线性退火"""
        args = self.args
        progress = min(1.0, self.total_steps / args.total_timesteps)
        # 在前 anneal_frac 比例内从 beta0 退火到 0
        if progress < args.kl_anneal_frac:
            return args.beta_kl * (1.0 - progress / args.kl_anneal_frac)
        return 0.0

    def train_step(self, bufs: List[RolloutBuffer],
                   last_vs: List[float]) -> Dict[str, float]:
        args = self.args

        # 每个 buffer 独立计算 GAE, 然后合并
        all_adv, all_ret = [], []
        all_s, all_a, all_lp = [], [], []
        for buf, lv in zip(bufs, last_vs):
            adv, ret = buf.compute_gae(lv, gamma=args.gamma,
                                       lam=args.gae_lambda)
            all_adv.append(adv)
            all_ret.append(ret)
            all_s.append(np.array(buf.states))
            all_a.append(np.array(buf.actions))
            all_lp.append(np.array(buf.logprobs))

        # concat & to tensor
        advantages = torch.tensor(
            np.concatenate(all_adv), dtype=torch.float32,
            device=self.device)
        returns = torch.tensor(
            np.concatenate(all_ret), dtype=torch.float32,
            device=self.device)
        s = torch.tensor(
            np.concatenate(all_s), dtype=torch.float32,
            device=self.device)
        a = torch.tensor(
            np.concatenate(all_a), dtype=torch.float32,
            device=self.device)
        old_lp = torch.tensor(
            np.concatenate(all_lp), dtype=torch.float32,
            device=self.device)

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8)

        beta_kl = self._get_beta_kl()
        total_loss = 0.0
        sum_actor = 0.0
        sum_critic = 0.0
        sum_ent = 0.0
        sum_kl = 0.0
        sum_clip_frac = 0.0
        n_up = 0

        for _ in range(args.n_epochs):
            perm = torch.randperm(len(s), device=self.device)
            for st in range(0, len(s), args.batch_size):
                idx = perm[st:st + args.batch_size]
                b_s, b_a = s[idx], a[idx]
                b_old_lp = old_lp[idx]
                b_adv, b_ret = advantages[idx], returns[idx]

                new_lp, ent = self.model.evaluate_actions(b_s, b_a)
                ratio = torch.exp(new_lp - b_old_lp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1 - args.clip_range,
                    1 + args.clip_range) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                v = self.model.get_value(b_s)
                critic_loss = F.mse_loss(v, b_ret)

                entropy = ent.mean()

                # clip fraction
                clip_frac = ((ratio - 1.0).abs()
                             > args.clip_range).float().mean()

                # KL penalty vs BC policy
                kl_loss = 0.0
                if beta_kl > 0:
                    bc_lp, _ = self.bc_model.evaluate_actions(
                        b_s, b_a)
                    kl_loss = (bc_lp - new_lp).mean()

                loss = (actor_loss
                        + args.vf_coef * critic_loss
                        - args.ent_coef * entropy
                        + beta_kl * kl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), args.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                sum_actor += actor_loss.item()
                sum_critic += critic_loss.item()
                sum_ent += entropy.item()
                sum_kl += (kl_loss.item()
                           if isinstance(kl_loss, torch.Tensor)
                           else kl_loss)
                sum_clip_frac += clip_frac.item()
                n_up += 1

        n = max(1, n_up)
        return {
            "loss": total_loss / n,
            "actor_loss": sum_actor / n,
            "critic_loss": sum_critic / n,
            "entropy": sum_ent / n,
            "kl": sum_kl / n,
            "clip_frac": sum_clip_frac / n,
            "beta_kl": beta_kl,
        }

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        rews, succs = [], 0
        all_comp: List[Dict[str, float]] = []
        ep_steps_list, capture_steps_list = [], []

        for ep in range(n_episodes):
            obs = self.eval_env.reset()
            eta = self.eval_env.auv_env.inner.dynamics.eta
            pd = float(np.linalg.norm(
                np.array([eta[0], eta[1]]) - self.eval_env.e))
            done, ep_r = False, 0.0
            ep_comp = {"r_event": 0, "r_shaping": 0,
                       "r_sg_obs": 0, "r_time": 0}
            ep_step = 0
            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a, _ = self.model.get_action(
                        obs_t, deterministic=True)
                a_np = a[0].cpu().numpy()
                obs, _, done, info = self.eval_env.step(a_np)
                eta = self.eval_env.auv_env.inner.dynamics.eta
                p = np.array([eta[0], eta[1]])
                cd = float(np.linalg.norm(p - self.eval_env.e))
                r, r_info = compute_high_reward(
                    self.eval_env, pd, cd,
                    info.get("captured", False),
                    info.get("out_of_bounds", False),
                    info.get("collision", False),
                    subgoal=info.get("corrected_subgoal", None),
                    done=done,
                    **self.reward_kwargs,
                )
                ep_r += r
                for k in ep_comp:
                    ep_comp[k] += r_info.get(k, 0.0)
                pd = cd
                ep_step += 1
                if info.get("captured", False):
                    succs += 1
                    capture_steps_list.append(ep_step)
            rews.append(ep_r)
            ep_steps_list.append(ep_step)
            all_comp.append(ep_comp)

        comp_means = {}
        for k in ["r_event", "r_shaping", "r_sg_obs", "r_time"]:
            comp_means[f"eval_{k}"] = float(
                np.mean([c[k] for c in all_comp]))
        return {
            "mean_reward": float(np.mean(rews)),
            "std_reward": float(np.std(rews)),
            "success_rate": succs / max(1, n_episodes),
            "mean_steps": float(np.mean(ep_steps_list)),
            "mean_capture_steps": (
                float(np.mean(capture_steps_list))
                if capture_steps_list else 0.0),
            **comp_means,
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.args.obs_dim,
            "act_dim": self.args.act_dim,
            "hidden": list(self.args.hidden),
            "total_steps": self.total_steps,
            "config": vars(self.args),
        }, path)
        # 保存历史到 JSON (恢复用)
        hist_path = os.path.join(self.ckpt_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f)
        print(f"Saved: {path}")

    def _get_lr(self, update, n_updates):
        """线性 LR 衰减"""
        args = self.args
        frac = 1.0 - (update - 1) / max(1, n_updates)
        return args.lr_end + (args.lr - args.lr_end) * frac

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train(self):
        args = self.args
        steps_per_update = args.n_steps * self.n_envs
        n_updates = max(1, args.total_timesteps // steps_per_update)
        norm_tag = " [reward-norm]" if self.reward_rms else ""
        print(f"Online PPO: {n_updates} updates, "
              f"{args.n_steps} steps × {self.n_envs} envs"
              f" = {steps_per_update}/update{norm_tag}")
        print(f"Reward params: {self.reward_kwargs}")
        t0 = time.time()
        best_eval_sr = -1.0

        for update in range(self.start_update, n_updates + 1):
            # LR 衰减
            cur_lr = self._get_lr(update, n_updates)
            self._set_lr(cur_lr)

            bufs, last_vs, r_info = self.collect_rollouts(args.n_steps)

            # reward normalization
            if self.reward_rms is not None:
                all_r = []
                for buf in bufs:
                    all_r.extend(buf.rewards)
                if all_r:
                    self.reward_rms.update(all_r)
                    for buf in bufs:
                        buf.rewards = [r / self.reward_rms.std
                                       for r in buf.rewards]
                    last_vs = [v / self.reward_rms.std for v in last_vs]

            info = self.train_step(bufs, last_vs)
            for buf in bufs:
                buf.clear()

            sr = self.success_count / max(1, self.episode_count)

            if update % 10 == 0 or update == self.start_update:
                elapsed = time.time() - t0
                fps = (self.total_steps
                       - (self.start_update - 1) * steps_per_update
                       ) / max(1, elapsed)
                print(
                    f"  Update {update}/{n_updates} | "
                    f"steps={self.total_steps} | "
                    f"ep_rew={r_info['mean_rew']:.1f} | "
                    f"ep_len={r_info['mean_len']:.0f} | "
                    f"sr={sr:.2%} | "
                    f"pi={info['actor_loss']:.3f} "
                    f"vf={info['critic_loss']:.1f} "
                    f"ent={info['entropy']:.3f} "
                    f"kl={info['kl']:.3f} "
                    f"clip={info['clip_frac']:.2f} | "
                    f"beta={info['beta_kl']:.3f} "
                    f"lr={cur_lr:.1e} | "
                    f"FPS={fps:.0f}")
                print(
                    f"    [Reward] event={r_info['r_event']:.1f} "
                    f"shaping={r_info['r_shaping']:.1f} "
                    f"sg_obs={r_info['r_sg_obs']:.1f} "
                    f"time={r_info['r_time']:.1f}")

            eval_info = None
            if self.total_steps % args.eval_freq < steps_per_update:
                eval_info = self.evaluate(args.eval_episodes)
                print(f"    [Eval] rew={eval_info['mean_reward']:.2f}"
                      f"±{eval_info['std_reward']:.1f}"
                      f"  sr={eval_info['success_rate']:.2%}"
                      f"  steps={eval_info['mean_steps']:.0f}"
                      f"  cap_steps={eval_info['mean_capture_steps']:.0f}")
                print(f"    [Eval Reward] "
                      f"event={eval_info['eval_r_event']:.1f} "
                      f"shaping={eval_info['eval_r_shaping']:.1f} "
                      f"sg_obs={eval_info['eval_r_sg_obs']:.1f} "
                      f"time={eval_info['eval_r_time']:.1f}")
                if eval_info["success_rate"] > best_eval_sr:
                    best_eval_sr = eval_info["success_rate"]
                    self.save_checkpoint("best")

            if self.total_steps % args.save_freq < steps_per_update:
                self.save_checkpoint("latest")

            self.history.append({
                "update": update,
                "steps": self.total_steps,
                "loss": info["loss"],
                "actor_loss": info["actor_loss"],
                "critic_loss": info["critic_loss"],
                "entropy": info["entropy"],
                "kl": info["kl"],
                "clip_frac": info["clip_frac"],
                "beta_kl": info["beta_kl"],
                "lr": cur_lr,
                "train_sr": sr,
                "train_rew": r_info["mean_rew"],
                "train_ep_len": r_info["mean_len"],
                "train_r_event": r_info["r_event"],
                "train_r_shaping": r_info["r_shaping"],
                "train_r_sg_obs": r_info["r_sg_obs"],
                "train_r_time": r_info["r_time"],
                "eval_rew": (eval_info["mean_reward"]
                             if eval_info else None),
                "eval_sr": (eval_info["success_rate"]
                            if eval_info else None),
                "eval_r_event": (eval_info.get("eval_r_event")
                                 if eval_info else None),
                "eval_r_shaping": (eval_info.get("eval_r_shaping")
                                   if eval_info else None),
                "eval_r_sg_obs": (eval_info.get("eval_r_sg_obs")
                                  if eval_info else None),
                "eval_r_time": (eval_info.get("eval_r_time")
                                if eval_info else None),
            })

        self.save_checkpoint("final")
        self.plot_curves()
        if self.use_vec:
            self.vec_env.close()
        print("Online PPO fine-tuning done.")

    def plot_curves(self):
        if not self.history:
            return
        steps = [h["steps"] for h in self.history]
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))

        def _plot(ax, key, title, **kw):
            vals = [h.get(key, 0) for h in self.history]
            ax.plot(steps, vals, lw=1, **kw)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

        # row 0: rewards
        _plot(axes[0, 0], "train_rew", "Train Ep Reward")
        evals_r = [(h["steps"], h["eval_rew"])
                   for h in self.history if h["eval_rew"] is not None]
        if evals_r:
            axes[0, 0].plot(*zip(*evals_r), "o-", ms=3,
                            color="tab:orange", label="eval")
            axes[0, 0].legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(steps, [h["train_sr"] for h in self.history], lw=1)
        evals_s = [(h["steps"], h["eval_sr"])
                   for h in self.history if h["eval_sr"] is not None]
        if evals_s:
            ax.plot(*zip(*evals_s), "o-", ms=3, label="eval")
        ax.set_title("Success Rate")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        _plot(axes[0, 2], "train_ep_len", "Episode Length")

        # row 1: reward components (TRAIN)
        _plot(axes[1, 0], "train_r_event", "R_event (train)",
              color="tab:green")
        evals_re = [(h["steps"], h["eval_r_event"])
                    for h in self.history
                    if h.get("eval_r_event") is not None]
        if evals_re:
            axes[1, 0].plot(*zip(*evals_re), "o-", ms=3,
                            color="tab:orange", label="eval")
            axes[1, 0].legend(fontsize=8)

        _plot(axes[1, 1], "train_r_shaping", "R_shaping (train)",
              color="tab:blue")
        evals_rs = [(h["steps"], h["eval_r_shaping"])
                    for h in self.history
                    if h.get("eval_r_shaping") is not None]
        if evals_rs:
            axes[1, 1].plot(*zip(*evals_rs), "o-", ms=3,
                            color="tab:orange", label="eval")
            axes[1, 1].legend(fontsize=8)

        _plot(axes[1, 2], "train_r_sg_obs", "R_sg_obs (train)",
              color="tab:red")
        evals_ro = [(h["steps"], h["eval_r_sg_obs"])
                    for h in self.history
                    if h.get("eval_r_sg_obs") is not None]
        if evals_ro:
            axes[1, 2].plot(*zip(*evals_ro), "o-", ms=3,
                            color="tab:orange", label="eval")
            axes[1, 2].legend(fontsize=8)

        # row 2: loss / policy stats
        _plot(axes[2, 0], "actor_loss", "Actor Loss")
        _plot(axes[2, 1], "critic_loss", "Critic Loss")
        _plot(axes[2, 2], "entropy", "Entropy")

        # row 3: KL / clip / LR
        _plot(axes[3, 0], "kl", "KL vs BC")
        _plot(axes[3, 1], "clip_frac", "Clip Fraction")
        ax_lr = axes[3, 2]
        ax_lr.plot(steps, [h.get("lr", self.args.lr)
                           for h in self.history],
                   lw=1, color="tab:purple")
        ax_lr.set_title("Learning Rate")
        ax_lr.grid(True, alpha=0.25)

        plt.suptitle(f"Online PPO Fine-tuning — {self.exp_name}")
        plt.tight_layout()
        out = os.path.join(self.log_dir, "online_ppo_curves.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


# ─────────── main ───────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--low-level-ckpt", type=str, required=True)
    pa.add_argument("--pretrain-ckpt", type=str, default=None,
                    help="Offline pretrained checkpoint (best.pth)")
    pa.add_argument("--resume", type=str, default=None,
                    help="Resume from this checkpoint (latest.pth)")
    pa.add_argument("--experiment-name", type=str, default=None)
    pa.add_argument("--obs-dim", type=int, default=21)
    pa.add_argument("--act-dim", type=int, default=2)
    pa.add_argument("--hidden", type=int, nargs="+",
                    default=[256, 128])
    pa.add_argument("--total-timesteps", type=int, default=500_000)
    pa.add_argument("--n-steps", type=int, default=256)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--n-epochs", type=int, default=3)
    pa.add_argument("--lr", type=float, default=1e-5)
    pa.add_argument("--lr-end", type=float, default=1e-6,
                    help="LR at end of training (linear decay)")
    pa.add_argument("--gamma", type=float, default=0.99)
    pa.add_argument("--gae-lambda", type=float, default=0.95)
    pa.add_argument("--clip-range", type=float, default=0.1)
    pa.add_argument("--ent-coef", type=float, default=0.01)
    pa.add_argument("--vf-coef", type=float, default=0.5)
    pa.add_argument("--max-grad-norm", type=float, default=0.5)
    pa.add_argument("--beta-kl", type=float, default=0.3,
                    help="Initial KL penalty coefficient")
    pa.add_argument("--kl-anneal-frac", type=float, default=0.3,
                    help="Fraction of training to anneal KL to 0")
    pa.add_argument("--eval-freq", type=int, default=10000)
    pa.add_argument("--eval-episodes", type=int, default=20)
    pa.add_argument("--save-freq", type=int, default=50000)
    # 环境参数
    pa.add_argument("--world-size", type=float, default=30.0)
    pa.add_argument("--catch-radius", type=float, default=1.0)
    pa.add_argument("--subgoal-range", type=float, default=0.7)
    pa.add_argument("--low-steps", type=int, default=50)
    pa.add_argument("--max-steps", type=int, default=6000)
    pa.add_argument("--evader-mode", type=str, default="medium")
    pa.add_argument("--normalize-reward", action="store_true",
                    help="Normalize rewards by running std")
    pa.add_argument("--seed", type=int, default=42)
    # 并行
    pa.add_argument("--num-envs", type=int, default=4,
                    help="Number of parallel envs (SubprocVecEnv)")
    # 奖励权重 (可调)
    pa.add_argument("--r-catch", type=float, default=150.0,
                    help="Capture reward (+)")
    pa.add_argument("--r-fail", type=float, default=-50.0,
                    help="Failure penalty (OOB/collision)")
    pa.add_argument("--alpha-shaping", type=float, default=5.0,
                    help="PBRS distance shaping coefficient")
    pa.add_argument("--c-sg", type=float, default=10.0,
                    help="Subgoal near-obstacle penalty coeff")
    pa.add_argument("--d-sg-safe", type=float, default=1.5,
                    help="Subgoal safe distance to obstacle")
    pa.add_argument("--c-path", type=float, default=5.0,
                    help="Path-through-obstacle penalty")
    pa.add_argument("--lam-time", type=float, default=0.02,
                    help="Per-step time penalty")
    args = pa.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = OnlinePPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
