"""
SubprocVecEnv  ——  多进程并行环境
=======================================
每个 worker 在独立进程中运行 PursuitEnvRealAUVFixed + reward 计算,
主进程只做高层 NN 推理 (GPU batch), 实现近线性加速。
"""

import multiprocessing as mp
import numpy as np
import torch
import sys
import os


def _worker_fn(conn, project_root, low_ckpt, nn_device, env_kwargs, reward_kwargs,
               env_version="v4"):
    """Worker 进程: 创建环境 + 处理 step/reset/close 命令。"""
    sys.path.insert(0, project_root)
    from training.collect_expert_trajectories import compute_high_reward

    if env_version == "v5":
        from high_tabular.hrl_v3_real_auv_direct_thrust import (
            load_v5_nn, PursuitEnvDirectThrust,
        )
        nn_policy = load_v5_nn(low_ckpt, torch.device(nn_device))
        env = PursuitEnvDirectThrust(
            nn_policy=nn_policy,
            device=torch.device(nn_device),
            **env_kwargs,
        )
    else:
        from high_tabular.hrl_v3_real_auv_fixed import (
            load_v4_nn, PursuitEnvRealAUVFixed,
        )
        nn_policy = load_v4_nn(low_ckpt, torch.device(nn_device))
        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy,
            device=torch.device(nn_device),
            **env_kwargs,
        )
    prev_dist = None

    try:
        while True:
            cmd, data = conn.recv()

            if cmd == "reset":
                obs = env.reset()
                eta = env.auv_env.inner.dynamics.eta
                prev_dist = float(np.linalg.norm(
                    np.array([eta[0], eta[1]]) - env.e))
                conn.send(obs)

            elif cmd == "step":
                action = data
                obs, _env_r, done, info = env.step(action)

                eta = env.auv_env.inner.dynamics.eta
                p = np.array([eta[0], eta[1]])
                curr_dist = float(np.linalg.norm(p - env.e))

                r, r_info = compute_high_reward(
                    env, prev_dist, curr_dist,
                    info.get("captured", False),
                    info.get("out_of_bounds", False),
                    info.get("collision", False),
                    subgoal=info.get("corrected_subgoal", None),
                    done=done,
                    **reward_kwargs,
                )

                # 把奖励分项放进 info
                info.update(r_info)

                if done:
                    # 自动重置, 但先告诉主进程这个 episode 结束了
                    obs = env.reset()
                    eta = env.auv_env.inner.dynamics.eta
                    prev_dist = float(np.linalg.norm(
                        np.array([eta[0], eta[1]]) - env.e))
                else:
                    prev_dist = curr_dist

                conn.send((obs, r, done, info))

            elif cmd == "close":
                break
    finally:
        conn.close()


class SubprocVecEnv:
    """多进程向量化环境。

    - 每个 worker 独立创建 env + low-level NN (CPU)
    - 主进程批量 GPU 推理后分发动作
    - 自动 auto-reset
    """

    def __init__(self, n_envs, project_root, low_ckpt,
                 env_kwargs, reward_kwargs, nn_device="cpu",
                 env_version="v4"):
        self.n_envs = n_envs
        self.env_version = env_version
        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.processes = []

        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            kw = dict(env_kwargs)
            kw["seed"] = kw.get("seed", 42) + i * 1000

            p = ctx.Process(
                target=_worker_fn,
                args=(child_conn, project_root, low_ckpt,
                      nn_device, kw, reward_kwargs, env_version),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(p)

    def reset(self):
        for conn in self.parent_conns:
            conn.send(("reset", None))
        return np.array([conn.recv() for conn in self.parent_conns])

    def step(self, actions):
        """actions: (n_envs, act_dim)"""
        for conn, a in zip(self.parent_conns, actions):
            conn.send(("step", a))
        results = [conn.recv() for conn in self.parent_conns]
        obs, rews, dones, infos = zip(*results)
        return np.array(obs), np.array(rews), np.array(dones), list(infos)

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
