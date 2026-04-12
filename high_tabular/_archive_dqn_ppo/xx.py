#!/usr/bin/env python3  
"""  
2D Pursuit-Evasion with Rectangular Obstacles — DQN (GPU)  
==========================================================  
验证 §1.2: 追击者利用地形 + 加速度优势 截击逃逸者  
  
核心机制:  
  - 矩形障碍物阻挡运动（碰撞反弹/滑行）  
  - 逃逸者被逼近障碍物 → 被迫转弯 → 因加速度劣势减速 → 追击者拉近  
  - 地形利用奖励 (Eq.27): 逃逸者离障碍物越近，追击者获正奖励  
  
运行:  
  pip install torch numpy matplotlib  
  python pursuit_evasion_obstacle_dqn.py  
"""  
  
import numpy as np  
import math  
import random  
from collections import deque  
from typing import Dict, Tuple, Optional, List  
  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
  
# ============ 自动选择设备 ============  
if torch.cuda.is_available():  
    DEVICE = torch.device("cuda")  
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")  
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  
    DEVICE = torch.device("mps")  
    print("[INFO] Using Apple MPS")  
else:  
    DEVICE = torch.device("cpu")  
    print("[INFO] Using CPU")  
  
  
# ================================================================  
#  1. 矩形障碍物  
# ================================================================  
class RectObstacle:  
    """轴对齐矩形障碍物"""  
    def __init__(self, cx: float, cy: float, w: float, h: float):  
        self.cx, self.cy = cx, cy  
        self.w, self.h = w, h  
        self.x_min = cx - w / 2  
        self.x_max = cx + w / 2  
        self.y_min = cy - h / 2  
        self.y_max = cy + h / 2  
  
    def contains(self, p: np.ndarray) -> bool:  
        return (self.x_min <= p[0] <= self.x_max and  
                self.y_min <= p[1] <= self.y_max)  
  
    def distance(self, p: np.ndarray) -> float:  
        """点到矩形边界的最短距离（内部为负）"""  
        dx = max(self.x_min - p[0], 0, p[0] - self.x_max)  
        dy = max(self.y_min - p[1], 0, p[1] - self.y_max)  
        if dx == 0 and dy == 0:  
            # 在内部：返回到最近边的负距离  
            return -min(p[0] - self.x_min, self.x_max - p[0],  
                        p[1] - self.y_min, self.y_max - p[1])  
        return math.sqrt(dx ** 2 + dy ** 2)  
  
    def nearest_boundary_point(self, p: np.ndarray) -> np.ndarray:  
        """从 p 到矩形边界的最近点"""  
        cx = np.clip(p[0], self.x_min, self.x_max)  
        cy = np.clip(p[1], self.y_min, self.y_max)  
        return np.array([cx, cy])  
  
    def resolve_collision(self, p_old: np.ndarray, p_new: np.ndarray,  
                          v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        碰撞处理：如果 p_new 在障碍物内，推回到表面并反弹/滑行。  
        返回修正后的 (position, velocity)。  
        """  
        if not self.contains(p_new):  
            return p_new, v  
  
        # 找到从 p_old 进入的那一面，推回  
        new_p = p_new.copy()  
        new_v = v.copy()  
  
        # 到各面的距离  
        d_left   = abs(p_new[0] - self.x_min)  
        d_right  = abs(p_new[0] - self.x_max)  
        d_bottom = abs(p_new[1] - self.y_min)  
        d_top    = abs(p_new[1] - self.y_max)  
        d_min = min(d_left, d_right, d_bottom, d_top)  
  
        margin = 0.05  
        if d_min == d_left:  
            new_p[0] = self.x_min - margin  
            new_v[0] = min(new_v[0], 0) * 0.1  # 弹性很小，近似滑行  
        elif d_min == d_right:  
            new_p[0] = self.x_max + margin  
            new_v[0] = max(new_v[0], 0) * 0.1  
        elif d_min == d_bottom:  
            new_p[1] = self.y_min - margin  
            new_v[1] = min(new_v[1], 0) * 0.1  
        elif d_min == d_top:  
            new_p[1] = self.y_max + margin  
            new_v[1] = max(new_v[1], 0) * 0.1  
  
        return new_p, new_v  
  
  
# ================================================================  
#  2. 环境  
# ================================================================  
def wrap_angle(a: float) -> float:  
    return (a + np.pi) % (2 * np.pi) - np.pi  
  
  
class PursuitEvasionObstacle:  
    """  
    带矩形障碍物的 2D 追逃环境  
  
    Arena: [-L, L] x [-L, L]  
    障碍物: 可配置的矩形列表  
    """  
  
    def __init__(self, cfg: Optional[Dict] = None):  
        c = cfg or {}  
  
        # ---- 动力学 ----  
        self.v_max_p: float = c.get("v_max_p", 1.0)  
        self.v_max_e: float = c.get("v_max_e", 1.5)  
        self.a_max_p: float = c.get("a_max_p", 2.0)  
        self.a_max_e: float = c.get("a_max_e", 0.5)  
  
        # ---- 时间 ----  
        self.dt: float  = c.get("dt", 0.1)  
        self.K: int     = c.get("K", 10)  
        self.N_max: int = c.get("N_max", 100)  
  
        # ---- Arena ----  
        self.L: float = c.get("L", 15.0)  
  
        # ---- 障碍物 ----  
        default_obs = [  
            {"cx": 0, "cy": 0, "w": 8, "h": 2},   # 中央长方形  
        ]  
        obs_list = c.get("obstacles", default_obs)  
        self.obstacles: List[RectObstacle] = [  
            RectObstacle(**o) for o in obs_list  
        ]  
  
        # ---- 几何 ----  
        self.D_c: float      = c.get("D_c", 1.0)  
        self.D_escape: float = c.get("D_escape", 28.0)  
        self.l_high: float   = c.get("l_high", 5.0)  
        self.n_actions: int  = c.get("n_actions", 8)  
  
        # ---- 奖励 ----  
        self.R_win: float     = c.get("R_win", 200.0)  
        self.R_lose: float    = c.get("R_lose", 80.0)  
        self.w_close: float   = c.get("w_close", 8.0)  
        self.w_int: float     = c.get("w_int", 0.5)  
        self.w_terrain: float = c.get("w_terrain", 3.0)  
        self.w_energy: float  = c.get("w_energy", 0.05)  
        self.c_time: float    = c.get("c_time", 0.3)  
        self.tau_max: float   = c.get("tau_max", 5.0)  
  
        # ---- 地形射线 (§1.2.2 terrain obs) ----  
        self.n_rays: int      = c.get("n_rays", 8)  
        self.ray_max: float   = c.get("ray_max", 15.0)  
  
        # ---- 逃逸策略 ----  
        self.evader_policy: str = c.get("evader_policy", "smart_avoid")  
  
        # 状态维度  
        # pursuer: 5 + evader: 4 + rel: 5 + pursuer_rays: 8 + evader_rays: 8  
        #   + evader_obs_dist: 1 = 31  
        self.obs_dim = 5 + 4 + 5 + self.n_rays * 2 + 1  
  
        self.reset()  
  
    def _in_any_obstacle(self, p: np.ndarray) -> bool:  
        return any(o.contains(p) for o in self.obstacles)  
  
    def _min_obs_dist(self, p: np.ndarray) -> float:  
        if not self.obstacles:  
            return self.ray_max  
        return min(o.distance(p) for o in self.obstacles)  
  
    def _cast_rays(self, p: np.ndarray, n: int) -> np.ndarray:  
        """从 p 向 n 个均匀方向投射射线，返回到最近障碍物/边界距离"""  
        dists = np.full(n, self.ray_max)  
        for i in range(n):  
            ang = i * 2 * np.pi / n  
            d_hat = np.array([np.cos(ang), np.sin(ang)])  
            # 步进式射线检测  
            for t_step in np.linspace(0.2, self.ray_max, 60):  
                test = p + t_step * d_hat  
                # 边界  
                if abs(test[0]) > self.L or abs(test[1]) > self.L:  
                    dists[i] = t_step  
                    break  
                # 障碍物  
                if self._in_any_obstacle(test):  
                    dists[i] = t_step  
                    break  
        return dists  
  
    def _resolve_all_collisions(self, p_old, p_new, v):  
        """对所有障碍物 + 边界做碰撞检测"""  
        for obs in self.obstacles:  
            p_new, v = obs.resolve_collision(p_old, p_new, v)  
        # 边界  
        for dim in [0, 1]:  
            if p_new[dim] < -self.L:  
                p_new[dim] = -self.L + 0.05  
                v[dim] = max(v[dim], 0) * 0.1  
            elif p_new[dim] > self.L:  
                p_new[dim] = self.L - 0.05  
                v[dim] = min(v[dim], 0) * 0.1  
        return p_new, v  
  
    def _safe_spawn(self) -> np.ndarray:  
        """在无障碍位置生成点"""  
        for _ in range(500):  
            p = np.random.uniform(-self.L * 0.8, self.L * 0.8, 2)  
            if not self._in_any_obstacle(p):  
                return p  
        return np.array([self.L * 0.7, self.L * 0.7])  
  
    # ----------------------------------------------------------  
    def reset(self, seed=None) -> np.ndarray:  
        if seed is not None:  
            np.random.seed(seed)  
            random.seed(seed)  
  
        # 追击者  
        self.p_p = self._safe_spawn()  
        self.v_p = np.zeros(2)  
  
        # 逃逸者：距追击者 5~10  
        for _ in range(200):  
            ang  = np.random.uniform(0, 2 * np.pi)  
            dist = np.random.uniform(5, 10)  
            cand = self.p_p + dist * np.array([np.cos(ang), np.sin(ang)])  
            if (not self._in_any_obstacle(cand) and  
                    np.all(np.abs(cand) < self.L * 0.9)):  
                self.p_e = cand  
                break  
        else:  
            self.p_e = self._safe_spawn()  
  
        e_ang = np.random.uniform(0, 2 * np.pi)  
        self.v_e = 0.5 * self.v_max_e * np.array([np.cos(e_ang), np.sin(e_ang)])  
  
        self.step_n = 0  
        self.prev_avec = np.zeros(2)  
        self.done = False  
        self.info = {}  
        return self._get_obs()  
  
    # ----------------------------------------------------------  
    def _get_obs(self) -> np.ndarray:  
        """向量化观测 → 用于神经网络"""  
        dp  = self.p_e - self.p_p  
        d   = np.linalg.norm(dp) + 1e-8  
        hat = dp / d  
  
        psi_p = np.arctan2(self.v_p[1], self.v_p[0])  
        sp    = np.linalg.norm(self.v_p)  
        psi_e = np.arctan2(self.v_e[1], self.v_e[0])  
        se    = np.linalg.norm(self.v_e)  
  
        alpha = np.arctan2(dp[1], dp[0])  
  
        if se > 1e-6:  
            cos_b = np.clip(np.dot(self.v_e, dp) / (se * d), -1, 1)  
            beta  = np.arccos(cos_b)  
        else:  
            beta = np.pi / 2  
  
        v_close   = np.dot(self.v_p - self.v_e, hat)  
        delta_psi = wrap_angle(alpha - psi_p)  
  
        # 射线  
        rays_p = self._cast_rays(self.p_p, self.n_rays) / self.ray_max  
        rays_e = self._cast_rays(self.p_e, self.n_rays) / self.ray_max  
  
        # 逃逸者到最近障碍物距离  
        d_e_obs = self._min_obs_dist(self.p_e) / self.ray_max  
  
        obs = np.concatenate([  
            # pursuer (5)  
            [self.p_p[0] / self.L, self.p_p[1] / self.L,  
             psi_p / np.pi, sp / self.v_max_p,  
             np.sin(psi_p)],  # 加个 sin 增加表达力  
            # evader (4)  
            [self.p_e[0] / self.L, self.p_e[1] / self.L,  
             psi_e / np.pi, se / self.v_max_e],  
            # relative (5)  
            [d / 20.0, alpha / np.pi, beta / np.pi,  
             v_close / 3.0, delta_psi / np.pi],  
            # terrain rays (n_rays * 2)  
            rays_p, rays_e,  
            # evader obstacle dist (1)  
            [d_e_obs],  
        ]).astype(np.float32)  
        return obs  
  
    def _get_obs_dict(self) -> Dict:  
        """字典版（用于奖励计算）"""  
        dp  = self.p_e - self.p_p  
        d   = np.linalg.norm(dp) + 1e-8  
        hat = dp / d  
        alpha = np.arctan2(dp[1], dp[0])  
        se = np.linalg.norm(self.v_e)  
        return dict(  
            p_p=self.p_p.copy(), v_p=self.v_p.copy(),  
            p_e=self.p_e.copy(), v_e=self.v_e.copy(),  
            d_pe=d, alpha_pe=alpha,  
            d_e_obs=self._min_obs_dist(self.p_e),  
        )  
  
    # ----------------------------------------------------------  
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:  
        obs0 = self._get_obs_dict()  
        d0   = obs0["d_pe"]  
        d_e_obs0 = obs0["d_e_obs"]  
        alpha = obs0["alpha_pe"]  
  
        # 动作 → 子目标  
        offset = action_idx * (2 * np.pi / self.n_actions)  
        world_ang = alpha + offset  
        avec = np.array([np.cos(world_ang), np.sin(world_ang)])  
        goal = self.p_p + self.l_high * avec  
  
        # K 底层步  
        captured = escaped = oob = False  
        for _ in range(self.K):  
            self._ctrl_pursuer(goal)  
            self._ctrl_evader()  
            d_now = np.linalg.norm(self.p_e - self.p_p)  
            if d_now < self.D_c:  
                captured = True; break  
            if d_now > self.D_escape:  
                escaped = True; break  
  
        self.step_n += 1  
        obs1 = self._get_obs_dict()  
        d1   = obs1["d_pe"]  
        d_e_obs1 = obs1["d_e_obs"]  
  
        # ---- 奖励 ----  
        r = 0.0  
  
        # (i) 捕获/失败  
        if captured:  
            r += self.R_win  
        elif escaped or self.step_n >= self.N_max:  
            r -= self.R_lose  
  
        # (ii) 接近  
        r += self.w_close * (d0 - d1)  
  
        # (iii) 截击几何  
        taus = np.linspace(0, self.tau_max, 20)  
        preds = obs0["p_e"][None, :] + taus[:, None] * obs0["v_e"][None, :]  
        dists = np.linalg.norm(obs1["p_p"][None, :] - preds, axis=1)  
        r -= self.w_int * min(dists.min(), 20.0)  
  
        # (iv) 地形利用 (Eq.27): 逃逸者离障碍物变近 → 正奖励  
        r += self.w_terrain * (d_e_obs0 - d_e_obs1)  
  
        # (v) 能量  
        r -= self.w_energy * np.sum((avec - self.prev_avec) ** 2)  
  
        # (vi) 时间  
        r -= self.c_time  
  
        self.prev_avec = avec.copy()  
        terminated = captured or escaped  
        truncated  = (self.step_n >= self.N_max) and not terminated  
        self.done  = terminated or truncated  
        self.info  = dict(captured=captured, escaped=escaped,  
                          step_n=self.step_n, d_pe=d1,  
                          d_e_obs=d_e_obs1)  
        return self._get_obs(), r, terminated, truncated, self.info  
  
    # ----------------------------------------------------------  
    def _ctrl_pursuer(self, goal):  
        dp   = goal - self.p_p  
        dist = np.linalg.norm(dp) + 1e-8  
        d_dir = dp / dist  
        d_spd = min(self.v_max_p, dist / (3 * self.dt + 1e-8))  
        dv    = d_dir * d_spd - self.v_p  
        mag   = np.linalg.norm(dv)  
        if mag > self.a_max_p * self.dt:  
            dv *= self.a_max_p * self.dt / mag  
        self.v_p += dv  
        spd = np.linalg.norm(self.v_p)  
        if spd > self.v_max_p:  
            self.v_p *= self.v_max_p / spd  
        p_old = self.p_p.copy()  
        self.p_p += self.v_p * self.dt  
        self.p_p, self.v_p = self._resolve_all_collisions(  
            p_old, self.p_p, self.v_p)  
  
    def _ctrl_evader(self):  
        dp   = self.p_e - self.p_p  
        dist = np.linalg.norm(dp) + 1e-8  
        away = dp / dist  
  
        if self.evader_policy == "smart_avoid":  
            # 远离追击者 + 远离障碍物 + 远离边界  
            force = away * 1.0  
  
            # 障碍物斥力  
            for obs in self.obstacles:  
                d_obs = obs.distance(self.p_e)  
                if d_obs < 4.0:  
                    bp = obs.nearest_boundary_point(self.p_e)  
                    diff = self.p_e - bp  
                    nd = np.linalg.norm(diff) + 1e-8  
                    force += (diff / nd) * (2.0 / (d_obs + 0.3))  
  
            # 边界斥力  
            for dim in [0, 1]:  
                if self.p_e[dim] > self.L * 0.7:  
                    force[dim] -= 1.5  
                elif self.p_e[dim] < -self.L * 0.7:  
                    force[dim] += 1.5  
  
            # 侧向分量（利用速度优势走弧线）  
            tang = np.array([-away[1], away[0]])  
            if np.dot(tang, self.v_e) < 0:  
                tang = -tang  
            force += tang * 0.3  
  
            fn = np.linalg.norm(force)  
            d_dir = force / fn if fn > 1e-8 else away  
        else:  
            d_dir = away  
  
        dv  = d_dir * self.v_max_e - self.v_e  
        mag = np.linalg.norm(dv)  
        if mag > self.a_max_e * self.dt:  
            dv *= self.a_max_e * self.dt / mag  
        self.v_e += dv  
        spd = np.linalg.norm(self.v_e)  
        if spd > self.v_max_e:  
            self.v_e *= self.v_max_e / spd  
        p_old = self.p_e.copy()  
        self.p_e += self.v_e * self.dt  
        self.p_e, self.v_e = self._resolve_all_collisions(  
            p_old, self.p_e, self.v_e)  
  
  
# ================================================================  
#  3. DQN 网络  
# ================================================================  
class DQN(nn.Module):  
    def __init__(self, obs_dim: int, n_actions: int, hidden=256):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(obs_dim, hidden), nn.ReLU(),  
            nn.Linear(hidden, hidden),  nn.ReLU(),  
            nn.Linear(hidden, hidden),  nn.ReLU(),  
            nn.Linear(hidden, n_actions),  
        )  
  
    def forward(self, x):  
        return self.net(x)  
  
  
# ================================================================  
#  4. Replay Buffer  
# ================================================================  
class ReplayBuffer:  
    def __init__(self, capacity=200000):  
        self.buf = deque(maxlen=capacity)  
  
    def push(self, s, a, r, s2, done):  
        self.buf.append((s, a, r, s2, done))  
  
    def sample(self, batch_size):  
        batch = random.sample(self.buf, batch_size)  
        s, a, r, s2, d = zip(*batch)  
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),  
                np.array(s2), np.array(d, dtype=np.float32))  
  
    def __len__(self):  
        return len(self.buf)  
  
  
# ================================================================  
#  5. DQN Agent  
# ================================================================  
class DQNAgent:  
    def __init__(self, obs_dim, n_actions, cfg=None):  
        c = cfg or {}  
        self.n_actions = n_actions  
        self.gamma     = c.get("gamma", 0.99)  
        self.lr        = c.get("lr", 3e-4)  
        self.batch_sz  = c.get("batch_size", 256)  
        self.eps       = c.get("eps_start", 1.0)  
        self.eps_end   = c.get("eps_end", 0.05)  
        self.eps_decay = c.get("eps_decay", 80000)  
        self.tau       = c.get("tau", 0.005)  
        self.steps     = 0  
  
        self.policy_net = DQN(obs_dim, n_actions).to(DEVICE)  
        self.target_net = DQN(obs_dim, n_actions).to(DEVICE)  
        self.target_net.load_state_dict(self.policy_net.state_dict())  
        self.target_net.eval()  
  
        self.opt    = optim.Adam(self.policy_net.parameters(), lr=self.lr)  
        self.buffer = ReplayBuffer(c.get("buffer_size", 200000))  
  
        print(f"  DQN params: {sum(p.numel() for p in self.policy_net.parameters()):,}")  
  
    def act(self, obs: np.ndarray, greedy=False) -> int:  
        if not greedy:  
            self.eps = self.eps_end + (1.0 - self.eps_end) * math.exp(-self.steps / self.eps_decay)       
            self.steps += 1  
            if random.random() < self.eps:  
                return random.randint(0, self.n_actions - 1)  
        with torch.no_grad():  
            q = self.policy_net(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))  
            return q.argmax(1).item()  
  
    def store(self, s, a, r, s2, done):  
        self.buffer.push(s, a, r, s2, done)  
  
    def train_step(self) -> Optional[float]:  
        if len(self.buffer) < self.batch_sz * 4:  
            return None  
  
        s, a, r, s2, d = self.buffer.sample(self.batch_sz)  
        s_t  = torch.FloatTensor(s).to(DEVICE)  
        a_t  = torch.LongTensor(a).to(DEVICE)  
        r_t  = torch.FloatTensor(r).to(DEVICE)  
        s2_t = torch.FloatTensor(s2).to(DEVICE)  
        d_t  = torch.FloatTensor(d).to(DEVICE)  
  
        # 当前 Q  
        q_vals = self.policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)  
  
        # Double DQN target  
        with torch.no_grad():  
            best_a = self.policy_net(s2_t).argmax(1)  
            q_next = self.target_net(s2_t).gather(1, best_a.unsqueeze(1)).squeeze(1)  
            target = r_t + self.gamma * q_next * (1 - d_t)  
  
        loss = F.smooth_l1_loss(q_vals, target)  
  
        self.opt.zero_grad()  
        loss.backward()  
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  
        self.opt.step()  
  
        # Soft update target  
        for p, tp in zip(self.policy_net.parameters(),  
                         self.target_net.parameters()):  
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)  
  
        return loss.item()  
  
  
# ================================================================  
#  6. 基线策略  
# ================================================================  
def policy_pure_pursuit(obs_vec, env):  
    return 0  
  
def policy_random(obs_vec, env):  
    return random.randint(0, env.n_actions - 1)  
  
def policy_lead(obs_vec, env):  
    dp = env.p_e - env.p_p  
    d  = np.linalg.norm(dp) + 1e-8  
    alpha = np.arctan2(dp[1], dp[0])  
    tau = d / (env.v_max_p + 1e-8) * 0.5  
    pred = env.p_e + tau * env.v_e  
    dp2  = pred - env.p_p  
    ang  = np.arctan2(dp2[1], dp2[0])  
    rel  = wrap_angle(ang - alpha)  
    idx  = int(round(rel / (2 * np.pi / env.n_actions))) % env.n_actions  
    return idx  
  
  
# ================================================================  
#  7. 训练循环  
# ================================================================  
def train_dqn(cfg, n_episodes=15000, log_every=1000):  
    env   = PursuitEvasionObstacle(cfg)  
    agent = DQNAgent(env.obs_dim, env.n_actions, cfg.get("dqn", {}))  
  
    caps, rews, losses = [], [], []  
  
    for ep in range(1, n_episodes + 1):  
        obs = env.reset()  
        ep_r = 0.0  
        while not env.done:  
            a = agent.act(obs)  
            obs2, r, term, trunc, info = env.step(a)  
            agent.store(obs, a, r, obs2, float(term or trunc))  
            loss = agent.train_step()  
            if loss is not None:  
                losses.append(loss)  
            obs = obs2  
            ep_r += r  
  
        caps.append(int(info.get("captured", False)))  
        rews.append(ep_r)  
  
        if ep % log_every == 0:  
            w = min(log_every, len(caps))  
            cr = np.mean(caps[-w:])  
            ar = np.mean(rews[-w:])  
            al = np.mean(losses[-w*20:]) if losses else 0  
            print(f"  ep {ep:6d} | cap={cr:.3f} | rew={ar:+8.1f} | "  
                  f"loss={al:.4f} | ε={agent.eps:.3f}")  
  
    return env, agent, caps, rews, losses  
  
  
# ================================================================  
#  8. 评估  
# ================================================================  
def evaluate(cfg, policy_fn, n=300, label=""):  
    env = PursuitEvasionObstacle(cfg)  
    cap = 0  
    for _ in range(n):  
        obs = env.reset()  
        while not env.done:  
            a = policy_fn(obs, env)  
            obs, *_, info = env.step(a)  
        if info.get("captured"):  
            cap += 1  
    rate = cap / n  
    print(f"  {label:25s}  capture = {rate:.3f}  ({cap}/{n})")  
    return rate  
  
  
# ================================================================  
#  9. 可视化  
# ================================================================  
def plot_episode(env, policy_fn, title="", filename="traj.png"):  
    obs = env.reset(seed=42)  
    traj_p, traj_e = [env.p_p.copy()], [env.p_e.copy()]  
    while not env.done:  
        a = policy_fn(obs, env)  
        obs, *_, info = env.step(a)  
        traj_p.append(env.p_p.copy())  
        traj_e.append(env.p_e.copy())  
  
    traj_p = np.array(traj_p)  
    traj_e = np.array(traj_e)  
  
    fig, ax = plt.subplots(figsize=(10, 10))  
  
    # 障碍物  
    for obs_rect in env.obstacles:  
        rect = patches.Rectangle(  
            (obs_rect.x_min, obs_rect.y_min),  
            obs_rect.w, obs_rect.h,  
            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.6,  
            label='Obstacle'  
        )  
        ax.add_patch(rect)  
  
    # 轨迹  
    ax.plot(traj_p[:, 0], traj_p[:, 1], 'b-o', ms=2, lw=1.5,  
            label='Pursuer', zorder=5)  
    ax.plot(traj_e[:, 0], traj_e[:, 1], 'r-s', ms=2, lw=1.5,  
            label='Evader', zorder=5)  
  
    # 起点终点  
    ax.plot(*traj_p[0],  'b^', ms=14, zorder=6)  
    ax.plot(*traj_e[0],  'r^', ms=14, zorder=6)  
    ax.plot(*traj_p[-1], 'bD', ms=10, zorder=6)  
    ax.plot(*traj_e[-1], 'rD', ms=10, zorder=6)  
  
    if info.get("captured"):  
        c = plt.Circle(traj_e[-1], env.D_c, fill=False,  
                        color='green', ls='--', lw=2)  
        ax.add_patch(c)  
        status = "[CAPTURED ✓]"  
        color = 'green'  
    else:  
        status = "[ESCAPED/TIMEOUT ✗]"  
        color = 'red'  
  
    ax.set_xlim(-env.L - 1, env.L + 1)  
    ax.set_ylim(-env.L - 1, env.L + 1)  
    ax.set_aspect('equal')  
    ax.set_title(f"{title}  {status}", fontsize=14, color=color)  
    ax.legend(loc='upper left')  
    ax.grid(True, alpha=0.3)  
  
    # 边界  
    border = patches.Rectangle((-env.L, -env.L), 2*env.L, 2*env.L,  
                                fill=False, edgecolor='navy', lw=2, ls='--')  
    ax.add_patch(border)  
  
    fig.tight_layout()  
    fig.savefig(filename, dpi=150)  
    plt.close(fig)  
    print(f"  Saved → {filename}")  
  
  
def plot_curves(caps, rews, losses, w=500, filename="dqn_curves.png"):  
    def smooth(x, win):  
        if len(x) == 0: return []  
        c = np.cumsum(x)  
        out = np.zeros(len(x))  
        for i in range(len(x)):  
            lo = max(0, i - win + 1)  
            out[i] = (c[i] - (c[lo-1] if lo > 0 else 0)) / (i - lo + 1)  
        return out  
  
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)  
  
    axes[0].plot(smooth(caps, w), 'b-', lw=0.8)  
    axes[0].set_ylabel("Capture Rate")  
    axes[0].set_ylim(-0.05, 1.05)  
    axes[0].grid(True, alpha=0.3)  
  
    axes[1].plot(smooth(rews, w), 'r-', lw=0.8)  
    axes[1].set_ylabel("Episode Reward")  
    axes[1].grid(True, alpha=0.3)  
  
    if losses:  
        axes[2].plot(smooth(losses, w*5), 'g-', lw=0.5)  
        axes[2].set_ylabel("Loss")  
        axes[2].set_yscale("log")  
    axes[2].set_xlabel("Episode / Step")  
    axes[2].grid(True, alpha=0.3)  
  
    fig.suptitle("DQN Training Curves (2D Pursuit-Evasion with Obstacles)",  
                 fontsize=14)  
    fig.tight_layout()  
    fig.savefig(filename, dpi=150)  
    plt.close(fig)  
    print(f"  Saved → {filename}")  
  
  
# ================================================================  
#  10. 主程序  
# ================================================================  
def main():  
    print("=" * 70)  
    print(" 2D Pursuit-Evasion with Obstacles — DQN on", DEVICE)  
    print(" 验证: 追击者利用地形 + 加速度优势 实现截击捕获")  
    print("=" * 70)  
  
    cfg = dict(  
        # 动力学  
        v_max_p=1.0,   v_max_e=1.5,  
        a_max_p=2.0,   a_max_e=0.5,  
        # 时间  
        dt=0.1, K=10, N_max=100,  
        # Arena  
        L=15.0,  
        # 障碍物：中央长方形 + 两个小方块  
        obstacles=[  
            {"cx": 0, "cy": 0, "w": 10, "h": 2.5},    # 中央长墙  
            {"cx": -6, "cy": 6, "w": 3, "h": 3},       # 左上方块  
            {"cx": 7, "cy": -5, "w": 3, "h": 4},       # 右下方块  
        ],  
        # 几何  
        D_c=1.0, D_escape=28.0, l_high=5.0, n_actions=8,  
        # 奖励  
        R_win=200.0, R_lose=80.0,  
        w_close=8.0, w_int=0.5,  
        w_terrain=3.0,       # 地形利用奖励权重  
        w_energy=0.05, c_time=0.3,  
        tau_max=5.0,  
        # 射线  
        n_rays=8, ray_max=15.0,  
        # 逃逸策略  
        evader_policy="smart_avoid",  
        # DQN 超参  
        dqn=dict(  
            gamma=0.99, lr=3e-4, batch_size=256,  
            eps_start=1.0, eps_end=0.05, eps_decay=60000,  
            tau=0.005, buffer_size=200000,  
        ),  
    )  
  
    # ---- 基线 ----  
    print("\n[1] Baselines (before training)")  
    evaluate(cfg, policy_random, n=300, label="Random")  
    evaluate(cfg, policy_pure_pursuit, n=300, label="Pure Pursuit")  
    evaluate(cfg, policy_lead, n=300, label="Lead Pursuit (geom)")  
  
    # ---- 训练 ----  
    N_EP = 15000  
    print(f"\n[2] Training DQN ({N_EP} episodes on {DEVICE})...")  
    env, agent, caps, rews, losses = train_dqn(cfg, n_episodes=N_EP,  
                                                log_every=1000)  
  
    # ---- 评估 ----  
    print("\n[3] Post-Training Evaluation")  
    evaluate(cfg, policy_random, n=300, label="Random")  
    evaluate(cfg, policy_pure_pursuit, n=300, label="Pure Pursuit")  
    evaluate(cfg, policy_lead, n=300, label="Lead Pursuit (geom)")  
  
    def dqn_policy(obs, env_):  
        return agent.act(obs, greedy=True)  
    evaluate(cfg, dqn_policy, n=300, label="DQN (greedy)")  
  
    # ---- 可视化 ----  
    print("\n[4] Visualization")  
    plot_curves(caps, rews, losses, w=500, filename="dqn_curves.png")  
  
    env_vis = PursuitEvasionObstacle(cfg)  
    for i in range(4):  
        env_vis.reset(seed=100 + i)  
        plot_episode(env_vis, dqn_policy,  
                     title=f"DQN #{i+1}", filename=f"traj_dqn_{i+1}.png")  
        env_vis.reset(seed=100 + i)  
        plot_episode(env_vis, policy_pure_pursuit,  
                     title=f"Pure Pursuit #{i+1}",  
                     filename=f"traj_pp_{i+1}.png")  
  
    # ---- 地形效果对比 ----  
    print("\n[5] Ablation: with vs without obstacles")  
    cfg_no_obs = {**cfg, "obstacles": []}  
    print("  --- No obstacles ---")  
    _, agent_no, caps_no, _, _ = train_dqn(cfg_no_obs, n_episodes=10000,  
                                            log_every=10000)  
    def dqn_no_policy(obs, env_):  
        return agent_no.act(obs, greedy=True)  
    evaluate(cfg_no_obs, dqn_no_policy, n=300, label="DQN (no obstacles)")  
    evaluate(cfg_no_obs, policy_pure_pursuit, n=300, label="PP (no obstacles)")  
  
    print("  --- With obstacles ---")  
    evaluate(cfg, dqn_policy, n=300, label="DQN (with obstacles)")  
    evaluate(cfg, policy_pure_pursuit, n=300, label="PP (with obstacles)")  
  
    print("\n" + "=" * 70)  
    print(" Done! Check .png files.")  
    print("=" * 70)  
  
  
if __name__ == "__main__":  
    main()  
