# 二维追逃：运动学与马尔科夫决策过程（MDP）

本文档对应当前 2D 追逃任务与场景（`PursuitEvasion2DAsymmetricEnvV2DynamicsAccel` + `pe_2d_dynamics`）的运动学描述与 MDP 形式化。

---

## 一、二维追逃运动学

### 1.1 通用动力学模型（双方共用形式）

追击者与逃脱者均采用 **2D 质点动力学**（`pe_2d_dynamics.Dynamics2D`），来源于 AUV 的 2D 简化：

**动力学方程：**

\[
M \, \mathbf{a} = \boldsymbol{\tau} - D_{\ell}\, \mathbf{v} - D_q\, \|\mathbf{v}\|\, \mathbf{v}
\]

- \(\mathbf{p}\in\mathbb{R}^2\)：位置  
- \(\mathbf{v}\in\mathbb{R}^2\)：速度  
- \(\mathbf{a}\in\mathbb{R}^2\)：加速度  
- \(\boldsymbol{\tau}\in\mathbb{R}^2\)：推力（控制输入）  
- \(M\)：有效质量（含附加质量）  
- \(D_{\ell}\)：线性阻尼系数  
- \(D_q\)：二次阻尼系数  

即：

\[
\mathbf{a} = \frac{1}{M}\bigl( \boldsymbol{\tau} - D_{\ell}\mathbf{v} - D_q\|\mathbf{v}\|\mathbf{v} \bigr)
\]

时间步长 `dt=0.05`，积分采用 **RK4**（与 `core/dynamics/auv_dynamics` 一致）；可选欧拉。  
速度在步进后可施加上限裁剪 \(\|\mathbf{v}\|\leq v_{\max}\)。

---

### 1.2 追击者 (Pursuer)

- **控制接口**：离散动作 \(a_p \in \{0,1,\ldots,8\}\)  
  - \(a_p=0\)：零推力 \(\boldsymbol{\tau}=\mathbf{0}\)  
  - \(a_p=1\sim 8\)：8 个方向**满推力**，方向角 \(\theta = (a_p-1)\cdot 2\pi/8\)（从东逆时针）  
  - 推力向量：\(\boldsymbol{\tau} = F_{\max}\, (\cos\theta,\,\sin\theta)^\top\)

- **动力学参数**（`create_pursuer_dynamics`，默认）：  
  - \(v_{\max}=1.2,\; a_{\max}=2.5\)（速度劣势、加速度优势）  
  - \(M=15,\; F_{\max}=a_{\max}\cdot M=37.5\)  
  - \(D_{\ell}=4,\; D_q=12\)

- **边界**：若某轴出界 \([-W,W]\)（\(W=\texttt{world\_size}\)），该轴位置 clamp 到边界，该轴速度分量乘以 0.3（模拟碰撞反弹）。

---

### 1.3 逃脱者 (Evader)

- **控制接口**：由内部策略给出**期望加速度** \(\mathbf{a}_{\mathrm{des}}\)，再通过动力学反推推力并积分。  
  - 推力反解：\(\boldsymbol{\tau} = M\,\mathbf{a}_{\mathrm{des}} - F_{\mathrm{damp}}(\mathbf{v})\)，再按最大推力限幅。  
  - 存在**航向角** \(\psi_e\) 与**最大转向率** \(\dot{\psi}_{\max}\)（默认 2.0 rad/s），期望加速度方向会先对齐到当前允许的航向再送入动力学。

- **动力学参数**（`create_evader_dynamics`，默认）：  
  - \(v_{\max}=1.4,\; a_{\max}=0.6\)（速度优势、加速度劣势）  
  - \(M=20,\; F_{\max}=12\)  
  - \(D_{\ell}=6,\; D_q=18\)

- **难度系数**：有效参数为 \(v_{\max}^{\mathrm{eff}}=v_{\max}\cdot(0.5+0.5\,\texttt{difficulty})\)，\(a_{\max}^{\mathrm{eff}}\) 同理。

- **边界**：与追击者相同（出界 clamp + 速度分量 ×0.3）。

---

### 1.4 小结（运动学）

| 对象   | 控制输入           | 动力学形式                         | 速度/加速度特点     |
|--------|--------------------|------------------------------------|---------------------|
| 追击者 | 离散 9 动作 → 推力 | \(M\mathbf{a}=\boldsymbol{\tau}+F_{\mathrm{damp}}\)，RK4 | \(v_{\max}=1.2,\,a_{\max}=2.5\) |
| 逃脱者 | 期望加速度 → 推力  | 同上 + 航向与转向率约束            | \(v_{\max}=1.4,\,a_{\max}=0.6\) |

世界为 2D 平面，范围 \([-W,W]^2\)（默认 \(W=10\)），时间步 `dt=0.05`。

---

## 二、马尔科夫决策过程（MDP）形式化

从**追击者**视角，将环境建模为 MDP（逃脱者视为固定或随机策略，即“对手模型”）。

### 2.1 状态空间 \(\mathcal{S}\)

- **环境内部状态**（不完全被智能体观测）：  
  - 追击者：\(\mathbf{p}_p,\ \mathbf{v}_p,\ \mathbf{a}_p\)  
  - 逃脱者：\(\mathbf{p}_e,\ \mathbf{v}_e,\ \psi_e\)  
  - 步数 \(t\)，以及用于 evader 延迟的 buffer 等  

- **观测空间** \(\mathcal{O}\subseteq\mathbb{R}^{22}\)（`_get_observation` 返回，归一化/裁剪到 \([-10,10]\)）：  

| 索引 | 含义 |
|------|------|
| 0,1  | 追击者位置 \(\mathbf{p}_p/W\) |
| 2,3  | 追击者速度 \(\mathbf{v}_p/v_{\max}^{\mathrm{scale}}\) |
| 4,5  | 相对位置 \((\mathbf{p}_e-\mathbf{p}_p)/W\) |
| 6,7  | 相对速度 \((\mathbf{v}_e-\mathbf{v}_p)/v_{\max}^{\mathrm{scale}}\) |
| 8    | 距离 \(\|\mathbf{p}_e-\mathbf{p}_p\|/W\) |
| 9    | 接近速度（沿相对位置方向） |
| 10,11| 逃脱者速度方向（单位向量） |
| 12–15| 追击者到四边边界的距离/\(W\)（上、下、左、右） |
| 16   | 逃脱者到最近边界的距离/\(W\) |
| 17   | 追击者速度大小（归一化） |
| 18   | 逃脱者速度大小（归一化） |
| 19   | 追击者速度与指向逃脱者方向的夹角（点积） |
| 20,21| 追击者加速度 \(\mathbf{a}_p/a_{\max}^{\mathrm{scale}}\) |

若把“观测”视为对状态的确定函数，则 MDP 的状态可视为该 22 维观测（或等价地用内部状态，视建模习惯）。

---

### 2.2 动作空间 \(\mathcal{A}\)

\[
\mathcal{A} = \{0,1,\ldots,8\}
\]

- \(0\)：无推力  
- \(1\sim 8\)：8 个方向满推力（见 1.2）

---

### 2.3 状态转移 \(\mathbb{P}(s'|s,a)\)

1. **追击者**：  
   \(a\to\) 推力方向与大小 → `Dynamics2D.step(pos, vel, thrust, v_max)` → 得到 \((\mathbf{p}_p',\mathbf{v}_p',\mathbf{a}_p)\)；再应用边界规则。

2. **逃脱者**：  
   由 `evader_mode`（random / simple / smart / expert）和当前状态得到 \(\mathbf{a}_{\mathrm{des}}\)（可能带反应延迟和噪声），再通过 `thrust_for_desired_acc` + `step` 得到 \((\mathbf{p}_e',\mathbf{v}_e')\) 并更新 \(\psi_e\)；再应用边界规则。

3. **步数**：\(t'=t+1\)。

因此转移由动力学方程 + 边界规则 + 逃脱者策略唯一确定；若逃脱者策略带随机（如 random、noise），则 \(\mathbb{P}(s'|s,a)\) 为在该随机性下的条件分布。

---

### 2.4 奖励函数 \(r(s,a,s')\)

在 `_compute_reward(hit_boundary)` 中实现，每步返回标量 \(r\)，由多部分组成（默认系数见构造函数）：

| 项 | 含义 | 默认系数 |
|----|------|----------|
| \(r_{\mathrm{improve}}\) | 距离缩短奖励：\(\lambda_{\mathrm{improve}}\cdot(d_{\mathrm{prev}}-d)\) | 5.0 |
| \(r_{\mathrm{closing}}\) | 接近速度（沿相对位置）：\(\lambda_{\mathrm{closing}}\cdot\mathrm{clip}(\mathrm{closing\_speed},-1,1)\) | 1.0 |
| \(r_{\mathrm{align}}\) | 追击者速度与指向逃脱者方向对齐 | 0.5 |
| \(r_{\mathrm{boundary\_trap}}\) | 逃脱者靠近边界时的“逼角”奖励 | 0.3 |
| \(r_{\mathrm{time}}\) | 每步时间惩罚 | -0.05 |
| \(r_{\mathrm{boundary\_hit}}\) | 追击者撞边界惩罚（若 hit_boundary） | -0.1 |
| \(r_{\mathrm{proximity}}\) | 距离 < 2 时的近距离奖励 | 0.2×(2−d) |
| \(r_{\mathrm{catch}}\) | 捕获成功奖励（\(d < R_{\mathrm{catch}}\)） | 100.0 |

\(d=\|\mathbf{p}_e-\mathbf{p}_p\|\)，\(R_{\mathrm{catch}}=0.5\)（默认）。  
总奖励 \(r = \sum r_{\bullet}\)。

---

### 2.5 终止与截断

- **终止 (terminated)**：捕获，即 \(d < R_{\mathrm{catch}}\)。  
- **截断 (truncated)**：\(t \geq T_{\max}\)（默认 \(T_{\max}=500\)）。

回报为 \(\sum_{k=0}^{\tau-1} \gamma^k r_k\)（\(\tau\) 为终止或截断步），通常 \(\gamma\in(0,1]\) 在算法中设定。

---

### 2.6 MDP 五元组小结

| 要素 | 符号 | 内容 |
|------|------|------|
| 状态 | \(\mathcal{S}\) | 22 维观测（或等价内部状态），连续、有界 |
| 动作 | \(\mathcal{A}\) | \(\{0,1,\ldots,8\}\)，离散 |
| 转移 | \(\mathbb{P}(s'|s,a)\) | 由 2D 动力学 + 边界 + 逃脱者策略（及随机性）决定 |
| 奖励 | \(r(s,a,s')\) | 见上表，稠密 + 捕获时大额正奖励 |
| 折扣 | \(\gamma\) | 由训练脚本设定（如 0.99） |

**注意**：逃脱者策略依赖当前状态（及延迟、噪声），因此从追击者视角，转移概率是定义良好的；若把逃脱者视为“环境的一部分”，则整体仍是 MDP。

---

## 三、速度/加速度约束（仅设计上限）

- **设计约束**：只要求**最大**速度、**最大**加速度满足「追击者速度劣于逃脱者、加速度优于逃脱者」，即  
  - 追击者 v_max < 逃脱者 v_max  
  - 追击者 a_max > 逃脱者 a_max  
  不要求每时每刻追击者都比逃脱者慢，只要两者的**上限**满足上述关系即可。  
- **实现**：在动力学/环境中对速度、加速度做**限幅**（每步后 ‖v‖ ≤ v_max，‖a‖ ≤ a_max），不额外做「时刻比对方慢」的约束。

## 四、AUV 配置 + 双环 PID+RL 运动学（与 low_level 一致）

- **动力学**：`pe_2d_auv_config.py` 从 `configs/robot_config.yaml` 读取质量、阻尼、推力及 pursuer/evader 的 `max_speed`、`max_acceleration`，构建 2D 动力学；step 时施加 v_max、a_max 硬约束。  
- **双环 PID**：采用与 `core/control/dual_loop_pid.py` 中 **IntegratedController** 相同的平移双环逻辑，从 `configs/low_level_config_2d.yaml` 读取：  
  - `action_space.velocity_controller`：Kp_vel、Kd_vel、max_force（速度环 PD）；  
  - `action_space.goal_slowdown`：enabled、radius、min_scale（近目标减速）；  
  - `action_space.action_distance`：surge、sway（每步位移，translation_mode=distance 时 desired_vel = linear_cmd/Ts）。  
  实现位于 `pe_2d_dual_loop_pid.py` 的 `DualLoopPID2D`：外环 动作→linear_cmd→desired_vel，内环 force = Kp_vel*(desired_vel−vel)−Kd_vel*vel 并限幅。  
- **环境**：`env_pe_2d_auv_pid.py` 的 `PursuitEvasion2DAUVPIDEnv`：RL 离散动作 → `DualLoopPID2D.compute_thrust(action, pos, vel, dt, target_position=evader_pos)` → 力 → `Dynamics2D.step(τ, v_max, a_max)`；**默认 world_size=5**。  
- **入口**：`from pursuit_evasion_2d import PursuitEvasion2DAUVPIDEnv, DualLoopPID2D`。

## 五、代码位置速查

- 动力学方程与积分、硬约束：`pursuit_evasion_2d/pe_2d_dynamics.py`（`Dynamics2D`、`_action_to_thrust_direction`、`create_pursuer_dynamics`、`create_evader_dynamics`）  
- AUV 配置动力学：`pursuit_evasion_2d/pe_2d_auv_config.py`（`create_pursuer_dynamics_from_config`、`create_evader_dynamics_from_config`）  
- 环境与 MDP：`pursuit_evasion_2d/env_asymmetric_pe_v2_dynamics_accel.py`（`PursuitEvasion2DAsymmetricEnvV2DynamicsAccel`）；AUV+PID 小世界：`env_pe_2d_auv_pid.py`（`PursuitEvasion2DAUVPIDEnv`）  
- 观测与奖励：各环境中的 `_get_observation`、`_compute_reward`、`step`、`reset`

---

## 六、训练（推荐：与「到达固定目标」同套 low_level）

**正确建模**：追逃任务应与「到达固定目标」使用**同一套**底层环境与运动学（完整 LowLevelAUVEnv + IntegratedController + low_level_config_2d），仅把目标从固定点改为每步更新的逃脱者位置，这样训练量与时长（约 2h+、2M 步）才与到达固定目标相当。

**推荐命令**（在项目根目录 `UnderwaterPursuitEvasion` 下）：

```bash
# 2D 追逃，与 train_low_level_2d_v2 同构：2M 步、相同 n_steps/n_envs/网络
python training/train_pe_low_level_2d.py --total-timesteps 2000000
python training/train_pe_low_level_2d.py --experiment-name pe2d_ll_2m
```

- **环境**：`core/environment/low_level_env_2d_pursuit_evasion.py` 的 `LowLevelAUV2DPursuitEvasionEnv`（内层 = `LowLevelAUVEnv`，目标 = 逃脱者当前位置；逃脱者用 config 的 v_max/a_max 限幅）。  
- **训练脚本**：`training/train_pe_low_level_2d.py`（与 `train_low_level_2d_v2` 同构：2M 步、1024 n_steps、相同网络与 PPO 超参）。  
- **输出**：`checkpoints/<experiment_name>/best_model.pth`、`final_model.pth`。

**场景参数（便于多组对比）**：  
`--init-distance-range 2,6`、`--world-size 10`、`--catch-radius 0.5`、`--max-episode-seconds 90`、`--n-envs 4`。  
**多组并行**：`python training/run_pe_2d_parallel.py --all` 会按场景变体（baseline / small_world / easy_catch / closer_start / short_ep / small_easy）同时跑多组（默认最多 2 个并行），日志在 `visualization/logs/pe2d_<场景>/`，便于比较效果。

**训练结果可视化**（low_level 2D 追逃）：  
用 `training/visualize_pe_low_level_2d.py` 加载 `best_model.pth`，在对应环境中跑若干 episode，生成轨迹图与距离曲线。  
- 单实验：`python training/visualize_pe_low_level_2d.py --experiment-name pe2d_small_easy --episodes 5`  
- 全部 pe2d_*：`python training/visualize_pe_low_level_2d.py --all --episodes 3`  
- 输出目录：`visualization/logs/pe2d_<实验名>/plots/`（`trajectory_epN.png`、`all_trajectories.png`）。

---

以下为**此前**在简化 2D 动力学上的做法（训练很快结束，与「到达固定目标」不等价，仅作参考）：

在 **PursuitEvasion2DAUVPIDEnv**（简化 AUV 配置 + 双环 PID，默认 5×5 世界）上训练追击者策略，使用 PPO，观测 22 维、动作 9。

**训练脚本**：`pursuit_evasion_2d/train_pe_2d_auv_pid.py`

**推荐命令**（在项目根目录 `UnderwaterPursuitEvasion` 下执行）：

```bash
cd UnderwaterPursuitEvasion
python pursuit_evasion_2d/train_pe_2d_auv_pid.py
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 1500000 | 总环境步数 |
| `--world-size` | 5.0 | 世界边长（5×5 小范围试验；可改为 10） |
| `--n-envs` | 16 | 并行环境数 |
| `--n-steps` | 256 | 每次更新每环境步数 |
| `--batch-size` | 128 | PPO mini-batch 大小 |
| `--lr` | 3e-4 | 学习率 |
| `--no-curriculum` | — | 关闭课程学习（固定难度） |
| `--initial-difficulty` | 0.2 | 课程初始难度 |
| `--experiment-name` | 自动 | 实验名，用于 checkpoint 子目录 |
| `--seed` | 42 | 随机种子 |

**示例**：

```bash
# 默认：5×5 世界，1.5M 步，课程学习
python pursuit_evasion_2d/train_pe_2d_auv_pid.py

# 大世界 10×10，2M 步，自定义实验名
python pursuit_evasion_2d/train_pe_2d_auv_pid.py --world-size 10 --total-timesteps 2000000 --experiment-name pe2d_auv_10x10

# 固定难度 0.6，不课程学习
python pursuit_evasion_2d/train_pe_2d_auv_pid.py --no-curriculum --initial-difficulty 0.6
```

**输出**：  
- 权重与优化器：`checkpoints/pe2d_auv_pid/<experiment_name>/best.pth`、`final.pth`、`checkpoint_*.pth`  
- 训练曲线：`checkpoints/pe2d_auv_pid/<experiment_name>/logs/training_log.json`

**可视化**（跑完训练后看轨迹、速度、加速度、距离）：

```bash
# 指定 checkpoint，生成轨迹图 + 可选 GIF，默认输出到 visualization/logs/pe2d_auv_pid_vis/
python pursuit_evasion_2d/visualize_pe_2d_auv_pid.py --ckpt checkpoints/pe2d_auv_pid/<experiment_name>/best.pth --episodes 5

# 10x10 世界、不生成 GIF（仅 PNG）
python pursuit_evasion_2d/visualize_pe_2d_auv_pid.py --ckpt checkpoints/pe2d_auv_pid/pe2d_auv_10x10/best.pth --world-size 10 --episodes 4 --no-gif

# 指定输出目录
python pursuit_evasion_2d/visualize_pe_2d_auv_pid.py --ckpt checkpoints/pe2d_auv_pid/pe2d_auv_10x10/best.pth --out-dir visualization/logs/my_pe2d_vis --episodes 3
```

每集生成一张四宫格图：左上轨迹（蓝=追击者、红=逃脱者），右上速度 \|v\| 与 v_max 线，左下加速度 \|a\| 与 a_max 线，右下追击者–逃脱者距离与捕获半径；未加 `--no-gif` 时还会生成每集 GIF 动画。
