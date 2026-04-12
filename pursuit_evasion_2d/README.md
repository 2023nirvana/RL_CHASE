# 二维多维非对称追逃博弈 (Pursuit-Evasion 2D)

## 研究思路

现有追逃研究多只考虑速度差异。本模块研究：**当追击者最大速度不如逃脱者、但最大加速度优于逃脱者时**，借助地形或预测策略，利用短时机动性完成追捕。

- **追击者 (P)**：`v_max` 较小、`a_max` 较大  
- **逃脱者 (E)**：`v_max` 较大、`a_max` 较小  

动力学为二维双积分器，速度与加速度分别受 `v_max`、`a_max` 约束；观测中包含**加速度**，便于策略利用机动性。

## 目录与文件

| 文件 | 说明 |
|------|------|
| `env_asymmetric_pe.py` | 二维非对称追逃环境（双积分器 + P/E 不同 v_max, a_max） |
| `env_low_level_target.py` | 低层目标到达环境，观测含 pos_error, vel, **acc** |
| `evader_policy.py` | 脚本逃脱者（朝远离追击者方向跑） |
| `train_low_level_accel.py` | 训练低层策略（目标到达，含加速度观测） |
| `train_pe_flat.py` | 扁平训练：追击者 PPO vs 脚本逃脱者 |
| `train_hrl_pe.py` | 分层 HRL：高层输出子目标，低层执行 |
| `config_pe.yaml` | 默认超参与环境参数 |

## 终端运行命令（可直接复制）

**先进入项目目录：**
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
```

以下命令为单行，复制到终端即可运行，**输出会实时打印**，便于观察指标。

### 1. 训练低层策略（含加速度观测）

每 **50 个 update** 打印一次：`update | eval reward | train success rate`

```bash
python -m pursuit_evasion_2d.train_low_level_accel --experiment-name pe2d_lowlevel --total-timesteps 500000 --n-envs 8 --v-max 1.0 --a-max 2.0 --seed 42
```

- **eval reward**：评估回合平均回报，越高越好  
- **train success rate**：训练中到达目标的比例  
- checkpoint 保存到 `checkpoints/pe2d/pe2d_lowlevel/`

### 2. 扁平追逃：直接训练追击者 vs 脚本逃脱者

每 **100 个 update** 打印一次：`update | mean_rew | catch_rate`

```bash
python -m pursuit_evasion_2d.train_pe_flat --experiment-name pe2d_flat --total-timesteps 1000000 --pursuer-v-max 1.0 --pursuer-a-max 2.0 --evader-v-max 1.5 --evader-a-max 0.8 --seed 42
```

- **mean_rew**：近期回合平均奖励（含追到时的 100 分）  
- **catch_rate**：追捕成功率，主要关注指标  
- 默认非对称：追击者 v_max=1.0, a_max=2.0；逃脱者 v_max=1.5, a_max=0.8  

### 3. 分层 HRL 追逃（需先完成步骤 1）

每 **50 个 update** 打印一次：`hrl update | mean_rew | catch_rate`

```bash
python -m pursuit_evasion_2d.train_hrl_pe --low-level-ckpt checkpoints/pe2d/pe2d_lowlevel/final.pth --experiment-name pe2d_hrl --total-timesteps 500000 --seed 42
```

- 若低层实验名不是 `pe2d_lowlevel`，请把 `final.pth` 前的路径改成你的 `checkpoints/pe2d/<实验名>/`  
- 高层输出子目标，低层执行若干步，观察 **catch_rate** 是否随训练上升  

---

以上命令均为前台运行，直接在当前终端看日志即可；如需后台跑可自行加 `nohup ... &` 或 `tee log.txt`。

## 观测与动作

- **追逃环境 (P)**：观测 `[self_pos, self_vel, self_acc, other_pos, other_vel]`（10 维）；动作 9 个（0=不加速，1~8=8 方向以 a_max 加速）。
- **低层目标环境**：观测 `[pos_error(2), vel(2), acc(2)]`（6 维）；动作同上 9 个。

## PE V2 动力学版（基于 low_level_env 动力学）

在保留原有 `env_asymmetric_pe_v2`（简化双积分器）不修改的前提下，提供基于 **low_level 动力学** 的版本：

- **`pe_2d_dynamics.py`**：二维动力学模块，仿照 `core/dynamics/auv_dynamics`，`M*a = τ - D_linear*v - D_quadratic*|v|*v`，含质量、线性/二次阻尼，RK4 积分，**加速度由动力学方程计算**。
- **`env_asymmetric_pe_v2_dynamics.py`**：追逃环境 V2 动力学版，动作为推力方向，状态与观测含加速度；观测 **22 维**（比简化版多 2 维加速度）。
- **`train_pe_v2_dynamics.py`**：在动力学环境上训练追击者（课程学习 + PPO），checkpoint 存到 `checkpoints/pe2d_v2_dynamics/`。
- **`eval_pe_v2_dynamics.py`**：加载动力学版 checkpoint，在动力学环境下评估并生成轨迹/动画。

**训练（动力学版，动作为推力方向）：**
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
python -m pursuit_evasion_2d.train_pe_v2_dynamics --env dynamics --experiment-name pe2d_v2_dyn --total-timesteps 2000000 --n-envs 16 --seed 42
```

**训练（PID 版，RL 输出目标点，2D 双环 PID 算 τ）：**
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
python -m pursuit_evasion_2d.train_pe_v2_dynamics --env pid --experiment-name pe2d_v2_pid --total-timesteps 2000000 --n-envs 16 --seed 42
```
checkpoint 会存到 `checkpoints/pe2d_v2_pid/<experiment-name>/`。

**评估（动力学版）：**
```bash
python -m pursuit_evasion_2d.eval_pe_v2_dynamics --checkpoint checkpoints/pe2d_v2_dynamics/pe2d_v2_dyn/best.pth --save-dir checkpoints/pe2d_v2_dynamics/pe2d_v2_dyn/evaluation
```

**评估（PID 版）：** 与动力学版用同一脚本，加 `--env pid` 即可在 PID 环境上评估。
```bash
python -m pursuit_evasion_2d.eval_pe_v2_dynamics --env pid --checkpoint checkpoints/pe2d_v2_pid/pe2d_v2_pid/best.pth --save-dir checkpoints/pe2d_v2_pid/pe2d_v2_pid/evaluation
```

简化版（双积分器）仍用 `train_pe_v2.py` / `eval_pe_v2.py` 和 `env_asymmetric_pe_v2`（观测 20 维）。

### PE V2 PID 版（RL 输出目标点，2D 双环 PID 算 τ）

与 low_level 输出逻辑一致：RL 只输出**目标点**（离散 9 个：0=保持，1~8=方向上的目标），由 **2D 双环 PID** 根据目标与当前 pos/vel 计算推力 τ，再交给同一套 2D 动力学积分；观测 22 维（含速度、加速度）。

- **`pe_2d_pid_controller.py`**：2D 位置-速度双环，`target_pos` → 期望速度 → 速度环 PD → τ
- **`env_asymmetric_pe_v2_pid.py`**：`PursuitEvasion2DAsymmetricEnvV2PID`，动作 0~8 表示目标点，内部用 PID 算 τ、Dynamics2D 积分
- 训练时使用同一脚本：`--env pid` 即用 PID 环境，checkpoint 存到 `checkpoints/pe2d_v2_pid/`。

## RL and nonlinearity

Dynamics model C(nu) and quadratic damping explicitly; controller is linear PID; RL only chooses high-level discrete actions. See docs/RL_and_Nonlinearity.md.

## 与现有 2D 代码的关系

本模块独立于 `core/environment/low_level_env_2d*.py` 和 `training/train_2d*.py`，不修改原有文件，仅新增 `pursuit_evasion_2d/` 下的环境与训练脚本。

---

## 快速复制（仅命令，先 cd 到 UnderwaterPursuitEvasion）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
```

**1) 低层策略**
```bash
python -m pursuit_evasion_2d.train_low_level_accel --experiment-name pe2d_lowlevel --total-timesteps 500000 --n-envs 8 --v-max 1.0 --a-max 2.0 --seed 42
```

**2) 扁平追逃**
```bash
python -m pursuit_evasion_2d.train_pe_flat --experiment-name pe2d_flat --total-timesteps 1000000 --pursuer-v-max 1.0 --pursuer-a-max 2.0 --evader-v-max 1.5 --evader-a-max 0.8 --seed 42
```

**3) 分层 HRL（先跑完 1 再跑，路径按实际实验名改）**
```bash
python -m pursuit_evasion_2d.train_hrl_pe --low-level-ckpt checkpoints/pe2d/pe2d_lowlevel/final.pth --experiment-name pe2d_hrl --total-timesteps 500000 --seed 42
```
