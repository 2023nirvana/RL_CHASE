# 2D 追逃训练命令（完整 2D 运动学：PID + RL）

**工作目录：先进入项目根目录**
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
```

---

## 一、标准流程（低层 → 扁平追逃 → HRL）

### 1. 训练低层策略（含加速度观测，每 50 update 打印 eval reward | success rate）
```bash
python -m pursuit_evasion_2d.train_low_level_accel --experiment-name pe2d_lowlevel --total-timesteps 500000 --n-envs 8 --v-max 1.0 --a-max 2.0 --seed 42
```
- checkpoint: `checkpoints/pe2d/pe2d_lowlevel/`

### 2. 扁平追逃（追击者 PPO vs 脚本逃脱者，每 100 update 打印 mean_rew | catch_rate）
```bash
python -m pursuit_evasion_2d.train_pe_flat --experiment-name pe2d_flat --total-timesteps 1000000 --pursuer-v-max 1.0 --pursuer-a-max 2.0 --evader-v-max 1.5 --evader-a-max 0.8 --seed 42
```

### 3. 分层 HRL（需先完成步骤 1，每 50 update 打印 mean_rew | catch_rate）
```bash
python -m pursuit_evasion_2d.train_hrl_pe --low-level-ckpt checkpoints/pe2d/pe2d_lowlevel/final.pth --experiment-name pe2d_hrl --total-timesteps 500000 --seed 42
```

---

## 二、或用 run_commands.sh 一键执行

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

# 步骤 1：低层
bash pursuit_evasion_2d/run_commands.sh 1

# 步骤 2：扁平追逃
bash pursuit_evasion_2d/run_commands.sh 2

# 步骤 3：HRL（需先跑完 1）
bash pursuit_evasion_2d/run_commands.sh 3
```

---

## 三、PE V2 动力学 / PID 版（RL 目标点 + 2D 双环 PID）

- **动力学版**（动作为推力方向，完整 2D 动力学积分）：
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
python -m pursuit_evasion_2d.train_pe_v2_dynamics --env dynamics --experiment-name pe2d_v2_dyn --total-timesteps 2000000 --n-envs 16 --seed 42
```
- checkpoint: `checkpoints/pe2d_v2_dynamics/pe2d_v2_dyn/`

- **PID 版**（RL 只输出目标点，2D 双环 PID 算 τ，与 low_level 输出逻辑一致）：
```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
python -m pursuit_evasion_2d.train_pe_v2_dynamics --env pid --experiment-name pe2d_v2_pid --total-timesteps 2000000 --n-envs 16 --seed 42
```
- checkpoint: `checkpoints/pe2d_v2_pid/pe2d_v2_pid/`

---

## 四、评估命令（可选）

**扁平/HRL 无单独 eval 脚本时**，主要看训练日志里的 catch_rate。

**PE V2 动力学版评估：**
```bash
python -m pursuit_evasion_2d.eval_pe_v2_dynamics --checkpoint checkpoints/pe2d_v2_dynamics/pe2d_v2_dyn/best.pth --save-dir checkpoints/pe2d_v2_dynamics/pe2d_v2_dyn/evaluation
```

**PE V2 PID 版评估：**
```bash
python -m pursuit_evasion_2d.eval_pe_v2_dynamics --env pid --checkpoint checkpoints/pe2d_v2_pid/pe2d_v2_pid/best.pth --save-dir checkpoints/pe2d_v2_pid/pe2d_v2_pid/evaluation
```
