#!/bin/bash
# 二维非对称追逃 - 终端运行训练并实时看指标
# 用法: 先 cd 到 UnderwaterPursuitEvasion，再执行:
#   bash pursuit_evasion_2d/run_commands.sh 1   # 低层
#   bash pursuit_evasion_2d/run_commands.sh 2   # 扁平追逃
#   bash pursuit_evasion_2d/run_commands.sh 3     # HRL（需先跑完 1）

cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"
echo ""

case "${1:-0}" in
  1)
    echo "========== 1. 训练低层策略（每 50 update 打印 eval reward | success rate）=========="
    python -m pursuit_evasion_2d.train_low_level_accel --experiment-name pe2d_lowlevel --total-timesteps 500000 --n-envs 8 --v-max 1.0 --a-max 2.0 --seed 42
    ;;
  2)
    echo "========== 2. 扁平追逃（每 100 update 打印 mean_rew | catch_rate）=========="
    python -m pursuit_evasion_2d.train_pe_flat --experiment-name pe2d_flat --total-timesteps 1000000 --pursuer-v-max 1.0 --pursuer-a-max 2.0 --evader-v-max 1.5 --evader-a-max 0.8 --seed 42
    ;;
  3)
    echo "========== 3. 分层 HRL（每 50 update 打印 mean_rew | catch_rate）=========="
    if [ ! -f "checkpoints/pe2d/pe2d_lowlevel/final.pth" ]; then
      echo "请先运行: bash pursuit_evasion_2d/run_commands.sh 1"
      exit 1
    fi
    python -m pursuit_evasion_2d.train_hrl_pe --low-level-ckpt checkpoints/pe2d/pe2d_lowlevel/final.pth --experiment-name pe2d_hrl --total-timesteps 500000 --seed 42
    ;;
  *)
    echo "用法: bash pursuit_evasion_2d/run_commands.sh <1|2|3>"
    echo "  1 = 低层策略（含加速度观测）"
    echo "  2 = 扁平追逃（追击者 vs 脚本逃脱者）"
    echo "  3 = 分层 HRL（需先完成 1）"
    exit 1
    ;;
esac
