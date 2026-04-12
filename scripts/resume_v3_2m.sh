#!/bin/bash
# Resume v3 → 2M steps: 从 latest.pth checkpoint 继续训练到 2M
set -e
cd "$(dirname "$0")/.."

LOW="low_near/v4_tight_heading_terminal/checkpoints/best.pth"

# ── Common (与原实验一致，仅 total-timesteps 改为 2M) ──
COMMON="--obs-dim 21 --act-dim 2 --hidden 256 128 \
  --total-timesteps 2000000 --n-steps 2048 --num-envs 4 \
  --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 0.5 \
  --normalize-reward --r-catch 200.0 --r-fail -50.0 \
  --alpha-shaping 5.0 --d-sg-safe 1.5 \
  --eval-freq 20000 --eval-episodes 20 --save-freq 50000 \
  --world-size 30.0 --catch-radius 1.0 --subgoal-range 0.7 \
  --low-steps 50 --max-steps 6000 --evader-mode medium"

# ── Group A: Conservative ──
GA="--lr 1e-5 --lr-end 5e-7 --clip-range 0.1 --n-epochs 3 \
  --beta-kl 0.2 --kl-anneal-frac 0.5 --ent-coef 0.01 \
  --c-sg 3.0 --c-path 2.0 --lam-time 0.02"

# ── Group B: Aggressive ──
GB="--lr 3e-5 --lr-end 1e-6 --clip-range 0.15 --n-epochs 4 \
  --beta-kl 0.0 --kl-anneal-frac 0.0 --ent-coef 0.02 \
  --c-sg 1.0 --c-path 1.0 --lam-time 0.01"

echo "$(date): Resuming 6 experiments from latest.pth to 2M steps..."
mkdir -p logs

# BPPO-A (使用 latest.pth)
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_bppo_A/latest.pth" \
  --experiment-name "v3_bppo_A" --seed 42 \
  $COMMON $GA \
  >> logs/v3_bppo_A.log 2>&1 &
P1=$!; echo "  BPPO-A  PID=$P1"

# BPPO-B
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_bppo_B/latest.pth" \
  --experiment-name "v3_bppo_B" --seed 42 \
  $COMMON $GB \
  >> logs/v3_bppo_B.log 2>&1 &
P2=$!; echo "  BPPO-B  PID=$P2"

# IQL-A
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_iql_A/latest.pth" \
  --experiment-name "v3_iql_A" --seed 42 \
  $COMMON $GA \
  >> logs/v3_iql_A.log 2>&1 &
P3=$!; echo "  IQL-A   PID=$P3"

# IQL-B
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_iql_B/latest.pth" \
  --experiment-name "v3_iql_B" --seed 42 \
  $COMMON $GB \
  >> logs/v3_iql_B.log 2>&1 &
P4=$!; echo "  IQL-B   PID=$P4"

# TD3BC-A
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_td3bc_A/latest.pth" \
  --experiment-name "v3_td3bc_A" --seed 42 \
  $COMMON $GA \
  >> logs/v3_td3bc_A.log 2>&1 &
P5=$!; echo "  TD3BC-A PID=$P5"

# TD3BC-B
python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" \
  --resume "checkpoints/v3_td3bc_B/latest.pth" \
  --experiment-name "v3_td3bc_B" --seed 42 \
  $COMMON $GB \
  >> logs/v3_td3bc_B.log 2>&1 &
P6=$!; echo "  TD3BC-B PID=$P6"

echo ""
echo "$(date): All 6 experiments resumed. Waiting..."
wait $P1 $P2 $P3 $P4 $P5 $P6
echo "$(date): All 6 experiments finished (2M steps)!"
