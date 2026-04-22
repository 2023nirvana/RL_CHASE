#!/bin/bash
# ==========================================================
# V5 底层 Online PPO 微调 - 3 算法 × 2 超参组 = 6 实验并行
# 完全 mirror run_online_v3_all.sh，只替换底层为 v5。
# ==========================================================
set -e
cd "$(dirname "$0")/.."

LOW="low_near/v5_direct_thrust/checkpoints/best.pth"
BPPO_CK="checkpoints/bppo_v5_2m/best.pth"
IQL_CK="checkpoints/iql_v5_2m/best.pth"
TD3BC_CK="checkpoints/td3bc_v5_2m/best.pth"

# ── Common ──
COMMON="--env-version v5 \
  --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
  --v-max-evader 0.85  --a-max-evader 0.25 \
  --obs-dim 21 --act-dim 2 --hidden 256 128 \
  --total-timesteps 500000 --n-steps 2048 --num-envs 4 \
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

LOGDIR="logs/v5/online"
mkdir -p "$LOGDIR"
echo "$(date): Starting 6 v5 experiments in parallel..."
echo "===== 日志目录: $LOGDIR ====="

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$BPPO_CK" \
  --experiment-name "v5_bppo_A" --seed 42 \
  $COMMON $GA > "$LOGDIR/bppo_A.log" 2>&1 &
P1=$!; echo "  BPPO-A  PID=$P1  log=$LOGDIR/bppo_A.log"

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$BPPO_CK" \
  --experiment-name "v5_bppo_B" --seed 42 \
  $COMMON $GB > "$LOGDIR/bppo_B.log" 2>&1 &
P2=$!; echo "  BPPO-B  PID=$P2  log=$LOGDIR/bppo_B.log"

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$IQL_CK" \
  --experiment-name "v5_iql_A" --seed 42 \
  $COMMON $GA > "$LOGDIR/iql_A.log" 2>&1 &
P3=$!; echo "  IQL-A   PID=$P3  log=$LOGDIR/iql_A.log"

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$IQL_CK" \
  --experiment-name "v5_iql_B" --seed 42 \
  $COMMON $GB > "$LOGDIR/iql_B.log" 2>&1 &
P4=$!; echo "  IQL-B   PID=$P4  log=$LOGDIR/iql_B.log"

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$TD3BC_CK" \
  --experiment-name "v5_td3bc_A" --seed 42 \
  $COMMON $GA > "$LOGDIR/td3bc_A.log" 2>&1 &
P5=$!; echo "  TD3BC-A PID=$P5  log=$LOGDIR/td3bc_A.log"

python -u training/online_ppo_finetune.py \
  --low-level-ckpt "$LOW" --pretrain-ckpt "$TD3BC_CK" \
  --experiment-name "v5_td3bc_B" --seed 42 \
  $COMMON $GB > "$LOGDIR/td3bc_B.log" 2>&1 &
P6=$!; echo "  TD3BC-B PID=$P6  log=$LOGDIR/td3bc_B.log"

# 记录 PID 方便事后 kill / 查进度
printf "bppo_A %d\nbppo_B %d\niql_A %d\niql_B %d\ntd3bc_A %d\ntd3bc_B %d\n" \
  $P1 $P2 $P3 $P4 $P5 $P6 > "$LOGDIR/pids.txt"

echo ""
echo "$(date): All 6 launched. Waiting..."
wait $P1 $P2 $P3 $P4 $P5 $P6
echo "$(date): All 6 v5 experiments finished!"
