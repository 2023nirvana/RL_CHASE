#!/bin/bash
# V5 离线预训练 - 并行版 (TD3BC + IQL + BPPO 同时跑)
# 原顺序脚本: run_all_offline_v5.sh (约 15-18h)
# 并行版:     约 5-6h (取决于最慢算法 BPPO)
set -e
cd "$(dirname "$0")/.."

DATASET="${DATASET:-data/expert_trajectories/expert_dataset.npz}"
LOW_V5="low_near/v5_direct_thrust/checkpoints/best.pth"
COMMON="--obs-dim 21 --act-dim 2 --hidden 256 128 \
  --normalize-reward --val-ratio 0.1 --seed 42"
V5_ARGS="--env-version v5 \
  --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
  --v-max-evader 0.85  --a-max-evader 0.25"

LOGDIR="logs/v5/offline"
mkdir -p "$LOGDIR"

echo "============================================"
echo " V5 Offline Training (PARALLEL): 3 algorithms"
echo " $(date)"
echo " Log dir: $LOGDIR"
echo "============================================"

# ─── TD3+BC (2M) ───
PYTHONUNBUFFERED=1 python -u training/td3bc_pretrain.py \
    --dataset "$DATASET" --experiment-name td3bc_v5_2m \
    $COMMON \
    --actor-lr 1e-4 --critic-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 --n-steps 2000000 \
    --policy-noise 0.2 --noise-clip 0.5 \
    --policy-delay 2 --alpha-ratio 2.5 \
    --log-every 500 --print-every 10000 --save-every 100000 \
    --eval-every 50000 --eval-episodes 15 \
    --low-level-ckpt "$LOW_V5" \
    $V5_ARGS > "$LOGDIR/td3bc_v5.log" 2>&1 &
P1=$!; echo "  TD3BC PID=$P1  log=$LOGDIR/td3bc_v5.log"

# ─── IQL (2M) ───
PYTHONUNBUFFERED=1 python -u training/iql_pretrain.py \
    --dataset "$DATASET" --experiment-name iql_v5_2m \
    $COMMON \
    --actor-lr 1e-4 --critic-lr 1e-4 --value-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 --n-steps 2000000 \
    --expectile 0.7 --beta 3.0 \
    --log-every 500 --print-every 10000 --save-every 100000 \
    --eval-every 50000 --eval-episodes 15 \
    --low-level-ckpt "$LOW_V5" \
    $V5_ARGS > "$LOGDIR/iql_v5.log" 2>&1 &
P2=$!; echo "  IQL   PID=$P2  log=$LOGDIR/iql_v5.log"

# ─── BPPO (V:300k Q:300k BC:300k BPPO:3k) ───
PYTHONUNBUFFERED=1 python -u training/bppo_pretrain.py \
    --dataset "$DATASET" --experiment-name bppo_v5_2m \
    $COMMON \
    --value-lr 1e-4 --critic-lr 1e-4 \
    --bc-lr 3e-4 --bppo-lr 3e-5 \
    --gamma 0.99 --tau 0.005 --batch-size 256 \
    --v-steps 300000 --q-steps 300000 \
    --bc-steps 300000 --bppo-steps 3000 \
    --clip-ratio 0.2 \
    --log-every 500 --print-every 10000 --save-every 100000 \
    --eval-every 50000 --eval-episodes 15 \
    --low-level-ckpt "$LOW_V5" \
    $V5_ARGS > "$LOGDIR/bppo_v5.log" 2>&1 &
P3=$!; echo "  BPPO  PID=$P3  log=$LOGDIR/bppo_v5.log"

printf "td3bc %d\niql %d\nbppo %d\n" $P1 $P2 $P3 > "$LOGDIR/pids_offline.txt"

echo ""
echo "$(date): All 3 launched in parallel. Waiting..."
wait $P1 $P2 $P3
echo "$(date): V5 offline pretraining DONE (all 3)"
