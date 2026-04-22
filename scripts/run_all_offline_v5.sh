#!/bin/bash
# ==========================================================
# V5 底层 + TD3BC/IQL/BPPO 离线预训练 (2M 步, 对齐 v4 版)
# ==========================================================
# 完全 mirror scripts/run_all_offline.sh (TD3BC/IQL/BPPO 三段),
# 只改: --env-version v5 + --low-level-ckpt 指向 v5 底层 + 加 v5 env 参数。
# ==========================================================
set -e
cd "$(dirname "$0")/.."

DATASET="${DATASET:-data/expert_trajectories/expert_dataset.npz}"
LOW_V5="low_near/v5_direct_thrust/checkpoints/best.pth"
COMMON="--obs-dim 21 --act-dim 2 --hidden 256 128 \
  --normalize-reward --val-ratio 0.1 --seed 42"

# v5 env 参数
V5_ARGS="--env-version v5 \
  --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
  --v-max-evader 0.85  --a-max-evader 0.25"

LOGDIR="logs/v5/offline"
mkdir -p "$LOGDIR"

echo "============================================"
echo " V5 Offline Training: TD3BC / IQL / BPPO (2M)"
echo " Dataset   : $DATASET"
echo " Low-level : $LOW_V5"
echo " Log dir   : $LOGDIR"
echo " $(date)"
echo "============================================"

# ─── 1. TD3+BC (2M steps) ───
echo ""
echo ">>> [1/3] TD3+BC (2M steps)   log=$LOGDIR/td3bc_v5.log"
PYTHONUNBUFFERED=1 python -u training/td3bc_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name td3bc_v5_2m \
    $COMMON \
    --actor-lr 1e-4 --critic-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 --n-steps 2000000 \
    --policy-noise 0.2 --noise-clip 0.5 \
    --policy-delay 2 --alpha-ratio 2.5 \
    --log-every 500 --print-every 10000 --save-every 100000 \
    --eval-every 50000 --eval-episodes 15 \
    --low-level-ckpt "$LOW_V5" \
    $V5_ARGS 2>&1 | tee "$LOGDIR/td3bc_v5.log"

# ─── 2. IQL (2M steps) ───
echo ""
echo ">>> [2/3] IQL (2M steps)   log=$LOGDIR/iql_v5.log"
PYTHONUNBUFFERED=1 python -u training/iql_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name iql_v5_2m \
    $COMMON \
    --actor-lr 1e-4 --critic-lr 1e-4 --value-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 --n-steps 2000000 \
    --expectile 0.7 --beta 3.0 \
    --log-every 500 --print-every 10000 --save-every 100000 \
    --eval-every 50000 --eval-episodes 15 \
    --low-level-ckpt "$LOW_V5" \
    $V5_ARGS 2>&1 | tee "$LOGDIR/iql_v5.log"

# ─── 3. BPPO (V:300k + Q:300k + BC:300k + BPPO:3k) ───
echo ""
echo ">>> [3/3] BPPO (V:300k Q:300k BC:300k BPPO:3k)   log=$LOGDIR/bppo_v5.log"
PYTHONUNBUFFERED=1 python -u training/bppo_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name bppo_v5_2m \
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
    $V5_ARGS 2>&1 | tee "$LOGDIR/bppo_v5.log"

echo ""
echo "============================================"
echo " V5 离线预训练完成  $(date)"
echo "   TD3BC: checkpoints/td3bc_v5_2m/best.pth"
echo "   IQL  : checkpoints/iql_v5_2m/best.pth"
echo "   BPPO : checkpoints/bppo_v5_2m/best.pth"
echo "============================================"
