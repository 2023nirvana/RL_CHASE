#!/bin/bash
set -e
cd /root/autodl-tmp/pursue_20260302

DATASET="data/expert_trajectories/expert_dataset.npz"
LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
COMMON="--obs-dim 21 --act-dim 2 --hidden 256 128 --normalize-reward --val-ratio 0.1 --seed 42"

echo "============================================"
echo " Offline Training: 7 algorithms"
echo " Dataset: $DATASET"
echo " $(date)"
echo "============================================"

# ─── 1. TD3+BC (2M steps) ───
echo ""
echo ">>> [1/7] TD3+BC  (2M steps)"
PYTHONUNBUFFERED=1 python -u training/td3bc_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name td3bc_v4_2m \
    $COMMON \
    --actor-lr 1e-4 \
    --critic-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 \
    --n-steps 2000000 \
    --policy-noise 0.2 --noise-clip 0.5 \
    --policy-delay 2 \
    --alpha-ratio 2.5 \
    --log-every 500 \
    --print-every 10000 \
    --save-every 100000 \
    --eval-every 50000 \
    --eval-episodes 15 \
    --low-level-ckpt "$LOW_CKPT"

# ─── 2. IQL (2M steps) ───
echo ""
echo ">>> [2/7] IQL  (2M steps)"
PYTHONUNBUFFERED=1 python -u training/iql_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name iql_v2_2m \
    $COMMON \
    --actor-lr 1e-4 \
    --critic-lr 1e-4 \
    --value-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 \
    --n-steps 2000000 \
    --expectile 0.7 \
    --beta 3.0 \
    --log-every 500 \
    --print-every 10000 \
    --save-every 100000 \
    --eval-every 50000 \
    --eval-episodes 15 \
    --low-level-ckpt "$LOW_CKPT"

# ─── 3. AWAC (2M steps) ───
echo ""
echo ">>> [3/7] AWAC  (2M steps)"
PYTHONUNBUFFERED=1 python -u training/awac_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name awac_v2_2m \
    $COMMON \
    --actor-lr 1e-4 \
    --critic-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 \
    --n-steps 2000000 \
    --beta 1.0 \
    --log-every 500 \
    --print-every 10000 \
    --save-every 100000 \
    --eval-every 50000 \
    --eval-episodes 15 \
    --low-level-ckpt "$LOW_CKPT"

# ─── 4. REBRAC (2M steps) ───
echo ""
echo ">>> [4/7] REBRAC  (2M steps)"
PYTHONUNBUFFERED=1 python -u training/rebrac_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name rebrac_v2_2m \
    $COMMON \
    --actor-lr 1e-4 \
    --critic-lr 1e-4 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 \
    --n-steps 2000000 \
    --actor-reg 0.1 \
    --critic-reg 0.5 \
    --log-every 500 \
    --print-every 10000 \
    --save-every 100000 \
    --eval-every 50000 \
    --eval-episodes 15 \
    --low-level-ckpt "$LOW_CKPT"

# ─── 5. BPPO (V:300k + Q:300k + BC:300k + BPPO:3k) ───
echo ""
echo ">>> [5/7] BPPO  (V:300k Q:300k BC:300k BPPO:3k)"
PYTHONUNBUFFERED=1 python -u training/bppo_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name bppo_v2_2m \
    $COMMON \
    --value-lr 1e-4 \
    --critic-lr 1e-4 \
    --bc-lr 3e-4 \
    --bppo-lr 3e-5 \
    --gamma 0.99 --tau 0.005 \
    --batch-size 256 \
    --v-steps 300000 \
    --q-steps 300000 \
    --bc-steps 300000 \
    --bppo-steps 3000 \
    --clip-ratio 0.2 \
    --log-every 500 \
    --print-every 10000 \
    --save-every 100000 \
    --eval-every 50000 \
    --eval-episodes 15 \
    --low-level-ckpt "$LOW_CKPT"

# ─── 6. BC (500 epochs) ───
echo ""
echo ">>> [6/7] BC  (500 epochs)"
PYTHONUNBUFFERED=1 python -u training/bc_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name bc_v2_500ep \
    --obs-dim 21 --act-dim 2 --hidden 256 128 \
    --n-epochs 500 \
    --batch-size 512 \
    --lr 3e-4 \
    --gamma 0.99 \
    --val-ratio 0.1 \
    --mse-weight 0.7 \
    --train-critic \
    --seed 42

# ─── 7. Offline PPO (2000 iters × 2 epochs) ───
echo ""
echo ">>> [7/7] Offline PPO  (2000 iters)"
PYTHONUNBUFFERED=1 python -u training/offline_ppo_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name offline_ppo_v2_2k \
    --obs-dim 21 --act-dim 2 --hidden 256 128 \
    --n-iters 2000 \
    --n-epochs 2 \
    --batch-size 512 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --target-kl 0.01

echo ""
echo "============================================"
echo " All offline training done!  $(date)"
echo "============================================"
