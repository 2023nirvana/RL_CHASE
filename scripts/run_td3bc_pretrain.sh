#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "===== Phase 1: TD3+BC Offline Pretrain (v2: subgoal penalty + eval) ====="

python training/td3bc_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name td3bc_phase1_v2 \
    --obs-dim 21 \
    --act-dim 2 \
    --hidden 256 128 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 256 \
    --n-steps 100000 \
    --policy-noise 0.2 \
    --noise-clip 0.5 \
    --policy-delay 2 \
    --alpha-ratio 2.5 \
    --log-every 100 \
    --print-every 2000 \
    --save-every 10000 \
    --eval-every 5000 \
    --eval-episodes 10 \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --seed 42

echo "===== Done ====="
