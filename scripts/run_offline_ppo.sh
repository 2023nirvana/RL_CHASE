#!/bin/bash
# Phase 1: Offline PPO 预训练
set -e
cd "$(dirname "$0")/.."

DATASET="data/expert_trajectories/expert_dataset.npz"

python training/offline_ppo_pretrain.py \
    --dataset "$DATASET" \
    --experiment-name "offline_ppo_phase1" \
    --hidden 256 128 \
    --lr 3e-4 \
    --n-iters 500 \
    --n-epochs 2 \
    --batch-size 512 \
    --clip-range 0.2 \
    --target-kl 0.01 \
    --ent-coef 0.01

echo "Done. Checkpoints in checkpoints/offline_ppo_phase1/"
