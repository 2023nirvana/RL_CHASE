#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "===== Phase 1: Behavioral Cloning Pretrain ====="

python training/bc_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name bc_phase1 \
    --obs-dim 21 \
    --act-dim 2 \
    --hidden 256 128 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --n-epochs 300 \
    --batch-size 256 \
    --mse-weight 0.7 \
    --train-critic \
    --vf-coef 0.5 \
    --gamma 0.99 \
    --patience 50 \
    --log-every 5 \
    --seed 42

echo "===== Done ====="
