#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "===== Phase 1: TD3+BC Offline Pretrain v3 (更多数据 + 子目标惩罚) ====="
echo ""

python -u training/td3bc_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name td3bc_phase1_v3 \
    --obs-dim 21 \
    --act-dim 2 \
    --hidden 256 128 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 256 \
    --n-steps 150000 \
    --policy-noise 0.2 \
    --noise-clip 0.5 \
    --policy-delay 2 \
    --alpha-ratio 2.5 \
    --log-every 100 \
    --print-every 2000 \
    --save-every 10000 \
    --eval-every 5000 \
    --eval-episodes 15 \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --seed 42

echo ""
echo "===== 训练完成 ====="
echo "训练曲线: visualization/logs/td3bc_phase1_v3/td3bc_curves.png"
echo "最优模型: checkpoints/td3bc_phase1_v3/best.pth"
