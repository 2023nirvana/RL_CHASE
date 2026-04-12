#!/bin/bash
# ============================================================
# 完整数据收集流水线: 大量基础收集 + DAgger 迭代
# ============================================================
# Step 1: 基础专家收集 500eps × 9策略 (~200k transitions)
# Step 2: DAgger 4轮 (Round 0 纯专家 + Round 1-3 混合策略)
# ============================================================
set -e
cd "$(dirname "$0")/.."

LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"

echo "============================================================"
echo "Step 1: 大量基础专家数据收集"
echo "  500 episodes × 9 strategies, only-success"
echo "============================================================"

python -u training/collect_expert_trajectories.py \
    --low-level-ckpt "$LOW_CKPT" \
    --episodes-per-strategy 500 \
    --strategies pp lead pn apn cb dp expert_1.0 expert_2.0 expert_3.0 \
    --only-success \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --max-steps 6000 \
    --num-workers 9 \
    --output-dir data/expert_trajectories \
    --seed 20000

echo ""
echo "============================================================"
echo "Step 2: DAgger 迭代收集 (4 rounds)"
echo "  Round 0: beta=1.0 纯专家, 合并到已有数据"
echo "  Round 1-3: 学习策略 rollout + 专家标注"
echo "============================================================"

python -u training/dagger_collect.py \
    --low-level-ckpt "$LOW_CKPT" \
    --n-rounds 4 \
    --episodes-per-round 700 \
    --beta-decay 0.3 \
    --beta-min 0.1 \
    --only-success-round0 \
    --expert-strategies pp lead pn apn cb dp expert_1.0 expert_2.0 expert_3.0 \
    --bc-epochs 60 \
    --bc-batch-size 256 \
    --bc-lr 3e-4 \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --max-steps 6000 \
    --num-workers 9 \
    --output-dir data/expert_trajectories \
    --seed 42

echo ""
echo "============================================================"
echo "数据收集完成!"
echo "最终数据集: data/expert_trajectories/expert_dataset.npz"
echo "============================================================"
