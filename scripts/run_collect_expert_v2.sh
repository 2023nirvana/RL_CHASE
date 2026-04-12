#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "===== 收集更多专家轨迹 (7策略 × 300 episodes, 合并旧数据) ====="
echo "预计 ~30-45 分钟"
echo ""

python -u training/collect_expert_trajectories.py \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --episodes-per-strategy 300 \
    --only-success \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --max-steps 6000 \
    --output-dir data/expert_trajectories \
    --seed 10000 \
    --strategies pp lead pn apn cb dp expert_1.0 expert_2.0 expert_3.0 \
    --merge

echo ""
echo "===== 数据收集完成 ====="
echo "数据位置: data/expert_trajectories/expert_dataset.npz"
echo "统计图: data/expert_trajectories/dataset_stats.png"
