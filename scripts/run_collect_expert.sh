#!/bin/bash
# Phase 0: 收集专家轨迹
set -e
cd "$(dirname "$0")/.."

LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
OUTPUT_DIR="data/expert_trajectories"

python training/collect_expert_trajectories.py \
    --low-level-ckpt "$LOW_CKPT" \
    --episodes-per-strategy 100 \
    --only-success \
    --evader-mode medium \
    --strategies lead pn apn cb expert \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --output-dir "$OUTPUT_DIR" \
    --seed 0

echo "Done. Dataset saved to $OUTPUT_DIR"
