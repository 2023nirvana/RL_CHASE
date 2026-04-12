#!/bin/bash
# 对两个新模型生成 enhanced_vis 可视化
# 训练完成后运行此脚本

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo " Enhanced Visualization for New Models"
echo "=========================================="

# --- Moretight (reach_threshold=0.05) ---
EXP1="low_near/v4_tight_heading_terminal_Moretight"
if [ -f "$EXP1/checkpoints/best.pth" ]; then
    echo ""
    echo "[1/2] Generating enhanced_vis for Moretight ..."
    python low_near/visualize_v3_enhanced.py \
        --exp-dir "$EXP1" \
        --episodes 50 \
        --gif-episodes 5 \
        --fps 20 \
        --seed 1234 \
        --reach-threshold 0.05 \
        --init-velocity-range 1.0
    echo "[1/2] Done."
else
    echo "[1/2] SKIP: $EXP1/checkpoints/best.pth not found (training not finished?)"
fi

# --- No-timepenalty (reach_threshold=0.10) ---
EXP2="low_near/v4_tight_heading_terminal_No-timepenalty"
if [ -f "$EXP2/checkpoints/best.pth" ]; then
    echo ""
    echo "[2/2] Generating enhanced_vis for No-timepenalty ..."
    python low_near/visualize_v3_enhanced.py \
        --exp-dir "$EXP2" \
        --episodes 50 \
        --gif-episodes 5 \
        --fps 20 \
        --seed 1234 \
        --reach-threshold 0.10 \
        --init-velocity-range 1.0
    echo "[2/2] Done."
else
    echo "[2/2] SKIP: $EXP2/checkpoints/best.pth not found (training not finished?)"
fi

echo ""
echo "=========================================="
echo " All visualizations complete."
echo "=========================================="
