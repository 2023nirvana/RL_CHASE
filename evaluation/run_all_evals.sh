#!/bin/bash
# ============================================================================
# 一键评估所有模型的脚本
# ============================================================================
# 使用方法：
#   cd UnderwaterPursuitEvasion
#   bash evaluation/run_all_evals.sh
#
# 或指定参数：
#   bash evaluation/run_all_evals.sh --episodes 20 --save-gif
# ============================================================================

set -e

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "AUV Model Evaluation Suite"
echo "Project: $PROJECT_ROOT"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# 默认参数
EPISODES=10
TARGET_MOTION="constant"
SAVE_GIF=""
VERBOSE="--verbose"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --target-motion)
            TARGET_MOTION="$2"
            shift 2
            ;;
        --save-gif)
            SAVE_GIF="--save-gif"
            shift
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Parameters:"
echo "  Episodes: $EPISODES"
echo "  Target Motion: $TARGET_MOTION"
echo "  Save GIF: ${SAVE_GIF:-no}"
echo ""

# ============================================
# 方法1: 使用综合评估脚本（推荐）
# ============================================
echo "=============================================="
echo "[1/1] Running Comprehensive Evaluation..."
echo "=============================================="

python evaluation/run_model_evaluation.py \
    --all \
    --episodes "$EPISODES" \
    --target-motion "$TARGET_MOTION" \
    $SAVE_GIF \
    $VERBOSE

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo "Results saved in: visualization/eval_reports/"
echo ""

# ============================================
# 可选：分别运行各个评估脚本（备用）
# ============================================

# # 评估 2D 动态 MLP
# echo ">>> Evaluating 2D Dynamic MLP..."
# python evaluation/eval_2d_dynamic.py \
#     --checkpoint checkpoints/dynamic_mlp/best_model.pth \
#     --target-motion "$TARGET_MOTION" \
#     --episodes "$EPISODES" \
#     --output-prefix dynamic_mlp \
#     $SAVE_GIF

# # 评估 2D 动态 LSTM
# echo ">>> Evaluating 2D Dynamic LSTM..."
# python evaluation/eval_2d_dynamic.py \
#     --checkpoint checkpoints/dynamic_lstm/best_model.pth \
#     --use-lstm \
#     --target-motion "$TARGET_MOTION" \
#     --episodes "$EPISODES" \
#     --output-prefix dynamic_lstm \
#     $SAVE_GIF

# # 评估帧堆叠模型
# for STACK in 4 5 6 8; do
#     CKPT="checkpoints/dynamic_stack${STACK}/best_model.pth"
#     if [ -f "$CKPT" ]; then
#         echo ">>> Evaluating FrameStack-${STACK}..."
#         python evaluation/eval_2d_dynamic_framestack.py \
#             --checkpoint "$CKPT" \
#             --n-stack "$STACK" \
#             --target-motion "$TARGET_MOTION" \
#             --episodes "$EPISODES" \
#             --output-prefix "stack${STACK}" \
#             $SAVE_GIF
#     fi
# done

# # 评估 3D 固定目标
# for DIST in 3 5; do
#     CKPT="checkpoints/3d_dist_${DIST}m/best_model.pth"
#     if [ -f "$CKPT" ]; then
#         echo ">>> Evaluating 3D Fixed Target (${DIST}m)..."
#         python evaluation/eval_3d.py \
#             --checkpoint "$CKPT" \
#             --target-distance "${DIST}.0" \
#             --episodes "$EPISODES" \
#             --output-prefix "3d_${DIST}m"
#     fi
# done
