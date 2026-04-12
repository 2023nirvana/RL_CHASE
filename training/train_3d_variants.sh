#!/bin/bash
# 3D模型并行训练脚本 - 多个变体同时训练
# ============================================
#
# 使用方法：
#   bash training/train_3d_variants.sh --all --target-distance 5.0
#   bash training/train_3d_variants.sh --variants 3d_large_net 3d_high_lr

cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

TARGET_DISTANCE=5.0
TOTAL_TIMESTEPS=2000000
SEED=42
MAX_PARALLEL=4

# 解析参数
TRAIN_ALL=false
VARIANTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            TRAIN_ALL=true
            shift
            ;;
        --variants)
            VARIANTS="$2"
            shift 2
            ;;
        --target-distance)
            TARGET_DISTANCE="$2"
            shift 2
            ;;
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "3D Model Variants Parallel Training"
echo "=========================================="
echo "Target Distance: ${TARGET_DISTANCE}m"
echo "Total Timesteps: ${TOTAL_TIMESTEPS}"
echo "Max Parallel: ${MAX_PARALLEL}"
echo "=========================================="

# 定义变体配置（通过环境变量传递）
declare -A VARIANTS_CONFIG

# 变体1: Baseline
VARIANTS_CONFIG[3d_baseline]="--experiment-name 3d_baseline"

# 变体2: Large Network (256, 128, 64)
VARIANTS_CONFIG[3d_large_net]="--experiment-name 3d_large_net --hidden-sizes 256 128 64"

# 变体3: Deep Network (128, 128, 64, 32)
VARIANTS_CONFIG[3d_deep_net]="--experiment-name 3d_deep_net --hidden-sizes 128 128 64 32"

# 变体4: High Learning Rate
VARIANTS_CONFIG[3d_high_lr]="--experiment-name 3d_high_lr --learning-rate 0.001"

# 变体5: Low Clip Range
VARIANTS_CONFIG[3d_low_clip]="--experiment-name 3d_low_clip --clip-range 0.1"

# 变体6: High Entropy
VARIANTS_CONFIG[3d_high_ent]="--experiment-name 3d_high_ent --ent-coef 0.1"

# 变体7: More Environments
VARIANTS_CONFIG[3d_more_envs]="--experiment-name 3d_more_envs --n-envs 8"

# 变体8: Longer Steps
VARIANTS_CONFIG[3d_long_steps]="--experiment-name 3d_long_steps --n-steps 2048"

# 变体9: Large Batch
VARIANTS_CONFIG[3d_large_batch]="--experiment-name 3d_large_batch --batch-size 512"

# 变体10: Combined (Large Net + More Envs + Long Steps)
VARIANTS_CONFIG[3d_combined]="--experiment-name 3d_combined --hidden-sizes 256 128 64 --n-envs 8 --n-steps 2048"

# 确定要训练的变体
if [ "$TRAIN_ALL" = true ]; then
    VARIANTS_LIST=("3d_baseline" "3d_large_net" "3d_deep_net" "3d_high_lr" "3d_low_clip" "3d_high_ent" "3d_more_envs" "3d_long_steps" "3d_large_batch" "3d_combined")
elif [ -n "$VARIANTS" ]; then
    IFS=' ' read -ra VARIANTS_LIST <<< "$VARIANTS"
else
    echo "Available variants:"
    for variant in "${!VARIANTS_CONFIG[@]}"; do
        echo "  $variant"
    done
    echo ""
    echo "Use --all to train all variants or --variants <name1> <name2> ..."
    exit 1
fi

echo "Training variants: ${VARIANTS_LIST[@]}"
echo ""

# 创建日志目录
LOG_DIR="visualization/logs/3d_variants_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 启动并行训练
PIDS=()
for variant in "${VARIANTS_LIST[@]}"; do
    if [ -z "${VARIANTS_CONFIG[$variant]}" ]; then
        echo "Warning: Unknown variant $variant, skipping"
        continue
    fi
    
    echo "Starting training: $variant"
    
    # 构建训练命令
    CMD="python training/train_3d.py \
        --target-distance $TARGET_DISTANCE \
        --total-timesteps $TOTAL_TIMESTEPS \
        --seed $SEED \
        ${VARIANTS_CONFIG[$variant]}"
    
    # 在后台运行，重定向输出到日志文件
    LOG_FILE="$LOG_DIR/${variant}.log"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    
    echo "  PID: $PID, Log: $LOG_FILE"
    
    # 控制并行数量
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        sleep 5
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
            fi
        done
        PIDS=("${PIDS[@]}")
    done
done

# 等待所有进程完成
echo ""
echo "Waiting for all training jobs to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
    echo "  Process $PID completed"
done

echo ""
echo "=========================================="
echo "All training jobs completed!"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
