#!/bin/bash
# 3D模型变体训练命令集合
# ============================================
# 使用方法：在不同的终端窗口中分别运行这些命令
# 每个命令对应一个变体，可以独立跟踪训练效果

cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

TARGET_DISTANCE=5.0
TOTAL_TIMESTEPS=2000000
SEED=42

echo "=========================================="
echo "3D Model Variants Training Commands"
echo "=========================================="
echo "Target Distance: ${TARGET_DISTANCE}m"
echo "Total Timesteps: ${TOTAL_TIMESTEPS}"
echo "=========================================="
echo ""
echo "Copy and run each command in a separate terminal:"
echo ""

# 变体1: Baseline
echo "# =========================================="
echo "# Variant 1: Baseline (128, 64)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_baseline \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED}"
echo ""

# 变体2: Large Network
echo "# =========================================="
echo "# Variant 2: Large Network (256, 128, 64)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_large_net \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --hidden-sizes 256 128 64"
echo ""

# 变体3: Deep Network
echo "# =========================================="
echo "# Variant 3: Deep Network (128, 128, 64, 32)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_deep_net \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --hidden-sizes 128 128 64 32"
echo ""

# 变体4: High Learning Rate
echo "# =========================================="
echo "# Variant 4: High Learning Rate (1e-3)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_high_lr \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --learning-rate 0.001"
echo ""

# 变体5: Low Clip Range
echo "# =========================================="
echo "# Variant 5: Low Clip Range (0.1)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_low_clip \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --clip-range 0.1"
echo ""

# 变体6: High Entropy
echo "# =========================================="
echo "# Variant 6: High Entropy Coef (0.1)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_high_ent \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --ent-coef 0.1"
echo ""

# 变体7: More Environments
echo "# =========================================="
echo "# Variant 7: More Environments (8)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_more_envs \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --n-envs 8"
echo ""

# 变体8: Longer Steps
echo "# =========================================="
echo "# Variant 8: Longer Steps (2048)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_long_steps \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --n-steps 2048"
echo ""

# 变体9: Large Batch
echo "# =========================================="
echo "# Variant 9: Large Batch (512)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_large_batch \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --batch-size 512"
echo ""

# 变体10: Combined
echo "# =========================================="
echo "# Variant 10: Combined (Large Net + More Envs + Long Steps)"
echo "# =========================================="
echo "python training/train_3d.py \\"
echo "    --experiment-name 3d_combined \\"
echo "    --target-distance ${TARGET_DISTANCE} \\"
echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
echo "    --seed ${SEED} \\"
echo "    --hidden-sizes 256 128 64 \\"
echo "    --n-envs 8 \\"
echo "    --n-steps 2048"
echo ""

echo "=========================================="
echo "All commands listed above"
echo "=========================================="
