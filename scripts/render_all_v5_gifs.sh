#!/bin/bash
# 为 6 个 v5 online 实验渲染 GIF
set -e
cd "$(dirname "$0")/.."
mkdir -p visualization/gif_out

unset OMP_NUM_THREADS

for exp in v5_bppo_A v5_bppo_B v5_iql_A v5_iql_B v5_td3bc_A v5_td3bc_B; do
    ckpt="checkpoints/$exp/best.pth"
    out="visualization/gif_out/${exp#v5_}.gif"
    if [ ! -f "$ckpt" ]; then
        echo "SKIP $exp: $ckpt missing"
        continue
    fi
    echo ""
    echo "========== $exp =========="
    python visualization/v5_rollout_gif.py \
        --ckpt "$ckpt" --output "$out" \
        --episodes 3 --fps 20 --skip 2 2>&1 \
        | grep -E "^\[ep|success|SUCCESS|FAIL|pursuer|evader |Picked|Saved|max=" \
        | head -20
done
echo ""
echo "DONE. GIFs in visualization/gif_out/"
ls -lh visualization/gif_out/*.gif
