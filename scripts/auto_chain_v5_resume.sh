#!/bin/bash
# V5 续训自动链：resume online→500k  →  extend 2M
# 用法: nohup bash scripts/auto_chain_v5_resume.sh > logs/v5/auto_chain_resume.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v5
log() { echo "[$(date '+%F %T')] $*"; }

log "============================================"
log " V5 Auto Chain (RESUME): online→500k → extend 2M"
log "============================================"

# 检查 latest.pth
for exp in v5_bppo_A v5_bppo_B v5_iql_A v5_iql_B v5_td3bc_A v5_td3bc_B; do
    f="checkpoints/$exp/latest.pth"
    if [ ! -f "$f" ]; then
        log "ERROR: missing $f"; exit 1
    fi
    log "  OK  $f"
done

# ── Phase 1: 续训到 500k ──
log ""
log "Phase 1/2: resume online → 500k ..."
bash scripts/resume_online_v5_to_500k.sh
RC1=$?
log "resume 阶段退出码: $RC1"
if [ "$RC1" -ne 0 ]; then log "resume 失败，终止"; exit $RC1; fi

# 检查 final.pth
log ""
log "检查 final.pth 是否生成："
for exp in v5_bppo_A v5_bppo_B v5_iql_A v5_iql_B v5_td3bc_A v5_td3bc_B; do
    f="checkpoints/$exp/final.pth"
    if [ -f "$f" ]; then log "  OK  $f"; else log "  MISSING $f"; fi
done

# ── Phase 2: extend 到 2M ──
log ""
log "Phase 2/2: extend → 2M ..."
bash scripts/extend_v5_2m.sh
RC2=$?
log "extend 阶段退出码: $RC2"

log "============================================"
log " V5 Resume Chain 全部完成"
log "============================================"
