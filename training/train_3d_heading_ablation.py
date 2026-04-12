"""
3D 朝向/奖励多组参数并行训练 - 大小参数都有，便于对比选优
================================================================

在 3d_combined 基线上，多组朝向相关参数同时训练，对比成功率和轨迹朝向效果后选一组。

思路包括：
  - 只调 heading_coef（小/中/大）
  - 加强“对准才前进”奖励 direction_coef
  - 严格模式：仅当朝向<30°才给前进奖励（strict_direction_heading）
  - 反朝向惩罚：朝向>90°却前进时小惩罚（heading_penalty_wrong）

使用：
  # 并行训练全部 8 组（建议 4~8 核或 2 卡）
  python training/train_3d_heading_ablation.py --all --max-parallel 4

  # 只跑其中几组
  python training/train_3d_heading_ablation.py --variants 3d_habl_h05 3d_habl_h07 3d_habl_h05_strict

  # 总步数可改（默认 200万）
  python training/train_3d_heading_ablation.py --all --total-timesteps 1000000
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 3d_combined 基线：大网络 + 8 envs + 2048 steps
BASE_ARGS = [
    "--hidden-sizes", "256", "128", "64",
    "--n-envs", "8",
    "--n-steps", "2048",
    "--target-distance", "5.0",
    "--seed", "42",
]

# 多组参数：大小都有，便于对比
HEADING_ABLATION_VARIANTS = {
    "3d_habl_h03": {
        "description": "heading_coef=0.3 (弱，对照)",
        "extra": ["--heading-coef", "0.3"],
    },
    "3d_habl_h05": {
        "description": "heading_coef=0.5 (当前默认)",
        "extra": ["--heading-coef", "0.5"],
    },
    "3d_habl_h07": {
        "description": "heading_coef=0.7 (强)",
        "extra": ["--heading-coef", "0.7"],
    },
    "3d_habl_h06_d08": {
        "description": "heading=0.6, direction_coef=0.8",
        "extra": ["--heading-coef", "0.6", "--direction-coef", "0.8"],
    },
    "3d_habl_h05_strict": {
        "description": "heading=0.5 + 严格: 仅<30°才奖励前进",
        "extra": ["--heading-coef", "0.5", "--strict-direction-heading"],
    },
    "3d_habl_h07_strict": {
        "description": "heading=0.7 + 严格: 仅<30°才奖励前进",
        "extra": ["--heading-coef", "0.7", "--strict-direction-heading"],
    },
    "3d_habl_h05_penalty": {
        "description": "heading=0.5 + 反朝向惩罚 0.1",
        "extra": ["--heading-coef", "0.5", "--heading-penalty-wrong", "0.1"],
    },
    "3d_habl_h07_penalty": {
        "description": "heading=0.7 + 反朝向惩罚 0.08",
        "extra": ["--heading-coef", "0.7", "--heading-penalty-wrong", "0.08"],
    },
}


def run_one_variant(
    variant_name: str,
    variant_config: Dict,
    total_timesteps: int,
    seed: int,
    gpu_id: int = None,
) -> bool:
    """跑单组实验：调用 train_3d.py 子进程"""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "training", "train_3d.py"),
        "--experiment-name", variant_name,
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
    ] + BASE_ARGS + variant_config["extra"]

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(f"[{variant_name}] {line}", end="", flush=True)
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        print(f"[{variant_name}] Error: {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="3D heading/reward ablation: multiple param groups in parallel"
    )
    parser.add_argument("--all", action="store_true", help="Run all heading ablation variants")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Variant names, e.g. 3d_habl_h05 3d_habl_h07_strict",
    )
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel jobs")
    args = parser.parse_args()

    if args.all:
        names = list(HEADING_ABLATION_VARIANTS.keys())
    elif args.variants:
        names = [n for n in args.variants if n in HEADING_ABLATION_VARIANTS]
        if len(names) != len(args.variants):
            unknown = set(args.variants) - set(HEADING_ABLATION_VARIANTS.keys())
            print("Unknown variants:", unknown)
            print("Available:", list(HEADING_ABLATION_VARIANTS.keys()))
    else:
        print("Heading ablation variants:")
        for k, v in HEADING_ABLATION_VARIANTS.items():
            print(f"  {k:25s}  {v['description']}")
        print("\nUse --all or --variants <name1> <name2> ...")
        return

    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0

    print(f"\n{'='*70}")
    print(f"3D Heading Ablation: {len(names)} variants, max_parallel={args.max_parallel}")
    print(f"Total timesteps={args.total_timesteps}, seed={args.seed}")
    if n_gpu:
        print(f"GPUs: {n_gpu}")
    print(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {}
        for i, name in enumerate(names):
            gpu_id = i % n_gpu if n_gpu else None
            fut = ex.submit(
                run_one_variant,
                name,
                HEADING_ABLATION_VARIANTS[name],
                args.total_timesteps,
                args.seed + i,
                gpu_id,
            )
            futures[fut] = name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                ok = fut.result()
                print(f"\n[{name}] {'OK' if ok else 'FAIL'}\n", flush=True)
            except Exception as e:
                print(f"\n[{name}] Error: {e}\n", flush=True)

    print(f"\n{'='*70}")
    print("Heading ablation runs finished. Compare logs under visualization/logs/<name>")
    print("and checkpoints under checkpoints/<name>, then pick one.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
