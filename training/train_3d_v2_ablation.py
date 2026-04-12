"""
3D V2 多组参数并行训练 - 解决"倒着开"问题
==========================================

在 3d_habl_h07_strict 基线上，测试不同的：
1. backward_facing_target_penalty（后退惩罚强度）
2. target_distance（目标距离）
3. heading_coef 配置

使用：
  # 并行训练全部组
  python training/train_3d_v2_ablation.py --all --max-parallel 4

  # 只跑其中几组
  python training/train_3d_v2_ablation.py --variants 3d_v2_bp05_d12 3d_v2_bp08_d15

  # 查看所有可用配置
  python training/train_3d_v2_ablation.py
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 基线参数（基于 3d_habl_h07_strict）
BASE_ARGS = [
    "--hidden-sizes", "256", "128", "64",
    "--n-envs", "8",
    "--n-steps", "2048",
    "--heading-coef", "0.7",
    "--strict-direction-heading",
    "--seed", "42",
]

# 多组参数配置
V2_ABLATION_VARIANTS = {
    # === 组1：不同后退惩罚强度，中等距离 (12m) ===
    "3d_v2_bp03_d12": {
        "description": "后退惩罚=0.3, 距离=12m (轻惩罚)",
        "extra": ["--backward-penalty", "0.3", "--target-distance", "12.0"],
    },
    "3d_v2_bp05_d12": {
        "description": "后退惩罚=0.5, 距离=12m (中惩罚，推荐)",
        "extra": ["--backward-penalty", "0.5", "--target-distance", "12.0"],
    },
    "3d_v2_bp08_d12": {
        "description": "后退惩罚=0.8, 距离=12m (重惩罚)",
        "extra": ["--backward-penalty", "0.8", "--target-distance", "12.0"],
    },
    
    # === 组2：不同后退惩罚强度，远距离 (15m) ===
    "3d_v2_bp05_d15": {
        "description": "后退惩罚=0.5, 距离=15m",
        "extra": ["--backward-penalty", "0.5", "--target-distance", "15.0", "--max-steps", "5000"],
    },
    "3d_v2_bp08_d15": {
        "description": "后退惩罚=0.8, 距离=15m",
        "extra": ["--backward-penalty", "0.8", "--target-distance", "15.0", "--max-steps", "5000"],
    },
    
    # === 组3：不同 heading_coef 配置 ===
    "3d_v2_h05_bp05": {
        "description": "heading=0.5 + 后退惩罚=0.5, 距离=12m",
        "extra": ["--heading-coef", "0.5", "--backward-penalty", "0.5", "--target-distance", "12.0"],
        "override_base": True,  # 需要覆盖 base 中的 heading-coef
    },
    "3d_v2_h09_bp05": {
        "description": "heading=0.9 + 后退惩罚=0.5, 距离=12m (强朝向)",
        "extra": ["--heading-coef", "0.9", "--backward-penalty", "0.5", "--target-distance", "12.0"],
        "override_base": True,
    },
    
    # === 组4：加上原有的 heading_penalty_wrong ===
    "3d_v2_bp05_wp01": {
        "description": "后退惩罚=0.5 + 反朝向惩罚=0.1, 距离=12m",
        "extra": ["--backward-penalty", "0.5", "--heading-penalty-wrong", "0.1", "--target-distance", "12.0"],
    },
}


def run_one_variant(
    variant_name: str,
    variant_config: dict,
    total_timesteps: int,
    seed: int,
    gpu_id: int = None,
) -> bool:
    """运行单组实验"""
    # 构建命令
    base_args = BASE_ARGS.copy()
    
    # 如果需要覆盖 base 参数
    if variant_config.get("override_base"):
        # 移除 base 中的 heading-coef
        try:
            idx = base_args.index("--heading-coef")
            base_args.pop(idx)  # 移除 --heading-coef
            base_args.pop(idx)  # 移除其值
        except ValueError:
            pass
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "training", "train_3d_v2.py"),
        "--experiment-name", variant_name,
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
    ] + base_args + variant_config["extra"]

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
        description="3D V2 ablation: multiple param groups in parallel"
    )
    parser.add_argument("--all", action="store_true", help="Run all V2 ablation variants")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Variant names, e.g. 3d_v2_bp05_d12 3d_v2_bp08_d15",
    )
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel jobs")
    args = parser.parse_args()

    if args.all:
        names = list(V2_ABLATION_VARIANTS.keys())
    elif args.variants:
        names = [n for n in args.variants if n in V2_ABLATION_VARIANTS]
        if len(names) != len(args.variants):
            unknown = set(args.variants) - set(V2_ABLATION_VARIANTS.keys())
            print("Unknown variants:", unknown)
            print("Available:", list(V2_ABLATION_VARIANTS.keys()))
    else:
        print("=" * 70)
        print("3D V2 Ablation Variants (解决'倒着开'问题)")
        print("=" * 70)
        for k, v in V2_ABLATION_VARIANTS.items():
            print(f"  {k:22s}  {v['description']}")
        print("\nUsage:")
        print("  --all                    Run all variants")
        print("  --variants <name1> ...   Run specific variants")
        print("  --max-parallel N         Max parallel jobs (default: 4)")
        print("  --total-timesteps N      Total timesteps (default: 2000000)")
        return

    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0

    print(f"\n{'=' * 70}")
    print(f"3D V2 Ablation: {len(names)} variants, max_parallel={args.max_parallel}")
    print(f"Total timesteps={args.total_timesteps}, seed={args.seed}")
    if n_gpu:
        print(f"GPUs: {n_gpu}")
    print(f"{'=' * 70}")
    
    print("\nVariants to run:")
    for name in names:
        print(f"  - {name}: {V2_ABLATION_VARIANTS[name]['description']}")
    print()

    with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {}
        for i, name in enumerate(names):
            gpu_id = i % n_gpu if n_gpu else None
            fut = ex.submit(
                run_one_variant,
                name,
                V2_ABLATION_VARIANTS[name],
                args.total_timesteps,
                args.seed + i,
                gpu_id,
            )
            futures[fut] = name

        results = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                ok = fut.result()
                results[name] = "OK" if ok else "FAIL"
                print(f"\n[{name}] {'OK' if ok else 'FAIL'}\n", flush=True)
            except Exception as e:
                results[name] = f"Error: {e}"
                print(f"\n[{name}] Error: {e}\n", flush=True)

    print(f"\n{'=' * 70}")
    print("3D V2 Ablation Summary")
    print(f"{'=' * 70}")
    for name in names:
        status = results.get(name, "Unknown")
        desc = V2_ABLATION_VARIANTS[name]['description']
        print(f"  {name:22s} [{status:4s}]  {desc}")
    print(f"\nCheckpoints: checkpoints/<variant_name>/")
    print(f"Visualizations: visualization/logs/<variant_name>/plots/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
