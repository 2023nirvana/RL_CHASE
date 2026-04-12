"""
3D V2 消融实验 V3 - 8-10 组，提高到达成功率与精度
==================================================

基于推荐（3d_v2_h09_bp05 表现最好），设计消融实验，每组仅改一个维度。
不修改原有 train_3d_v2.py 与 low_level_env_3d_v2.py。

新增文件：
- training/train_3d_v2_ablation_v3.py（本文件）
- core/environment/low_level_env_3d_v2_ablation.py（高精度 variant 使用）
- docs/3d_kinematics_and_simplifications.md（理论文档）

用法：
  python training/train_3d_v2_ablation_v3.py --all --max-parallel 4
  python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_h09 3d_v2_abl_prec03
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 基线：基于 3d_v2_h09_bp05 的最佳配置
BASE_ARGS = [
    "--hidden-sizes", "256", "128", "64",
    "--n-envs", "8",
    "--n-steps", "2048",
    "--heading-coef", "0.9",   # h09：之前最佳
    "--strict-direction-heading",
    "--backward-penalty", "0.5",
    "--seed", "42",
]

# 8-10 组消融配置（每组仅改一个维度）
ABLATION_VARIANTS = {
    # === 1. 奖励/惩罚参数 ===
    "3d_v2_abl_bp06": {
        "description": "消融: backward_penalty=0.6（略增）",
        "extra": ["--backward-penalty", "0.6", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
    "3d_v2_abl_bp04": {
        "description": "消融: backward_penalty=0.4（略减）",
        "extra": ["--backward-penalty", "0.4", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
    "3d_v2_abl_h08": {
        "description": "消融: heading_coef=0.8",
        "extra": ["--heading-coef", "0.8", "--backward-penalty", "0.5", "--target-distance", "12.0"],
        "override_base": ["heading_coef"],
        "use_subprocess": True,
    },
    "3d_v2_abl_wp02": {
        "description": "消融: heading_penalty_wrong=0.2",
        "extra": ["--heading-penalty-wrong", "0.2", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
    # === 2. 到达精度（需消融 env） ===
    "3d_v2_abl_prec03": {
        "description": "消融: success_threshold=0.3（高精度）",
        "extra": ["--target-distance", "12.0"],
        "use_subprocess": False,  # 需用 ablation env
        "success_threshold": 0.3,
    },
    # === 3. 网络结构 ===
    "3d_v2_abl_net_large": {
        "description": "消融: 网络 [512,256,128]",
        "extra": ["--hidden-sizes", "512", "256", "128", "--target-distance", "12.0"],
        "override_base": ["hidden_sizes"],
        "use_subprocess": True,
    },
    "3d_v2_abl_net_deep": {
        "description": "消融: 网络 [256,256,128,64]",
        "extra": ["--hidden-sizes", "256", "256", "128", "64", "--target-distance", "12.0"],
        "override_base": ["hidden_sizes"],
        "use_subprocess": True,
    },
    # === 4. 训练超参 ===
    "3d_v2_abl_lr_low": {
        "description": "消融: lr=1e-4",
        "extra": ["--learning-rate", "0.0001", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
    "3d_v2_abl_ent08": {
        "description": "消融: ent_coef=0.08（略增探索）",
        "extra": ["--ent-coef", "0.08", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
    "3d_v2_abl_nenvs16": {
        "description": "消融: n_envs=16",
        "extra": ["--n-envs", "16", "--target-distance", "12.0"],
        "use_subprocess": True,
    },
}


def _build_base_args(variant_config):
    """根据 override_base 构建 base args"""
    base = list(BASE_ARGS)
    if not variant_config.get("override_base"):
        return base
    overrides = variant_config["override_base"]
    if "heading_coef" in overrides:
        try:
            idx = base.index("--heading-coef")
            base.pop(idx)
            base.pop(idx)
        except ValueError:
            pass
    if "hidden_sizes" in overrides:
        try:
            idx = base.index("--hidden-sizes")
            for _ in range(4):
                base.pop(idx)
        except (ValueError, IndexError):
            pass
    return base


def run_subprocess_variant(variant_name, variant_config, total_timesteps, seed, gpu_id):
    """通过 subprocess 调用 train_3d_v2.py"""
    base = _build_base_args(variant_config)
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "training", "train_3d_v2.py"),
        "--experiment-name", variant_name,
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
    ] + base + variant_config["extra"]

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


def run_ablation_env_variant(variant_name, variant_config, total_timesteps, seed, gpu_id):
    """使用消融 env（success_threshold）直接运行"""
    import yaml
    import numpy as np
    import torch

    from training.train_3d_v2 import Trainer3DV2
    from core.environment.low_level_env_3d_v2_ablation import (
        LowLevelAUV3DEnvV2Ablation,
        VecLowLevelAUV3DEnvV2Ablation,
    )

    class Trainer3DV2Ablation(Trainer3DV2):
        """仅覆盖 _init_envs，使用消融 env"""

        def _init_envs(self):
            from core.environment.low_level_env_3d_v2_ablation import (
                LowLevelAUV3DEnvV2Ablation,
                VecLowLevelAUV3DEnvV2Ablation,
            )
            tc = self.config['training']
            self.num_envs = self.args.n_envs if self.args.n_envs else tc.get('n_envs', 8)
            success_threshold = getattr(self.args, 'success_threshold', 0.5)

            env_kwargs = dict(
                target_distance=self.args.target_distance,
                heading_coef=self.args.heading_coef,
                direction_coef=self.args.direction_coef,
                strict_direction_heading=self.args.strict_direction_heading,
                heading_penalty_wrong=self.args.heading_penalty_wrong,
                backward_facing_target_penalty=self.args.backward_penalty,
                max_steps=self.args.max_steps,
            )
            self.env_kwargs = {**env_kwargs, 'success_threshold': success_threshold}

            self.train_env = VecLowLevelAUV3DEnvV2Ablation(
                num_envs=self.num_envs, success_threshold=success_threshold, **env_kwargs
            )
            self.eval_env = LowLevelAUV3DEnvV2Ablation(
                success_threshold=success_threshold, **env_kwargs
            )
            self.obs_dim = self.train_env.observation_space.shape[0]
            self.num_actions = self.train_env.action_space.n

            print(f"3D V2 Ablation Env: success_threshold={success_threshold}m")
            print(f"  obs={self.obs_dim}, actions={self.num_actions}")

    success_threshold = variant_config.get("success_threshold", 0.5)
    base = _build_base_args(variant_config)

    class Args:
        pass

    args = Args()
    args.experiment_name = variant_name
    args.target_distance = 12.0
    args.max_steps = 4000
    args.heading_coef = 0.9
    args.direction_coef = 0.6
    args.strict_direction_heading = True
    args.heading_penalty_wrong = 0.0
    args.backward_penalty = 0.5
    args.total_timesteps = total_timesteps
    args.seed = seed
    args.hidden_sizes = [256, 128, 64]
    args.learning_rate = None
    args.clip_range = None
    args.ent_coef = None
    args.n_envs = 8
    args.n_steps = 2048
    args.batch_size = None
    args.resume = None
    args.success_threshold = success_threshold

    for i in range(0, len(base), 2):
        if i + 1 >= len(base):
            break
        k, v = base[i], base[i + 1]
        if k == "--heading-coef":
            args.heading_coef = float(v)
        elif k == "--backward-penalty":
            args.backward_penalty = float(v)
        elif k == "--n-envs":
            args.n_envs = int(v)
        elif k == "--n-steps":
            args.n_steps = int(v)

    for i in range(0, len(variant_config["extra"]), 2):
        if i + 1 >= len(variant_config["extra"]):
            break
        k, v = variant_config["extra"][i], variant_config["extra"][i + 1]
        if k == "--target-distance":
            args.target_distance = float(v)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['training']['total_timesteps'] = total_timesteps

    try:
        trainer = Trainer3DV2Ablation(config, args)
        trainer.train()
        return True
    except Exception as e:
        print(f"[{variant_name}] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def run_one_variant(variant_name, variant_config, total_timesteps, seed, gpu_id):
    if variant_config.get("use_subprocess", True):
        return run_subprocess_variant(
            variant_name, variant_config, total_timesteps, seed, gpu_id
        )
    return run_ablation_env_variant(
        variant_name, variant_config, total_timesteps, seed, gpu_id
    )


def main():
    parser = argparse.ArgumentParser(
        description="3D V2 Ablation V3: 8-10 variants for higher success rate"
    )
    parser.add_argument("--all", action="store_true", help="Run all ablation variants")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Variant names, e.g. 3d_v2_abl_h09 3d_v2_abl_prec03",
    )
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()

    if args.all:
        names = list(ABLATION_VARIANTS.keys())
    elif args.variants:
        names = [n for n in args.variants if n in ABLATION_VARIANTS]
        if len(names) != len(args.variants):
            unknown = set(args.variants) - set(ABLATION_VARIANTS.keys())
            print("Unknown variants:", unknown)
            print("Available:", list(ABLATION_VARIANTS.keys()))
    else:
        print("=" * 70)
        print("3D V2 Ablation V3 (提高到达成功率/精度)")
        print("=" * 70)
        for k, v in ABLATION_VARIANTS.items():
            sub = " [需 ablation env]" if not v.get("use_subprocess", True) else ""
            print(f"  {k:25s}  {v['description']}{sub}")
        print("\nUsage:")
        print("  --all                    Run all variants")
        print("  --variants <name1> ...   Run specific variants")
        print("  --max-parallel N         Max parallel jobs (default: 4)")
        return

    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0

    print(f"\n{'=' * 70}")
    print(f"3D V2 Ablation V3: {len(names)} variants, max_parallel={args.max_parallel}")
    print(f"Total timesteps={args.total_timesteps}, seed={args.seed}")
    print(f"{'=' * 70}\n")

    # 分离：ablation env  variants 在主进程运行（避免 CUDA fork 问题）
    subprocess_names = [n for n in names if ABLATION_VARIANTS[n].get("use_subprocess", True)]
    ablation_names = [n for n in names if not ABLATION_VARIANTS[n].get("use_subprocess", True)]

    results = {}

    for i, name in enumerate(ablation_names):
        gpu_id = i % n_gpu if n_gpu else None
        try:
            ok = run_ablation_env_variant(
                name, ABLATION_VARIANTS[name], args.total_timesteps, args.seed + i, gpu_id
            )
            results[name] = "OK" if ok else "FAIL"
            print(f"\n[{name}] {'OK' if ok else 'FAIL'}\n", flush=True)
        except Exception as e:
            results[name] = f"Error: {e}"
            print(f"\n[{name}] Error: {e}\n", flush=True)

    if subprocess_names:
        with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
            futures = {}
            for i, name in enumerate(subprocess_names):
                gpu_id = i % n_gpu if n_gpu else None
                fut = ex.submit(
                    run_subprocess_variant,
                    name,
                    ABLATION_VARIANTS[name],
                    args.total_timesteps,
                    args.seed + len(ablation_names) + i,
                    gpu_id,
                )
                futures[fut] = name

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
    print("3D V2 Ablation V3 Summary")
    print(f"{'=' * 70}")
    for name in names:
        status = results.get(name, "Unknown")
        desc = ABLATION_VARIANTS[name]["description"]
        print(f"  {name:25s} [{status:4s}]  {desc}")
    print(f"\nCheckpoints: checkpoints/<variant_name>/")
    print(f"Visualizations: visualization/logs/<variant_name>/plots/")


if __name__ == "__main__":
    main()
