"""
2D 追逃多组并行训练 - 同时跑多组场景/难度，便于对比效果
========================================================================

场景变体（可比较）：
- baseline: 默认 (world=10, 初始距离 2~6, catch=0.5)
- small_world: 缩小场景 (world=5, 初始 1~3)，更容易碰到边界、回合更短
- easy_catch: 更容易抓到 (catch_radius=0.8, 初始 2~4)
- closer_start: 更近开局 (初始 1~3)，便于学收敛
- short_ep: 短回合 (max_episode_seconds=45)，加快单局

使用：
    # 并行跑全部场景（默认最多 2 个同时，避免单卡 OOM）
    python training/run_pe_2d_parallel.py --all

    # 只跑指定场景
    python training/run_pe_2d_parallel.py --scenarios baseline small_world easy_catch

    # 总步数、并行数
    python training/run_pe_2d_parallel.py --all --total-timesteps 500000 --max-parallel 2
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 场景名 -> 描述 + 传给 train_pe_low_level_2d.py 的参数（仅场景相关）
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "默认 (world=10, init_dist 2~6, catch=0.5)",
        "init_distance_range": "2,6",
        "world_size": 10.0,
        "catch_radius": 0.5,
        "max_episode_seconds": 90.0,
    },
    "small_world": {
        "description": "缩小场景 (world=5, init 1~3)",
        "init_distance_range": "1,3",
        "world_size": 5.0,
        "catch_radius": 0.5,
        "max_episode_seconds": 90.0,
    },
    "easy_catch": {
        "description": "更容易抓到 (catch=0.8, init 2~4)",
        "init_distance_range": "2,4",
        "world_size": 10.0,
        "catch_radius": 0.8,
        "max_episode_seconds": 90.0,
    },
    "closer_start": {
        "description": "更近开局 (init 1~3)",
        "init_distance_range": "1,3",
        "world_size": 10.0,
        "catch_radius": 0.5,
        "max_episode_seconds": 90.0,
    },
    "short_ep": {
        "description": "短回合 (45s)",
        "init_distance_range": "2,6",
        "world_size": 10.0,
        "catch_radius": 0.5,
        "max_episode_seconds": 45.0,
    },
    "small_easy": {
        "description": "缩小+易抓 (world=5, catch=0.8, init 1~3)",
        "init_distance_range": "1,3",
        "world_size": 5.0,
        "catch_radius": 0.8,
        "max_episode_seconds": 60.0,
    },
}


def build_cmd(
    scenario_name: str,
    scenario_config: Dict[str, Any],
    total_timesteps: int,
    seed: int,
    experiment_name: str,
) -> List[str]:
    script = os.path.join(PROJECT_ROOT, "training", "train_pe_low_level_2d.py")
    cmd = [
        sys.executable,
        script,
        "--experiment-name", experiment_name,
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--init-distance-range", scenario_config["init_distance_range"],
        "--world-size", str(scenario_config["world_size"]),
        "--catch-radius", str(scenario_config["catch_radius"]),
        "--max-episode-seconds", str(scenario_config["max_episode_seconds"]),
    ]
    return cmd


def run_one(
    scenario_name: str,
    scenario_config: Dict[str, Any],
    total_timesteps: int,
    seed: int,
    log_path: str,
) -> subprocess.Popen:
    exp_name = f"pe2d_{scenario_name}"
    cmd = build_cmd(scenario_name, scenario_config, total_timesteps, seed, exp_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"# scenario: {scenario_name}\n# {scenario_config['description']}\n# cmd: {' '.join(cmd)}\n\n")
    log_file.flush()
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def main():
    parser = argparse.ArgumentParser(description="2D 追逃多组场景并行训练")
    parser.add_argument("--all", action="store_true", help="跑全部场景")
    parser.add_argument("--scenarios", nargs="+", help="指定场景名，如 baseline small_world")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="每组总步数")
    parser.add_argument("--seed", type=int, default=42, help="基础 seed，每组 +1")
    parser.add_argument("--max-parallel", type=int, default=2, help="最多同时跑几个（单卡建议 1~2）")
    parser.add_argument("--print-commands", action="store_true", help="只打印每组单独命令，可复制到不同终端实时看输出")
    args = parser.parse_args()

    if args.all:
        names = list(SCENARIOS.keys())
    elif args.scenarios:
        names = [n for n in args.scenarios if n in SCENARIOS]
        if len(names) != len(args.scenarios):
            unknown = set(args.scenarios) - set(SCENARIOS.keys())
            print("Unknown scenarios (ignored):", unknown)
    else:
        if not args.print_commands:
            print("可用场景:")
            for name, cfg in SCENARIOS.items():
                print(f"  {name:15s} - {cfg['description']}")
            print("\n用法: --all 或 --scenarios <名1> <名2> ...")
            print("      --print-commands 只打印命令，不执行")
        return

    # 只打印命令，方便在不同终端各开一个看实时效果
    if args.print_commands:
        print("# 先在一个终端执行 cd，再复制下面每一行到不同终端运行\n")
        print(f"cd {PROJECT_ROOT}\n")
        for i, scenario_name in enumerate(names):
            cfg = SCENARIOS[scenario_name]
            exp_name = f"pe2d_{scenario_name}"
            parts = [
                "python", "training/train_pe_low_level_2d.py",
                "--experiment-name", exp_name,
                "--total-timesteps", str(args.total_timesteps),
                "--seed", str(args.seed + i),
                "--init-distance-range", cfg["init_distance_range"],
                "--world-size", str(cfg["world_size"]),
                "--catch-radius", str(cfg["catch_radius"]),
                "--max-episode-seconds", str(cfg["max_episode_seconds"]),
            ]
            print(f"# {scenario_name}: {cfg['description']}")
            print(" ".join(parts))
            print()
        return

    log_dir = os.path.join(PROJECT_ROOT, "visualization", "logs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    running: List[tuple] = []
    total = len(names)
    idx = 0

    print(f"共 {total} 组: {names}")
    print(f"每组 {args.total_timesteps:,} 步, 最多并行 {args.max_parallel}")
    print(f"日志: {log_dir}/pe2d_<场景>/train_{ts}.log\n")

    for i, scenario_name in enumerate(names):
        if len(running) >= args.max_parallel:
            # 先等至少一个结束再启动新的
            while len(running) >= args.max_parallel:
                still = []
                for name, p, path in running:
                    if p.poll() is None:
                        still.append((name, p, path))
                    else:
                        code = p.returncode
                        print(f"  [结束] {name} (exit {code})")
                running = still
                if len(running) >= args.max_parallel:
                    time.sleep(10)
        cfg = SCENARIOS[scenario_name]
        exp_name = f"pe2d_{scenario_name}"
        log_path = os.path.join(log_dir, exp_name, f"train_{ts}.log")
        proc = run_one(scenario_name, cfg, args.total_timesteps, args.seed + i, log_path)
        running.append((scenario_name, proc, log_path))
        print(f"  [启动] {scenario_name} -> {log_path}")

    while running:
        still = []
        for name, p, path in running:
            if p.poll() is None:
                still.append((name, p, path))
            else:
                code = p.returncode
                status = "ok" if code == 0 else f"exit {code}"
                print(f"  [结束] {name} ({status})")
        running = still
        if running:
            time.sleep(10)

    print("\n全部结束。对比可看: checkpoints/pe2d_<场景>/ 与 visualization/logs/pe2d_<场景>/")


if __name__ == "__main__":
    main()
