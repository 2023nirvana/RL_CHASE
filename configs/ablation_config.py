#!/usr/bin/env python3
"""
消融实验配置
============

定义各类消融实验的参数组合。
"""

ABLATION_EXPERIMENTS = {
    # ========== A1: 预训练方法消融 ==========
    "pretrain_method": {
        "description": "Compare different offline pretraining methods",
        "baseline": "td3bc",
        "experiments": [
            {
                "name": "no_pretrain",
                "pretrain_ckpt": None,  # 从零开始
                "description": "No pretraining, direct online PPO",
            },
            {
                "name": "td3bc",
                "pretrain_ckpt": "checkpoints/td3bc_v4_2m/best.pth",
                "description": "TD3+BC pretraining (baseline)",
            },
            {
                "name": "iql",
                "pretrain_ckpt": "checkpoints/iql_v2_2m/best.pth",
                "description": "IQL pretraining",
            },
            {
                "name": "bppo",
                "pretrain_ckpt": "checkpoints/bppo_v2_2m/best.pth",
                "description": "Behavior-regularized PPO pretraining",
            },
        ],
    },
    
    # ========== A2: KL约束消融 ==========
    "kl_constraint": {
        "description": "Ablate KL divergence constraint strength",
        "baseline": "beta_0.3",
        "experiments": [
            {
                "name": "beta_0.0",
                "beta_kl": 0.0,
                "description": "No KL constraint",
            },
            {
                "name": "beta_0.1",
                "beta_kl": 0.1,
                "description": "Weak KL constraint",
            },
            {
                "name": "beta_0.3",
                "beta_kl": 0.3,
                "description": "Medium KL constraint (baseline)",
            },
            {
                "name": "beta_0.5",
                "beta_kl": 0.5,
                "description": "Strong KL constraint",
            },
            {
                "name": "beta_1.0",
                "beta_kl": 1.0,
                "description": "Very strong KL constraint",
            },
        ],
    },
    
    # ========== A3: Subgoal范围消融 ==========
    "subgoal_range": {
        "description": "Ablate subgoal range parameter",
        "baseline": "sg_0.7",
        "experiments": [
            {
                "name": "sg_0.3",
                "subgoal_range": 0.3,
                "description": "Very small subgoal range",
            },
            {
                "name": "sg_0.5",
                "subgoal_range": 0.5,
                "description": "Small subgoal range",
            },
            {
                "name": "sg_0.7",
                "subgoal_range": 0.7,
                "description": "Medium subgoal range (baseline)",
            },
            {
                "name": "sg_1.0",
                "subgoal_range": 1.0,
                "description": "Large subgoal range",
            },
            {
                "name": "sg_2.0",
                "subgoal_range": 2.0,
                "description": "Very large subgoal range",
            },
        ],
    },
    
    # ========== A4: 逃跑者难度消融 ==========
    "evader_difficulty": {
        "description": "Ablate evader difficulty level",
        "baseline": "medium",
        "experiments": [
            {
                "name": "simple",
                "evader_mode": "simple",
                "description": "Simple evader (reactive)",
            },
            {
                "name": "medium",
                "evader_mode": "medium",
                "description": "Medium evader (baseline)",
            },
            {
                "name": "hard",
                "evader_mode": "hard",
                "description": "Hard evader (aggressive)",
            },
            {
                "name": "unicycle_medium",
                "evader_mode": "unicycle_medium",
                "description": "Unicycle sampling evader",
            },
        ],
    },
    
    # ========== A5: Low-level步数消融 ==========
    "low_level_steps": {
        "description": "Ablate number of low-level steps per high-level action",
        "baseline": "ls_50",
        "experiments": [
            {
                "name": "ls_25",
                "low_steps": 25,
                "description": "25 low-level steps (faster decisions)",
            },
            {
                "name": "ls_50",
                "low_steps": 50,
                "description": "50 low-level steps (baseline)",
            },
            {
                "name": "ls_100",
                "low_steps": 100,
                "description": "100 low-level steps (slower decisions)",
            },
        ],
    },
    
    # ========== A6: 奖励权重消融 ==========
    "reward_shaping": {
        "description": "Ablate reward shaping parameters",
        "baseline": "default",
        "experiments": [
            {
                "name": "no_shaping",
                "alpha_shaping": 0.0,
                "c_sg": 0.0,
                "c_path": 0.0,
                "description": "No reward shaping (sparse only)",
            },
            {
                "name": "default",
                "alpha_shaping": 5.0,
                "c_sg": 10.0,
                "c_path": 5.0,
                "description": "Default shaping (baseline)",
            },
            {
                "name": "strong_shaping",
                "alpha_shaping": 10.0,
                "c_sg": 20.0,
                "c_path": 10.0,
                "description": "Strong reward shaping",
            },
        ],
    },
    
    # ========== A7: 水流干扰消融 ==========
    "water_current": {
        "description": "Ablate water current disturbance level",
        "baseline": "level_0",
        "experiments": [
            {
                "name": "level_0",
                "current_config": "configs/current_level_0.yaml",
                "description": "No water current (baseline)",
            },
            {
                "name": "level_1",
                "current_config": "configs/current_level_1.yaml",
                "description": "Weak water current",
            },
            {
                "name": "level_2",
                "current_config": "configs/current_level_2.yaml",
                "description": "Medium water current",
            },
            {
                "name": "level_3",
                "current_config": "configs/current_level_3.yaml",
                "description": "Strong water current",
            },
        ],
    },
    
    # ========== A8: 网络架构消融 ==========
    "network_architecture": {
        "description": "Ablate network architecture",
        "baseline": "256_128",
        "experiments": [
            {
                "name": "128_64",
                "hidden": (128, 64),
                "description": "Small network",
            },
            {
                "name": "256_128",
                "hidden": (256, 128),
                "description": "Medium network (baseline)",
            },
            {
                "name": "512_256",
                "hidden": (512, 256),
                "description": "Large network",
            },
            {
                "name": "256_256_128",
                "hidden": (256, 256, 128),
                "description": "Deeper network",
            },
        ],
    },
}

# 常用组合实验
COMBINATION_EXPERIMENTS = {
    "full_pipeline": {
        "description": "Full training pipeline with different configurations",
        "phases": [
            {"name": "collect_expert", "script": "scripts/run_collect_expert_v2.sh"},
            {"name": "offline_pretrain", "script": "scripts/run_td3bc_pretrain_v3.sh"},
            {"name": "online_finetune", "script": "scripts/run_online_ppo_td3bc_v3.sh"},
        ],
    },
    
    "water_current_robustness": {
        "description": "Train with water current and test robustness",
        "train_current_levels": [0, 1, 2],
        "test_current_levels": [0, 1, 2, 3],
    },
}


def get_ablation_config(experiment_type: str, variant_name: str) -> dict:
    """获取特定消融实验的配置"""
    if experiment_type not in ABLATION_EXPERIMENTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    exp_group = ABLATION_EXPERIMENTS[experiment_type]
    for exp in exp_group["experiments"]:
        if exp["name"] == variant_name:
            return exp
    
    raise ValueError(f"Unknown variant: {variant_name} in {experiment_type}")


def list_all_experiments():
    """列出所有消融实验"""
    print("=" * 60)
    print("Available Ablation Experiments")
    print("=" * 60)
    
    for exp_type, exp_group in ABLATION_EXPERIMENTS.items():
        print(f"\n{exp_type}: {exp_group['description']}")
        print(f"  Baseline: {exp_group['baseline']}")
        print("  Variants:")
        for exp in exp_group["experiments"]:
            baseline_marker = " (baseline)" if exp["name"] == exp_group["baseline"] else ""
            print(f"    - {exp['name']}: {exp['description']}{baseline_marker}")


if __name__ == "__main__":
    list_all_experiments()
