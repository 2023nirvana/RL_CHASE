"""  
批量实验脚本  
============  
  
运行多个实验进行比较  
  
使用方法:  
    python training/run_experiments.py --config experiments/ablation.yaml  
"""  
  
import os  
import sys  
import argparse  
import yaml  
import subprocess  
import time  
from datetime import datetime  
from typing import Dict, List  
import json  
  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
  
class ExperimentRunner:  
    """  
    实验运行器  
      
    管理和运行多个实验  
    """  
      
    def __init__(self, config: Dict):  
        """  
        初始化  
          
        Args:  
            config: 实验配置  
        """  
        self.config = config  
        self.experiments = config.get('experiments', [])  
        self.base_config = config.get('base_config', 'configs/low_level_config.yaml')  
        self.output_dir = config.get('output_dir', 'experiments/results')  
          
        os.makedirs(self.output_dir, exist_ok=True)  
          
    def run_all(self):  
        """运行所有实验"""  
        results = []  
          
        for i, exp_config in enumerate(self.experiments):  
            print(f"\n{'='*60}")  
            print(f"Running experiment {i+1}/{len(self.experiments)}: {exp_config['name']}")  
            print(f"{'='*60}")  
              
            result = self.run_experiment(exp_config)  
            results.append(result)  
              
            # 保存中间结果  
            self._save_results(results)  
              
        # 生成报告  
        self._generate_report(results)  
          
        return results  
      
    def run_experiment(self, exp_config: Dict) -> Dict:  
        """  
        运行单个实验  
          
        Args:  
            exp_config: 实验配置  
              
        Returns:  
            result: 实验结果  
        """  
        name = exp_config['name']  
        seed = exp_config.get('seed', 42)  
        timesteps = exp_config.get('timesteps', 500000)  
          
        # 修改配置  
        modified_config = self._create_modified_config(exp_config)  
          
        # 保存临时配置  
        temp_config_path = os.path.join(self.output_dir, f'{name}_config.yaml')  
        with open(temp_config_path, 'w') as f:  
            yaml.dump(modified_config, f)  
              
        # 构建命令  
        cmd = [  
            sys.executable,  
            os.path.join(PROJECT_ROOT, 'training/train_low_level.py'),  
            '--config', temp_config_path,  
            '--experiment-name', name,  
            '--seed', str(seed),  
            '--total-timesteps', str(timesteps)  
        ]  
          
        # 运行  
        start_time = time.time()  
          
        try:  
            process = subprocess.run(cmd, capture_output=True, text=True)  
            success = process.returncode == 0  
              
            if not success:  
                print(f"Experiment failed: {process.stderr}")  
                  
        except Exception as e:  
            print(f"Error running experiment: {e}")  
            success = False  
              
        elapsed_time = time.time() - start_time  
          
        # 收集结果  
        result = {  
            'name': name,  
            'config': exp_config,  
            'success': success,  
            'elapsed_time': elapsed_time,  
            'timestamp': datetime.now().isoformat()  
        }  
          
        # 读取训练结果  
        if success:  
            result.update(self._read_experiment_results(name))  
              
        return result  
      
    def _create_modified_config(self, exp_config: Dict) -> Dict:  
        """创建修改后的配置"""  
        # 加载基础配置  
        with open(os.path.join(PROJECT_ROOT, self.base_config), 'r') as f:  
            config = yaml.safe_load(f)  
              
        # 应用修改  
        modifications = exp_config.get('modifications', {})  
        for key_path, value in modifications.items():  
            self._set_nested(config, key_path, value)  
              
        return config  
      
    @staticmethod  
    def _set_nested(d: Dict, key_path: str, value):  
        """设置嵌套字典值"""  
        keys = key_path.split('.')  
        for key in keys[:-1]:  
            d = d.setdefault(key, {})  
        d[keys[-1]] = value  
          
    def _read_experiment_results(self, name: str) -> Dict:  
        """读取实验结果"""  
        result = {}  
          
        # 尝试读取训练数据  
        data_path = os.path.join(  
            PROJECT_ROOT, 'visualization/logs', name, 'plots/training_data.json'  
        )  
          
        if os.path.exists(data_path):  
            with open(data_path, 'r') as f:  
                data = json.load(f)  
                  
            if 'episode/reward' in data:  
                rewards = data['episode/reward']['values']  
                result['final_reward'] = rewards[-1] if rewards else 0  
                result['best_reward'] = max(rewards) if rewards else 0  
                result['mean_reward'] = sum(rewards[-100:]) / min(100, len(rewards)) if rewards else 0  
                  
        return result  
      
    def _save_results(self, results: List[Dict]):  
        """保存结果"""  
        results_path = os.path.join(self.output_dir, 'results.json')  
        with open(results_path, 'w') as f:  
            json.dump(results, f, indent=2)  
              
    def _generate_report(self, results: List[Dict]):  
        """生成实验报告"""  
        report_path = os.path.join(self.output_dir, 'report.md')  
          
        with open(report_path, 'w') as f:  
            f.write("# Experiment Report\n\n")  
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")  
              
            f.write("## Summary\n\n")  
            f.write("| Experiment | Success | Final Reward | Best Reward | Time |\n")  
            f.write("|------------|---------|--------------|-------------|------|\n")  
              
            for r in results:  
                f.write(f"| {r['name']} | {'✓' if r['success'] else '✗'} | "  
                       f"{r.get('final_reward', 'N/A'):.2f} | "  
                       f"{r.get('best_reward', 'N/A'):.2f} | "  
                       f"{r['elapsed_time']/3600:.2f}h |\n")  
                  
            f.write("\n## Configurations\n\n")  
              
            for r in results:  
                f.write(f"### {r['name']}\n\n")  
                f.write("```yaml\n")  
                yaml.dump(r['config'], f, default_flow_style=False)  
                f.write("```\n\n")  
                  
        print(f"\nReport saved to {report_path}")  
  
  
def create_default_experiments():  
    """创建默认实验配置"""  
    return {  
        'base_config': 'configs/low_level_config.yaml',  
        'output_dir': 'experiments/results',  
        'experiments': [  
            {  
                'name': 'baseline',  
                'seed': 42,  
                'timesteps': 500000,  
                'modifications': {}  
            },  
            {  
                'name': 'higher_lr',  
                'seed': 42,  
                'timesteps': 500000,  
                'modifications': {  
                    'training.learning_rate': 1e-3  
                }  
            },  
            {  
                'name': 'lower_lr',  
                'seed': 42,  
                'timesteps': 500000,  
                'modifications': {  
                    'training.learning_rate': 1e-4  
                }  
            },  
            {  
                'name': 'more_entropy',  
                'seed': 42,  
                'timesteps': 500000,  
                'modifications': {  
                    'training.ent_coef': 0.05  
                }  
            },  
            {  
                'name': 'larger_network',  
                'seed': 42,  
                'timesteps': 500000,  
                'modifications': {  
                    'network.feature_extractor.hidden_sizes': [512, 512],  
                    'network.lstm.hidden_size': 512  
                }  
            }  
        ]  
    }  
  
  
def parse_args():  
    parser = argparse.ArgumentParser(description="Run multiple experiments")  
    parser.add_argument('--config', type=str, default=None,  
                       help='Experiments config file')  
    parser.add_argument('--create-default', action='store_true',  
                       help='Create default experiments config')  
    return parser.parse_args()  
  
  
def main():  
    args = parse_args()  
      
    if args.create_default:  
        config = create_default_experiments()  
        config_path = os.path.join(PROJECT_ROOT, 'experiments/ablation.yaml')  
        os.makedirs(os.path.dirname(config_path), exist_ok=True)  
        with open(config_path, 'w') as f:  
            yaml.dump(config, f, default_flow_style=False)  
        print(f"Default experiments config created: {config_path}")  
        return  
          
    if args.config:  
        config_path = os.path.join(PROJECT_ROOT, args.config)  
    else:  
        config_path = os.path.join(PROJECT_ROOT, 'experiments/ablation.yaml')  
          
    if not os.path.exists(config_path):  
        print(f"Config file not found: {config_path}")  
        print("Use --create-default to create a default config")  
        return  
          
    with open(config_path, 'r') as f:  
        config = yaml.safe_load(f)  
          
    runner = ExperimentRunner(config)  
    runner.run_all()  
  
  
if __name__ == "__main__":  
    main()  
