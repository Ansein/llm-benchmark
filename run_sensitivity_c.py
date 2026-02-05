"""
场景C敏感度分析实验控制器
支持3×3参数网格（τ_mean × σ），多个模型并行测试
只运行迭代学习模式（配置B、C、D）
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluators.llm_client import load_model_configs
from src.evaluators.scenario_c_sensitivity_helpers import (
    run_config_B, run_config_C, run_config_D,
    compute_all_metrics
)


def run_sensitivity_experiment(
    tau_mean_values: List[float],
    sigma_values: List[float],
    model_names: List[str],
    num_trials: int = 3,
    num_rounds: int = 20,
    output_dir: str = "sensitivity_results/scenario_c",
    config_file: str = "configs/model_configs.json"
):
    """
    运行场景C敏感度实验
    
    Args:
        tau_mean_values: τ_mean值列表
        sigma_values: σ值列表
        model_names: 模型名称列表
        num_trials: 每组参数的重复次数
        num_rounds: 迭代学习轮数
        output_dir: 输出目录
        config_file: 模型配置文件
    """
    print(f"\n{'='*80}")
    print(f"场景C敏感度分析 - 迭代学习模式")
    print(f"{'='*80}")
    print(f"参数网格:")
    print(f"  τ_mean值: {tau_mean_values}")
    print(f"  σ值: {sigma_values}")
    print(f"模型列表: {model_names}")
    print(f"重复次数: {num_trials}")
    print(f"迭代轮数: {num_rounds}")
    print(f"总实验数: {len(tau_mean_values)} × {len(sigma_values)} × {len(model_names)} × {num_trials} = {len(tau_mean_values) * len(sigma_values) * len(model_names) * num_trials}")
    
    # 加载模型配置
    model_configs = load_model_configs(config_file)
    
    # 验证所有模型配置存在
    for model_name in model_names:
        if model_name not in model_configs:
            available = list(model_configs.keys())
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {available}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"sensitivity_3x3_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存实验配置
    config = {
        "experiment_type": "sensitivity_analysis",
        "scenario": "C",
        "design": "3x3_grid",
        "tau_mean_values": tau_mean_values,
        "sigma_values": sigma_values,
        "models": model_names,
        "num_trials": num_trials,
        "num_rounds": num_rounds,
        "timestamp": timestamp,
    }
    
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验配置已保存: {config_path}")
    
    # 收集所有结果
    all_results = []
    total_experiments = len(tau_mean_values) * len(sigma_values) * len(model_names) * num_trials
    current_exp = 0
    
    # 遍历所有参数组合
    for tau_mean in tau_mean_values:
        for sigma in sigma_values:
            print(f"\n{'='*80}")
            print(f"参数组合: τ_mean={tau_mean}, σ={sigma}")
            print(f"{'='*80}")
            
            # 加载对应的GT文件
            gt_filename = f"scenario_c_tau{tau_mean:.1f}_sigma{sigma:.1f}.json"
            gt_path = Path("data/ground_truth/sensitivity_c") / gt_filename
            
            if not gt_path.exists():
                print(f"❌ GT文件不存在: {gt_path}")
                print("请先运行: python scripts/generate_scenario_c_sensitivity_gt.py")
                continue
            
            # 加载GT
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            print(f"GT文件: {gt_path}")
            print(f"  最优策略: {gt_data['optimal_strategy']['anonymization_star']}")
            print(f"  中介利润*: {gt_data['optimal_strategy']['intermediary_profit_star']:.4f}")
            print(f"  社会福利*: {gt_data['equilibrium']['social_welfare']:.4f}")
            
            # 遍历所有模型
            for model_name in model_names:
                print(f"\n--- 模型: {model_name} ---")
                
                # 创建模型专属目录
                model_dir = exp_dir / model_name / f"tau{tau_mean:.1f}_sigma{sigma:.1f}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # 重复多次试验
                for trial in range(1, num_trials + 1):
                    current_exp += 1
                    print(f"\n  [试验 {trial}/{num_trials}] ({current_exp}/{total_experiments})")
                    
                    try:
                        # 运行三个配置
                        results = {}
                        
                        # 配置B: 理性中介 × LLM消费者
                        print("    运行配置B...")
                        result_b = run_config_B(
                            model_name=model_name,
                            gt_data=gt_data,
                            num_rounds=num_rounds,
                            output_dir=str(model_dir / f"trial{trial}")
                        )
                        results['config_B'] = result_b
                        
                        # 配置C: LLM中介 × 理性消费者
                        print("    运行配置C...")
                        result_c = run_config_C(
                            model_name=model_name,
                            gt_data=gt_data,
                            num_rounds=num_rounds,
                            output_dir=str(model_dir / f"trial{trial}")
                        )
                        results['config_C'] = result_c
                        
                        # 配置D: LLM中介 × LLM消费者
                        print("    运行配置D...")
                        result_d = run_config_D(
                            model_name=model_name,
                            gt_data=gt_data,
                            num_rounds=num_rounds,
                            output_dir=str(model_dir / f"trial{trial}")
                        )
                        results['config_D'] = result_d
                        
                        # 计算综合指标
                        metrics = compute_all_metrics(results, gt_data)
                        
                        # 保存单次实验结果
                        experiment_result = {
                            "sensitivity_params": {
                                "tau_mean": tau_mean,
                                "sigma": sigma
                            },
                            "model_name": model_name,
                            "trial": trial,
                            "timestamp": datetime.now().isoformat(),
                            "results": results,
                            "metrics": metrics,
                            "gt_summary": {
                                "anonymization_star": gt_data['optimal_strategy']['anonymization_star'],
                                "intermediary_profit_star": gt_data['optimal_strategy']['intermediary_profit_star'],
                                "social_welfare": gt_data['equilibrium']['social_welfare']
                            }
                        }
                        
                        all_results.append(experiment_result)
                        
                        # 保存到文件
                        result_file = model_dir / f"trial{trial}_result.json"
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
                        
                        print(f"    ✅ 实验成功")
                        print(f"    配置B最终利润: {result_b['final_profit']:.4f}")
                        print(f"    配置C最终利润: {result_c['final_profit']:.4f}")
                        print(f"    配置D最终利润: {result_d['final_profit']:.4f}")
                        
                    except Exception as e:
                        print(f"    ❌ 实验失败: {e}")
                        import traceback
                        traceback.print_exc()
    
    # 保存汇总结果
    summary_path = exp_dir / "summary_all_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 实验完成！")
    print(f"{'='*80}")
    print(f"总实验数: {len(all_results)}/{total_experiments}")
    print(f"结果目录: {exp_dir}")
    print(f"  - 配置文件: experiment_config.json")
    print(f"  - 汇总结果: summary_all_results.json")
    print(f"  - 模型结果: {model_name}/ (每个模型一个子目录)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="场景C敏感度分析实验")
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='模型名称列表（空格分隔）'
    )
    
    parser.add_argument(
        '--num-trials',
        type=int,
        default=3,
        help='每组参数的重复次数（默认3次）'
    )
    
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=20,
        help='迭代学习轮数（默认20轮）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sensitivity_results/scenario_c',
        help='输出目录（默认: sensitivity_results/scenario_c）'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default='configs/model_configs.json',
        help='模型配置文件路径'
    )
    
    args = parser.parse_args()
    
    # 参数网格（固定）
    tau_mean_values = [0.5, 1.0, 1.5]
    sigma_values = [0.5, 1.0, 2.0]
    
    # 运行实验
    run_sensitivity_experiment(
        tau_mean_values=tau_mean_values,
        sigma_values=sigma_values,
        model_names=args.models,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        config_file=args.config_file
    )


if __name__ == "__main__":
    main()
