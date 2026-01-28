"""
场景B敏感度分析实验控制器 - 多模型版本
支持3×3参数网格，多个模型并行测试
"""
import argparse
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from src.evaluators.llm_client import create_llm_client
from run_prompt_experiments import CustomScenarioBEvaluator, PromptVersionParser


def run_sensitivity_experiment(
    rho_values: List[float],
    v_ranges: List[Tuple[float, float]],
    prompt_version: str,
    model_names: List[str],
    num_trials: int = 3,
    output_dir: str = "sensitivity_results/scenario_b"
):
    """
    运行敏感度实验（方案D：多模型版本）
    
    Args:
        rho_values: rho值列表
        v_ranges: v范围列表 [(v_min, v_max), ...]
        prompt_version: 提示词版本（单个）
        model_names: 模型名称列表
        num_trials: 重复次数
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"场景B敏感度分析 - 方案D（多模型×多试验）")
    print(f"{'='*80}")
    print(f"参数网格:")
    print(f"  ρ值: {rho_values}")
    print(f"  v范围: {v_ranges}")
    print(f"提示词版本: {prompt_version}")
    print(f"模型列表: {model_names}")
    print(f"重复次数: {num_trials}")
    print(f"总实验数: {len(rho_values)} × {len(v_ranges)} × {len(model_names)} × {num_trials} = {len(rho_values) * len(v_ranges) * len(model_names) * num_trials}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"sensitivity_3x3_{prompt_version}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存实验配置
    config = {
        "experiment_type": "sensitivity_analysis",
        "design": "3x3_grid",
        "rho_values": rho_values,
        "v_ranges": [[v[0], v[1]] for v in v_ranges],
        "prompt_version": prompt_version,
        "models": model_names,
        "num_trials": num_trials,
        "timestamp": timestamp,
    }
    
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验配置已保存: {config_path}")
    
    # 初始化提示词解析器
    parser = PromptVersionParser()
    prompts = parser.get_version(prompt_version)
    
    # 收集所有结果
    all_results = []
    total_experiments = len(rho_values) * len(v_ranges) * len(model_names) * num_trials
    current_exp = 0
    
    # 遍历所有参数组合
    for rho in rho_values:
        for v_min, v_max in v_ranges:
            print(f"\n{'='*80}")
            print(f"参数组合: ρ={rho}, v=[{v_min}, {v_max}]")
            print(f"{'='*80}")
            
            # 加载对应的GT文件
            gt_filename = f"scenario_b_rho{rho:.1f}_v{v_min:.1f}-{v_max:.1f}.json"
            gt_path = Path("data/ground_truth/sensitivity_b") / gt_filename
            
            if not gt_path.exists():
                print(f"❌ GT文件不存在: {gt_path}")
                print(f"   请先运行: python scripts/generate_sensitivity_b_gt.py")
                continue
            
            # 遍历所有模型
            for model_name in model_names:
                print(f"\n--- 模型: {model_name} ---")
                
                # 为该模型创建子目录
                model_dir = exp_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                # 运行多次试验
                for trial_idx in range(num_trials):
                    current_exp += 1
                    print(f"\n  [试验 {trial_idx+1}/{num_trials}] ({current_exp}/{total_experiments})")
                    
                    # 创建日志目录
                    log_dir = exp_dir / "llm_logs" / model_name / \
                              f"rho{rho:.1f}_v{v_min:.1f}-{v_max:.1f}_trial{trial_idx+1}"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # 创建LLM客户端
                        llm_client = create_llm_client(model_name, log_dir=str(log_dir))
                        
                        # 创建评估器
                        evaluator = CustomScenarioBEvaluator(
                            llm_client=llm_client,
                            ground_truth_path=str(gt_path),
                            custom_system_prompt=prompts["system"],
                            custom_user_prompt_template=prompts["user_template"],
                            use_theory_platform=True
                        )
                        
                        # 运行评估（num_trials=1，因为我们在外层循环重复）
                        results = evaluator.simulate_static_game(num_trials=1)
                        
                        # 添加元信息
                        results["sensitivity_params"] = {
                            "rho": rho,
                            "v_min": v_min,
                            "v_max": v_max,
                        }
                        results["experiment_meta"] = {
                            "model_name": model_name,
                            "prompt_version": prompt_version,
                            "trial_index": trial_idx + 1,
                            "timestamp": datetime.now().isoformat(),
                        }
                        
                        # 保存单个结果
                        result_filename = f"result_rho{rho:.1f}_v{v_min:.1f}-{v_max:.1f}_trial{trial_idx+1}.json"
                        result_path = model_dir / result_filename
                        
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        
                        # 打印关键指标
                        metrics = results.get("metrics", {})
                        eq_quality = results.get("equilibrium_quality", {})
                        
                        llm_share_rate = metrics.get("llm", {}).get("share_rate", 0)
                        gt_share_rate = metrics.get("ground_truth", {}).get("share_rate", 0)
                        jaccard = eq_quality.get("share_set_similarity", 0)
                        
                        print(f"    分享率: LLM={llm_share_rate:.2%}, GT={gt_share_rate:.2%}")
                        print(f"    Jaccard相似度: {jaccard:.3f}")
                        print(f"    ✅ 已保存: {result_filename}")
                        
                        # 收集到汇总
                        all_results.append(results)
                        
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
    print(f"  - 模型结果: {'/'.join(model_names)}/")
    print(f"  - LLM日志: llm_logs/")
    print("="*80)
    
    # 生成快速统计
    print_quick_statistics(all_results, model_names)
    
    return all_results, exp_dir


def print_quick_statistics(results: List[Dict], model_names: List[str]):
    """打印快速统计摘要"""
    print(f"\n{'='*80}")
    print("快速统计摘要")
    print("="*80)
    
    # 按模型统计
    for model in model_names:
        model_results = [r for r in results if r.get("experiment_meta", {}).get("model_name") == model]
        
        if not model_results:
            continue
        
        # 提取指标
        jaccards = [r.get("equilibrium_quality", {}).get("share_set_similarity", 0) for r in model_results]
        share_rate_errors = [abs(
            r.get("metrics", {}).get("llm", {}).get("share_rate", 0) - 
            r.get("metrics", {}).get("ground_truth", {}).get("share_rate", 0)
        ) for r in model_results]
        
        print(f"\n模型: {model}")
        print(f"  实验数: {len(model_results)}")
        print(f"  Jaccard相似度: 均值={np.mean(jaccards):.3f}, 标准差={np.std(jaccards):.3f}")
        print(f"  分享率误差: 均值={np.mean(share_rate_errors):.3%}, 标准差={np.std(share_rate_errors):.3%}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="场景B敏感度分析 - 方案D（多模型）")
    parser.add_argument("--models", nargs="+", 
                        default=["deepseek-v3.2", "gpt-5.1", "qwen-plus"],
                        help="模型名称列表")
    parser.add_argument("--prompt-version", type=str, default="b.v4", 
                        help="提示词版本（单个）")
    parser.add_argument("--num-trials", type=int, default=3, 
                        help="每个组合重复次数")
    parser.add_argument("--output-dir", type=str, 
                        default="sensitivity_results/scenario_b",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 固定的参数网格
    rho_values = [0.3, 0.6, 0.9]
    v_ranges = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
    
    # 运行实验
    run_sensitivity_experiment(
        rho_values=rho_values,
        v_ranges=v_ranges,
        prompt_version=args.prompt_version,
        model_names=args.models,
        num_trials=args.num_trials,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
