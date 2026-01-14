"""
主评估脚本
批量运行多个模型在不同场景下的评估

# python run_evaluation.py --scenarios B --models grok-3-mini gpt-4.1-mini deepseek-v3 --num-trials 5 --max-iterations 15
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.evaluators import create_llm_client, ScenarioAEvaluator, ScenarioBEvaluator


def run_single_evaluation(
    scenario: str,
    model_name: str,
    num_trials: int = 3,
    max_iterations: int = 10,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    运行单个场景的评估
    
    Args:
        scenario: 场景名称 ("A" 或 "B")
        model_name: 模型配置名称
        num_trials: 每个决策的重复次数
        max_iterations: 最大迭代次数
        output_dir: 输出目录
    
    Returns:
        评估结果字典
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始评估: 场景{scenario} | 模型: {model_name}")
    print(f"{'='*80}")
    
    try:
        # 创建LLM客户端
        llm_client = create_llm_client(model_name)
        
        # 根据场景选择评估器
        if scenario == "A":
            evaluator = ScenarioAEvaluator(llm_client)
            # 运行评估
            results = evaluator.simulate_llm_equilibrium(
                num_trials=num_trials,
                max_iterations=max_iterations
            )
        elif scenario == "B":
            evaluator = ScenarioBEvaluator(llm_client)
            # 运行评估（并行博弈模式，参数名为max_rounds）
            results = evaluator.simulate_llm_equilibrium(
                num_trials=num_trials,
                max_rounds=max_iterations
            )
        else:
            raise ValueError(f"不支持的场景: {scenario}")
        
        # 打印摘要
        evaluator.print_evaluation_summary(results)
        
        # 保存结果
        output_path = Path(output_dir) / f"eval_scenario_{scenario}_{model_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(results, str(output_path))
        
        return results
    
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_evaluation(
    scenarios: List[str],
    model_names: List[str],
    num_trials: int = 3,
    max_iterations: int = 10,
    output_dir: str = "evaluation_results"
):
    """
    批量运行评估
    
    Args:
        scenarios: 场景列表 (["A", "B"])
        model_names: 模型名称列表
        num_trials: 每个决策的重复次数
        max_iterations: 最大迭代次数
        output_dir: 输出目录
    """
    print(f"\n{'#'*80}")
    print(f"🎯 批量评估开始")
    print(f"{'#'*80}")
    print(f"场景: {scenarios}")
    print(f"模型: {model_names}")
    print(f"每个决策重复次数: {num_trials}")
    print(f"最大迭代次数: {max_iterations}")
    print(f"输出目录: {output_dir}")
    
    # 收集所有结果
    all_results = []
    
    for scenario in scenarios:
        for model_name in model_names:
            result = run_single_evaluation(
                scenario=scenario,
                model_name=model_name,
                num_trials=num_trials,
                max_iterations=max_iterations,
                output_dir=output_dir
            )
            
            if result:
                all_results.append({
                    "scenario": scenario,
                    "model_name": model_name,
                    "result": result
                })
    
    # 生成汇总报告
    generate_summary_report(all_results, output_dir)
    
    print(f"\n{'#'*80}")
    print(f"✅ 批量评估完成！")
    print(f"{'#'*80}")


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    生成汇总报告
    
    Args:
        all_results: 所有评估结果
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"📊 生成汇总报告")
    print(f"{'='*80}")
    
    # 准备表格数据
    summary_data = []
    
    for item in all_results:
        scenario = item["scenario"]
        model_name = item["model_name"]
        result = item["result"]
        
        metrics = result["metrics"]
        labels = result["labels"]
        
        row = {
            "场景": scenario,
            "模型": model_name,
            "收敛": "✅" if result["converged"] else "❌",
            "迭代次数": result["iterations"],
        }
        
        # 场景A的指标
        if scenario == "A":
            row.update({
                "披露率_LLM": f"{metrics['llm']['disclosure_rate']:.2%}",
                "披露率_GT": f"{metrics['ground_truth']['disclosure_rate']:.2%}",
                "利润MAE": f"{metrics['deviations']['profit_mae']:.3f}",
                "CS_MAE": f"{metrics['deviations']['cs_mae']:.3f}",
                "福利MAE": f"{metrics['deviations']['welfare_mae']:.3f}",
                "披露率分桶匹配": "✅" if labels["llm_disclosure_rate_bucket"] == labels["gt_disclosure_rate_bucket"] else "❌",
                "过度披露匹配": "✅" if labels["llm_over_disclosure"] == labels["gt_over_disclosure"] else "❌"
            })
        
        # 场景B的指标
        elif scenario == "B":
            row.update({
                "分享率_LLM": f"{metrics['llm']['share_rate']:.2%}",
                "分享率_GT": f"{metrics['ground_truth']['share_rate']:.2%}",
                "利润MAE": f"{metrics['deviations']['profit_mae']:.4f}",
                "福利MAE": f"{metrics['deviations']['welfare_mae']:.4f}",
                "泄露MAE": f"{metrics['deviations']['total_leakage_mae']:.4f}",
                "泄露分桶匹配": "✅" if labels["llm_leakage_bucket"] == labels["gt_leakage_bucket"] else "❌",
                "过度分享匹配": "✅" if labels["llm_over_sharing"] == labels["gt_over_sharing"] else "❌"
            })
        
        summary_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    
    # 打印表格
    print("\n" + df.to_string(index=False))
    
    # 保存为CSV
    csv_path = Path(output_dir) / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 汇总报告已保存到: {csv_path}")
    
    # 保存完整JSON
    json_path = Path(output_dir) / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"💾 完整结果已保存到: {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行LLM benchmark评估")
    
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["A", "B"],
        choices=["A", "B"],
        help="要评估的场景列表"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt-4.1-mini"],
        help="要评估的模型列表（配置名称）"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="每个决策的重复次数（用于评估稳定性）"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="寻找均衡的最大迭代次数"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="输出目录"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="单次评估模式（用于测试）"
    )
    
    args = parser.parse_args()
    
    if args.single:
        # 单次评估模式
        run_single_evaluation(
            scenario=args.scenarios[0],
            model_name=args.models[0],
            num_trials=args.num_trials,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir
        )
    else:
        # 批量评估模式
        run_batch_evaluation(
            scenarios=args.scenarios,
            model_names=args.models,
            num_trials=args.num_trials,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
