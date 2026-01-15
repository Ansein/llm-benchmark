"""
主评估脚本
批量运行多个模型在不同场景下的评估

# python run_evaluation.py --scenarios B --models grok-3-mini gpt-4.1-mini deepseek-v3 deepseek-r1 --num-trials 1 --max-iterations 15
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.evaluators import create_llm_client, ScenarioAEvaluator, ScenarioBEvaluator


def load_existing_results(output_dir: str = "evaluation_results") -> List[Dict[str, Any]]:
    """
    从输出目录加载已有的评估结果
    
    Args:
        output_dir: 输出目录
    
    Returns:
        评估结果列表
    """
    print(f"\n{'='*80}")
    print(f"[加载结果] 从 {output_dir} 加载已有评估结果")
    print(f"{'='*80}")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"[错误] 目录不存在: {output_dir}")
        return []
    
    all_results = []
    
    # 查找所有评估结果文件
    result_files = list(output_path.glob("eval_scenario_*.json"))
    
    if not result_files:
        print(f"[错误] 在 {output_dir} 中没有找到评估结果文件")
        return []
    
    print(f"[信息] 找到 {len(result_files)} 个结果文件")
    
    for result_file in result_files:
        try:
            # 从文件名解析场景和模型
            # 文件名格式: eval_scenario_A_model-name.json 或 eval_scenario_B_model-name.json
            filename = result_file.stem  # 去掉 .json
            parts = filename.split("_")
            
            if len(parts) >= 3:
                scenario = parts[2]  # "A" 或 "B"
                model_name = "_".join(parts[3:])  # 剩余部分是模型名
                
                # 加载结果
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                all_results.append({
                    "scenario": scenario,
                    "model_name": model_name,
                    "result": result
                })
                
                print(f"  [OK] 加载: 场景{scenario} | {model_name}")
            else:
                print(f"  [跳过] {result_file.name} (文件名格式不符)")
        
        except Exception as e:
            print(f"  [失败] {result_file.name} - {e}")
    
    print(f"\n[完成] 成功加载 {len(all_results)} 个评估结果")
    return all_results


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
    print(f"[开始评估] 场景{scenario} | 模型: {model_name}")
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
    print(f"[批量评估] 开始")
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
    print(f"[完成] 批量评估完成！")
    print(f"{'#'*80}")


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    生成汇总报告
    
    Args:
        all_results: 所有评估结果
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"[汇总报告] 生成中")
    print(f"{'='*80}")
    
    # 准备表格数据
    summary_data = []
    
    for item in all_results:
        scenario = item["scenario"]
        model_name = item["model_name"]
        result = item["result"]
        
        # 检查必需字段
        if "metrics" not in result:
            print(f"  [警告] 跳过 {scenario}-{model_name}: 缺少 metrics 字段")
            continue
            
        metrics = result["metrics"]
        labels = result.get("labels", {})  # 使用get，如果没有则为空字典
        iterations = result.get("iterations", result.get("rounds", "N/A"))  # 兼容旧版本
        
        row = {
            "场景": scenario,
            "模型": model_name,
            "收敛": "[是]" if result.get("converged", False) else "[否]",
            "迭代次数": iterations,
        }
        
        # 场景A的指标
        if scenario == "A":
            row.update({
                "披露率_LLM": f"{metrics['llm']['disclosure_rate']:.2%}",
                "披露率_GT": f"{metrics['ground_truth']['disclosure_rate']:.2%}",
                "利润MAE": f"{metrics['deviations']['profit_mae']:.3f}",
                "CS_MAE": f"{metrics['deviations']['cs_mae']:.3f}",
                "福利MAE": f"{metrics['deviations']['welfare_mae']:.3f}",
                "披露率分桶匹配": "[是]" if labels.get("llm_disclosure_rate_bucket") == labels.get("gt_disclosure_rate_bucket") else "[否]" if labels else "N/A",
                "过度披露匹配": "[是]" if labels.get("llm_over_disclosure") == labels.get("gt_over_disclosure") else "[否]" if labels else "N/A"
            })
        
        # 场景B的指标
        elif scenario == "B":
            row.update({
                "分享率_LLM": f"{metrics['llm']['share_rate']:.2%}",
                "分享率_GT": f"{metrics['ground_truth']['share_rate']:.2%}",
                "利润MAE": f"{metrics['deviations']['profit_mae']:.4f}",
                "福利MAE": f"{metrics['deviations']['welfare_mae']:.4f}",
                "泄露MAE": f"{metrics['deviations']['total_leakage_mae']:.4f}",
                "泄露分桶匹配": "[是]" if labels.get("llm_leakage_bucket") == labels.get("gt_leakage_bucket") else "[否]" if labels else "N/A",
                "过度分享匹配": "[是]" if labels.get("llm_over_sharing") == labels.get("gt_over_sharing") else "[否]" if labels else "N/A"
            })
        
        summary_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    
    # 打印表格
    print("\n" + df.to_string(index=False))
    
    # 保存为CSV
    csv_path = Path(output_dir) / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 汇总报告已保存到: {csv_path}")
    
    # 保存完整JSON
    json_path = Path(output_dir) / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[保存] 完整结果已保存到: {json_path}")


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
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="仅生成汇总报告（使用已有的评估结果，不重新运行LLM）"
    )
    
    args = parser.parse_args()
    
    if args.summary_only:
        # 仅生成报告模式
        print(f"\n{'#'*80}")
        print(f"[汇总报告模式] 使用已有评估结果")
        print(f"{'#'*80}")
        
        all_results = load_existing_results(args.output_dir)
        
        if all_results:
            generate_summary_report(all_results, args.output_dir)
            print(f"\n{'#'*80}")
            print(f"[成功] 汇总报告生成完成！")
            print(f"{'#'*80}")
        else:
            print(f"\n[错误] 没有找到可用的评估结果，无法生成报告")
    elif args.single:
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
