"""
场景C评估器测试脚本

演示如何使用评估器，包含模拟的LLM代理。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from src.evaluators.evaluate_scenario_c import ScenarioCEvaluator


# ============================================================================
# 模拟的LLM代理（用于演示）
# ============================================================================

def mock_llm_consumer_rational(consumer_params, m, anonymization):
    """
    模拟LLM消费者：完全理性（用于验证评估器）
    
    决策规则：如果补偿m > 隐私成本tau_i，则参与
    """
    # 简化的理性决策（真实应该计算期望效用）
    # 这里假设ΔU ≈ m - tau_i（简化）
    delta_u_approx = m - consumer_params['tau_i']
    return delta_u_approx > 0


def mock_llm_consumer_optimistic(consumer_params, m, anonymization):
    """
    模拟LLM消费者：过度乐观（低估隐私成本）
    """
    # 低估隐私成本50%
    perceived_tau = consumer_params['tau_i'] * 0.5
    delta_u_approx = m - perceived_tau
    return delta_u_approx > 0


def mock_llm_consumer_pessimistic(consumer_params, m, anonymization):
    """
    模拟LLM消费者：过度悲观（高估隐私成本）
    """
    # 高估隐私成本50%
    perceived_tau = consumer_params['tau_i'] * 1.5
    delta_u_approx = m - perceived_tau
    return delta_u_approx > 0


def mock_llm_intermediary_rational(market_params):
    """
    模拟LLM中介：接近理性
    
    返回接近理论最优的策略
    """
    # 对于common_preferences，理论最优约为 m*=0.5, anonymized
    # 这里模拟LLM略微偏离
    m = 0.6  # 略高于最优
    anon = "anonymized"
    return m, anon


def mock_llm_intermediary_exploitative(market_params):
    """
    模拟LLM中介：倾向剥削
    
    选择高补偿+identified策略
    """
    m = 1.5  # 高补偿
    anon = "identified"  # 允许价格歧视
    return m, anon


def mock_llm_intermediary_conservative(market_params):
    """
    模拟LLM中介：保守策略
    
    选择低补偿+anonymized
    """
    m = 0.3  # 低补偿
    anon = "anonymized"
    return m, anon


# ============================================================================
# 测试函数
# ============================================================================

def test_config_B(evaluator, consumer_type="rational"):
    """测试配置B：理性中介 × LLM消费者"""
    print("\n" + "🔬 "*30)
    print(f"测试配置B：理性中介 × LLM消费者（{consumer_type}）")
    print("🔬 "*30)
    
    # 选择消费者代理
    consumer_agents = {
        "rational": mock_llm_consumer_rational,
        "optimistic": mock_llm_consumer_optimistic,
        "pessimistic": mock_llm_consumer_pessimistic,
    }
    
    llm_consumer = consumer_agents.get(consumer_type, mock_llm_consumer_rational)
    
    # 评估
    results_B = evaluator.evaluate_config_B(
        llm_consumer_agent=llm_consumer,
        verbose=True
    )
    
    return results_B


def test_config_C(evaluator, intermediary_type="rational"):
    """测试配置C：LLM中介 × 理性消费者"""
    print("\n" + "🔬 "*30)
    print(f"测试配置C：LLM中介（{intermediary_type}）× 理性消费者")
    print("🔬 "*30)
    
    # 选择中介代理
    intermediary_agents = {
        "rational": mock_llm_intermediary_rational,
        "exploitative": mock_llm_intermediary_exploitative,
        "conservative": mock_llm_intermediary_conservative,
    }
    
    llm_intermediary = intermediary_agents.get(intermediary_type, mock_llm_intermediary_rational)
    
    # 评估
    results_C = evaluator.evaluate_config_C(
        llm_intermediary_agent=llm_intermediary,
        verbose=True
    )
    
    return results_C


def test_config_D(evaluator, consumer_type="rational", intermediary_type="rational"):
    """测试配置D：LLM中介 × LLM消费者"""
    print("\n" + "🔬 "*30)
    print(f"测试配置D：LLM中介（{intermediary_type}）× LLM消费者（{consumer_type}）")
    print("🔬 "*30)
    
    # 选择代理
    consumer_agents = {
        "rational": mock_llm_consumer_rational,
        "optimistic": mock_llm_consumer_optimistic,
        "pessimistic": mock_llm_consumer_pessimistic,
    }
    
    intermediary_agents = {
        "rational": mock_llm_intermediary_rational,
        "exploitative": mock_llm_intermediary_exploitative,
        "conservative": mock_llm_intermediary_conservative,
    }
    
    llm_consumer = consumer_agents.get(consumer_type, mock_llm_consumer_rational)
    llm_intermediary = intermediary_agents.get(intermediary_type, mock_llm_intermediary_rational)
    
    # 评估
    results_D = evaluator.evaluate_config_D(
        llm_intermediary_agent=llm_intermediary,
        llm_consumer_agent=llm_consumer,
        verbose=True
    )
    
    return results_D


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主测试程序"""
    print("=" * 70)
    print("场景C评估器测试")
    print("=" * 70)
    
    # 1. 加载Ground Truth
    gt_path = "data/ground_truth/scenario_c_common_preferences_optimal.json"
    print(f"\n加载Ground Truth: {gt_path}")
    
    evaluator = ScenarioCEvaluator(gt_path)
    
    print(f"\n理论基准（配置A）:")
    print(f"  m* = {evaluator.gt_A['optimal_strategy']['m_star']:.4f}")
    print(f"  anonymization* = {evaluator.gt_A['optimal_strategy']['anonymization_star']}")
    print(f"  r* = {evaluator.gt_A['optimal_strategy']['r_star']:.4f}")
    print(f"  中介利润* = {evaluator.gt_A['optimal_strategy']['intermediary_profit_star']:.4f}")
    print(f"  社会福利 = {evaluator.gt_A['equilibrium']['social_welfare']:.4f}")
    
    # 2. 测试配置B（不同类型的消费者）
    results_B_rational = test_config_B(evaluator, "rational")
    results_B_optimistic = test_config_B(evaluator, "optimistic")
    results_B_pessimistic = test_config_B(evaluator, "pessimistic")
    
    # 3. 测试配置C（不同类型的中介）
    results_C_rational = test_config_C(evaluator, "rational")
    results_C_exploitative = test_config_C(evaluator, "exploitative")
    results_C_conservative = test_config_C(evaluator, "conservative")
    
    # 4. 测试配置D
    results_D = test_config_D(evaluator, "rational", "rational")
    
    # 5. 生成报告
    print("\n" + "=" * 70)
    print("生成综合报告")
    print("=" * 70)
    
    df = evaluator.generate_report(
        results_B=results_B_rational,
        results_C=results_C_rational,
        results_D=results_D,
        output_path="evaluation_results/scenario_c_test_report.csv"
    )
    
    print("\n报告预览:")
    print(df.to_string(index=False))
    
    # 6. 保存详细结果
    detailed_results = {
        "config_B": {
            "rational": results_B_rational,
            "optimistic": results_B_optimistic,
            "pessimistic": results_B_pessimistic,
        },
        "config_C": {
            "rational": results_C_rational,
            "exploitative": results_C_exploitative,
            "conservative": results_C_conservative,
        },
        "config_D": results_D,
    }
    
    output_json = "evaluation_results/scenario_c/scenario_c_test_detailed.json"
    from pathlib import Path
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_json}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
