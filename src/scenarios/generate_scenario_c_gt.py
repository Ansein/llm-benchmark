"""
生成场景C的Ground Truth数据（完整模式）

自动生成所有配置：
1. 最优GT（论文理论解）：Common Experience + Common Preferences
2. 条件均衡（研究用）：2x2 对比配置（固定m=1.0）

位置: src/scenarios/generate_scenario_c_gt.py
运行: python -m src.scenarios.generate_scenario_c_gt
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from pathlib import Path
from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    generate_ground_truth,
    generate_conditional_equilibrium,
    evaluate_intermediary_strategy,
    generate_consumer_data
)
from src.scenarios.scenario_c_social_data_optimization import (
    optimize_intermediary_policy_personalized
)


def generate_optimal_gt_common_experience():
    """生成Common Experience的最优Ground Truth（论文理论解）"""
    print("\n" + "⭐ "*30)
    print("生成Common Experience最优Ground Truth（论文理论解）")
    print("⭐ "*30)
    
    params_base = {
        'N': 20,
        'data_structure': 'common_experience',
        # ⚠️ 不包含 m 和 anonymization，由中介优化求解
        'mu_theta': 5.0,
        'sigma_theta': 1.0,
        'sigma': 1.0,
        'tau_dist': 'normal',
        'tau_mean': 1.0,
        'tau_std': 0.3,
        'c': 0.0,
        'participation_timing': 'ex_ante',
        'seed': 42
    }
    
    gt = generate_ground_truth(
        params_base=params_base,
        m_grid=np.linspace(0, 3, 31),
        max_iter=20,
        num_mc_samples=50,
        num_outcome_samples=20
    )
    
    # 保存
    output_path = Path(__file__).parent.parent.parent / "data" / "ground_truth" / "scenario_c_common_experience_optimal.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 已保存到: {output_path}")
    print(f"\n最优策略:")
    print(f"  m* = {gt['optimal_strategy']['m_star']:.4f}")
    print(f"  anonymization* = {gt['optimal_strategy']['anonymization_star']}")
    print(f"  r* = {gt['optimal_strategy']['r_star']:.4f}")
    print(f"  中介利润* = {gt['optimal_strategy']['intermediary_profit_star']:.4f}")
    print(f"  社会福利* = {gt['equilibrium']['social_welfare']:.4f}")
    
    return gt


def generate_optimal_gt_common_preferences():
    """生成Common Preferences的最优Ground Truth（论文理论解）- 使用连续优化"""
    print("\n" + "⭐ "*30)
    print("生成Common Preferences最优Ground Truth（论文理论解）")
    print("使用连续优化（个性化补偿m_i）")
    print("⭐ "*30)
    
    params_base = {
        'N': 20,
        'data_structure': 'common_preferences',
        # ⚠️ 不包含 m 和 anonymization，由中介优化求解
        'mu_theta': 5.0,
        'sigma_theta': 1.0,
        'sigma': 1.0,
        'tau_dist': 'normal',
        'tau_mean': 1.0,
        'tau_std': 0.3,
        'c': 0.0,
        'participation_timing': 'ex_ante',
        'seed': 42
    }
    
    # ✅ 使用混合优化（网格搜索初始化 + 连续优化）
    print("\n使用混合优化方法（网格搜索 + scipy L-BFGS-B）...")
    result = optimize_intermediary_policy_personalized(
        params_base=params_base,
        policies=['identified', 'anonymized'],  # 优化两个策略，选利润最高的
        optimization_method='hybrid',  # 混合方法：网格搜索找起点 + scipy优化
        m_bounds=(0.0, 3.0),
        num_mc_samples=30,  # MC样本数
        max_iter=20,  # 固定点迭代次数
        grid_size=21,  # 网格搜索点数
        n_jobs=-1,  # 使用所有CPU核心并行加速
        verbose=True,
        seed=42
    )
    
    # 转换为标准GT格式
    # 处理no_participation情况
    if result['anonymization_star'] == 'no_participation':
        gt = {
            "optimal_strategy": {
                "m_star": 0.0,
                "m_star_mean": 0.0,
                "m_star_std": 0.0,
                "anonymization_star": "no_participation",
                "intermediary_profit_star": 0.0,
                "r_star": 0.0,
                "delta_u_star": 0.0,
                "m_0_star": 0.0,
                "optimization_method": "hybrid"
            },
            "equilibrium": {
                "consumer_surplus": 0.0,
                "producer_profit": 0.0,
                "intermediary_profit": 0.0,
                "social_welfare": 0.0,
                "gini_coefficient": 0.0,
                "price_discrimination_index": 0.0
            },
            "data_transaction": {
                "m_0": 0.0,
                "producer_profit_with_data": 0.0,
                "producer_profit_no_data": 0.0,
                "producer_profit_gain": 0.0,
                "expected_num_participants": 0.0,
                "intermediary_cost": 0.0
            },
            "sample_data": {},
            "sample_participation": [],
            "optimization_info": {
                "method": "hybrid",
                "message": "All policies unprofitable, intermediary does not participate"
            }
        }
    else:
        # ✅ 调用 evaluate_intermediary_strategy 获取完整的均衡信息
        print("\n评估最优策略的完整均衡信息...")
        eval_result = evaluate_intermediary_strategy(
            m=result['m_star_vector'],
            anonymization=result['anonymization_star'],
            params_base=params_base,
            num_mc_samples=30,
            max_iter=20,
            seed=42
        )
        
        # 生成样本数据（用于LLM评估）
        params_optimal = ScenarioCParams(
            m=result['m_star_vector'],
            anonymization=result['anonymization_star'],
            **params_base
        )
        rng_sample = np.random.default_rng(42)
        sample_data = generate_consumer_data(params_optimal, rng=rng_sample)
        
        # 生成参与决策（基于delta_u阈值）
        delta_u = eval_result.delta_u
        if params_base['tau_dist'] == 'normal':
            tau_samples = rng_sample.normal(params_base['tau_mean'], params_base['tau_std'], params_base['N'])
            sample_participation = (tau_samples <= delta_u).tolist()
        else:
            sample_participation = [delta_u > 0] * params_base['N']
        
        gt = {
            "params_base": params_base,  # ✅ 添加params_base（评估器需要）
            "optimal_strategy": {
                "m_star": result['m_star_vector'].tolist() if isinstance(result['m_star_vector'], np.ndarray) else result['m_star_vector'],
                "m_star_mean": float(np.mean(result['m_star_vector'])),  # 添加均值用于对比
                "m_star_std": float(np.std(result['m_star_vector'])),    # 添加标准差
                "anonymization_star": result['anonymization_star'],
                "intermediary_profit_star": float(result['profit_star']),
                "r_star": float(eval_result.r_star),
                "delta_u_star": float(eval_result.delta_u),
                "m_0_star": float(eval_result.m_0),
                "optimization_method": "hybrid"  # 标记优化方法
            },
            "equilibrium": {
                "consumer_surplus": float(eval_result.consumer_surplus),
                "producer_profit": float(eval_result.producer_profit_with_data),
                "intermediary_profit": float(eval_result.intermediary_profit),
                "social_welfare": float(eval_result.social_welfare),
                "gini_coefficient": float(eval_result.gini_coefficient),
                "price_discrimination_index": float(eval_result.price_discrimination_index)
            },
            "data_transaction": {
                "m_0": float(eval_result.m_0),
                "producer_profit_with_data": float(eval_result.producer_profit_with_data),
                "producer_profit_no_data": float(eval_result.producer_profit_no_data),
                "producer_profit_gain": float(eval_result.producer_profit_gain),
                "expected_num_participants": float(eval_result.num_participants),
                "intermediary_cost": float(eval_result.intermediary_cost)
            },
            "sample_data": {
                "w": sample_data.w.tolist(),
                "s": sample_data.s.tolist(),
                "theta": float(sample_data.theta) if sample_data.theta is not None else None,
                "epsilon": float(sample_data.epsilon) if sample_data.epsilon is not None else None,
            },
            "sample_participation": sample_participation,
            "optimization_info": {
                "method": "hybrid",
                "m_bounds": [0.0, 3.0],
                "convergence": result.get('results_by_policy', {}).get(result['anonymization_star'], {}).get('info', {})
            }
        }
    
    # 保存
    output_path = Path(__file__).parent.parent.parent / "data" / "ground_truth" / "scenario_c_common_preferences_optimal.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 已保存到: {output_path}")
    print(f"\n最优策略:")
    print(f"  m* = 向量 (均值={gt['optimal_strategy']['m_star_mean']:.4f}, 标准差={gt['optimal_strategy']['m_star_std']:.4f})")
    print(f"  anonymization* = {gt['optimal_strategy']['anonymization_star']}")
    print(f"  r* = {gt['optimal_strategy']['r_star']:.4f}")
    print(f"  中介利润* = {gt['optimal_strategy']['intermediary_profit_star']:.4f}")
    print(f"  社会福利* = {gt['equilibrium']['social_welfare']:.4f}")
    
    return gt


def generate_conditional_equilibria_for_comparison():
    """生成2x2对比配置的条件均衡（用于研究策略空间）"""
    print("\n" + "🔬 "*30)
    print("生成2x2条件均衡（研究用）")
    print("🔬 "*30)
    
    results = {}
    output_dir = Path(__file__).parent.parent.parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 固定m=1.0，对比2种数据结构 × 2种匿名化策略
    for data_structure in ["common_preferences", "common_experience"]:
        for anonymization in ["identified", "anonymized"]:
            config_name = f"{data_structure}_{anonymization}"
            print(f"\n生成配置: {config_name} (m=1.0)")
            
            params = ScenarioCParams(
                N=20,
                m=1.0,  # 固定策略
                anonymization=anonymization,  # 固定策略
                data_structure=data_structure,
                mu_theta=5.0,
                sigma_theta=1.0,
                sigma=1.0,
                tau_dist='normal',
                tau_mean=1.0,
                tau_std=0.3,
                c=0.0,
                participation_timing='ex_ante',
                seed=42
            )
            
            gt = generate_conditional_equilibrium(
                params,
                max_iter=20,
                num_mc_samples=50,
                num_outcome_samples=20
            )
            
            # 保存
            output_path = output_dir / f"scenario_c_{config_name}_m1.0.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gt, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ 保存到: {output_path}")
            results[config_name] = gt
    
    return results


def main():
    """主函数：生成所有Ground Truth配置"""
    print("=" * 70)
    print("场景C Ground Truth 生成器 - 简化模式")
    print("=" * 70)
    print("\n将生成：")
    print("  • 最优GT - Common Preferences（论文理论解，使用混合优化+个性化补偿）")
    print()
    
    try:
        # 只生成common_preferences最优GT
        print("\n" + "=" * 70)
        print("生成最优Ground Truth（论文理论解）")
        print("=" * 70)
        
        generate_optimal_gt_common_preferences()
        
        print("\n" + "=" * 70)
        print("✅ Ground Truth生成完成！")
        print("=" * 70)
        print("\n生成文件：")
        print("  • scenario_c_common_preferences_optimal.json")
        print()
    
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Windows上multiprocessing需要这个保护
    main()
