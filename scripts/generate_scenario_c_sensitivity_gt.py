"""
场景C敏感度分析 - Ground Truth生成器
生成3×3参数网格的理论解：τ_mean × σ
只生成common_preferences数据结构
"""
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    evaluate_intermediary_strategy,
    generate_consumer_data
)
from src.scenarios.scenario_c_social_data_optimization import (
    optimize_intermediary_policy_personalized
)


def generate_sensitivity_gt():
    """生成敏感度分析的所有GT配置"""
    
    # 参数网格
    tau_mean_values = [0.5, 1.0, 1.5]
    sigma_values = [0.5, 1.0, 2.0]
    
    # 固定参数
    params_base_fixed = {
        'N': 20,
        'data_structure': 'common_preferences',
        'tau_std': 0.3,
        'mu_theta': 5.0,
        'sigma_theta': 1.0,
        'c': 0.0,
        'participation_timing': 'ex_ante',
        'tau_dist': 'normal',
    }
    
    print("="*80)
    print("场景C敏感度分析 - Ground Truth生成")
    print("="*80)
    print("参数网格：")
    print(f"  τ_mean值: {tau_mean_values}")
    print(f"  σ值: {sigma_values}")
    print(f"  固定参数: N={params_base_fixed['N']}, data_structure={params_base_fixed['data_structure']}")
    print(f"  输出目录: data/ground_truth/sensitivity_c")
    print("="*80)
    print()
    
    # 创建输出目录
    output_dir = Path("data/ground_truth/sensitivity_c")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = len(tau_mean_values) * len(sigma_values)
    current = 0
    
    # 遍历所有参数组合
    for tau_mean in tau_mean_values:
        for sigma in sigma_values:
            current += 1
            print(f"[{current}/{total_configs}] 生成 GT: τ_mean={tau_mean}, σ={sigma}")
            
            # 构建完整参数
            params_base = {
                **params_base_fixed,
                'tau_mean': tau_mean,
                'sigma': sigma,
                'seed': 42
            }
            
            print(f"  参数: N={params_base['N']}, τ_mean={tau_mean}, σ={sigma}")
            
            try:
                # 使用混合优化求解最优策略
                result = optimize_intermediary_policy_personalized(
                    params_base=params_base,
                    policies=['identified', 'anonymized'],
                    optimization_method='hybrid',
                    m_bounds=(0.0, 3.0),
                    num_mc_samples=30,
                    max_iter=20,
                    grid_size=21,
                    n_jobs=-1,
                    verbose=False,
                    seed=42
                )
                
                # 处理no_participation情况
                if result['anonymization_star'] == 'no_participation':
                    print("  ⚠️  所有策略不盈利，中介不参与")
                    gt = {
                        "params_base": params_base,
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
                    # 评估最优策略的完整均衡信息
                    eval_result = evaluate_intermediary_strategy(
                        m=result['m_star_vector'],
                        anonymization=result['anonymization_star'],
                        params_base=params_base,
                        num_mc_samples=30,
                        max_iter=20,
                        seed=42
                    )
                    
                    # 生成样本数据
                    params_optimal = ScenarioCParams(
                        m=result['m_star_vector'],
                        anonymization=result['anonymization_star'],
                        **params_base
                    )
                    rng_sample = np.random.default_rng(42)
                    sample_data = generate_consumer_data(params_optimal, rng=rng_sample)
                    
                    # 生成参与决策
                    delta_u = eval_result.delta_u
                    tau_samples = rng_sample.normal(params_base['tau_mean'], params_base['tau_std'], params_base['N'])
                    sample_participation = (tau_samples <= delta_u).tolist()
                    
                    gt = {
                        "params_base": params_base,
                        "optimal_strategy": {
                            "m_star": result['m_star_vector'].tolist() if isinstance(result['m_star_vector'], np.ndarray) else result['m_star_vector'],
                            "m_star_mean": float(np.mean(result['m_star_vector'])),
                            "m_star_std": float(np.std(result['m_star_vector'])),
                            "anonymization_star": result['anonymization_star'],
                            "intermediary_profit_star": float(result['profit_star']),
                            "r_star": float(eval_result.r_star),
                            "delta_u_star": float(eval_result.delta_u),
                            "m_0_star": float(eval_result.m_0),
                            "optimization_method": "hybrid"
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
                    
                    print(f"  求解成功!")
                    print(f"    最优策略: {result['anonymization_star']}")
                    print(f"    m* 均值: {gt['optimal_strategy']['m_star_mean']:.4f}")
                    print(f"    中介利润*: {gt['optimal_strategy']['intermediary_profit_star']:.4f}")
                    print(f"    社会福利*: {gt['equilibrium']['social_welfare']:.4f}")
                    print(f"    参与人数: {gt['data_transaction']['expected_num_participants']:.1f}/{params_base['N']}")
                
                # 保存GT文件
                filename = f"scenario_c_tau{tau_mean:.1f}_sigma{sigma:.1f}.json"
                output_path = output_dir / filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(gt, f, indent=2, ensure_ascii=False)
                
                print(f"  ✅ 已保存: {output_path}")
                
            except Exception as e:
                print(f"  ❌ 求解失败: {e}")
                import traceback
                traceback.print_exc()
            
            print()
    
    print("="*80)
    print(f"✅ 完成！共生成 {total_configs} 个GT配置")
    print(f"输出目录: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    generate_sensitivity_gt()
