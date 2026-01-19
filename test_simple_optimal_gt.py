"""
简单测试最优GT生成（使用更宽松的参数避免不收敛）
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from src.scenarios.scenario_c_social_data import generate_ground_truth

print("=" * 70)
print("简单测试最优GT生成")
print("=" * 70)

params_base = {
    'N': 10,  # 减少N
    'data_structure': 'common_experience',
    'mu_theta': 5.0,
    'sigma_theta': 1.0,
    'sigma': 1.0,
    'tau_dist': 'normal',
    'tau_mean': 1.0,
    'tau_std': 0.5,  # 增加异质性
    'c': 0.0,
    'participation_timing': 'ex_ante',
    'seed': 42
}

print("\n基础参数:")
for key, value in params_base.items():
    print(f"  {key} = {value}")

print("\n开始生成...")

try:
    gt = generate_ground_truth(
        params_base=params_base,
        m_grid=np.linspace(0.5, 2.5, 5),  # 只测试5个点，从0.5开始
        max_iter=100,  # 增加迭代次数
        num_mc_samples=20,  # 减少MC样本
        num_outcome_samples=5  # 减少outcome样本
    )
    
    print("\n" + "=" * 70)
    print("✅ 成功！")
    print("=" * 70)
    
    print(f"\n最优策略:")
    print(f"  m* = {gt['optimal_strategy']['m_star']:.4f}")
    print(f"  anonymization* = {gt['optimal_strategy']['anonymization_star']}")
    print(f"  r* = {gt['optimal_strategy']['r_star']:.4f}")
    print(f"  中介利润* = {gt['optimal_strategy']['intermediary_profit_star']:.4f}")
    
    print(f"\n市场均衡:")
    print(f"  社会福利 = {gt['equilibrium']['social_welfare']:.4f}")
    print(f"  消费者剩余 = {gt['equilibrium']['consumer_surplus']:.4f}")
    print(f"  生产者利润 = {gt['equilibrium']['producer_profit']:.4f}")
    
    print(f"\n数据交易:")
    print(f"  m_0 = {gt['data_transaction']['m_0']:.4f}")
    
    print(f"\n候选策略数量: {len(gt['all_candidates'])}")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
