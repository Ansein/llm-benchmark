"""
测试m_0内生化修改

验证：
1. generate_ground_truth 能否正确计算内生m_0
2. m_0 估计是否包含在GT输出中
3. 中介利润计算是否使用内生m_0
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    generate_ground_truth
)

print("=" * 70)
print("测试m_0内生化修改")
print("=" * 70)

# 测试配置：Common Experience + Identified
params = ScenarioCParams(
    N=20,
    m=1.0,
    # ⭐ 注意：不设置m_0，让它自动计算
    data_structure='common_experience',
    anonymization='identified',
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

print("\n参数配置:")
print(f"  N = {params.N}")
print(f"  m = {params.m}")
print(f"  数据结构 = {params.data_structure}")
print(f"  匿名化 = {params.anonymization}")
print(f"  tau异质性 = {params.tau_dist}(μ={params.tau_mean}, σ={params.tau_std})")

print("\n" + "━" * 70)
print("生成Ground Truth（内部会自动计算内生m_0）")
print("━" * 70)

try:
    gt = generate_ground_truth(
        params,
        max_iter=20,
        tol=1e-3,
        num_mc_samples=30,  # 为快速测试减少
        num_outcome_samples=10  # 为快速测试减少
    )
    
    print("\n" + "=" * 70)
    print("✅ GT生成成功！")
    print("=" * 70)
    
    # 检查m0_estimation字段
    if "m0_estimation" in gt:
        print("\n✅ m0_estimation字段存在")
        m0_est = gt["m0_estimation"]
        
        print(f"\n内生m_0信息:")
        print(f"  m_0 = {m0_est['m_0']:.4f}")
        print(f"  delta_profit_mean = {m0_est['delta_profit_mean']:.4f}")
        print(f"  delta_profit_std = {m0_est['delta_profit_std']:.4f}")
        print(f"  expected_num_participants = {m0_est['expected_num_participants']:.2f}")
        print(f"  expected_intermediary_profit = {m0_est['expected_intermediary_profit']:.4f}")
        
        # 验证m_0 > 0
        if m0_est['m_0'] > 0:
            print(f"\n  ✓ m_0 > 0（数据有价值）")
        else:
            print(f"\n  ✗ m_0 = 0（异常）")
        
        # 验证中介利润计算
        m0 = m0_est['m_0']
        cost = params.m * m0_est['expected_num_participants']
        profit = m0 - cost
        
        print(f"\n中介利润验证:")
        print(f"  收入 (m_0) = {m0:.4f}")
        print(f"  成本 (m × E[N]) = {cost:.4f}")
        print(f"  净利润 (R) = {profit:.4f}")
        print(f"  存储的profit = {m0_est['expected_intermediary_profit']:.4f}")
        
        if abs(profit - m0_est['expected_intermediary_profit']) < 0.01:
            print(f"  ✓ 中介利润计算一致")
        else:
            print(f"  ✗ 中介利润计算不一致")
    else:
        print("\n❌ m0_estimation字段不存在")
    
    # 检查expected_outcome
    if "expected_outcome" in gt:
        print(f"\n✅ expected_outcome字段存在")
        exp = gt["expected_outcome"]
        
        print(f"\n期望市场结果:")
        print(f"  consumer_surplus = {exp['consumer_surplus']:.4f}")
        print(f"  producer_profit = {exp['producer_profit']:.4f}")
        print(f"  intermediary_profit = {exp['intermediary_profit']:.4f}")
        print(f"  social_welfare = {exp['social_welfare']:.4f}")
        
        # 验证福利分解
        sw_computed = exp['consumer_surplus'] + exp['producer_profit'] + exp['intermediary_profit']
        sw_stored = exp['social_welfare']
        
        if abs(sw_computed - sw_stored) < 0.01:
            print(f"  ✓ 社会福利 = CS + PS + IS")
        else:
            print(f"  ✗ 社会福利分解不一致")
            print(f"    计算: {sw_computed:.4f}")
            print(f"    存储: {sw_stored:.4f}")
    
    # 检查sample_outcome
    if "sample_outcome" in gt:
        print(f"\n✅ sample_outcome字段存在")
        samp = gt["sample_outcome"]
        
        print(f"\n示例市场结果:")
        print(f"  num_participants = {samp['num_participants']}")
        print(f"  intermediary_profit = {samp['intermediary_profit']:.4f}")
        print(f"  social_welfare = {samp['social_welfare']:.4f}")
    
    print("\n" + "=" * 70)
    print("📊 关键对比")
    print("=" * 70)
    
    if "m0_estimation" in gt:
        print(f"\n内生m_0方法:")
        print(f"  m_0 = {gt['m0_estimation']['m_0']:.4f} (Ex-Ante期望，MC-200次)")
        print(f"  期望中介利润 = {gt['m0_estimation']['expected_intermediary_profit']:.4f}")
        print(f"  期望社会福利 = {gt['expected_outcome']['social_welfare']:.4f}")
        
        print(f"\n旧方法（假设m_0=0）对比:")
        old_IS = 0 - params.m * gt['m0_estimation']['expected_num_participants']
        old_SW = (gt['expected_outcome']['consumer_surplus'] + 
                  gt['expected_outcome']['producer_profit'] + old_IS)
        print(f"  m_0 = 0.0 (外生假设)")
        print(f"  中介利润 = {old_IS:.4f}")
        print(f"  社会福利 = {old_SW:.4f}")
        
        print(f"\n改进:")
        print(f"  Δ中介利润 = {gt['m0_estimation']['expected_intermediary_profit'] - old_IS:.4f}")
        print(f"  Δ社会福利 = {gt['expected_outcome']['social_welfare'] - old_SW:.4f}")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
