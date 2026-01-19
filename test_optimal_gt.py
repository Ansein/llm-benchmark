"""
测试最优Ground Truth生成（论文理论解）

验证新的 generate_ground_truth 函数：
- 中介优化作为GT生成的第一步
- 输出包含最优策略和完整均衡
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
from src.scenarios.scenario_c_social_data import generate_ground_truth

print("=" * 70)
print("测试最优Ground Truth生成（论文理论解）")
print("=" * 70)

# 基础参数（不包含m和anonymization）
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

print("\n基础参数:")
for key, value in params_base.items():
    print(f"  {key} = {value}")

print("\n" + "━" * 70)
print("生成最优Ground Truth（内部会求解中介最优策略）")
print("━" * 70)

try:
    gt = generate_ground_truth(
        params_base=params_base,
        m_grid=np.linspace(0, 3, 13),  # 快速测试：13个点
        max_iter=50,  # 增加迭代次数避免不收敛
        num_mc_samples=30,
        num_outcome_samples=10
    )
    
    print("\n" + "=" * 70)
    print("✅ GT生成成功！")
    print("=" * 70)
    
    # 验证最优策略
    if "optimal_strategy" in gt:
        print("\n✅ optimal_strategy字段存在")
        opt = gt["optimal_strategy"]
        
        print(f"\n最优策略:")
        print(f"  m* = {opt['m_star']:.4f}")
        print(f"  anonymization* = {opt['anonymization_star']}")
        print(f"  r* = {opt['r_star']:.4f}")
        print(f"  m_0* = {opt['m_0_star']:.4f}")
        print(f"  中介利润* = {opt['intermediary_profit_star']:.4f}")
        
        # 验证m*在合理范围内
        if 0 <= opt['m_star'] <= 3:
            print(f"  ✓ m* 在合理范围 [0, 3]")
        else:
            print(f"  ✗ m* 超出范围")
        
        # 验证r*在[0,1]
        if 0 <= opt['r_star'] <= 1:
            print(f"  ✓ r* 在合理范围 [0, 1]")
        else:
            print(f"  ✗ r* 超出范围")
    else:
        print("\n❌ optimal_strategy字段不存在")
    
    # 验证均衡结果
    if "equilibrium" in gt:
        print(f"\n✅ equilibrium字段存在")
        eq = gt["equilibrium"]
        
        print(f"\n市场均衡:")
        print(f"  consumer_surplus = {eq['consumer_surplus']:.4f}")
        print(f"  producer_profit = {eq['producer_profit']:.4f}")
        print(f"  intermediary_profit = {eq['intermediary_profit']:.4f}")
        print(f"  social_welfare = {eq['social_welfare']:.4f}")
        
        # 验证福利分解
        sw_computed = eq['consumer_surplus'] + eq['producer_profit'] + eq['intermediary_profit']
        sw_stored = eq['social_welfare']
        
        if abs(sw_computed - sw_stored) < 0.01:
            print(f"  ✓ SW = CS + PS + IS")
        else:
            print(f"  ✗ SW分解不一致")
    
    # 验证数据交易信息
    if "data_transaction" in gt:
        print(f"\n✅ data_transaction字段存在")
        dt = gt["data_transaction"]
        
        print(f"\n数据交易:")
        print(f"  m_0 = {dt['m_0']:.4f}")
        print(f"  producer_profit_gain = {dt['producer_profit_gain']:.4f}")
        print(f"  expected_num_participants = {dt['expected_num_participants']:.2f}")
        
        if dt['m_0'] > 0:
            print(f"  ✓ m_0 > 0（数据有价值）")
    
    # 验证候选策略
    if "all_candidates" in gt:
        print(f"\n✅ all_candidates字段存在")
        print(f"  候选策略数量: {len(gt['all_candidates'])}")
        
        # 找到中介利润最高的策略
        best_candidate = max(gt['all_candidates'], key=lambda x: x['intermediary_profit'])
        print(f"\n  最优候选（验证）:")
        print(f"    m = {best_candidate['m']:.4f}")
        print(f"    anonymization = {best_candidate['anonymization']}")
        print(f"    intermediary_profit = {best_candidate['intermediary_profit']:.4f}")
        
        # 验证与optimal_strategy一致
        if (abs(best_candidate['m'] - gt['optimal_strategy']['m_star']) < 0.001 and
            best_candidate['anonymization'] == gt['optimal_strategy']['anonymization_star']):
            print(f"  ✓ 最优候选与optimal_strategy一致")
        else:
            print(f"  ✗ 不一致")
    
    # 验证示例数据
    if "sample_data" in gt and "sample_participation" in gt:
        print(f"\n✅ sample_data和sample_participation字段存在")
        print(f"  示例参与率: {sum(gt['sample_participation'])/len(gt['sample_participation']):.2%}")
        print(f"  用途: LLM评估")
    
    # 验证metadata
    if "metadata" in gt:
        print(f"\n✅ metadata字段存在")
        meta = gt["metadata"]
        print(f"  生成方法: {meta['generation_method']}")
        print(f"  是否最优策略: {meta['is_optimal_strategy']}")
    
    # 保存到文件
    output_file = "data/ground_truth/test_optimal_gt.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 保存到: {output_file}")
    
    print("\n" + "=" * 70)
    print("📊 关键结果总结")
    print("=" * 70)
    print(f"\n中介最优决策:")
    print(f"  选择补偿: m* = {gt['optimal_strategy']['m_star']:.4f}")
    print(f"  选择策略: {gt['optimal_strategy']['anonymization_star']}")
    print(f"  获得利润: {gt['optimal_strategy']['intermediary_profit_star']:.4f}")
    
    print(f"\n消费者反应:")
    print(f"  参与率: r* = {gt['optimal_strategy']['r_star']:.2%}")
    
    print(f"\n市场结果:")
    print(f"  社会福利: {gt['equilibrium']['social_welfare']:.4f}")
    print(f"  消费者剩余: {gt['equilibrium']['consumer_surplus']:.4f}")
    print(f"  生产者利润: {gt['equilibrium']['producer_profit']:.4f}")
    
    print(f"\n数据交易:")
    print(f"  生产者支付m_0: {gt['data_transaction']['m_0']:.4f}")
    print(f"  中介支付成本: {gt['data_transaction']['intermediary_cost']:.4f}")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
