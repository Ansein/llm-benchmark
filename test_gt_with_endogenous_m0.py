"""
测试使用内生m_0的GT生成
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.scenarios.scenario_c_social_data import ScenarioCParams, generate_ground_truth
import json

print("=" * 70)
print("测试GT生成（完全内生m_0）")
print("=" * 70)

# MVP配置
print("\n生成MVP配置...")
params = ScenarioCParams(
    N=20,
    m=1.0,
    # ⭐ 不传入m_0！它会自动计算
    data_structure='common_preferences',
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

print(f"参数: N={params.N}, m={params.m}, {params.data_structure}, {params.anonymization}")

try:
    gt = generate_ground_truth(
        params,
        max_iter=20,
        tol=1e-3,
        num_mc_samples=30,
        num_outcome_samples=10
    )
    
    print("\n✅ GT生成成功！")
    
    # 验证m0_estimation存在
    if "m0_estimation" in gt:
        m0_info = gt["m0_estimation"]
        print(f"\n内生m_0信息:")
        print(f"  m_0 = {m0_info['m_0']:.4f}")
        print(f"  delta_profit_mean = {m0_info['delta_profit_mean']:.4f}")
        print(f"  expected_intermediary_profit = {m0_info['expected_intermediary_profit']:.4f}")
        print(f"  method = {m0_info['method']}")
        
        # 保存到文件
        output_file = "data/ground_truth/test_mvp_endogenous_m0.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 保存到: {output_file}")
    else:
        print("\n❌ m0_estimation字段不存在")
    
    # 验证params中没有m_0
    if "m_0" in gt["params"]:
        print(f"\n⚠️ 警告: params中仍包含m_0字段")
    else:
        print(f"\n✅ params中不包含m_0（符合预期）")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
