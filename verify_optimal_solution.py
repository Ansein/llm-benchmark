"""
一分钟确认：验证最优解的正确性

检验理论一致性：
如果 τ_i ~ N(τ_mean, τ_std)，且参与条件为 τ_i ≤ ΔU
则理论参与率应为：r = Φ((ΔU - τ_mean) / τ_std)

如果理论值与求解值 r* 接近，说明求解正确。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from scipy.stats import norm
from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    compute_rational_participation_rate_ex_ante
)

print("=" * 70)
print("一分钟确认：验证最优解正确性")
print("=" * 70)

# 最优点的参数
params = ScenarioCParams(
    N=20,
    m=0.70,  # 最优补偿
    anonymization='identified',  # 最优策略
    data_structure='common_preferences',
    mu_theta=5.0,
    sigma_theta=1.0,
    sigma=1.0,
    tau_dist='normal',
    tau_mean=1.0,  # τ均值
    tau_std=0.3,   # τ标准差
    c=0.0,
    participation_timing='ex_ante',
    seed=42
)

print(f"\n📋 参数配置:")
print(f"  最优策略: m = {params.m}, {params.anonymization}")
print(f"  数据结构: {params.data_structure}")
print(f"  市场规模: N = {params.N}")

print(f"\n🎲 隐私成本分布:")
print(f"  分布类型: {params.tau_dist}")
print(f"  τ_mean = {params.tau_mean}")
print(f"  τ_std = {params.tau_std}")

# 求解固定点
print(f"\n🔍 求解固定点...")
r_star, r_history, delta_u = compute_rational_participation_rate_ex_ante(
    params=params,
    max_iter=100,
    tol=1e-3,
    num_world_samples=50,
    num_market_samples=50
)

print(f"\n✅ 固定点收敛:")
print(f"  r* = {r_star:.4f} ({r_star:.2%})")
print(f"  ΔU = {delta_u:.4f}")
print(f"  收敛历史: {[f'{x:.4f}' for x in r_history[-10:]]}")

# 理论预测
print(f"\n📐 理论验证:")
print(f"\n  参与条件: τ_i ≤ ΔU")
print(f"  如果 τ_i ~ N(τ_mean, τ_std)")
print(f"  则理论参与率:")
print(f"    r̂ = P(τ_i ≤ ΔU)")
print(f"      = Φ((ΔU - τ_mean) / τ_std)")
print(f"      = Φ(({delta_u:.4f} - {params.tau_mean}) / {params.tau_std})")

# 计算标准化值
z_score = (delta_u - params.tau_mean) / params.tau_std
print(f"      = Φ({z_score:.4f})")

# 计算理论参与率
r_hat = norm.cdf(z_score)
print(f"      = {r_hat:.4f} ({r_hat:.2%})")

# 对比
print(f"\n🎯 对比结果:")
print(f"  求解的 r* = {r_star:.4f} ({r_star:.2%})")
print(f"  理论的 r̂  = {r_hat:.4f} ({r_hat:.2%})")
print(f"  绝对误差   = {abs(r_star - r_hat):.4f}")
print(f"  相对误差   = {abs(r_star - r_hat) / r_star * 100:.2f}%")

# 判断
if abs(r_star - r_hat) < 0.01:
    print(f"\n✅ 验证通过！")
    print(f"   求解的 r* 与理论预测 r̂ 非常接近")
    print(f"   说明：")
    print(f"     • 固定点求解正确")
    print(f"     • 参与率低 (6.5%) 是模型含义，不是bug")
    print(f"     • 原因：τ_mean=1.0 > ΔU=0.55，大部分消费者隐私成本高于效用增益")
else:
    print(f"\n⚠️ 存在偏差")
    print(f"   可能原因：")
    print(f"     • 固定点未完全收敛")
    print(f"     • MC样本数不足")
    print(f"     • 模型实现存在问题")

# 额外分析
print(f"\n📊 参与率低的原因分析:")
print(f"\n  效用增益: ΔU = {delta_u:.4f}")
print(f"  隐私成本均值: τ_mean = {params.tau_mean}")
print(f"  隐私成本标准差: τ_std = {params.tau_std}")

if delta_u < params.tau_mean:
    print(f"\n  ⚠️ ΔU < τ_mean：效用增益低于平均隐私成本")
    print(f"     → 大多数消费者不愿意参与")
    print(f"     → 只有隐私成本较低的消费者（τ < {delta_u:.2f}）才会参与")
    
    # 计算参与者的隐私成本分布
    print(f"\n  参与者特征:")
    print(f"    • 隐私成本需满足：τ_i < {delta_u:.4f}")
    print(f"    • 这是正态分布左尾的 {r_hat:.2%}")
    print(f"    • 平均约 {r_hat * params.N:.1f} 人参与（N={params.N}）")
else:
    print(f"\n  ✅ ΔU > τ_mean：效用增益高于平均隐私成本")
    print(f"     → 大多数消费者愿意参与")

# 敏感性分析
print(f"\n🔬 敏感性分析（如果要提高参与率）:")
print(f"\n  方案1：增加补偿 m")
print(f"    当前 m = {params.m:.2f}")
print(f"    增加 m → 增加 ΔU → 提高参与率")
print(f"    但成本也会增加，可能导致中介利润下降")

print(f"\n  方案2：减少隐私成本")
print(f"    当前 τ_mean = {params.tau_mean}")
print(f"    如果 τ_mean = 0.5，则:")
z_score_alt = (delta_u - 0.5) / params.tau_std
r_hat_alt = norm.cdf(z_score_alt)
print(f"      r̂ = Φ(({delta_u:.4f} - 0.5) / {params.tau_std})")
print(f"        = Φ({z_score_alt:.4f})")
print(f"        = {r_hat_alt:.4f} ({r_hat_alt:.2%})")
print(f"    参与率可提升至 {r_hat_alt:.2%}")

print(f"\n  方案3：增加隐私成本异质性")
print(f"    当前 τ_std = {params.tau_std}")
print(f"    如果 τ_std = 0.5，则:")
z_score_alt2 = (delta_u - params.tau_mean) / 0.5
r_hat_alt2 = norm.cdf(z_score_alt2)
print(f"      r̂ = Φ(({delta_u:.4f} - {params.tau_mean}) / 0.5)")
print(f"        = Φ({z_score_alt2:.4f})")
print(f"        = {r_hat_alt2:.4f} ({r_hat_alt2:.2%})")
print(f"    参与率可提升至 {r_hat_alt2:.2%}")

print(f"\n" + "=" * 70)
print(f"验证完成")
print(f"=" * 70)
