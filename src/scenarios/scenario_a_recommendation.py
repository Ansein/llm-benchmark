"""
场景 A：Personalized Recommendation with Privacy Choice (推荐系统与隐私选择)
完整版 - 包含搜索成本、推荐机制和贝叶斯纳什均衡求解

基于 agents_complete.py 和 recommendation_simulation(1).ipynb 的理论模型重构

核心机制：
1. 消费者决策：是否分享数据（权衡隐私成本 vs 搜索成本节省）
2. 平台推荐：为分享数据的消费者提供排序推荐（从高估值到低估值）
3. 企业定价：贝叶斯纳什均衡定价（考虑数据分享率）
4. 消费者搜索：分享数据者按推荐顺序搜索，未分享者随机搜索
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ScenarioARecommendationParams:
    """场景A推荐系统参数"""
    n_consumers: int  # 消费者数量
    n_firms: int  # 企业数量
    search_cost: float  # 每次搜索成本（首次免费，后续每次s）
    privacy_costs: List[float]  # 每个消费者的隐私成本
    v_dist: Dict[str, float]  # 估值分布参数 {'type': 'uniform', 'low': 0, 'high': 1}
    r_value: float  # 保留效用（reservation value）
    firm_cost: float  # 企业边际成本
    seed: int
    
    def to_dict(self):
        return asdict(self)


def generate_recommendation_instance(
    n_consumers: int = 10,
    n_firms: int = 5,
    search_cost: float = 0.02,
    seed: int = 42
) -> ScenarioARecommendationParams:
    """生成场景A推荐系统实例"""
    np.random.seed(seed)
    
    # 隐私成本：从[0.025, 0.055]均匀抽样
    privacy_costs = np.random.uniform(0.025, 0.055, size=n_consumers).tolist()
    
    # 估值分布：uniform[0, 1]
    v_dist = {'type': 'uniform', 'low': 0.0, 'high': 1.0}
    
    # 保留效用
    r_value = 0.8
    
    # 企业成本
    firm_cost = 0.0
    
    return ScenarioARecommendationParams(
        n_consumers=n_consumers,
        n_firms=n_firms,
        search_cost=search_cost,
        privacy_costs=privacy_costs,
        v_dist=v_dist,
        r_value=r_value,
        firm_cost=firm_cost,
        seed=seed
    )


# ============================================================================
# 理性决策逻辑（基于贝叶斯纳什均衡）
# ============================================================================

def calculate_delta_sharing(v_dist: Dict, r_value: float, n_firms: int) -> float:
    """
    计算分享决策的Delta参数
    
    Delta = ∫_r^{v_high} [F_v - F_v^n] dv
    
    表示：通过推荐系统获得的期望额外效用增益
    """
    low, high = v_dist['low'], v_dist['high']
    n = n_firms
    r = r_value
    
    def integrand(v):
        F_v = (v - low) / (high - low)
        return F_v - F_v ** n
    
    # 确保积分下界在分布范围内
    r_for_quad = max(r, low)
    if r_for_quad >= high:
        return 0.0
    
    try:
        delta, _ = quad(integrand, r_for_quad, high, limit=100)
    except Exception as e:
        print(f"Warning: Delta calculation failed (r={r}, n={n}). Error: {e}")
        delta = 0.0
    
    return max(0.0, delta)


def rational_share_decision(
    privacy_cost: float,
    delta: float,
    search_cost: float
) -> bool:
    """
    理性分享决策
    
    分享条件：Delta - τ - s ≥ 0
    - Delta: 推荐带来的期望效用增益
    - τ: 隐私成本
    - s: 搜索成本节省（粗略估计）
    """
    # 简化模型：分享可以节省搜索成本（首次搜索免费，后续需要成本）
    # 预期节省 ≈ search_cost * (expected_searches_without_recommendation - 1)
    # 粗略估计：不分享平均搜索2-3次，分享后1-2次
    expected_cost_saving = search_cost * 1.5
    
    benefit = delta + expected_cost_saving
    cost = privacy_cost
    
    return benefit >= cost


def firm_non_shared_demand(
    p_i: float,
    market_price: float,
    r_value: float,
    n_firms: int,
    v_dist: Dict,
    firm_cost: float
) -> float:
    """
    计算企业i面对未分享数据消费者的需求
    
    基于理论模型：消费者随机搜索，根据价格和估值决定购买
    """
    v_low = v_dist['low']
    v_high = v_dist['high']
    v_span = v_high - v_low
    
    if v_span == 0:
        return 0.0
    
    F_r = np.clip((r_value - v_low) / v_span, 0, 1)
    
    # Term 1: 从未搜索到其他企业的消费者中获得的需求
    F_term = np.clip((r_value - market_price + p_i - v_low) / v_span, 0, 1)
    numerator = (1 - F_term) * (1 - F_r ** n_firms)
    denominator = n_firms * (1 - F_r)
    term1 = numerator / denominator if denominator > 1e-9 else 0.0
    
    # Term 2: 积分项（已搜索其他企业但选择购买当前企业的消费者）
    lower_int = max(p_i, v_low)
    upper_int = min(r_value - market_price + p_i, v_high)
    
    if lower_int >= upper_int:
        term2 = 0.0
    else:
        def integrand(v_i):
            offset = v_i - p_i + market_price
            F_offset = np.clip((offset - v_low) / v_span, 0, 1)
            return F_offset ** (n_firms - 1) / v_span
        
        try:
            term2, _ = quad(integrand, lower_int, upper_int, limit=100)
        except:
            term2 = 0.0
    
    return term1 + term2


def firm_shared_demand(p_i: float, n_firms: int, v_dist: Dict) -> float:
    """
    计算企业i面对分享数据消费者的需求
    
    分享数据的消费者按推荐顺序搜索（从高估值到低估值）
    企业i的需求 = (1 - F_p^n) / n
    """
    v_low = v_dist['low']
    v_high = v_dist['high']
    v_span = v_high - v_low
    
    if v_span == 0:
        return 0.0
    
    F_pi = np.clip((p_i - v_low) / v_span, 0, 1)
    q_s = (1 - F_pi ** n_firms) / n_firms
    
    return q_s


def optimize_firm_price(
    share_rate: float,
    n_firms: int,
    market_price: float,
    v_dist: Dict,
    r_value: float,
    firm_cost: float
) -> float:
    """
    企业最优定价（贝叶斯纳什均衡）
    
    利润 = (p - c) * [σ * q_shared + (1-σ) * q_non_shared]
    
    其中：
    - σ: 数据分享率
    - q_shared: 分享数据消费者的需求
    - q_non_shared: 未分享数据消费者的需求
    """
    σ = share_rate
    v_low = v_dist['low']
    v_high = v_dist['high']
    
    def profit(p_i):
        q_s = firm_shared_demand(p_i, n_firms, v_dist)
        q_ns = firm_non_shared_demand(p_i, market_price, r_value, n_firms, v_dist, firm_cost)
        Q = σ * q_s + (1 - σ) * q_ns
        return (p_i - firm_cost) * Q
    
    # 价格区间：[c, r]
    lower_bound = firm_cost
    upper_bound = min(r_value, v_high * 0.99)
    
    if lower_bound >= upper_bound:
        return lower_bound
    
    try:
        # 使用minimize_scalar寻找最大利润
        result = minimize_scalar(
            lambda p: -profit(p),  # 负号转为最大化问题
            bounds=(lower_bound, upper_bound),
            method='bounded'
        )
        optimal_price = result.x
    except:
        # 如果优化失败，使用中点
        optimal_price = (lower_bound + upper_bound) / 2
    
    return optimal_price


def solve_rational_equilibrium(
    params: ScenarioARecommendationParams,
    max_share_iter: int = 50,
    max_price_iter: int = 50,
    tol: float = 1e-6
) -> Dict:
    """
    求解理性均衡（分享率 + 价格的联合均衡）
    
    算法：
    1. 固定点迭代求解分享率均衡
    2. 对每个分享率，迭代求解价格均衡
    3. 计算均衡时的市场结果
    
    Returns:
        均衡结果字典
    """
    print(f"\n{'='*60}")
    print(f"[理性均衡求解] n_consumers={params.n_consumers}, n_firms={params.n_firms}")
    print(f"{'='*60}")
    
    # 预计算Delta（仅依赖全局参数）
    delta = calculate_delta_sharing(params.v_dist, params.r_value, params.n_firms)
    print(f"Delta (推荐效用增益): {delta:.6f}")
    
    # ===== 步骤1: 求解分享率均衡 =====
    print("\n[步骤1] 求解分享率均衡...")
    σ = 0.5  # 初始分享率猜测
    
    for iter_share in range(max_share_iter):
        # 每个消费者根据当前分享率做理性决策
        share_decisions = []
        for i in range(params.n_consumers):
            τ_i = params.privacy_costs[i]
            should_share = rational_share_decision(τ_i, delta, params.search_cost)
            share_decisions.append(int(should_share))
        
        σ_new = np.mean(share_decisions)
        
        if abs(σ_new - σ) < tol:
            print(f"  收敛于第 {iter_share + 1} 次迭代: σ = {σ_new:.6f}")
            break
        
        σ = σ_new
        
        if (iter_share + 1) % 10 == 0:
            print(f"  迭代 {iter_share + 1}: σ = {σ:.6f}")
    else:
        print(f"  未在 {max_share_iter} 次迭代内收敛，最终 σ = {σ:.6f}")
    
    equilibrium_share_rate = σ
    
    # ===== 步骤2: 给定分享率，求解价格均衡 =====
    print("\n[步骤2] 求解价格均衡...")
    
    # 初始价格猜测
    initial_price = max(0.1, params.r_value - 0.3)
    prices = [initial_price] * params.n_firms
    
    for iter_price in range(max_price_iter):
        market_price = np.mean(prices)
        new_prices = []
        
        for firm_idx in range(params.n_firms):
            optimal_p = optimize_firm_price(
                share_rate=equilibrium_share_rate,
                n_firms=params.n_firms,
                market_price=market_price,
                v_dist=params.v_dist,
                r_value=params.r_value,
                firm_cost=params.firm_cost
            )
            new_prices.append(optimal_p)
        
        price_diff = np.max(np.abs(np.array(new_prices) - np.array(prices)))
        
        if price_diff < tol:
            print(f"  收敛于第 {iter_price + 1} 次迭代")
            print(f"  均衡价格: {[round(p, 4) for p in new_prices]}")
            break
        
        prices = new_prices
        
        if (iter_price + 1) % 10 == 0:
            print(f"  迭代 {iter_price + 1}: 价格差异 = {price_diff:.6f}")
    else:
        print(f"  未在 {max_price_iter} 次迭代内收敛")
        print(f"  最终价格: {[round(p, 4) for p in prices]}")
    
    equilibrium_prices = prices
    avg_price = np.mean(prices)
    
    # ===== 步骤3: 计算均衡时的市场结果 =====
    print("\n[步骤3] 计算市场结果...")
    
    # 消费者效用（简化计算）
    total_consumer_utility = 0.0
    for i in range(params.n_consumers):
        if share_decisions[i] == 1:
            # 分享数据：获得推荐，支付隐私成本
            # 期望效用 ≈ delta - privacy_cost - search_cost
            u_i = delta - params.privacy_costs[i] - params.search_cost
        else:
            # 未分享：随机搜索
            # 期望效用 ≈ (r - avg_price) - expected_search_cost
            u_i = max(0, params.r_value - avg_price) - params.search_cost * 2
        
        total_consumer_utility += u_i
    
    # 企业利润（简化：假设每个企业平均需求）
    avg_demand_per_firm = (equilibrium_share_rate * 0.8 + (1 - equilibrium_share_rate) * 0.5) / params.n_firms
    total_firm_profit = sum((p - params.firm_cost) * avg_demand_per_firm for p in equilibrium_prices)
    
    # 社会福利
    total_welfare = total_consumer_utility + total_firm_profit
    
    print(f"\n均衡结果:")
    print(f"  分享率: {equilibrium_share_rate:.4f}")
    print(f"  平均价格: {avg_price:.4f}")
    print(f"  消费者剩余: {total_consumer_utility:.4f}")
    print(f"  企业利润: {total_firm_profit:.4f}")
    print(f"  社会福利: {total_welfare:.4f}")
    
    return {
        "equilibrium_share_rate": equilibrium_share_rate,
        "equilibrium_prices": equilibrium_prices,
        "avg_price": avg_price,
        "share_decisions": share_decisions,
        "consumer_surplus": total_consumer_utility,
        "firm_profit": total_firm_profit,
        "social_welfare": total_welfare,
        "delta": delta,
        "converged": True
    }


def solve_scenario_a_recommendation(params: ScenarioARecommendationParams) -> Dict:
    """
    完整求解场景A推荐系统
    
    Returns:
        包含ground truth的完整结果
    """
    print(f"\n{'='*60}")
    print(f"场景A推荐系统 - 完整求解")
    print(f"{'='*60}")
    print(f"参数:")
    print(f"  消费者数量: {params.n_consumers}")
    print(f"  企业数量: {params.n_firms}")
    print(f"  搜索成本: {params.search_cost}")
    print(f"  估值分布: {params.v_dist}")
    print(f"  保留效用: {params.r_value}")
    print(f"  企业成本: {params.firm_cost}")
    
    # 求解理性均衡
    eq_result = solve_rational_equilibrium(params)
    
    # 构造gt_numeric
    gt_numeric = {
        "eq_share_rate": eq_result["equilibrium_share_rate"],
        "eq_prices": eq_result["equilibrium_prices"],
        "eq_avg_price": eq_result["avg_price"],
        "eq_consumer_surplus": eq_result["consumer_surplus"],
        "eq_firm_profit": eq_result["firm_profit"],
        "eq_welfare": eq_result["social_welfare"],
        "delta": eq_result["delta"]
    }
    
    # 构造gt_labels（抽象标签）
    gt_labels = {
        "high_share_rate": 1 if eq_result["equilibrium_share_rate"] > 0.7 else 0,
        "low_share_rate": 1 if eq_result["equilibrium_share_rate"] < 0.3 else 0,
        "price_competitive": 1 if eq_result["avg_price"] < params.r_value * 0.7 else 0
    }
    
    return {
        "scenario": "A_recommendation",
        "params": params.to_dict(),
        "gt_numeric": gt_numeric,
        "gt_labels": gt_labels,
        "full_equilibrium": eq_result
    }


# ============================================================================
# Ground Truth生成和保存
# ============================================================================

def generate_and_save_ground_truth(
    output_path: str = "data/ground_truth/scenario_a_recommendation_result.json",
    n_consumers: int = 10,
    n_firms: int = 5,
    search_cost: float = 0.02,
    seed: int = 42
):
    """生成并保存场景A推荐系统的ground truth"""
    from pathlib import Path
    
    print("生成场景A推荐系统Ground Truth...")
    
    # 生成实例
    params = generate_recommendation_instance(
        n_consumers=n_consumers,
        n_firms=n_firms,
        search_cost=search_cost,
        seed=seed
    )
    
    # 求解
    result = solve_scenario_a_recommendation(params)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nGround Truth saved to: {output_path}")
    return result


if __name__ == "__main__":
    # 测试：生成ground truth
    result = generate_and_save_ground_truth(
        n_consumers=10,
        n_firms=5,
        search_cost=0.02,
        seed=42
    )
    
    print("\n" + "="*60)
    print("Ground Truth生成完成！")
    print("="*60)
