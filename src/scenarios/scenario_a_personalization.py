"""
场景 A：Personalization & Privacy Choice（价格传导外部性）
基于"最终设计方案.md"的实现

核心逻辑：
1. 枚举所有可能的披露集合 D
2. 对每个 D，计算平台最优统一价格（针对未披露者）
3. 计算每个 D 下的消费者效用、平台利润、社会福利
4. 找到Nash均衡：对每个消费者，单边偏离不增益
5. 找到社会最优：最大化 W(D)
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import json


@dataclass
class ScenarioAParams:
    """场景A参数"""
    n: int  # 消费者数
    theta: List[float]  # 每个消费者的愿付
    c_privacy: List[float]  # 每个消费者的隐私成本
    seller_cost: float  # 卖家边际成本
    seed: int
    
    def to_dict(self):
        return asdict(self)


@dataclass
class OutcomeForD:
    """给定披露集合D的结果"""
    D: Set[int]  # 披露集合
    p_uniform: float  # 统一价格（针对未披露者）
    profits_disclosed: float  # 从披露者获得的利润
    profits_undisclosed: float  # 从未披露者获得的利润
    total_profit: float  # 总利润
    consumer_surplus: float  # 消费者剩余
    welfare: float  # 社会福利
    consumer_utilities: List[float]  # 每个消费者的效用


def generate_instance(n: int = 10, seed: int = 42) -> ScenarioAParams:
    """生成一个场景A的实例"""
    np.random.seed(seed)
    
    # 从离散网格抽样愿付
    theta_choices = [0.2, 0.4, 0.6, 0.8, 1.0]
    theta = np.random.choice(theta_choices, size=n).tolist()
    
    # 隐私成本：从[0.05, 0.3]均匀抽样
    c_privacy = np.random.uniform(0.05, 0.3, size=n).tolist()
    
    # 卖家成本
    seller_cost = 0.1
    
    return ScenarioAParams(
        n=n,
        theta=theta,
        c_privacy=c_privacy,
        seller_cost=seller_cost,
        seed=seed
    )


def solve_for_D(params: ScenarioAParams, D: Set[int]) -> OutcomeForD:
    """
    给定披露集合D，求解平台最优定价和福利
    
    披露者：平台设个性化价 p_i = theta_i（榨取全部剩余）
    未披露者：平台设统一价 p_u，通过枚举候选价格找最优
    """
    n = params.n
    theta = params.theta
    c_privacy = params.c_privacy
    cost = params.seller_cost
    
    # 未披露集合
    U = set(range(n)) - D
    
    # === 1. 对披露者：个性化定价 p_i = theta_i ===
    profits_disclosed = 0.0
    for i in D:
        p_i = theta[i]
        if theta[i] >= p_i:  # 购买
            profits_disclosed += (p_i - cost)
    
    # === 2. 对未披露者：找最优统一价 p_u ===
    if len(U) == 0:
        p_uniform = 0.0
        profits_undisclosed = 0.0
    else:
        # 候选价格：所有未披露者的theta值
        candidate_prices = sorted(set([theta[i] for i in U]))
        
        best_p = 0.0
        best_profit = 0.0
        
        for p in candidate_prices:
            profit = sum((p - cost) for i in U if theta[i] >= p)
            if profit > best_profit:
                best_profit = profit
                best_p = p
        
        p_uniform = best_p
        profits_undisclosed = best_profit
    
    total_profit = profits_disclosed + profits_undisclosed
    
    # === 3. 计算消费者效用 ===
    consumer_utilities = []
    for i in range(n):
        if i in D:
            # 披露者：p_i = theta_i，购买，效用 = (theta - p) - c_privacy = 0 - c_privacy
            u_i = -c_privacy[i]
        else:
            # 未披露者：如果购买（theta >= p_uniform），效用 = theta - p_uniform
            if theta[i] >= p_uniform:
                u_i = theta[i] - p_uniform
            else:
                u_i = 0.0
        consumer_utilities.append(u_i)
    
    consumer_surplus = sum(consumer_utilities)
    welfare = consumer_surplus + total_profit
    
    return OutcomeForD(
        D=D,
        p_uniform=p_uniform,
        profits_disclosed=profits_disclosed,
        profits_undisclosed=profits_undisclosed,
        total_profit=total_profit,
        consumer_surplus=consumer_surplus,
        welfare=welfare,
        consumer_utilities=consumer_utilities
    )


def check_nash_equilibrium(params: ScenarioAParams, D: Set[int], 
                          all_outcomes: Dict[frozenset, OutcomeForD]) -> bool:
    """
    检查D是否为Nash均衡：对每个i，单边偏离不增益
    """
    n = params.n
    outcome_D = all_outcomes[frozenset(D)]
    
    for i in range(n):
        # 当前i的效用
        u_i_current = outcome_D.consumer_utilities[i]
        
        # 尝试单边偏离
        if i in D:
            # i当前披露，尝试不披露
            D_prime = D - {i}
        else:
            # i当前不披露，尝试披露
            D_prime = D | {i}
        
        outcome_D_prime = all_outcomes[frozenset(D_prime)]
        u_i_deviate = outcome_D_prime.consumer_utilities[i]
        
        # 如果偏离能增益，则不是均衡
        if u_i_deviate > u_i_current + 1e-9:
            return False
    
    return True


def solve_scenario_a(params: ScenarioAParams) -> Dict:
    """
    完整求解场景A
    
    返回：
    - 所有披露集合的结果
    - Nash均衡集合
    - 社会最优集合
    - gt_numeric（数值真值）
    - gt_labels（抽象标签）
    """
    n = params.n
    
    # === 1. 枚举所有披露集合并计算结果 ===
    print(f"枚举所有 2^{n} = {2**n} 个披露集合...")
    all_outcomes = {}
    
    for size in range(n + 1):
        for D_tuple in itertools.combinations(range(n), size):
            D = set(D_tuple)
            outcome = solve_for_D(params, D)
            all_outcomes[frozenset(D)] = outcome
    
    # === 2. 找Nash均衡 ===
    print("寻找Nash均衡...")
    nash_equilibria = []
    for D_frozen, outcome in all_outcomes.items():
        D = set(D_frozen)
        if check_nash_equilibrium(params, D, all_outcomes):
            nash_equilibria.append((D, outcome))
    
    print(f"找到 {len(nash_equilibria)} 个Nash均衡")
    
    # 如果有多个均衡，选择社会福利最大的
    if len(nash_equilibria) > 0:
        eq_D, eq_outcome = max(nash_equilibria, key=lambda x: x[1].welfare)
    else:
        # 理论上应该存在均衡；如果没找到，取空集
        print("警告：未找到Nash均衡，使用空集")
        eq_D = set()
        eq_outcome = all_outcomes[frozenset()]
    
    # === 3. 找社会最优 ===
    print("寻找社会最优...")
    fb_D_frozen, fb_outcome = max(all_outcomes.items(), key=lambda x: x[1].welfare)
    fb_D = set(fb_D_frozen)
    
    # === 4. 构造 gt_numeric ===
    gt_numeric = {
        "eq_disclosure_set": list(eq_D),
        "eq_uniform_price": eq_outcome.p_uniform,
        "eq_profit": eq_outcome.total_profit,
        "eq_CS": eq_outcome.consumer_surplus,
        "eq_W": eq_outcome.welfare,
        "fb_disclosure_set": list(fb_D),
        "fb_uniform_price": fb_outcome.p_uniform,
        "fb_profit": fb_outcome.total_profit,
        "fb_CS": fb_outcome.consumer_surplus,
        "fb_W": fb_outcome.welfare,
        "externality_gap_W": fb_outcome.welfare - eq_outcome.welfare,
        "externality_gap_CS": fb_outcome.consumer_surplus - eq_outcome.consumer_surplus,
    }
    
    # === 5. 构造 gt_labels ===
    disclosure_rate = len(eq_D) / n
    if disclosure_rate < 0.33:
        disclosure_bucket = "low"
    elif disclosure_rate < 0.67:
        disclosure_bucket = "med"
    else:
        disclosure_bucket = "high"
    
    gt_labels = {
        "disclosure_rate_bucket": disclosure_bucket,
        "disclosure_rate": disclosure_rate,
        "over_disclosure": 1 if len(eq_D) > len(fb_D) else 0,
        "eq_size": len(eq_D),
        "fb_size": len(fb_D),
    }
    
    # 转换all_outcomes为可JSON序列化格式
    all_outcomes_serializable = {}
    for D_frozen, outcome in all_outcomes.items():
        D_list = sorted(list(D_frozen))
        outcome_dict = asdict(outcome)
        outcome_dict['D'] = D_list  # 将set转为list
        all_outcomes_serializable[str(D_list)] = outcome_dict
    
    return {
        "params": params.to_dict(),
        "gt_numeric": gt_numeric,
        "gt_labels": gt_labels,
        "all_outcomes": all_outcomes_serializable,
        "nash_equilibria_count": len(nash_equilibria),
    }


def main():
    """示例运行"""
    print("=" * 60)
    print("场景 A：Personalization & Privacy Choice")
    print("=" * 60)
    
    # 生成实例（小规模便于演示）
    params = generate_instance(n=8, seed=42)
    
    print(f"\n参数设置：")
    print(f"  消费者数 n = {params.n}")
    print(f"  愿付 theta = {params.theta}")
    print(f"  隐私成本 c = {params.c_privacy}")
    print(f"  卖家成本 = {params.seller_cost}")
    
    # 求解
    result = solve_scenario_a(params)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("求解结果")
    print("=" * 60)
    
    gt = result["gt_numeric"]
    labels = result["gt_labels"]
    
    print(f"\nNash均衡：")
    print(f"  披露集合: {gt['eq_disclosure_set']}")
    print(f"  披露率: {labels['disclosure_rate']:.2%}")
    print(f"  统一价格: {gt['eq_uniform_price']:.3f}")
    print(f"  平台利润: {gt['eq_profit']:.3f}")
    print(f"  消费者剩余: {gt['eq_CS']:.3f}")
    print(f"  社会福利: {gt['eq_W']:.3f}")
    
    print(f"\n社会最优（First-best）：")
    print(f"  披露集合: {gt['fb_disclosure_set']}")
    print(f"  统一价格: {gt['fb_uniform_price']:.3f}")
    print(f"  平台利润: {gt['fb_profit']:.3f}")
    print(f"  消费者剩余: {gt['fb_CS']:.3f}")
    print(f"  社会福利: {gt['fb_W']:.3f}")
    
    print(f"\n外部性分析：")
    print(f"  过度披露?: {'是' if labels['over_disclosure'] else '否'}")
    print(f"  均衡披露数 vs 最优披露数: {labels['eq_size']} vs {labels['fb_size']}")
    print(f"  福利损失: {gt['externality_gap_W']:.3f}")
    print(f"  消费者剩余损失: {gt['externality_gap_CS']:.3f}")
    
    # 保存结果
    output_file = "data/ground_truth/scenario_a_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
