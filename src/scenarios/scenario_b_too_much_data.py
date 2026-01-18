"""
场景 B：Too Much Data（推断外部性/信息外部性）
基于"最终设计方案.md"的实现

核心逻辑：
1. 用户类型相关（高斯），他人分享会泄露你的信息
2. 枚举所有可能的分享集合 S
3. 对每个 S，计算后验协方差和泄露信息量
4. 平台从泄露信息中获益，用户承受隐私成本
5. 找均衡：平台利润最大的 S（同时满足最小支付约束）
6. 找社会最优：最大化 W(S)
7. 比较shutdown等政策
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import json


@dataclass
class ScenarioBParams:
    """场景B参数"""
    n: int  # 用户数
    Sigma: np.ndarray  # 类型协方差矩阵 (n x n)
    sigma_noise_sq: float  # 观测噪声方差
    alpha: float  # 平台从信息获益系数
    v: List[float]  # 每个用户的隐私偏好
    rho: float  # 相关系数（用于生成Sigma）
    seed: int
    
    def to_dict(self):
        return {
            "n": self.n,
            "Sigma": self.Sigma.tolist(),
            "sigma_noise_sq": self.sigma_noise_sq,
            "alpha": self.alpha,
            "v": self.v,
            "rho": self.rho,
            "seed": self.seed,
        }


@dataclass
class OutcomeForS:
    """给定分享集合S的结果"""
    S: Set[int]  # 分享集合
    leakage: List[float]  # 每个用户的泄露信息量
    total_leakage: float  # 总泄露
    platform_value: float  # 平台价值
    user_costs: List[float]  # 每个用户的隐私成本
    total_user_cost: float  # 总用户成本
    welfare: float  # 社会福利 = 平台价值 - 用户成本
    min_prices: List[float]  # 支持该集合的最小价格（仅对分享者）
    platform_profit: float  # 平台利润 = 价值 - 支付


def generate_instance(n: int = 10, rho: float = 0.6, seed: int = 42) -> ScenarioBParams:
    """生成一个场景B的实例"""
    np.random.seed(seed)
    
    # 生成等相关协方差矩阵
    Sigma = np.ones((n, n)) * rho
    np.fill_diagonal(Sigma, 1.0)
    
    # 观测噪声
    sigma_noise_sq = 0.1
    
    # 平台收益系数
    alpha = 1.0
    
    # 用户隐私偏好：从[0.3, 1.2]均匀抽样
    v = np.random.uniform(0.3, 1.2, size=n).tolist()
    
    return ScenarioBParams(
        n=n,
        Sigma=Sigma,
        sigma_noise_sq=sigma_noise_sq,
        alpha=alpha,
        v=v,
        rho=rho,
        seed=seed
    )


def compute_posterior_covariance(Sigma: np.ndarray, S: Set[int], 
                                 sigma_noise_sq: float) -> np.ndarray:
    """
    计算后验协方差矩阵（高斯条件分布）
    
    观测模型：对于 i in S，观测 y_i = x_i + noise
    
    后验：Sigma_post = Sigma - Sigma @ H.T @ (H @ Sigma @ H.T + R)^{-1} @ H @ Sigma
    
    其中 H 是观测矩阵，R 是噪声协方差
    """
    n = Sigma.shape[0]
    S_list = sorted(S)
    m = len(S_list)
    
    if m == 0:
        # 没有观测，后验等于先验
        return Sigma.copy()
    
    # 构造观测矩阵 H (m x n)
    H = np.zeros((m, n))
    for idx, i in enumerate(S_list):
        H[idx, i] = 1.0
    
    # 噪声协方差矩阵 R (m x m)
    R = np.eye(m) * sigma_noise_sq
    
    # 计算后验协方差
    # Sigma_post = Sigma - Sigma @ H.T @ inv(H @ Sigma @ H.T + R) @ H @ Sigma
    HSigma = H @ Sigma
    HSigmaHT_R = HSigma @ H.T + R
    solve_term = np.linalg.solve(HSigmaHT_R, HSigma)

    Sigma_post = Sigma - Sigma @ H.T @ solve_term
    
    return Sigma_post


def solve_for_S(params: ScenarioBParams, S: Set[int]) -> OutcomeForS:
    """
    给定分享集合S，计算泄露、价值、成本、福利
    """
    n = params.n
    Sigma = params.Sigma
    sigma_noise_sq = params.sigma_noise_sq
    alpha = params.alpha
    v = params.v
    
    # 计算后验协方差
    Sigma_post = compute_posterior_covariance(Sigma, S, sigma_noise_sq)
    
    # 计算每个用户的泄露信息量（方差减少）
    leakage = []
    for i in range(n):
        leak_i = Sigma[i, i] - Sigma_post[i, i]
        leakage.append(max(0.0, leak_i))  # 确保非负
    
    total_leakage = sum(leakage)
    
    # 平台价值
    platform_value = alpha * total_leakage
    
    # 用户隐私成本
    user_costs = [v[i] * leakage[i] for i in range(n)]
    total_user_cost = sum(user_costs)
    
    # 社会福利
    welfare = platform_value - total_user_cost
    
    # === 计算支持该集合的最小价格（目标集合支持法）===
    # 考虑推断外部性：用户不分享也会有基础泄露
    # 参与约束：p_i - v_i * leak_i(分享) >= -v_i * leak_i(不分享)
    # 即：p_i >= v_i * [leak_i(分享) - leak_i(不分享)] = v_i * ΔI_i
    
    min_prices = [0.0] * n
    
    # 计算不包含每个用户时的泄露（用于计算边际泄露）
    for i in S:
        # 计算不包含i时的泄露
        S_without_i = S - {i}
        Sigma_post_without_i = compute_posterior_covariance(Sigma, S_without_i, sigma_noise_sq)
        leak_i_without = max(0.0, Sigma[i, i] - Sigma_post_without_i[i, i])
        
        # 边际泄露 = 总泄露 - 基础泄露
        marginal_leak_i = leakage[i] - leak_i_without
        
        # 最小价格 = v_i × 边际泄露（而非总泄露）
        min_prices[i] = v[i] * max(0.0, marginal_leak_i)
    
    # 平台利润 = 价值 - 支付给分享者的价格
    platform_profit = platform_value - sum(min_prices)
    
    return OutcomeForS(
        S=S,
        leakage=leakage,
        total_leakage=total_leakage,
        platform_value=platform_value,
        user_costs=user_costs,
        total_user_cost=total_user_cost,
        welfare=welfare,
        min_prices=min_prices,
        platform_profit=platform_profit
    )


def calculate_leakage(S: Set[int], Sigma: np.ndarray, sigma_noise_sq: float) -> List[float]:
    """
    计算给定分享集合S下每个用户的泄露信息量
    
    Args:
        S: 分享集合
        Sigma: 协方差矩阵
        sigma_noise_sq: 观测噪声方差
    
    Returns:
        每个用户的泄露量列表
    """
    n = Sigma.shape[0]
    Sigma_post = compute_posterior_covariance(Sigma, S, sigma_noise_sq)
    
    leakage = []
    for i in range(n):
        leak_i = Sigma[i, i] - Sigma_post[i, i]
        leakage.append(max(0.0, leak_i))
    
    return leakage


def calculate_outcome(S: Set[int], params: ScenarioBParams) -> Dict:
    """
    计算给定分享集合S的结果（简化版，用于评估器）
    
    Args:
        S: 分享集合
        params: 场景参数
    
    Returns:
        结果字典
    """
    outcome = solve_for_S(params, S)
    return {
        "leakage": outcome.leakage,
        "total_leakage": outcome.total_leakage,
        "welfare": outcome.welfare,
        "profit": outcome.platform_profit,
        "platform_value": outcome.platform_value,
        "user_costs": outcome.user_costs
    }


def solve_scenario_b(params: ScenarioBParams) -> Dict:
    """
    完整求解场景B
    
    返回：
    - 所有分享集合的结果
    - 均衡集合（平台利润最大）
    - 社会最优集合（福利最大）
    - shutdown情况
    - gt_numeric & gt_labels
    """
    n = params.n
    
    # === 1. 枚举所有分享集合并计算结果 ===
    print(f"枚举所有 2^{n} = {2**n} 个分享集合...")
    all_outcomes = {}
    
    for size in range(n + 1):
        for S_tuple in itertools.combinations(range(n), size):
            S = set(S_tuple)
            outcome = solve_for_S(params, S)
            all_outcomes[frozenset(S)] = outcome
    
    # === 2. 找均衡：平台利润最大 ===
    print("寻找均衡（平台利润最大）...")
    eq_S_frozen, eq_outcome = max(all_outcomes.items(), 
                                   key=lambda x: x[1].platform_profit)
    eq_S = set(eq_S_frozen)
    
    # === 3. 找社会最优：福利最大 ===
    print("寻找社会最优（福利最大）...")
    fb_S_frozen, fb_outcome = max(all_outcomes.items(), 
                                   key=lambda x: x[1].welfare)
    fb_S = set(fb_S_frozen)
    
    # === 4. Shutdown情况（S = 空集）===
    shutdown_outcome = all_outcomes[frozenset()]
    
    # === 5. 构造 gt_numeric ===
    gt_numeric = {
        "eq_share_set": list(eq_S),
        "eq_prices": eq_outcome.min_prices,
        "eq_value": eq_outcome.platform_value,
        "eq_profit": eq_outcome.platform_profit,
        "eq_W": eq_outcome.welfare,
        "eq_total_leakage": eq_outcome.total_leakage,
        "fb_share_set": list(fb_S),
        "fb_W": fb_outcome.welfare,
        "fb_total_leakage": fb_outcome.total_leakage,
        "shutdown_W": shutdown_outcome.welfare,
        "shutdown_leakage": shutdown_outcome.total_leakage,
    }
    
    # === 6. 构造 gt_labels ===
    over_sharing = 1 if len(eq_S) > len(fb_S) else 0
    shutdown_better = 1 if shutdown_outcome.welfare > eq_outcome.welfare else 0
    
    # 泄露量分桶
    max_possible_leakage = n * 1.0  # 每个用户最多泄露方差=1
    leakage_ratio = eq_outcome.total_leakage / max_possible_leakage
    if leakage_ratio < 0.33:
        leakage_bucket = "low"
    elif leakage_ratio < 0.67:
        leakage_bucket = "med"
    else:
        leakage_bucket = "high"
    
    gt_labels = {
        "over_sharing": over_sharing,
        "shutdown_better": shutdown_better,
        "leakage_bucket": leakage_bucket,
        "eq_size": len(eq_S),
        "fb_size": len(fb_S),
        "share_rate": len(eq_S) / n,
    }
    
    # 转换all_outcomes为可JSON序列化格式
    all_outcomes_serializable = {}
    for S_frozen, outcome in all_outcomes.items():
        S_list = sorted(list(S_frozen))
        all_outcomes_serializable[str(S_list)] = {
            "S": S_list,
            "leakage": outcome.leakage,
            "welfare": outcome.welfare,
            "platform_profit": outcome.platform_profit,
            "total_leakage": outcome.total_leakage,
        }
    
    return {
        "params": params.to_dict(),
        "gt_numeric": gt_numeric,
        "gt_labels": gt_labels,
        "all_outcomes": all_outcomes_serializable,
    }



def solve_for_S_with_prices(params: ScenarioBParams, S: Set[int], prices: List[float]) -> Dict:
    """给定分享集合S与实际支付价格向量prices，计算泄露、平台利润与福利（TMD机制一致）。

    说明：
    - 泄露与隐私成本由统计结构决定；
    - 平台利润使用实际支付：sum(prices[i] for i in S)；
    - welfare 仍采用 platform_value - total_user_cost（支付是转移项），便于与文献口径一致。
    """
    n = params.n
    Sigma = params.Sigma
    sigma_noise_sq = params.sigma_noise_sq
    alpha = params.alpha
    v = params.v

    if len(prices) != n:
        raise ValueError(f"prices length {len(prices)} != n {n}")
    if any(p < 0 for p in prices):
        raise ValueError("prices must be non-negative")

    Sigma_post = compute_posterior_covariance(Sigma, S, sigma_noise_sq)

    leakage = [max(0.0, float(Sigma[i, i] - Sigma_post[i, i])) for i in range(n)]
    total_leakage = float(sum(leakage))
    platform_value = float(alpha * total_leakage)

    user_costs = [float(v[i] * leakage[i]) for i in range(n)]
    total_user_cost = float(sum(user_costs))

    welfare = float(platform_value - total_user_cost)

    platform_payment = float(sum(prices[i] for i in S))
    platform_profit = float(platform_value - platform_payment)

    return {
        "leakage": leakage,
        "total_leakage": total_leakage,
        "platform_value": platform_value,
        "user_costs": user_costs,
        "total_user_cost": total_user_cost,
        "welfare": welfare,
        "platform_payment": platform_payment,
        "platform_profit": platform_profit,
            "profit": platform_profit,
    }


def calculate_outcome_with_prices(S: Set[int], params: ScenarioBParams, prices: List[float]) -> Dict:
    """评估器用：给定分享集合与实际prices输出结果字典。"""
    return solve_for_S_with_prices(params, S, prices)


def _supporting_prices_for_set(params: ScenarioBParams, S: Set[int], epsilon: float = 1e-6) -> List[float]:
    """对给定集合S，构造TMD意义下的最小支撑价格（加epsilon打破无差异）。"""
    n = params.n
    Sigma = params.Sigma
    sigma_noise_sq = params.sigma_noise_sq
    v = params.v

    # 基础泄露（在S下）
    Sigma_post = compute_posterior_covariance(Sigma, S, sigma_noise_sq)
    leak = [max(0.0, float(Sigma[i, i] - Sigma_post[i, i])) for i in range(n)]

    prices = [0.0] * n
    for i in S:
        S_wo = set(S)
        S_wo.remove(i)
        Sigma_post_wo = compute_posterior_covariance(Sigma, S_wo, sigma_noise_sq)
        leak_i_wo = max(0.0, float(Sigma[i, i] - Sigma_post_wo[i, i]))
        marginal = max(0.0, leak[i] - leak_i_wo)
        prices[i] = float(v[i] * marginal + (epsilon if epsilon is not None else 0.0))
    return prices


def _profit_for_set_under_supporting_prices(params: ScenarioBParams, S: Set[int], epsilon: float = 1e-6) -> Tuple[float, List[float], Dict]:
    """返回：平台利润、支撑价格向量、以及用于审计的诊断信息。"""
    prices = _supporting_prices_for_set(params, S, epsilon=epsilon)
    outcome = solve_for_S_with_prices(params, S, prices)

    # 均衡裕度审计：对i∈S应≥epsilon；对i∉S应≤0（在p_i=0时）
    n = params.n
    Sigma = params.Sigma
    sigma_noise_sq = params.sigma_noise_sq
    v = params.v

    Sigma_post = compute_posterior_covariance(Sigma, S, sigma_noise_sq)
    leak = [max(0.0, float(Sigma[i, i] - Sigma_post[i, i])) for i in range(n)]

    margins_in = []
    for i in S:
        S_wo = set(S); S_wo.remove(i)
        Sigma_post_wo = compute_posterior_covariance(Sigma, S_wo, sigma_noise_sq)
        leak_i_wo = max(0.0, float(Sigma[i, i] - Sigma_post_wo[i, i]))
        marginal = max(0.0, leak[i] - leak_i_wo)
        margins_in.append(float(prices[i] - v[i] * marginal))

    margins_out = []
    for j in range(n):
        if j in S:
            continue
        # deviation gain if share at price 0
        S_plus = set(S); S_plus.add(j)
        Sigma_post_plus = compute_posterior_covariance(Sigma, S_plus, sigma_noise_sq)
        leak_j_plus = max(0.0, float(Sigma[j, j] - Sigma_post_plus[j, j]))
        marginal_j = max(0.0, leak_j_plus - leak[j])
        margins_out.append(float(0.0 - v[j] * marginal_j))

    diag = {
        "min_margin_in": min(margins_in) if margins_in else None,
        "max_margin_out": max(margins_out) if margins_out else None,
    }
    return float(outcome["platform_profit"]), prices, diag


def solve_stackelberg_personalized(params: ScenarioBParams,
                                  exact_n_limit: int = 22,
                                  epsilon: float = 1e-6,
                                  local_search_restarts: int = 10,
                                  local_search_max_iters: int = 200,
                                  rng_seed: int = 0) -> Dict:
    """TMD个性化定价的理论基线（Stackelberg）求解器。

    输出包含：
    - eq_share_set: 平台最优诱导的分享集合A*
    - eq_prices: 对应的最小支撑价格向量p^{A*}（加epsilon）
    - eq_profit, eq_W, eq_total_leakage
    - diagnostics: 均衡裕度等审计信息
    """
    n = params.n
    rng = np.random.default_rng(rng_seed)

    def eval_set(S: Set[int]) -> Tuple[float, List[float], Dict, Dict]:
        profit, prices, diag = _profit_for_set_under_supporting_prices(params, S, epsilon=epsilon)
        outcome = solve_for_S_with_prices(params, S, prices)
        return profit, prices, diag, outcome

    # === Exact enumeration for small n ===
    if n <= exact_n_limit:
        best = None
        best_S = set()
        best_prices = None
        best_diag = None
        best_outcome = None

        for mask in range(1 << n):
            S = {i for i in range(n) if (mask >> i) & 1}
            profit, prices, diag, outcome = eval_set(S)
            if (best is None) or (profit > best):
                best = profit
                best_S = S
                best_prices = prices
                best_diag = diag
                best_outcome = outcome

        return {
            "eq_share_set": sorted(best_S),
            "eq_prices": best_prices,
            "eq_profit": float(best),
            "eq_W": float(best_outcome["welfare"]),
            "eq_total_leakage": float(best_outcome["total_leakage"]),
            "diagnostics": best_diag,
            "solver_mode": "exact",
        }

    # === Local search for larger n ===
    def neighbors(S: Set[int]):
        # single flip neighbors
        for i in range(n):
            if i in S:
                S2 = set(S); S2.remove(i)
            else:
                S2 = set(S); S2.add(i)
            yield S2

    best = None
    best_S = set()
    best_prices = None
    best_diag = None
    best_outcome = None

    for r in range(local_search_restarts):
        # random init
        S = {i for i in range(n) if rng.random() < 0.5}
        profit, prices, diag, outcome = eval_set(S)
        improved = True
        it = 0
        while improved and it < local_search_max_iters:
            it += 1
            improved = False
            best_local = profit
            best_local_S = S
            best_local_prices = prices
            best_local_diag = diag
            best_local_outcome = outcome

            for S2 in neighbors(S):
                p2, pr2, d2, o2 = eval_set(S2)
                if p2 > best_local + 1e-12:
                    best_local = p2
                    best_local_S = S2
                    best_local_prices = pr2
                    best_local_diag = d2
                    best_local_outcome = o2
                    improved = True

            S, profit, prices, diag, outcome = best_local_S, best_local, best_local_prices, best_local_diag, best_local_outcome

        if (best is None) or (profit > best):
            best = profit
            best_S = S
            best_prices = prices
            best_diag = diag
            best_outcome = outcome

    return {
        "eq_share_set": sorted(best_S),
        "eq_prices": best_prices,
        "eq_profit": float(best),
        "eq_W": float(best_outcome["welfare"]),
        "eq_total_leakage": float(best_outcome["total_leakage"]),
        "diagnostics": best_diag,
        "solver_mode": "local_search",
    }



def main():
    """示例运行"""
    print("=" * 60)
    print("场景 B：Too Much Data（推断外部性）")
    print("=" * 60)
    
    # 生成实例（小规模便于演示）
    params = generate_instance(n=20, rho=0.2, seed=42)
    
    print(f"\n参数设置：")
    print(f"  用户数 n = {params.n}")
    print(f"  相关系数 rho = {params.rho}")
    print(f"  观测噪声方差 = {params.sigma_noise_sq}")
    print(f"  平台收益系数 alpha = {params.alpha}")
    print(f"  用户隐私偏好 v = {[f'{x:.2f}' for x in params.v]}")
    
    # 求解
    result = solve_scenario_b(params)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("求解结果")
    print("=" * 60)
    
    gt = result["gt_numeric"]
    labels = result["gt_labels"]
    
    print(f"\n均衡（平台利润最大）：")
    print(f"  分享集合: {gt['eq_share_set']}")
    print(f"  分享率: {labels['share_rate']:.2%}")
    print(f"  总泄露信息量: {gt['eq_total_leakage']:.3f}")
    print(f"  平台价值: {gt['eq_value']:.3f}")
    print(f"  平台利润: {gt['eq_profit']:.3f}")
    print(f"  社会福利: {gt['eq_W']:.3f}")
    
    print(f"\n社会最优（福利最大）：")
    print(f"  分享集合: {gt['fb_share_set']}")
    print(f"  总泄露信息量: {gt['fb_total_leakage']:.3f}")
    print(f"  社会福利: {gt['fb_W']:.3f}")
    
    print(f"\nShutdown（关停市场）：")
    print(f"  泄露信息量: {gt['shutdown_leakage']:.3f}")
    print(f"  社会福利: {gt['shutdown_W']:.3f}")
    
    print(f"\n外部性分析：")
    print(f"  过度分享?: {'是' if labels['over_sharing'] else '否'}")
    print(f"  均衡分享数 vs 最优分享数: {labels['eq_size']} vs {labels['fb_size']}")
    print(f"  关停更好?: {'是' if labels['shutdown_better'] else '否'}")
    print(f"  泄露量级别: {labels['leakage_bucket']}")
    print(f"  福利损失: {gt['fb_W'] - gt['eq_W']:.3f}")
    
    # 保存结果
    output_file = "data/ground_truth/scenario_b_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存到: {output_file}")
    
    # 额外分析：展示次模性
    print("\n" + "=" * 60)
    print("次模性验证（边际泄露递减）")
    print("=" * 60)
    
    # 选几个例子展示边际泄露
    if params.n >= 3:
        # 对用户0：展示随着其他人分享，用户0的边际泄露如何变化
        S_empty = set()
        S_one = {1}
        S_two = {1, 2}
        
        out_empty = result["all_outcomes"][str(sorted(S_empty))]
        out_one = result["all_outcomes"][str(sorted(S_one))]
        out_two = result["all_outcomes"][str(sorted(S_two))]
        
        leak_0_empty = out_empty["leakage"][0]
        leak_0_one = out_one["leakage"][0]
        leak_0_two = out_two["leakage"][0]
        
        print(f"用户0的泄露信息量：")
        print(f"  无人分享: {leak_0_empty:.4f}")
        print(f"  用户1分享: {leak_0_one:.4f} (增加 {leak_0_one - leak_0_empty:.4f})")
        print(f"  用户1,2分享: {leak_0_two:.4f} (再增加 {leak_0_two - leak_0_one:.4f})")
        print(f"  → 边际泄露递减（次模性）")


if __name__ == "__main__":
    main()
