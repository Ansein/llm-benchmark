"""
场景C评估指标计算函数

所有指标都是完全量化的、客观的，基于行为数据与理论解的对比。
不包含主观评分、文本质量评估等需要人工判断的指标。
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import spearmanr, kendalltau


def compute_participation_metrics(
    llm_decisions: List[bool],
    theory_decisions: List[bool],
    r_theory: float
) -> Dict[str, float]:
    """
    计算参与率相关指标
    
    Args:
        llm_decisions: LLM的参与决策（N个布尔值）
        theory_decisions: 理论决策（N个布尔值）
        r_theory: 理论参与率
    
    Returns:
        包含所有参与率指标的字典
    """
    llm_decisions = np.array(llm_decisions, dtype=bool)
    theory_decisions = np.array(theory_decisions, dtype=bool)
    N = len(llm_decisions)
    
    # 1. 总体参与率
    r_llm = float(np.mean(llm_decisions))
    r_absolute_error = abs(r_llm - r_theory)
    r_relative_error = r_absolute_error / r_theory if r_theory > 0 else float('inf')
    
    # 2. 个体决策准确率
    correct_decisions = (llm_decisions == theory_decisions)
    individual_accuracy = float(np.mean(correct_decisions))
    
    # 3. 混淆矩阵指标
    TP = float(np.sum(theory_decisions & llm_decisions))
    TN = float(np.sum(~theory_decisions & ~llm_decisions))
    FP = float(np.sum(~theory_decisions & llm_decisions))
    FN = float(np.sum(theory_decisions & ~llm_decisions))
    
    # 避免除零
    n_positive = TP + FN
    n_negative = TN + FP
    
    true_positive_rate = TP / n_positive if n_positive > 0 else 0.0
    true_negative_rate = TN / n_negative if n_negative > 0 else 0.0
    false_positive_rate = FP / n_negative if n_negative > 0 else 0.0
    false_negative_rate = FN / n_positive if n_positive > 0 else 0.0
    
    return {
        "r_llm": r_llm,
        "r_theory": r_theory,
        "r_absolute_error": r_absolute_error,
        "r_relative_error": r_relative_error,
        "individual_accuracy": individual_accuracy,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        }
    }


def compute_consumer_metrics(
    llm_decisions: List[bool],
    theory_decisions: List[bool],
    outcome_llm: Dict[str, float],
    outcome_theory: Dict[str, float]
) -> Dict[str, float]:
    """
    计算面向消费者的量化指标
    
    Args:
        llm_decisions: LLM消费者决策
        theory_decisions: 理论决策
        outcome_llm: LLM场景市场结果
        outcome_theory: 理论场景市场结果
    
    Returns:
        消费者指标字典
    """
    participation = compute_participation_metrics(
        llm_decisions=llm_decisions,
        theory_decisions=theory_decisions,
        r_theory=float(np.mean(theory_decisions)) if len(theory_decisions) > 0 else 0.0
    )
    cs_llm = outcome_llm.get("consumer_surplus", 0.0)
    cs_theory = outcome_theory.get("consumer_surplus", 0.0)
    gini_llm = outcome_llm.get("gini_coefficient", 0.0)

    return {
        "individual_accuracy": participation["individual_accuracy"],
        "decision_confusion_matrix": participation["confusion_matrix"],
        "consumer_surplus_gap": cs_llm - cs_theory,
        "gini_consumer_surplus": gini_llm
    }


def compute_market_metrics(
    outcome_llm: Dict[str, float],
    outcome_theory: Dict[str, float]
) -> Dict[str, float]:
    """
    计算市场结果指标（福利、利润等）
    
    Args:
        outcome_llm: LLM场景下的市场结果
        outcome_theory: 理论场景下的市场结果
    
    Returns:
        市场指标字典
    """
    # 提取数值
    sw_llm = outcome_llm['social_welfare']
    sw_theory = outcome_theory['social_welfare']
    cs_llm = outcome_llm['consumer_surplus']
    cs_theory = outcome_theory['consumer_surplus']
    pp_llm = outcome_llm['producer_profit']
    pp_theory = outcome_theory['producer_profit']
    ip_llm = outcome_llm.get('intermediary_profit', 0.0)
    ip_theory = outcome_theory.get('intermediary_profit', 0.0)
    
    # 计算指标
    return {
        # 绝对值
        "social_welfare_llm": sw_llm,
        "social_welfare_theory": sw_theory,
        "social_welfare_diff": sw_llm - sw_theory,
        
        "consumer_surplus_llm": cs_llm,
        "consumer_surplus_theory": cs_theory,
        "consumer_surplus_diff": cs_llm - cs_theory,
        
        "producer_profit_llm": pp_llm,
        "producer_profit_theory": pp_theory,
        "producer_profit_diff": pp_llm - pp_theory,
        
        "intermediary_profit_llm": ip_llm,
        "intermediary_profit_theory": ip_theory,
        "intermediary_profit_diff": ip_llm - ip_theory,
        
        # 相对值
        "social_welfare_ratio": sw_llm / sw_theory if sw_theory > 0 else 0.0,
        "consumer_surplus_ratio": cs_llm / cs_theory if cs_theory > 0 else 0.0,
        "producer_profit_ratio": pp_llm / pp_theory if pp_theory > 0 else 0.0,
        "intermediary_profit_ratio": ip_llm / ip_theory if ip_theory > 0 else 0.0,
        
        # 效率损失
        "welfare_loss": max(0.0, sw_theory - sw_llm),
        "welfare_loss_percent": (sw_theory - sw_llm) / sw_theory * 100 if sw_theory > 0 else 0.0,
    }


def compute_inequality_metrics(
    outcome_llm: Dict[str, float],
    outcome_theory: Dict[str, float]
) -> Dict[str, float]:
    """
    计算不平等指标
    
    Args:
        outcome_llm: LLM场景下的市场结果
        outcome_theory: 理论场景下的市场结果
    
    Returns:
        不平等指标字典
    """
    gini_llm = outcome_llm.get('gini_coefficient', 0.0)
    gini_theory = outcome_theory.get('gini_coefficient', 0.0)
    pv_llm = outcome_llm.get('price_variance', 0.0)
    pv_theory = outcome_theory.get('price_variance', 0.0)
    pdi_llm = outcome_llm.get('price_discrimination_index', 0.0)
    pdi_theory = outcome_theory.get('price_discrimination_index', 0.0)
    
    return {
        "gini_llm": gini_llm,
        "gini_theory": gini_theory,
        "gini_diff": gini_llm - gini_theory,
        
        "price_variance_llm": pv_llm,
        "price_variance_theory": pv_theory,
        "price_variance_diff": pv_llm - pv_theory,
        
        "price_discrimination_index_llm": pdi_llm,
        "price_discrimination_index_theory": pdi_theory,
        "pdi_diff": pdi_llm - pdi_theory,
    }


def compute_strategy_metrics(
    m_llm: float,
    anon_llm: str,
    m_theory: float,
    anon_theory: str
) -> Dict[str, float]:
    """
    计算中介策略指标
    
    Args:
        m_llm: LLM选择的补偿（标量或向量）
        anon_llm: LLM选择的匿名化策略
        m_theory: 理论最优补偿（标量或向量）
        anon_theory: 理论最优匿名化策略
    
    Returns:
        策略指标字典
    """
    import numpy as np
    
    # 处理m_theory（可能是向量）
    if isinstance(m_theory, (list, np.ndarray)):
        m_theory_scalar = float(np.mean(m_theory))  # 使用均值对比
        m_theory_std = float(np.std(m_theory))
    else:
        m_theory_scalar = float(m_theory)
        m_theory_std = 0.0
    
    # 处理m_llm（可能是向量）
    if isinstance(m_llm, (list, np.ndarray)):
        m_llm_scalar = float(np.mean(m_llm))
        m_llm_std = float(np.std(m_llm))
    else:
        m_llm_scalar = float(m_llm)
        m_llm_std = 0.0
    
    m_absolute_error = abs(m_llm_scalar - m_theory_scalar)
    m_relative_error = m_absolute_error / m_theory_scalar if m_theory_scalar > 0 else float('inf')
    anon_match = int(anon_llm == anon_theory)
    strategy_match = int((m_absolute_error < 0.01) and (anon_llm == anon_theory))
    
    return {
        "m_llm": m_llm_scalar,
        "m_llm_std": m_llm_std,
        "m_theory": m_theory_scalar,
        "m_theory_std": m_theory_std,
        "m_absolute_error": m_absolute_error,
        "m_relative_error": m_relative_error,
        
        "anon_llm": anon_llm,
        "anon_theory": anon_theory,
        "anon_match": anon_match,
        
        "strategy_match": strategy_match,
    }


def compute_profit_metrics(
    profit_llm: float,
    profit_theory: float,
    cost_llm: float = None,
    cost_theory: float = None
) -> Dict[str, float]:
    """
    计算中介利润指标
    
    Args:
        profit_llm: LLM策略下的利润
        profit_theory: 理论最优利润
        cost_llm: LLM策略下的成本（可选）
        cost_theory: 理论成本（可选）
    
    Returns:
        利润指标字典
    """
    import numpy as np
    
    # 处理成本（可能是向量，需要转换为标量）
    if cost_llm is not None and isinstance(cost_llm, (list, np.ndarray)):
        cost_llm_scalar = float(np.sum(cost_llm))  # 总成本
    elif cost_llm is not None:
        cost_llm_scalar = float(cost_llm)
    else:
        cost_llm_scalar = None
        
    if cost_theory is not None and isinstance(cost_theory, (list, np.ndarray)):
        cost_theory_scalar = float(np.sum(cost_theory))  # 总成本
    elif cost_theory is not None:
        cost_theory_scalar = float(cost_theory)
    else:
        cost_theory_scalar = None
    
    profit_diff = profit_llm - profit_theory
    profit_ratio = profit_llm / profit_theory if profit_theory != 0 else 0.0
    profit_loss = max(0.0, profit_theory - profit_llm)
    profit_loss_percent = profit_loss / profit_theory * 100 if profit_theory > 0 else 0.0
    
    metrics = {
        "profit_llm": profit_llm,
        "profit_theory": profit_theory,
        "profit_diff": profit_diff,
        "profit_ratio": profit_ratio,
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent,
    }
    
    # 添加成本指标（如果提供）
    if cost_llm_scalar is not None and cost_theory_scalar is not None:
        metrics["cost_llm"] = cost_llm_scalar
        metrics["cost_theory"] = cost_theory_scalar
        metrics["cost_efficiency"] = cost_llm_scalar / cost_theory_scalar if cost_theory_scalar > 0 else 0.0
    
    return metrics


def compute_ranking_metrics(
    llm_ranking: List[int],
    theory_ranking: List[int]
) -> Dict[str, float]:
    """
    计算策略排序能力指标
    
    Args:
        llm_ranking: LLM的策略排序（按偏好从高到低）
        theory_ranking: 理论排序（按利润从高到低）
    
    Returns:
        排序指标字典
    """
    # Spearman相关系数
    spearman_corr, _ = spearmanr(llm_ranking, theory_ranking)
    
    # Kendall Tau相关系数
    kendall_tau_corr, _ = kendalltau(llm_ranking, theory_ranking)
    
    # Top-k准确率
    top_1_match = int(llm_ranking[0] == theory_ranking[0])
    top_2_match = int(theory_ranking[0] in llm_ranking[:2])
    top_3_match = int(theory_ranking[0] in llm_ranking[:3])
    
    return {
        "spearman_correlation": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "kendall_tau": float(kendall_tau_corr) if not np.isnan(kendall_tau_corr) else 0.0,
        "top_1_accuracy": top_1_match,
        "top_2_accuracy": top_2_match,
        "top_3_accuracy": top_3_match,
        "identified_best": top_1_match,
    }


def compute_interaction_metrics(
    outcome_D: Dict[str, float],
    outcome_A: Dict[str, float],
    outcome_B: Dict[str, float] = None,
    outcome_C: Dict[str, float] = None
) -> Dict[str, float]:
    """
    计算配置D的交互指标
    
    Args:
        outcome_D: 配置D（LLM×LLM）的结果
        outcome_A: 配置A（理性×理性）的结果
        outcome_B: 配置B（理性中介×LLM消费者）的结果（可选）
        outcome_C: 配置C（LLM中介×理性消费者）的结果（可选）
    
    Returns:
        交互指标字典
    """
    metrics = {}
    
    # 与理论解对比
    sw_D = outcome_D['social_welfare']
    sw_A = outcome_A['social_welfare']
    cs_D = outcome_D['consumer_surplus']
    cs_A = outcome_A['consumer_surplus']
    ip_D = outcome_D.get('intermediary_profit', 0.0)
    ip_A = outcome_A.get('intermediary_profit', 0.0)
    
    metrics["vs_theory"] = {
        "social_welfare_ratio": sw_D / sw_A if sw_A > 0 else 0.0,
        "welfare_loss": max(0.0, sw_A - sw_D),
        "welfare_loss_percent": (sw_A - sw_D) / sw_A * 100 if sw_A > 0 else 0.0,
        "cs_ratio": cs_D / cs_A if cs_A > 0 else 0.0,
        "ip_ratio": ip_D / ip_A if ip_A > 0 else 0.0,
    }
    
    # 剥削指标
    # >1 表示中介利润占比增加快于消费者剩余占比
    exploitation_indicator = (ip_D / ip_A) / (cs_D / cs_A) if (ip_A > 0 and cs_A > 0) else 1.0
    metrics["exploitation_indicator"] = exploitation_indicator
    
    # 与单边LLM对比
    if outcome_B is not None:
        sw_B = outcome_B['social_welfare']
        cs_B = outcome_B['consumer_surplus']
        
        metrics["vs_config_B"] = {
            "welfare_diff": sw_D - sw_B,
            "consumer_better_off": int(cs_D > cs_B),
        }
    
    if outcome_C is not None:
        sw_C = outcome_C['social_welfare']
        ip_C = outcome_C.get('intermediary_profit', 0.0)
        
        metrics["vs_config_C"] = {
            "welfare_diff": sw_D - sw_C,
            "intermediary_better_off": int(ip_D > ip_C),
        }
    
    # 交互效应（如果有B和C）
    if outcome_B is not None and outcome_C is not None:
        sw_B = outcome_B['social_welfare']
        sw_C = outcome_C['social_welfare']
        
        # 交互效应 = D的偏差 - (B的偏差 + C的偏差)
        interaction_effect = (sw_A - sw_D) - ((sw_A - sw_B) + (sw_A - sw_C))
        metrics["interaction_effect_welfare"] = interaction_effect
    
    return metrics


def compute_summary_statistics(metrics_dict: Dict) -> Dict[str, float]:
    """
    计算指标的汇总统计
    
    Args:
        metrics_dict: 包含所有指标的字典
    
    Returns:
        汇总统计字典
    """
    # 提取所有数值型指标
    def extract_numeric_values(d, prefix=""):
        values = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float, np.number)):
                values[full_key] = float(value)
            elif isinstance(value, dict):
                values.update(extract_numeric_values(value, full_key))
        return values
    
    numeric_values = extract_numeric_values(metrics_dict)
    
    # 计算统计量
    values_array = np.array(list(numeric_values.values()))
    
    return {
        "num_metrics": len(numeric_values),
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array)),
    }
