"""
场景C：中介LLM提示词生成器

基于论文《The Economics of Social Data》实现中介决策提示词
集成关键词提取，优化提示词长度
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .scenario_c_reason_keywords import (
    summarize_iteration_history,
    format_keywords_for_intermediary_prompt,
    analyze_compression_ratio
)


@dataclass
class IntermediaryContext:
    """中介决策上下文"""
    # 当前策略
    current_m: np.ndarray  # (N,) 个性化补偿
    current_anonymization: str  # 'identified' or 'anonymized'
    
    # 市场状态
    current_iteration: int
    current_participation_rate: float
    current_profit: float
    
    # 迭代历史
    iteration_history: List[Dict]  # 消费者理由历史
    
    # 理论参数（用于提示词）
    N: int  # 消费者数量
    theta_prior_mean: float  # 先验均值
    theta_prior_std: float  # 先验标准差
    tau_mean: float  # 平均隐私成本
    
    # 优化约束
    profit_constraint: bool = True  # 是否强制R>0
    m_bounds: Tuple[float, float] = (0.0, 5.0)  # m的取值范围


def generate_intermediary_prompt_with_keywords(
    context: IntermediaryContext,
    use_keywords: bool = True,
    max_keywords_per_category: int = 5,
    include_theory: bool = True,
    include_statistics: bool = True,
    language: str = 'zh'
) -> str:
    """
    生成中介决策提示词（使用关键词优化）
    
    Args:
        context: 中介决策上下文
        use_keywords: 是否使用关键词压缩（False则使用原始理由）
        max_keywords_per_category: 每个类别最多保留多少个关键词
        include_theory: 是否包含理论背景
        include_statistics: 是否包含统计信息
        language: 语言 ('zh' or 'en')
    
    Returns:
        提示词字符串
    """
    
    if language == 'zh':
        return _generate_prompt_zh(
            context, use_keywords, max_keywords_per_category,
            include_theory, include_statistics
        )
    else:
        return _generate_prompt_en(
            context, use_keywords, max_keywords_per_category,
            include_theory, include_statistics
        )


def _generate_prompt_zh(
    context: IntermediaryContext,
    use_keywords: bool,
    max_keywords_per_category: int,
    include_theory: bool,
    include_statistics: bool
) -> str:
    """生成中文提示词"""
    
    prompt_parts = []
    
    # ========== 角色定义 ==========
    prompt_parts.append("""# 角色：数据中介（Data Intermediary）

你是一个理性的数据中介，在社会数据市场中扮演关键角色。你的目标是最大化利润，同时考虑消费者的参与决策和生产者的数据需求。

## 你的任务
基于当前市场状态、消费者历史反馈和理论模型，决定下一轮的策略：
1. **补偿策略 (m_i)**: 给每个消费者的补偿（可以个性化）
2. **匿名化策略**: 选择 'identified'（实名）或 'anonymized'（匿名）

## 约束条件
""")
    
    if context.profit_constraint:
        prompt_parts.append(f"- **利润约束**: 必须保证 R = m_0 - Σ(m_i × a_i) > 0，否则不购买数据")
    
    prompt_parts.append(f"- **补偿范围**: {context.m_bounds[0]} ≤ m_i ≤ {context.m_bounds[1]}")
    
    # ========== 理论背景 ==========
    if include_theory:
        prompt_parts.append(f"""
---

## 理论框架

### 消费者效用函数
- **参与**: U_i(参与) = -(w - theta_i)^2 + m_i - tau_i
- **不参与**: U_i(不参与) = -(w_0 - theta_i)^2

其中：
- theta_i: 消费者i的真实偏好类型（未知）
- w: 生产者基于数据的定价
- w_0: 无数据时的定价
- m_i: 你给消费者i的补偿
- tau_i: 消费者i的隐私成本

### 理性参与条件
消费者i参与当且仅当：**delta_U_i = U_i(参与) - U_i(不参与) > 0**

关键影响因素：
1. **补偿 m_i**: 越高越倾向参与
2. **隐私成本 tau_i**: 越高越不倾向参与（平均 {context.tau_mean:.2f}）
3. **数据质量**: 信号越准确，数据价值越高
4. **参与率 r**: 影响数据价值和消费者预期
5. **匿名化策略**:
   - **identified**: 生产者可以个性化定价，隐私成本tau更高
   - **anonymized**: 生产者只能统一定价，隐私成本tau更低

### 你的利润函数
R = m_0 - sum(m_i * a_i)

其中：
- m_0: 生产者愿意支付的价格（取决于数据价值）
- a_i: 消费者i是否参与 (0 or 1)

**关键权衡**:
- 提高m_i -> 更多人参与 -> 数据价值增加 -> m_0增加，但成本也增加
- 选择anonymized -> 隐私成本降低 -> 参与率增加，但m_0降低（因为无法个性化定价）
""")
    
    # ========== 当前市场状态 ==========
    prompt_parts.append(f"""
---

## 当前市场状态

### 策略参数
- **迭代轮次**: 第 {context.current_iteration} 轮
- **当前补偿**: 平均 m = {np.mean(context.current_m):.3f}，标准差 = {np.std(context.current_m):.3f}
- **补偿范围**: [{np.min(context.current_m):.3f}, {np.max(context.current_m):.3f}]
- **匿名化策略**: {context.current_anonymization}

### 市场表现
- **参与率**: {context.current_participation_rate:.1%} ({int(context.current_participation_rate * context.N)}/{context.N} 人)
- **中介利润**: R = {context.current_profit:.3f}
""")
    
    if context.current_profit <= 0:
        prompt_parts.append(f"  [WARNING] **警告**: 当前利润为负或零！必须调整策略。")
    
    # ========== 消费者反馈历史 ==========
    if len(context.iteration_history) > 0:
        prompt_parts.append("\n---\n\n## 消费者历史反馈\n")
        
        if use_keywords:
            # 使用关键词总结
            summary = summarize_iteration_history(
                context.iteration_history,
                use_keywords=True,
                max_keywords_per_category=max_keywords_per_category
            )
            
            # 格式化关键词
            keywords_text = format_keywords_for_intermediary_prompt(summary)
            prompt_parts.append(keywords_text)
            
            # 压缩效果统计
            if include_statistics:
                analysis = analyze_compression_ratio(context.iteration_history)
                prompt_parts.append(f"\n**提示词压缩**: 原始 {analysis['original_length']} 字符 → "
                                  f"压缩 {analysis['compressed_length']} 字符 "
                                  f"(压缩比 {analysis['compression_ratio']:.1%})")
        else:
            # 使用原始理由（截断）
            max_display = 10
            prompt_parts.append(f"**最近 {min(max_display, len(context.iteration_history))} 条反馈**:\n")
            
            for i, record in enumerate(context.iteration_history[-max_display:], 1):
                participation_text = "✓ 参与" if record['participation'] else "✗ 不参与"
                prompt_parts.append(
                    f"{i}. 消费者{record['consumer_id']}: {participation_text} - {record['reason']}"
                )
    else:
        prompt_parts.append("\n---\n\n## 消费者历史反馈\n\n（首轮决策，暂无历史反馈）\n")
    
    # ========== 统计信息 ==========
    if include_statistics and len(context.iteration_history) > 0:
        # 计算趋势
        recent_participation = [r['participation'] for r in context.iteration_history[-context.N:]]
        recent_rate = sum(recent_participation) / len(recent_participation) if recent_participation else 0
        
        prompt_parts.append(f"""
---

## 趋势分析

- **总反馈记录数**: {len(context.iteration_history)}
- **最近一轮参与率**: {recent_rate:.1%}
- **参与率变化**: {(context.current_participation_rate - recent_rate):.1%}
""")
    
    # ========== 决策提示 ==========
    prompt_parts.append("""
---

## 请做出决策

基于以上信息，请回答以下问题：

### 1. 策略分析
- 当前策略的主要问题是什么？
- 消费者反馈揭示了哪些关键因素？
- 参与者和不参与者的主要区别是什么？

### 2. 下一轮策略
请提出下一轮的具体策略，包括：

**补偿策略 (m_i)**:
- 建议的平均补偿 m_avg = ?
- 是否需要个性化？如果需要，如何设计差异化？
- 理由：...

**匿名化策略**:
- 选择 'identified' 还是 'anonymized'？
- 理由：...

### 3. 预期效果
- 预期参与率：...
- 预期利润：...
- 风险分析：...

---

**注意**:
1. 必须给出具体的数值建议，不要只说"提高"或"降低"
2. 考虑利润约束 R > 0
3. 平衡短期利润和长期可持续性
4. 基于消费者反馈的关键词，针对性调整策略
""")
    
    return '\n'.join(prompt_parts)


def _generate_prompt_en(
    context: IntermediaryContext,
    use_keywords: bool,
    max_keywords_per_category: int,
    include_theory: bool,
    include_statistics: bool
) -> str:
    """生成英文提示词（简化版本）"""
    
    # 简化实现，主要结构类似中文版
    prompt = f"""# Role: Data Intermediary

You are a rational data intermediary in a social data market.

## Current Strategy
- Iteration: {context.current_iteration}
- Average compensation: m_avg = {np.mean(context.current_m):.3f}
- Anonymization: {context.current_anonymization}
- Participation rate: {context.current_participation_rate:.1%}
- Profit: R = {context.current_profit:.3f}

## Consumer Feedback
"""
    
    if len(context.iteration_history) > 0 and use_keywords:
        summary = summarize_iteration_history(
            context.iteration_history,
            use_keywords=True,
            max_keywords_per_category=max_keywords_per_category
        )
        keywords_text = format_keywords_for_intermediary_prompt(summary)
        prompt += keywords_text
    
    prompt += """

## Decision Required
Based on the above information, decide:
1. Next round's compensation (m_i)
2. Anonymization strategy ('identified' or 'anonymized')

Provide specific numerical recommendations and reasoning.
"""
    
    return prompt


# ========== 辅助函数 ==========

def create_intermediary_context_from_result(
    iteration: int,
    m_current: np.ndarray,
    anonymization_current: str,
    participation_rate: float,
    profit: float,
    iteration_history: List[Dict],
    N: int,
    theta_prior_mean: float = 50.0,
    theta_prior_std: float = 10.0,
    tau_mean: float = 2.0,
    profit_constraint: bool = True,
    m_bounds: Tuple[float, float] = (0.0, 5.0)
) -> IntermediaryContext:
    """
    从优化结果创建中介上下文
    
    便捷函数，用于从场景C的结果中提取信息并构建上下文
    """
    return IntermediaryContext(
        current_m=m_current,
        current_anonymization=anonymization_current,
        current_iteration=iteration,
        current_participation_rate=participation_rate,
        current_profit=profit,
        iteration_history=iteration_history,
        N=N,
        theta_prior_mean=theta_prior_mean,
        theta_prior_std=theta_prior_std,
        tau_mean=tau_mean,
        profit_constraint=profit_constraint,
        m_bounds=m_bounds
    )


def compare_prompt_lengths(
    context: IntermediaryContext,
    max_keywords_per_category: int = 5
) -> Dict:
    """
    比较使用关键词vs不使用关键词的提示词长度
    
    Returns:
        {
            'with_keywords': str,
            'without_keywords': str,
            'with_keywords_length': int,
            'without_keywords_length': int,
            'compression_ratio': float,
            'tokens_saved_estimate': int
        }
    """
    prompt_with_kw = generate_intermediary_prompt_with_keywords(
        context, use_keywords=True, max_keywords_per_category=max_keywords_per_category
    )
    prompt_without_kw = generate_intermediary_prompt_with_keywords(
        context, use_keywords=False
    )
    
    len_with = len(prompt_with_kw)
    len_without = len(prompt_without_kw)
    
    return {
        'with_keywords': prompt_with_kw,
        'without_keywords': prompt_without_kw,
        'with_keywords_length': len_with,
        'without_keywords_length': len_without,
        'compression_ratio': len_with / len_without if len_without > 0 else 1.0,
        'savings': len_without - len_with,
        'tokens_saved_estimate': (len_without - len_with) // 4  # 粗略估计
    }


# ========== 测试和示例 ==========

def example_intermediary_prompt():
    """示例：生成中介提示词"""
    
    # 模拟上下文
    N = 20
    iteration_history = [
        {
            'iteration': 1,
            'consumer_id': i,
            'participation': i % 3 != 0,  # 约2/3参与
            'reason': '补偿足够高，值得分享数据' if i % 3 != 0 else '隐私成本太高，补偿不够',
            'm': 1.0,
            'anonymization': 'anonymized'
        }
        for i in range(N)
    ] + [
        {
            'iteration': 2,
            'consumer_id': i,
            'participation': i % 2 == 0,  # 1/2参与
            'reason': '信任中介，收益大于成本' if i % 2 == 0 else '隐私泄露风险大，观望一下',
            'm': 1.2,
            'anonymization': 'identified'
        }
        for i in range(N)
    ]
    
    context = IntermediaryContext(
        current_m=np.full(N, 1.2),
        current_anonymization='identified',
        current_iteration=2,
        current_participation_rate=0.5,
        current_profit=0.5,
        iteration_history=iteration_history,
        N=N,
        theta_prior_mean=50.0,
        theta_prior_std=10.0,
        tau_mean=2.0
    )
    
    print("="*80)
    print("中介提示词生成示例")
    print("="*80)
    
    # 生成提示词（使用关键词）
    prompt = generate_intermediary_prompt_with_keywords(context, use_keywords=True)
    
    print("\n提示词（使用关键词压缩）:")
    print("-"*80)
    print(prompt)
    print("-"*80)
    
    # 比较长度
    print("\n" + "="*80)
    print("压缩效果对比")
    print("="*80)
    
    comparison = compare_prompt_lengths(context)
    print(f"使用关键词: {comparison['with_keywords_length']} 字符")
    print(f"不使用关键词: {comparison['without_keywords_length']} 字符")
    print(f"压缩比: {comparison['compression_ratio']:.1%}")
    print(f"节省字符: {comparison['savings']}")
    print(f"估计节省Token: {comparison['tokens_saved_estimate']}")


if __name__ == "__main__":
    example_intermediary_prompt()
