"""
场景C：消费者理由关键词提取和匹配

优化中介提示词，将消费者理由压缩为关键词，减少提示词长度。
"""
import re
from typing import List, Dict, Set, Tuple
from collections import Counter
import json
from pathlib import Path

# 导入专家词表
from .scenario_c_keywords_vocabulary import (
    PARTICIPATE_KEYWORDS,
    NOT_PARTICIPATE_KEYWORDS,
    VOCABULARY_METADATA
)


# ========== 第2步：关键词匹配函数 ==========

def extract_keywords(reason: str, participation: bool) -> List[str]:
    """
    从消费者理由中提取关键词
    
    Args:
        reason: 消费者的理由文本
        participation: True=参与，False=不参与
    
    Returns:
        匹配到的关键词列表（类别名）
    """
    if not reason or not reason.strip():
        return []
    
    reason_lower = reason.lower()
    matched_keywords = []
    
    # 选择对应的词表
    keyword_dict = PARTICIPATE_KEYWORDS if participation else NOT_PARTICIPATE_KEYWORDS
    
    # 遍历所有类别
    for category, patterns in keyword_dict.items():
        # 检查是否匹配该类别的任何模式
        for pattern in patterns:
            if pattern.lower() in reason_lower or pattern in reason:
                matched_keywords.append(category)
                break  # 匹配到一个即可，避免重复
    
    return matched_keywords


def extract_keywords_regex(reason: str, participation: bool) -> List[str]:
    """
    使用正则表达式提取关键词（更灵活）
    
    Args:
        reason: 消费者的理由文本
        participation: True=参与，False=不参与
    
    Returns:
        匹配到的关键词列表（类别名）
    """
    if not reason or not reason.strip():
        return []
    
    matched_keywords = []
    keyword_dict = PARTICIPATE_KEYWORDS if participation else NOT_PARTICIPATE_KEYWORDS
    
    for category, patterns in keyword_dict.items():
        # 构建正则表达式：匹配任意模式
        pattern_regex = '|'.join([re.escape(p) for p in patterns])
        if re.search(pattern_regex, reason, re.IGNORECASE):
            matched_keywords.append(category)
    
    return matched_keywords


# ========== 第3步：批量处理迭代历史 ==========

def summarize_iteration_history(
    iteration_history: List[Dict],
    use_keywords: bool = True,
    max_keywords_per_category: int = 5
) -> Dict:
    """
    总结迭代历史，提取关键统计信息
    
    Args:
        iteration_history: 迭代历史列表，每个元素包含：
            {
                'iteration': int,
                'consumer_id': int,
                'participation': bool,
                'reason': str,
                'm': float,
                'anonymization': str
            }
        use_keywords: 是否使用关键词（False则保留原文）
        max_keywords_per_category: 每个类别最多保留多少个关键词
    
    Returns:
        {
            'participate_reasons': {...},  # 参与理由统计
            'not_participate_reasons': {...},  # 不参与理由统计
            'statistics': {...}  # 统计信息
        }
    """
    # 分类存储
    participate_keywords_all = []
    not_participate_keywords_all = []
    participate_reasons_raw = []
    not_participate_reasons_raw = []
    
    # 遍历历史
    for record in iteration_history:
        participation = record.get('participation', False)
        reason = record.get('reason', '')
        
        if participation:
            participate_reasons_raw.append(reason)
            if use_keywords:
                keywords = extract_keywords_regex(reason, participation=True)
                participate_keywords_all.extend(keywords)
        else:
            not_participate_reasons_raw.append(reason)
            if use_keywords:
                keywords = extract_keywords_regex(reason, participation=False)
                not_participate_keywords_all.extend(keywords)
    
    # 统计关键词频率
    if use_keywords:
        participate_counter = Counter(participate_keywords_all)
        not_participate_counter = Counter(not_participate_keywords_all)
        
        # 取topK
        top_participate = dict(participate_counter.most_common(max_keywords_per_category))
        top_not_participate = dict(not_participate_counter.most_common(max_keywords_per_category))
        
        result = {
            'participate_reasons': {
                'keywords': top_participate,
                'total_count': len(participate_reasons_raw),
                'keyword_coverage': len(participate_keywords_all) / max(len(participate_reasons_raw), 1)
            },
            'not_participate_reasons': {
                'keywords': top_not_participate,
                'total_count': len(not_participate_reasons_raw),
                'keyword_coverage': len(not_participate_keywords_all) / max(len(not_participate_reasons_raw), 1)
            },
            'statistics': {
                'total_records': len(iteration_history),
                'participation_rate': len(participate_reasons_raw) / max(len(iteration_history), 1),
                'avg_reason_length_before': sum(len(r) for r in participate_reasons_raw + not_participate_reasons_raw) / max(len(iteration_history), 1),
                'avg_keywords_per_reason': (len(participate_keywords_all) + len(not_participate_keywords_all)) / max(len(iteration_history), 1)
            }
        }
    else:
        # 不使用关键词，返回原文（可选截断）
        result = {
            'participate_reasons': {
                'reasons': participate_reasons_raw[:max_keywords_per_category],
                'total_count': len(participate_reasons_raw)
            },
            'not_participate_reasons': {
                'reasons': not_participate_reasons_raw[:max_keywords_per_category],
                'total_count': len(not_participate_reasons_raw)
            },
            'statistics': {
                'total_records': len(iteration_history),
                'participation_rate': len(participate_reasons_raw) / max(len(iteration_history), 1)
            }
        }
    
    return result


# ========== 第4步：生成中介提示词 ==========

def format_keywords_for_intermediary_prompt(summary: Dict) -> str:
    """
    将关键词总结格式化为中介LLM的提示词
    
    Args:
        summary: summarize_iteration_history的输出
    
    Returns:
        格式化的提示词文本
    """
    prompt_parts = []
    
    # 参与理由
    participate_data = summary['participate_reasons']
    if 'keywords' in participate_data:
        keywords = participate_data['keywords']
        total = participate_data['total_count']
        prompt_parts.append(f"**参与理由** (共{total}条):")
        for keyword, count in sorted(keywords.items(), key=lambda x: -x[1]):
            prompt_parts.append(f"  - {keyword}: {count}次")
    
    # 不参与理由
    not_participate_data = summary['not_participate_reasons']
    if 'keywords' in not_participate_data:
        keywords = not_participate_data['keywords']
        total = not_participate_data['total_count']
        prompt_parts.append(f"\n**不参与理由** (共{total}条):")
        for keyword, count in sorted(keywords.items(), key=lambda x: -x[1]):
            prompt_parts.append(f"  - {keyword}: {count}次")
    
    # 统计信息
    stats = summary['statistics']
    prompt_parts.append(f"\n**统计信息**:")
    prompt_parts.append(f"  - 总记录数: {stats['total_records']}")
    prompt_parts.append(f"  - 参与率: {stats['participation_rate']:.1%}")
    
    return '\n'.join(prompt_parts)


# ========== 第5步：压缩比分析 ==========

def analyze_compression_ratio(iteration_history: List[Dict]) -> Dict:
    """
    分析使用关键词后的压缩比
    
    Returns:
        {
            'original_length': int,  # 原始总字符数
            'compressed_length': int,  # 压缩后总字符数
            'compression_ratio': float,  # 压缩比
            'original_prompt': str,  # 原始提示词
            'compressed_prompt': str  # 压缩后提示词
        }
    """
    # 原始方式：拼接所有理由
    original_reasons = [record.get('reason', '') for record in iteration_history]
    original_prompt = "消费者历史理由：\n" + "\n".join(f"- {r}" for r in original_reasons)
    
    # 关键词方式
    summary = summarize_iteration_history(iteration_history, use_keywords=True)
    compressed_prompt = format_keywords_for_intermediary_prompt(summary)
    
    return {
        'original_length': len(original_prompt),
        'compressed_length': len(compressed_prompt),
        'compression_ratio': len(compressed_prompt) / max(len(original_prompt), 1),
        'savings': len(original_prompt) - len(compressed_prompt),
        'original_prompt': original_prompt,
        'compressed_prompt': compressed_prompt
    }


# ========== 第6步：自动扩展词表（从语料学习） ==========

def extract_frequent_phrases(
    reasons: List[str],
    min_frequency: int = 3,
    max_phrase_length: int = 10
) -> List[Tuple[str, int]]:
    """
    从理由语料中提取高频短语（用于扩展词表）
    
    Args:
        reasons: 理由文本列表
        min_frequency: 最小出现频率
        max_phrase_length: 最大短语长度（字符数）
    
    Returns:
        [(phrase, frequency), ...]
    """
    # 简单的n-gram提取
    from collections import Counter
    
    phrase_counter = Counter()
    
    for reason in reasons:
        # 分词（简单按空格和标点）
        tokens = re.findall(r'[\w]+', reason)
        
        # 提取1-3gram
        for n in range(1, 4):
            for i in range(len(tokens) - n + 1):
                phrase = ''.join(tokens[i:i+n])
                if len(phrase) <= max_phrase_length:
                    phrase_counter[phrase] += 1
    
    # 过滤低频短语
    frequent = [(p, c) for p, c in phrase_counter.items() if c >= min_frequency]
    frequent.sort(key=lambda x: -x[1])
    
    return frequent


# ========== 示例和测试 ==========

def example_usage():
    """使用示例"""
    # 模拟迭代历史
    iteration_history = [
        {
            'iteration': 1,
            'consumer_id': 0,
            'participation': True,
            'reason': '补偿足够高，值得分享数据，而且匿名化保护很好',
            'm': 1.0,
            'anonymization': 'anonymized'
        },
        {
            'iteration': 1,
            'consumer_id': 1,
            'participation': False,
            'reason': '隐私成本太高，不想暴露个人信息，补偿也不够',
            'm': 1.0,
            'anonymization': 'identified'
        },
        {
            'iteration': 2,
            'consumer_id': 0,
            'participation': True,
            'reason': '看到别人也参与了，我也跟随参与，数据质量很重要',
            'm': 1.2,
            'anonymization': 'anonymized'
        },
        {
            'iteration': 2,
            'consumer_id': 1,
            'participation': False,
            'reason': '补偿太低，不划算，而且担心被识别',
            'm': 1.2,
            'anonymization': 'anonymized'
        },
        {
            'iteration': 3,
            'consumer_id': 2,
            'participation': True,
            'reason': '信任中介，收益大于成本，帮助提升市场效率',
            'm': 1.5,
            'anonymization': 'anonymized'
        },
        {
            'iteration': 3,
            'consumer_id': 3,
            'participation': False,
            'reason': '隐私泄露风险大，不信任平台，观望一下',
            'm': 1.5,
            'anonymization': 'identified'
        },
    ]
    
    print("="*80)
    print("关键词提取示例（使用专家词表：933个关键词，86个类别）")
    print("="*80)
    
    # 显示词表信息
    print(f"\n词表统计:")
    print(f"  参与理由: {VOCABULARY_METADATA['total_categories']['participate']}类, "
          f"{VOCABULARY_METADATA['total_keywords']['participate']}词")
    print(f"  不参与理由: {VOCABULARY_METADATA['total_categories']['not_participate']}类, "
          f"{VOCABULARY_METADATA['total_keywords']['not_participate']}词")
    
    # 1. 提取关键词
    print("\n" + "-"*80)
    print("单条理由关键词提取:")
    print("-"*80)
    for record in iteration_history:
        reason = record['reason']
        participation = record['participation']
        keywords = extract_keywords_regex(reason, participation)
        print(f"\n理由: {reason}")
        print(f"参与: {'是' if participation else '否'}")
        print(f"关键词: {keywords}")
    
    # 2. 总结历史
    print("\n" + "="*80)
    print("迭代历史总结（关键词模式）")
    print("="*80)
    summary = summarize_iteration_history(iteration_history, use_keywords=True, max_keywords_per_category=10)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 3. 生成提示词
    print("\n" + "="*80)
    print("中介提示词（用于LLM）")
    print("="*80)
    prompt = format_keywords_for_intermediary_prompt(summary)
    print(prompt)
    
    # 4. 压缩比分析
    print("\n" + "="*80)
    print("压缩效果分析")
    print("="*80)
    analysis = analyze_compression_ratio(iteration_history)
    print(f"原始长度: {analysis['original_length']} 字符")
    print(f"压缩后长度: {analysis['compressed_length']} 字符")
    print(f"压缩比: {analysis['compression_ratio']:.1%}")
    print(f"节省字符数: {analysis['savings']}")
    print(f"估计节省Token数: {analysis['savings'] // 4}")  # 粗略估计
    
    if analysis['savings'] > 0:
        print(f"[OK] 成功压缩 {100 - analysis['compression_ratio']*100:.1f}%")
    else:
        print(f"[WARNING] 样本太少，未体现压缩优势（需要更多迭代记录）")
    
    print("\n原始提示词（前200字符）:")
    print(analysis['original_prompt'][:200] + "...")
    
    print("\n压缩后提示词:")
    print(analysis['compressed_prompt'])


if __name__ == "__main__":
    example_usage()
