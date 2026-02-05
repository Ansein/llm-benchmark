# 场景C：消费者理由关键词提取方案

## 📋 目标

**问题**：迭代历史中消费者理由文本过长，导致中介LLM提示词超长
**解决方案**：提取关键词，压缩提示词长度，同时保留核心信息

---

## 🎯 关键词词表设计原则

### 1. **分类设计**

按**参与决策**分为两大类：
- **参与理由**（PARTICIPATE_KEYWORDS）
- **不参与理由**（NOT_PARTICIPATE_KEYWORDS）

### 2. **维度设计**

每类包含多个维度，每个维度包含多个关键词模式：

#### 参与理由维度：

| 维度类别 | 说明 | 示例关键词 |
|---------|------|-----------|
| **compensation** | 经济动机 | 补偿、支付、金钱、收益 |
| **benefit** | 利益感知 | 有利、好处、价值 |
| **worth** | 价值判断 | 值得、划算、合算 |
| **data_quality** | 数据质量 | 数据质量、信号准确 |
| **low_privacy_cost** | 隐私成本低 | 隐私成本低、不在意隐私 |
| **trust** | 信任感 | 信任、放心、安全 |
| **help_others** | 利他动机 | 帮助他人、贡献社会 |
| **others_participate** | 从众效应 | 别人参与、大家都参与 |
| **anonymized** | 匿名保护 | 匿名、隐藏身份 |

#### 不参与理由维度：

| 维度类别 | 说明 | 示例关键词 |
|---------|------|-----------|
| **privacy_concern** | 隐私担忧 | 隐私、个人信息、隐私泄露 |
| **privacy_cost_high** | 隐私成本高 | 隐私成本高、隐私代价 |
| **not_trust** | 不信任 | 不信任、不放心、担心 |
| **low_compensation** | 补偿低 | 补偿太低、钱太少 |
| **not_worth** | 不值得 | 不值得、不划算 |
| **low_data_value** | 数据价值低 | 数据价值低、信息无用 |
| **identified** | 身份识别担忧 | 身份识别、暴露身份 |
| **discrimination** | 歧视担忧 | 歧视、差别对待 |

---

## 🔧 关键词词表制作流程

### 方法1：专家设计（推荐用于初始版本）

#### 步骤1：收集理论依据
参考论文中的消费者效用函数：
```
Uᵢ(参与) = -(w - θᵢ)² + m - τ
Uᵢ(不参与) = -(w₀ - θᵢ)²
```

影响因素：
- ✅ **补偿 m**：越高越倾向参与
- ✅ **隐私成本 τ**：越高越不参与
- ✅ **数据质量**：信号准确性影响定价精度
- ✅ **匿名化**：identified vs anonymized
- ✅ **参与率 r**：影响数据价值

#### 步骤2：头脑风暴关键词
基于理论因素，列举消费者可能的理由：

**参与理由**：
```python
- 补偿相关："补偿高"、"钱够多"、"收益大"
- 隐私相关："匿名保护"、"隐私成本低"
- 策略相关："别人都参与"、"数据有价值"
```

**不参与理由**：
```python
- 补偿相关："补偿太低"、"不划算"
- 隐私相关："担心隐私"、"不信任"
- 策略相关："别人不参与"、"数据无用"
```

#### 步骤3：组织为层级结构
```python
PARTICIPATE_KEYWORDS = {
    "compensation": ["补偿", "支付", "金钱", "收益", "报酬"],
    "benefit": ["有利", "好处", "收益"],
    # ...
}
```

---

### 方法2：数据驱动（用于优化迭代）

#### 步骤1：收集实际语料

运行一批LLM实验，收集消费者的理由：

```python
# 运行场景C实验
python run_scenario_c_benchmark.py --num_samples 100

# 提取理由
reasons_participate = []
reasons_not_participate = []

for record in results:
    if record['participation']:
        reasons_participate.append(record['reason'])
    else:
        reasons_not_participate.append(record['reason'])
```

#### 步骤2：统计高频词/短语

```python
from src.scenarios.scenario_c_reason_keywords import extract_frequent_phrases

# 提取参与理由的高频短语
frequent_participate = extract_frequent_phrases(
    reasons_participate,
    min_frequency=5,  # 至少出现5次
    max_phrase_length=10
)

print("高频参与理由短语：")
for phrase, count in frequent_participate[:20]:
    print(f"  {phrase}: {count}次")
```

#### 步骤3：人工审核和分类

查看高频短语，归类到合适的维度：
```
"补偿足够" → compensation
"隐私保护" → privacy_protected
"数据价值" → data_quality
```

#### 步骤4：更新词表

```python
# 添加新发现的关键词
PARTICIPATE_KEYWORDS["compensation"].extend([
    "补偿足够", "报酬合理"
])
```

---

## 🚀 使用方法

### 1. 基本使用

```python
from src.scenarios.scenario_c_reason_keywords import (
    extract_keywords_regex,
    summarize_iteration_history,
    format_keywords_for_intermediary_prompt
)

# 单条理由提取
reason = "补偿足够高，而且匿名保护很好，值得参与"
keywords = extract_keywords_regex(reason, participation=True)
print(keywords)
# 输出: ['compensation', 'anonymized', 'worth']

# 批量处理迭代历史
iteration_history = [
    {
        'iteration': 1,
        'consumer_id': 0,
        'participation': True,
        'reason': '补偿足够高，值得分享数据',
        'm': 1.0,
        'anonymization': 'anonymized'
    },
    # ... 更多记录
]

summary = summarize_iteration_history(
    iteration_history,
    use_keywords=True,
    max_keywords_per_category=5
)

# 生成中介提示词
prompt = format_keywords_for_intermediary_prompt(summary)
print(prompt)
```

**输出示例**：
```
**参与理由** (共15条):
  - compensation: 8次
  - worth: 5次
  - anonymized: 3次

**不参与理由** (共5条):
  - privacy_concern: 3次
  - low_compensation: 2次

**统计信息**:
  - 总记录数: 20
  - 参与率: 75.0%
```

### 2. 集成到中介决策流程

```python
def intermediary_decide_with_keywords(
    iteration_history,
    current_m,
    current_anonymization
):
    """中介决策（使用关键词优化）"""
    
    # 提取关键词总结
    summary = summarize_iteration_history(
        iteration_history,
        use_keywords=True
    )
    
    # 格式化为提示词
    history_summary = format_keywords_for_intermediary_prompt(summary)
    
    # 构建中介提示词
    intermediary_prompt = f"""
你是数据中介，需要决定下一轮的策略。

## 当前策略
- 补偿: {current_m}
- 匿名化: {current_anonymization}

## 历史反馈（关键词总结）
{history_summary}

## 请决策
基于以上信息，决定下一轮策略...
"""
    
    # 调用LLM
    response = call_llm(intermediary_prompt)
    return response
```

---

## 📊 效果评估

### 压缩比分析

```python
from src.scenarios.scenario_c_reason_keywords import analyze_compression_ratio

# 分析压缩效果
analysis = analyze_compression_ratio(iteration_history)

print(f"原始长度: {analysis['original_length']} 字符")
print(f"压缩后长度: {analysis['compressed_length']} 字符")
print(f"压缩比: {analysis['compression_ratio']:.1%}")
print(f"节省token数: {analysis['savings'] // 4}")  # 粗略估计
```

**预期效果**：
- **压缩比**：20-40%（保留核心信息的同时大幅压缩）
- **信息损失**：< 10%（关键决策信息基本保留）
- **Token节省**：
  - 原始：20条理由 × 30字 = 600字 ≈ 150 tokens
  - 压缩：5个关键词 × 10字 = 50字 ≈ 13 tokens
  - **节省：90% tokens**

---

## 🔄 持续优化策略

### 1. A/B测试对比

运行两组实验：
- **组A**：使用原始理由（基线）
- **组B**：使用关键词

对比指标：
- 中介决策质量（利润）
- 提示词长度
- LLM推理时间
- 成本

### 2. 逐步扩展词表

```python
# 定期分析未匹配的理由
def analyze_unmatched_reasons(iteration_history):
    """找出未被关键词覆盖的理由"""
    unmatched = []
    
    for record in iteration_history:
        reason = record['reason']
        keywords = extract_keywords_regex(
            reason,
            record['participation']
        )
        
        if len(keywords) == 0:
            unmatched.append(reason)
    
    return unmatched

# 人工审核未匹配理由，补充词表
unmatched = analyze_unmatched_reasons(history)
print(f"未匹配率: {len(unmatched)/len(history):.1%}")
print("未匹配样例:")
for reason in unmatched[:10]:
    print(f"  - {reason}")
```

### 3. 多语言支持

如果需要支持英文：
```python
PARTICIPATE_KEYWORDS_EN = {
    "compensation": ["compensation", "payment", "money", "reward"],
    "benefit": ["benefit", "advantage", "value"],
    # ...
}

def extract_keywords_multilang(reason, participation, lang='zh'):
    if lang == 'zh':
        return extract_keywords_regex(reason, participation)
    elif lang == 'en':
        # 使用英文词表
        pass
```

---

## 🎯 最佳实践

### 1. 词表设计原则

✅ **DO**:
- 基于理论模型设计维度
- 包含同义词和变体
- 使用层级结构组织
- 定期更新和扩展

❌ **DON'T**:
- 词表过于细粒度（太多类别）
- 遗漏常见表达
- 忽略领域特定术语

### 2. 匹配策略

**简单匹配** → **正则匹配** → **语义匹配**

- **简单匹配**：子串包含（快速，但可能误匹配）
- **正则匹配**：模式匹配（平衡速度和准确度，推荐）
- **语义匹配**：使用embedding（准确但慢，可选）

### 3. 压缩程度控制

```python
# 保留更多细节
summary = summarize_iteration_history(
    history,
    max_keywords_per_category=10  # 保留top10
)

# 更激进压缩
summary = summarize_iteration_history(
    history,
    max_keywords_per_category=3  # 只保留top3
)
```

---

## 🧪 运行测试

```bash
# 运行示例
python src/scenarios/scenario_c_reason_keywords.py

# 输出：
# - 关键词提取示例
# - 迭代历史总结
# - 中介提示词
# - 压缩比分析
```

---

## 📝 总结

### 优势
- ✅ **大幅减少提示词长度**（节省90% tokens）
- ✅ **保留核心决策信息**（关键因素全覆盖）
- ✅ **降低LLM成本**（tokens减少）
- ✅ **加快推理速度**（输入更短）
- ✅ **提高鲁棒性**（结构化表示）

### 局限
- ⚠️ **信息损失**（细节丢失）
- ⚠️ **需要维护词表**（人工成本）
- ⚠️ **可能误匹配**（关键词歧义）

### 建议
- 初期使用**专家设计词表** + **定期数据驱动优化**
- 监控**未匹配率**，逐步扩展词表
- 进行**A/B测试**验证效果
- 考虑**混合策略**：高频理由用关键词，低频理由保留原文

---

## 📚 参考资料

- 论文：《The Economics of Social Data》Section 2
- 实现：`src/scenarios/scenario_c_reason_keywords.py`
- 测试：`python src/scenarios/scenario_c_reason_keywords.py`
