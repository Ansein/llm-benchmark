# 场景C评估器使用指南

## 概述

场景C评估器用于评估LLM代理在数据市场中的表现，特别是测试LLM对隐私外部性机制的理解能力。

### 支持的配置

| 配置 | 中介 | 消费者 | 用途 |
|------|------|--------|------|
| **A** | 理性 | 理性 | Ground Truth（理论基准） |
| **B** | 理性 | LLM | 测试消费者决策理解 |
| **C** | LLM | 理性 | 测试中介策略优化 |
| **D** | LLM | LLM | 测试LLM-LLM交互 |

---

## 快速开始

### 方式1：直接运行（推荐，真实LLM）⭐

```bash
# 1. 生成Ground Truth（如果还没有）
python -m src.scenarios.generate_scenario_c_gt

# 2. 直接运行评估器（使用 configs/model_configs.json 中配置的真实LLM）
python src/evaluators/evaluate_scenario_c.py
```

这会运行评估并输出 CSV/JSON 到 `evaluation_results/`。

#### 选择要评估的模型

在 `src/evaluators/evaluate_scenario_c.py` 中修改：

- `TARGET_MODEL = "gpt-4.1-mini"`

它会按 `configs/model_configs.json` 中的 `config_name` 精确匹配。

---

### 方式2：自定义LLM代理

### 1. 准备Ground Truth

首先生成理论基准（配置A）：

```bash
python -m src.scenarios.generate_scenario_c_gt
```

这会生成：
- `data/ground_truth/scenario_c_common_preferences_optimal.json`
- `data/ground_truth/scenario_c_common_experience_optimal.json`

### 2. 创建LLM代理

实现LLM消费者代理（示例）：

```python
def my_llm_consumer(consumer_params, m, anonymization):
    """
    LLM消费者决策函数
    
    Args:
        consumer_params: dict，包含 {theta_i, tau_i, w_i (可选)}
        m: float，补偿金额
        anonymization: str，"identified" 或 "anonymized"
    
    Returns:
        bool: True表示参与，False表示拒绝
    """
    # 调用你的LLM API
    prompt = f"""
    你是一个消费者，面临数据分享决策。
    
    你的参数：
    - 效用参数 θ = {consumer_params['theta_i']}
    - 隐私成本 τ = {consumer_params['tau_i']}
    
    数据中介提供：
    - 补偿金额：{m}
    - 隐私保护：{anonymization}
    
    请决定：是否参与数据分享？（仅回答"是"或"否"）
    """
    
    response = call_llm_api(prompt)
    return "是" in response or "参与" in response
```

实现LLM中介代理（示例）：

```python
def my_llm_intermediary(market_params):
    """
    LLM中介策略选择函数
    
    Args:
        market_params: dict，包含 {N, mu_theta, sigma_theta, tau_mean, tau_std, ...}
    
    Returns:
        tuple: (m, anonymization)
    """
    prompt = f"""
    你是数据中介，需要选择策略以最大化利润。
    
    市场参数：
    - 消费者数量：{market_params['N']}
    - 偏好均值：{market_params['mu_theta']}
    - 隐私成本均值：{market_params['tau_mean']}
    
    请选择：
    1. 补偿金额 m（建议范围0-3）
    2. 隐私策略（identified 或 anonymized）
    
    请以JSON格式回答：{{"m": 数值, "anonymization": "策略"}}
    """
    
    response = call_llm_api(prompt)
    result = json.loads(response)
    return result['m'], result['anonymization']
```

### 3. 运行评估

```python
from src.evaluators.evaluate_scenario_c import ScenarioCEvaluator

# 初始化评估器
evaluator = ScenarioCEvaluator(
    "data/ground_truth/scenario_c_common_preferences_optimal.json"
)

# 配置B：测试LLM消费者
results_B = evaluator.evaluate_config_B(
    llm_consumer_agent=my_llm_consumer,
    verbose=True
)

# 配置C：测试LLM中介
results_C = evaluator.evaluate_config_C(
    llm_intermediary_agent=my_llm_intermediary,
    verbose=True
)

# 配置D：测试双边LLM
results_D = evaluator.evaluate_config_D(
    llm_intermediary_agent=my_llm_intermediary,
    llm_consumer_agent=my_llm_consumer,
    verbose=True
)

# 生成报告
df = evaluator.generate_report(
    results_B=results_B,
    results_C=results_C,
    results_D=results_D,
    output_path="evaluation_results/my_llm_report.csv"
)

print(df)
```

---

## 评估指标

### 配置B指标（LLM消费者）

#### 参与率指标
```python
{
    "r_llm": 0.065,                    # LLM的参与率
    "r_theory": 0.049,                 # 理论参与率
    "r_absolute_error": 0.016,         # 绝对误差
    "r_relative_error": 0.337,         # 相对误差（33.7%）
    "individual_accuracy": 0.85,       # 个体决策准确率
    "true_positive_rate": 0.80,        # 真阳性率
    "false_positive_rate": 0.12,       # 假阳性率
}
```

#### 市场结果指标
```python
{
    "social_welfare_llm": 195.92,
    "social_welfare_theory": 202.04,
    "social_welfare_ratio": 0.97,      # 97%效率
    "welfare_loss": 6.12,
    "welfare_loss_percent": 3.03,      # 3%福利损失
    "consumer_surplus_ratio": 0.94,
    "producer_profit_ratio": 0.96,
}
```

### 配置C指标（LLM中介）

#### 策略指标
```python
{
    "m_llm": 0.6,
    "m_theory": 0.5,
    "m_absolute_error": 0.1,           # m偏差0.1
    "m_relative_error": 0.20,          # 相对误差20%
    "anon_llm": "anonymized",
    "anon_theory": "anonymized",
    "anon_match": 1,                   # 策略匹配
}
```

#### 利润指标
```python
{
    "profit_llm": 1.50,
    "profit_theory": 1.596,
    "profit_ratio": 0.94,              # 94%利润效率
    "profit_loss": 0.096,
    "profit_loss_percent": 6.0,        # 6%利润损失
}
```

### 配置D指标（双边LLM）

#### 与理论解对比
```python
{
    "vs_theory": {
        "m_error": 0.15,
        "anon_match": 1,
        "r_error": 0.025,
        "social_welfare_ratio": 0.92,
        "welfare_loss_percent": 8.37,
    }
}
```

#### 交互指标
```python
{
    "exploitation_indicator": 1.15,    # >1表示中介获利更多
    "interaction_effect_welfare": -2.5,
}
```

---

## 指标解读

### 参与率误差（r_error）

- **< 5%**: 优秀，LLM理解非常准确
- **5-15%**: 良好，基本理解机制
- **15-30%**: 一般，存在系统性偏差
- **> 30%**: 较差，决策与理论偏离较大

### 福利损失（welfare_loss_percent）

- **< 5%**: 优秀，接近理论最优
- **5-15%**: 良好，效率损失可接受
- **15-30%**: 一般，存在明显效率损失
- **> 30%**: 较差，严重偏离最优

### 个体准确率（individual_accuracy）

- **> 90%**: 优秀，大部分决策正确
- **80-90%**: 良好，多数决策正确
- **70-80%**: 一般，存在较多错误
- **< 70%**: 较差，错误率过高

### 利润效率（profit_ratio）

- **> 95%**: 优秀，策略接近最优
- **90-95%**: 良好，策略合理
- **80-90%**: 一般，有改进空间
- **< 80%**: 较差，策略选择不当

---

## 完整示例

```python
"""
完整的评估流程示例
"""

import json
from src.evaluators.evaluate_scenario_c import ScenarioCEvaluator

# 1. 初始化评估器
evaluator = ScenarioCEvaluator(
    "data/ground_truth/scenario_c_common_preferences_optimal.json"
)

print("理论基准:")
print(f"  m* = {evaluator.gt_A['optimal_strategy']['m_star']}")
print(f"  r* = {evaluator.gt_A['optimal_strategy']['r_star']}")

# 2. 测试多个LLM模型
models = ["gpt-4", "claude-3", "gemini-pro"]
all_results = {}

for model in models:
    print(f"\n评估模型: {model}")
    
    # 配置B
    results_B = evaluator.evaluate_config_B(
        llm_consumer_agent=lambda **kwargs: call_llm(model, "consumer", **kwargs),
        verbose=False
    )
    
    # 配置C
    results_C = evaluator.evaluate_config_C(
        llm_intermediary_agent=lambda **kwargs: call_llm(model, "intermediary", **kwargs),
        verbose=False
    )
    
    all_results[model] = {
        "B": results_B,
        "C": results_C,
    }
    
    print(f"  配置B - 参与率误差: {results_B['participation']['r_relative_error']:.2%}")
    print(f"  配置C - 利润效率: {results_C['profit']['profit_ratio']:.2%}")

# 3. 保存结果
with open("evaluation_results/all_models_comparison.json", 'w') as f:
    json.dump(all_results, f, indent=2)

# 4. 生成对比报告
import pandas as pd

comparison_data = []
for model in models:
    row = {
        "Model": model,
        "B_r_error": all_results[model]['B']['participation']['r_relative_error'],
        "B_accuracy": all_results[model]['B']['participation']['individual_accuracy'],
        "B_welfare_loss": all_results[model]['B']['market']['welfare_loss_percent'],
        "C_m_error": all_results[model]['C']['strategy']['m_relative_error'],
        "C_profit_ratio": all_results[model]['C']['profit']['profit_ratio'],
        "C_welfare_loss": all_results[model]['C']['market']['welfare_loss_percent'],
    }
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv("evaluation_results/models_comparison.csv", index=False)

print("\n模型对比:")
print(df_comparison.to_string(index=False))
```

---

## 注意事项

### 1. Ground Truth依赖

- 评估器依赖于理论GT，确保先运行`generate_scenario_c_gt.py`
- 不同的GT文件（common_preferences vs common_experience）会产生不同的基准

### 2. 随机性控制

- GT文件中包含固定的`seed`和`sample_data`
- 评估时使用相同的消费者参数，确保可对比性
- 如果LLM有随机性，建议多次运行取平均

### 3. LLM代理接口

评估器接受两种类型的代理：

**方式1：函数**
```python
def my_consumer(consumer_params, m, anonymization):
    return decision
```

**方式2：对象（带`.decide()`或`.choose_strategy()`方法）**
```python
class MyConsumer:
    def decide(self, consumer_params, m, anonymization):
        return decision

agent = MyConsumer()
evaluator.evaluate_config_B(agent)
```

### 4. 指标的客观性

- 所有指标都是**数值型**的，无主观评分
- 基于行为数据（决策、参与率、福利）直接计算
- 完全可复现

---

## 文件结构

```
src/evaluators/
├── evaluate_scenario_c.py          # 主评估器
├── scenario_c_metrics.py           # 指标计算函数
└── __init__.py

test_scenario_c_evaluator.py        # 测试脚本（含模拟代理）

evaluation_results/                  # 评估结果输出目录
├── scenario_c_test_report.csv      # 简要报告
└── scenario_c_test_detailed.json   # 详细结果

data/ground_truth/                   # Ground Truth文件
├── scenario_c_common_preferences_optimal.json
└── scenario_c_common_experience_optimal.json
```

---

## 运行测试

```bash
# 测试评估器（使用模拟LLM）
python test_scenario_c_evaluator.py

# 查看结果
cat evaluation_results/scenario_c_test_report.csv
```

---

## 常见问题

### Q1: 为什么需要Ground Truth？

A: Ground Truth提供理论最优解作为benchmark。评估指标都是LLM行为与理论解的对比。

### Q2: 配置B和配置C哪个更重要？

A: **配置B**是核心，测试LLM对隐私外部性机制的理解（消费者视角）。配置C测试优化能力（中介视角），也很重要但相对次要。

### Q3: 如何处理LLM的随机性？

A: 建议多次运行取平均。也可以在LLM调用时设置`temperature=0`以减少随机性。

### Q4: 评估器支持哪些LLM？

A: 评估器是模型无关的，只要LLM代理实现了指定的接口即可。支持任何能生成决策或策略的模型。

---

## 扩展开发

### 添加新指标

在`scenario_c_metrics.py`中添加新的计算函数：

```python
def compute_my_metric(llm_data, theory_data):
    """计算自定义指标"""
    return {
        "my_metric": some_calculation(llm_data, theory_data)
    }
```

然后在`evaluate_scenario_c.py`中调用。

### 支持新配置

在`ScenarioCEvaluator`类中添加新方法：

```python
def evaluate_config_E(self, ...):
    """评估新配置E"""
    pass
```

---

## 引用

如果使用本评估器，请引用：

```
场景C：《The Economics of Social Data》理论实现与LLM评估框架
```
