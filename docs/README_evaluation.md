# LLM Benchmark 评估系统使用说明

## 📋 系统概述

本评估系统用于测试不同LLM在隐私外部性场景下的决策能力。系统包含三个步骤：

1. **第1步（已完成）**：生成理论基准 (ground truth)
2. **第2步（本模块）**：让LLM做决策并模拟均衡
3. **第3步（本模块）**：计算LLM表现与理论基准的偏差

## 🗂️ 文件结构

```
benchmark/
├── llm_client.py                    # LLM客户端封装
├── evaluate_scenario_a.py           # 场景A评估器
├── evaluate_scenario_b.py           # 场景B评估器
├── run_evaluation.py                # 主评估脚本
├── model_configs.json               # 模型配置文件
├── scenario_a_result.json           # 场景A的ground truth
├── scenario_b_result.json           # 场景B的ground truth
└── evaluation_results/              # 评估结果输出目录
```

## 🔧 安装依赖

```bash
pip install openai pandas numpy
```

## 🚀 快速开始

### 1. 单个场景单个模型评估（测试）

```bash
# 测试场景A，使用gpt-4.1-mini
python run_evaluation.py --single --scenarios A --models gpt-4.1-mini

# 测试场景B，使用deepseek-v3
python run_evaluation.py --single --scenarios B --models deepseek-v3
```

### 2. 批量评估

```bash
# 评估所有场景和所有模型
python run_evaluation.py --scenarios A B --models gpt-4.1-mini deepseek-v3 grok-3-mini

# 自定义参数
python run_evaluation.py \
  --scenarios A B \
  --models gpt-4.1-mini deepseek-v3 \
  --num-trials 5 \
  --max-iterations 15 \
  --output-dir my_results
```

### 3. 参数说明

- `--scenarios`: 要评估的场景列表 (A, B)
- `--models`: 要评估的模型列表（需要在model_configs.json中配置）
- `--num-trials`: 每个决策重复次数（默认3，用于评估稳定性）
- `--max-iterations`: 寻找均衡的最大迭代次数（默认10）
- `--output-dir`: 结果输出目录（默认evaluation_results）
- `--single`: 单次评估模式（用于快速测试）

## 📊 评估指标

### 场景A：个性化定价与隐私选择

**偏差指标（MAE vs Ground Truth）**：
- 平台利润偏差
- 消费者剩余偏差
- 社会福利偏差
- 披露率偏差

**标签一致性**：
- 披露率分桶匹配（low/medium/high）
- 过度披露判断匹配

### 场景B：Too Much Data（推断外部性）

**偏差指标（MAE vs Ground Truth）**：
- 平台利润偏差
- 社会福利偏差
- 总信息泄露量偏差
- 分享率偏差

**标签一致性**：
- 泄露量分桶匹配（low/medium/high）
- 过度分享判断匹配
- 关停更优判断匹配

## 📈 输出文件

评估完成后会生成以下文件：

```
evaluation_results/
├── eval_scenario_A_gpt-4.1-mini.json      # 单个评估的详细结果
├── eval_scenario_B_gpt-4.1-mini.json
├── summary_report_20260113_150230.csv     # 汇总表格
└── all_results_20260113_150230.json       # 所有结果的完整JSON
```

### 单个评估结果包含：

```json
{
  "model_name": "gpt-4.1-mini",
  "llm_disclosure_set": [0, 2, 5],
  "gt_disclosure_set": [5],
  "convergence_history": [...],
  "converged": true,
  "iterations": 3,
  "metrics": {
    "llm": {...},
    "ground_truth": {...},
    "deviations": {
      "profit_mae": 0.123,
      "cs_mae": 0.456,
      ...
    }
  },
  "labels": {...}
}
```

## 🔍 评估流程

### 场景A流程

1. **初始化**：从空披露集合开始
2. **迭代决策**：
   - 每轮随机顺序遍历所有消费者
   - 每个消费者根据当前状态决定是否披露
   - LLM接收包含个人参数和市场状态的提示
3. **收敛检测**：没有消费者改变决策时收敛
4. **计算指标**：基于LLM均衡计算各项指标
5. **与GT比较**：计算MAE和标签一致性

### 场景B流程

类似场景A，但决策是"是否分享数据"，且提示中强调推断外部性。

## 🎯 关键设计

### 1. 多次试验机制

每个决策重复`num_trials`次（默认3次），然后多数投票，用于：
- 评估LLM决策的稳定性
- 减少随机性影响

### 2. 收敛检测

- 最多迭代`max_iterations`次（默认10次）
- 如果某轮没有任何代理改变决策，则认为收敛
- 未收敛会在结果中标记

### 3. 提示设计

- **清晰的场景描述**：包含关键经济学概念
- **个性化信息**：每个代理的参数和当前状态
- **计算辅助**：预先计算关键数值，帮助LLM理解
- **JSON格式输出**：便于解析和评分

## 🧪 测试建议

1. **先单个测试**：
   ```bash
   python run_evaluation.py --single --scenarios A --models gpt-4.1-mini --max-iterations 3
   ```

2. **检查输出**：查看`evaluation_results/`目录下的JSON文件

3. **调整参数**：
   - 如果不收敛，增加`--max-iterations`
   - 如果结果不稳定，增加`--num-trials`

4. **批量运行**：确认单个测试正常后，运行完整批量评估

## ⚠️ 注意事项

1. **API调用成本**：
   - 场景A (n=8): 约 8×3×10 = 240次LLM调用（最坏情况）
   - 场景B (n=10): 约 10×3×10 = 300次LLM调用（最坏情况）
   - 建议先用`--max-iterations 3`测试

2. **运行时间**：
   - 单个场景单个模型：约5-15分钟
   - 批量评估（2场景×3模型）：约30-90分钟

3. **错误处理**：
   - LLM调用失败时会自动重试
   - 如果某个评估失败，会跳过并继续下一个

## 📖 示例输出

```
==================================================
🤖 开始模拟LLM均衡 (模型: gpt-4.1-mini)
==================================================

--- 迭代 1 ---
当前披露集合: []

  消费者 0: 决策=0, 试验结果=[0, 0, 0]
  消费者 1: 决策=1, 试验结果=[1, 1, 1]
    ✅ 消费者1加入披露集合
  ...

✅ 在第3轮达到收敛！

==================================================
📊 评估结果摘要
==================================================

【披露集合比较】
  LLM均衡: [1, 3, 5]
  理论均衡: [5]
  收敛情况: ✅ 已收敛 (迭代3次)

【关键指标】
  平台利润:     LLM=3.800  |  GT=3.500  |  MAE=0.300
  消费者剩余:   LLM=1.000  |  GT=1.400  |  MAE=0.400
  社会福利:     LLM=4.800  |  GT=4.900  |  MAE=0.100
  披露率:       LLM=37.50%  |  GT=0.00%  |  MAE=37.50%

【标签一致性】
  披露率分桶:   LLM=medium  |  GT=low  |  ❌
  过度披露:     LLM=1  |  GT=0  |  ❌
```

## 🔗 相关文件

- `scenario_a_personalization.py`: 场景A的ground truth生成
- `scenario_b_too_much_data.py`: 场景B的ground truth生成
- `README_scenarios.md`: 场景说明和ground truth生成说明
- `最终设计方案.md`: 完整的benchmark设计方案
