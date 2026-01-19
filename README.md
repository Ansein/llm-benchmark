# LLM隐私外部性Benchmark系统说明

## 📋 系统概述

这是一个完整的LLM benchmark系统，用于评估不同大语言模型在隐私外部性场景下的决策能力。

### 三步流程

```
第1步 ✅ [已完成]          第2步 ✅ [已完成]         第3步 ✅ [已完成]
生成理论基准          →    LLM做决策模拟     →    计算偏差与评分
(Ground Truth)            (均衡求解)              (MAE + 标签)
```

## 🗂️ 项目目录结构

```
benchmark/
├── src/
│   ├── evaluators/
│   │   ├── evaluate_scenario_a.py
│   │   ├── evaluate_scenario_b.py
│   │   ├── evaluate_scenario_c.py          # 场景C评估器（可直接运行，支持真实LLM）⭐
│   │   └── scenario_c_metrics.py
│   └── scenarios/
│       ├── generate_scenario_c_gt.py       # 生成场景C Ground Truth
│       └── scenario_c_social_data.py       # 场景C理论求解器（Stackelberg + m0内生化）
├── configs/
│   └── model_configs.json                 # 模型配置（OpenAI兼容base_url等）
├── data/
│   └── ground_truth/                      # 生成的GT输出目录（自动创建）
├── evaluation_results/                    # 评估输出目录（自动创建/保存）
└── docs/
    ├── README_scenario_c_evaluator.md     # 场景C评估器使用说明 ⭐
    ├── README_evaluation.md
    └── README_scenarios.md
```

## 🚀 快速开始（场景C，真实LLM）

### 0) 安装依赖

```bash
pip install openai numpy pandas
```

Windows 建议先设置：`$env:PYTHONIOENCODING="utf-8"`

### 1) 生成 Ground Truth（只需一次）

```bash
python -m src.scenarios.generate_scenario_c_gt
```

### 2) 运行评估器（真实LLM）

```bash
python src/evaluators/evaluate_scenario_c.py
```

在 `src/evaluators/evaluate_scenario_c.py` 里修改 **`TARGET_MODEL = "gpt-4.1-mini"`** 来选择模型（按 `configs/model_configs.json` 的 `config_name` 匹配）。

## 📦 模块说明

### src/evaluators/
评估器模块，负责调用LLM并评估其决策能力
- `llm_client.py`: 封装OpenAI兼容的API接口
- `evaluate_scenario_a.py`: 场景A（个性化定价）评估器
- `evaluate_scenario_b.py`: 场景B（推断外部性）评估器

### src/scenarios/
场景生成器模块，负责生成Ground Truth
- `scenario_a_personalization.py`: 场景A的理论求解器
- `scenario_b_too_much_data.py`: 场景B的理论求解器

### data/
数据文件目录
- `ground_truth/`: 理论基准数据（由场景生成器生成）
- `test_results/`: 测试结果

### configs/
配置文件目录
- `model_configs.json`: LLM模型配置（API密钥、base_url等）

### docs/
文档目录
- `README_*.md`: 各种说明文档
- `README_scenario_c_evaluator.md`: 场景C评估器使用说明（推荐先看）

### papers/
论文PDF及提取的文本

## 🎯 三个评估场景

### 场景A：个性化定价与隐私选择

**外部性机制**：
- 消费者披露数据 → 平台个性化定价 → 影响其他消费者的价格和福利
- 个体决策未内生对他人的影响 → **过度披露**

**LLM任务**：
- 每个消费者决定是否披露数据
- 需要权衡：隐私成本 vs 可能的购买效用

**评估指标**：
- 披露率与理论均衡的偏差（MAE）
- 平台利润、消费者剩余、社会福利偏差
- 是否识别出"过度披露"现象

### 场景B：Too Much Data（推断外部性）

**外部性机制**：
- 用户类型相关 → 他人分享数据会"连带泄露"你的信息
- 边际信息次模性 → 平台压价 → **过度分享**

**LLM任务**：
- 每个用户决定是否分享数据
- 需要理解：即使不分享也会被推断（推断外部性）

**评估指标**：
- 分享率与理论均衡的偏差（MAE）
- 平台利润、社会福利、信息泄露量偏差
- 是否理解推断外部性

### 场景C：The Economics of Social Data（社会数据外部性）✨ **新增**

**外部性机制**：
- 社会数据特性 → 个人数据对他人有预测价值
- 搭便车问题 → 拒绝者仍能从参与者数据中学习
- 数据外部性 → 参与决策相互影响

**LLM任务**：
- 每个消费者决定是否向中介出售数据
- 需要权衡：补偿+学习收益 vs 价格歧视风险
- 需要理解：匿名化如何影响定价和参与意愿

**评估指标**：
- 参与率与理论均衡的偏差（MAE）
- 消费者剩余、生产者利润、社会福利偏差
- 是否理解搭便车机制
- 匿名化 vs 实名化下的行为差异

## 📊 评估方法

### 1. 模拟均衡过程

```python
初始状态: 空集合（无人披露/分享）
↓
迭代过程:
  - 随机顺序遍历所有代理
  - 每个代理基于当前状态做决策（调用LLM）
  - 更新集合
↓
收敛检测: 无代理改变决策 → 达到LLM均衡
↓
计算指标: 基于LLM均衡计算各项指标
```

### 2. 评估指标

**主评分：偏差指标（MAE）**
- MAE = |LLM结果 - 理论基准|
- 越小越好（0表示完美匹配）

**辅助评分：标签一致性**
- 结构标签：披露/分享率分桶（low/medium/high）
- 方向标签：过度披露/分享判断
- 政策标签：关停是否更优

**稳定性指标**
- 每个决策重复num_trials次
- 多数投票确定最终决策
- 记录决策的稳定性

### 3. 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_trials` | 3 | 每个决策重复次数 |
| `max_iterations` | 10 | 最大迭代次数 |
| `scenarios` | A, B | 要评估的场景 |
| `models` | gpt-4.1-mini | 要评估的模型 |

## 🚀 使用示例

### 场景C（推荐：真实LLM）

```bash
python -m src.scenarios.generate_scenario_c_gt
python src/evaluators/evaluate_scenario_c.py
```

### 单个场景评估

```bash
# 场景A
python run_evaluation.py --single --scenarios A --models gpt-4.1-mini

# 场景B
python run_evaluation.py --single --scenarios B --models deepseek-v3
```

### 批量评估（生成对比报告）

```bash
# 评估所有场景和所有模型
python run_evaluation.py \
  --scenarios A B \
  --models gpt-4.1-mini deepseek-v3 grok-3-mini
```

## 📈 测试结果（示例）

### 场景A测试结果

```
【披露集合比较】
  LLM均衡: []
  理论均衡: []
  收敛情况: ✅ 已收敛 (迭代1次)

【关键指标】
  平台利润:     LLM=3.500  |  GT=3.500  |  MAE=0.000
  消费者剩余:   LLM=1.400  |  GT=1.400  |  MAE=0.000
  社会福利:     LLM=4.900  |  GT=4.900  |  MAE=0.000
  披露率:       LLM=0.00%  |  GT=0.00%  |  MAE=0.00%

【标签一致性】
  披露率分桶:   LLM=low  |  GT=low  |  ✅
  过度披露:     LLM=0  |  GT=0  |  ✅
```

**解读**：GPT-4.1-mini在场景A中表现完美，完全理解了个性化定价的外部性机制。

### 场景B测试结果

```
【分享集合比较】
  LLM均衡: []
  理论均衡: [4, 5, 6]
  收敛情况: ✅ 已收敛 (迭代1次)

【关键指标】
  平台利润:     LLM=0.0000  |  GT=4.5620  |  MAE=4.5620
  社会福利:     LLM=0.0000  |  GT=1.8161  |  MAE=1.8161
  总泄露量:     LLM=0.0000  |  GT=5.6940  |  MAE=5.6940
  分享率:       LLM=0.00%  |  GT=37.50%  |  MAE=37.50%

【标签一致性】
  泄露量分桶:   LLM=low  |  GT=high  |  ❌
  过度分享:     LLM=0  |  GT=0  |  ✅
  关停更优:     LLM=0  |  GT=0  |  ✅
```

**解读**：GPT-4.1-mini在场景B中过于谨慎，没有理解"即使不分享也会被推断"的关键机制，导致所有用户都选择不分享。

## 💡 关键发现

1. **不同LLM的差异**：不同模型在理解经济学概念上有显著差异
2. **场景难度**：场景B（推断外部性）比场景A（价格传导）更难理解
3. **Benchmark价值**：能够定量评估LLM的经济学推理能力

## 📊 输出文件

### 单个评估结果

```json
{
  "model_name": "gpt-4.1-mini",
  "llm_disclosure_set": [0, 2, 5],
  "gt_disclosure_set": [5],
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

### 汇总报告（CSV）

| 场景 | 模型 | 收敛 | 迭代次数 | 利润MAE | CS_MAE | 福利MAE | 标签匹配 |
|------|------|------|----------|---------|--------|---------|----------|
| A | gpt-4.1-mini | ✅ | 1 | 0.000 | 0.000 | 0.000 | ✅ |
| B | gpt-4.1-mini | ✅ | 1 | 4.562 | - | 1.816 | ❌ |
| A | deepseek-v3 | ✅ | 2 | 0.150 | 0.200 | 0.100 | ✅ |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 🔧 技术细节

### LLM提示设计

每个提示包含：
1. **场景描述**：清晰解释外部性机制
2. **个性化信息**：当前代理的参数
3. **当前状态**：其他代理的决策
4. **计算辅助**：预先计算关键数值
5. **决策要求**：JSON格式输出

### 均衡求解策略

- **固定点迭代**：重复决策直到收敛
- **随机顺序**：避免顺序偏差
- **多次试验**：评估稳定性
- **收敛检测**：无代理改变决策

### 评分方法

- **MAE计算**：使用理论基准作为ground truth
- **标签评分**：结构/方向/桶化匹配
- **置信区间**：多次运行计算统计显著性（可选）

## 📚 相关论文

本系统基于以下论文设计：

1. **ICDE 2026**: "LLM Amplify Inequality in Data Market"（baseline）
2. **Rhodes & Zhou**: "Personalization and Privacy Choice"
3. **Acemoglu et al.**: "Too Much Data: Prices and Inefficiencies in Data Markets"
4. **Bergemann & Bonatti**: "The Economics of Social Data"
5. **Ichihashi**: "Data-enabled Learning, Network Effects, and Competitive Advantage"

## 🎯 下一步工作

- [x] 实现场景A和场景B的评估器
- [x] 完成LLM客户端封装
- [x] 测试完整评估流程
- [x] 实现场景C（Economics of Social Data）✨
- [x] 生成场景C Ground Truth
- [ ] （可选）将场景C集成到 run_evaluation.py 统一入口
- [ ] 实现场景D（Data-enabled Learning）
- [ ] 实现场景E（Platform-Data Broker Partnership）
- [ ] 添加解释题评估（keypoints + factor_ranks）
- [ ] 实现异质部署（不同代理使用不同LLM）
- [ ] 添加置信区间计算（多次运行）

## ⚠️ 注意事项

1. **API成本**：每个完整评估约需100-300次LLM调用
2. **运行时间**：单个场景约5-15分钟
3. **编码问题**（Windows）：运行前设置`$env:PYTHONIOENCODING="utf-8"`
4. **API密钥安全**：如果仓库公开，请不要在 `configs/model_configs.json` 中明文保存密钥；建议改为环境变量或使用私有仓库。

## 📞 联系方式

如有问题，请查看：
- `docs/README_scenario_c_evaluator.md` - 场景C评估器说明
- `docs/README_evaluation.md` - 评估系统说明
- `docs/README_scenarios.md` - 场景与GT生成说明

---

**祝评估顺利！** 🎉
