# 虚拟博弈（Fictitious Play）使用指南

## 概述

虚拟博弈是一个迭代学习算法，用于模拟多轮博弈中玩家如何根据历史观察更新信念并调整策略。

本项目支持两个场景的虚拟博弈：
- **场景B**：数据市场（推断外部性）- 用户分享决策
- **场景C**：社会数据（中介-消费者博弈）- 中介策略+消费者参与决策

### 核心特性

#### 场景B（单层博弈）
1. **固定价格机制**：平台在第0轮提供理论最优价格，之后保持不变
2. **信念更新**：用户根据最近N轮的历史观察更新对其他用户的分享概率估计
3. **迭代学习**：最多50轮，或连续3轮分享集合不变时提前终止
4. **成本控制**：使用deepseek-v3模型，每轮单次查询

#### 场景C（双层序贯博弈）

**三种FP配置**：

1. **配置B_FP（理性中介 × LLM消费者）**：
   - 中介：始终使用理论最优策略（m*, anon*）
   - 消费者：LLM，基于历史学习其他消费者的参与概率
   - 测试目标：LLM消费者在固定策略下的学习收敛能力
   - 查询次数：50轮 × 20个消费者 ≈ 1000次

2. **配置C_FP（LLM中介 × 理性消费者）**：
   - 中介：LLM，基于历史学习参与率趋势
   - 消费者：理性决策（给定中介策略后做最优反应）
   - 测试目标：LLM中介面对理性消费者的策略学习能力
   - 查询次数：50轮 × 1个中介 ≈ 50次

3. **配置D_FP（LLM中介 × LLM消费者）**：
   - 中介：LLM，基于历史学习参与率趋势
   - 消费者：LLM，基于历史学习其他消费者的参与概率
   - 测试目标：双方学习下的均衡收敛性
   - 查询次数：50轮 × 21个代理 ≈ 1050次

**共同特性**：
- 迭代学习：最多50轮，或连续3轮策略和参与集合不变时提前终止
- 成本控制：使用deepseek-v3.2模型

## 快速开始

### 推荐命令

**一条命令运行所有配置**（最高效）：

```bash
# 场景B：虚拟博弈
python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3.2

# 场景C：所有三个配置的虚拟博弈
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config all --model deepseek-v3.2
```

### 依赖安装

确保已安装可视化依赖：

```bash
pip install matplotlib seaborn
```

### 运行评估器

#### 场景B

```bash
# 运行虚拟博弈（使用默认参数）
python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3

# 自定义参数
python -m src.evaluators.evaluate_scenario_b \
    --mode fp \
    --model deepseek-v3 \
    --max_rounds 50 \
    --belief_window 10 \
    --num_trials 1

# 运行静态博弈（对比）
python -m src.evaluators.evaluate_scenario_b --mode static --model deepseek-v3
```

#### 场景C

```bash
# 一次运行所有三个配置（推荐！省时间）
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config all --model deepseek-v3.2

# 单独运行配置B_FP：理性中介 × LLM消费者（消费者学习）
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config B --model deepseek-v3.2

# 单独运行配置C_FP：LLM中介 × 理性消费者（中介学习）
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config C --model deepseek-v3.2

# 单独运行配置D_FP：LLM中介 × LLM消费者（双方学习，默认）
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config D --model deepseek-v3.2

# 自定义参数（运行所有配置）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model deepseek-v3.2 \
    --rounds 50 \
    --belief_window 10

# 运行现有多轮迭代（对比）
python -m src.evaluators.evaluate_scenario_c --mode iterative --model deepseek-v3.2 --rounds 20
```

## 场景C的 `--fp_config all` 说明

使用 `--fp_config all` 可以**一条命令运行所有三个配置**（B、C、D），非常高效！

### 为什么使用 `all`？

| 运行方式 | 命令次数 | 总查询次数 | 优点 | 缺点 |
|---------|---------|-----------|------|------|
| **单独运行** | 3次 | 2100次 | 灵活，可针对性测试 | 需要手动运行3次，时间戳不一致 |
| **all 选项** | 1次 | 2100次 | 自动化，结果统一时间戳 | 无法中途停止单个配置 |

**成本相同**：无论是运行3次还是用 `all`，总查询次数都是 2100 次

### 适用场景

**推荐使用 `all`**：
- ✅ 全面评估三种配置的学习表现
- ✅ 自动化测试流程
- ✅ 需要统一时间戳方便对比
- ✅ 一次性生成完整实验结果

**单独运行**：
- ⭕ 只关心某一方的学习能力（如只测试中介用C，只测试消费者用B）
- ⭕ 调试某个特定配置
- ⭕ 中途需要调整参数

## 参数说明

- `--mode`: 博弈模式
  - `static`: 静态博弈（一次性决策）
  - `fp`: 虚拟博弈（多轮迭代）
  
- `--model`: LLM模型
  - `deepseek-v3`: 推荐，成本低
  - `gpt-4.1-mini`: 备选
  - `grok-3-mini`: 备选
  
- `--max_rounds`: 虚拟博弈最大轮数（默认50）

- `--belief_window`: 信念窗口大小（默认10）
  - 用户只看最近N轮历史
  - 实际窗口 = min(当前轮数, belief_window)
  
- `--num_trials`: 每个决策的重复查询次数（默认1）
  - 建议保持为1以控制成本
  - 可以通过多次运行整个FP过程来分析稳定性

## 输出文件结构

### 场景B
结果将保存到按模式和模型组织的子文件夹中：

```
evaluation_results/
├── static_deepseek-v3/
│   ├── eval_20260122_143022.json
│   └── ...
├── fp_deepseek-v3/
│   ├── eval_20260122_143530.json
│   ├── eval_20260122_143530_share_rate.png
│   └── eval_20260122_143530_strategy_heatmap.png
└── ...
```

### 场景C
结果保存到scenario_c目录下的子文件夹，按配置分类：

```
evaluation_results/scenario_c/
├── fp_configB_deepseek-v3.2/          # 配置B_FP（理性中介×LLM消费者）
│   ├── eval_20260122_143530.json
│   ├── eval_20260122_143530_profit_rate.png
│   └── eval_20260122_143530_strategy_evolution.png
├── fp_configC_deepseek-v3.2/          # 配置C_FP（LLM中介×理性消费者）
│   └── ...
├── fp_configD_deepseek-v3.2/          # 配置D_FP（LLM×LLM）
│   └── ...
└── iterative结果/
    ├── scenario_c_common_preferences_deepseek-v3.2_XXX.csv
    └── ...
```

每次运行会生成带时间戳的JSON文件，便于管理多次实验结果。

### 自动生成的可视化

#### 场景B

运行虚拟博弈后，会自动生成两个可视化图表：

1. **分享率曲线图**（`_share_rate.png`）：
   - 蓝色曲线：每轮的分享率演化
   - 红色曲线：与理论均衡的Jaccard相似度
   - 绿色虚线：标注收敛点（如果收敛）

2. **用户策略热力图**（`_strategy_heatmap.png`）：
   - 行：用户ID（0到n-1）
   - 列：轮次（1到T）
   - 颜色：浅灰=不分享(0)，蓝色=分享(1)
   - 金色★：标注理论均衡中应分享的用户

#### 场景C

运行虚拟博弈后，会自动生成两个可视化图表：

1. **利润与参与率曲线**（`_profit_rate.png`）：
   - 蓝色曲线：每轮的中介利润演化
   - 红色曲线：每轮的消费者参与率
   - 绿色虚线：标注收敛点（如果收敛）

2. **策略演化图**（`_strategy_evolution.png`）：
   - 上图：补偿金额m的演化，红色虚线标注理论最优
   - 下图：匿名化策略的演化（绿=anonymized, 红=identified）

## 输出结果

### 收敛性分析

```json
{
  "convergence_analysis": {
    "converged": true,                    // 是否收敛
    "convergence_round": 23,              // 收敛轮数
    "final_stability": 0.95,              // 最后10轮稳定性
    "oscillation_detected": false,        // 是否检测到震荡
    "avg_hamming_distance_last10": 0.1,   // 最后10轮平均汉明距离
    "share_rate_trajectory": [...],       // 每轮分享率
    "similarity_trajectory": [...],       // 每轮与理论均衡的相似度
    "final_similarity_to_equilibrium": 0.85
  }
}
```

### 均衡质量指标

- **share_set_similarity**: 最终分享集合与理论均衡的Jaccard相似度
- **share_rate_error**: 分享率误差
- **welfare_mae**: 社会福利偏差
- **profit_mae**: 平台利润偏差

## 设计细节

### 信念更新规则

#### 场景B
使用**经验频率**估计分享概率：

```
P(用户i分享) = (最近N轮中用户i分享的次数) / N
```

- 初始信念：均匀分布（50%）
- 窗口大小：min(当前轮数, 10)

#### 场景C
双层信念更新：

**中介的信念**：
```
预期参与率 = (最近N轮的平均参与率)
- identified策略下的平均参与率
- anonymized策略下的平均参与率
```

**消费者的信念**：
```
P(消费者i参与) = (最近N轮中消费者i参与的次数) / N
```

- 初始信念：中介预期50%参与率，消费者预期其他人50%参与
- 窗口大小：min(当前轮数, 10)

### 提示词结构

虚拟博弈的提示词包含：

1. **用户私有信息**：报价、隐私偏好
2. **公共知识**：参数、推断外部性机制
3. **历史观察**：
   - 最近N轮的分享集合
   - 基于历史计算的分享概率估计
4. **决策提示**：基于历史和机制选择分享或不分享

注意：**不告诉用户当前轮数**，避免"时间感"干扰决策。

### 收敛判断

#### 场景B
**提前终止条件**：连续3轮分享集合完全相同

**稳定性度量**：
- 汉明距离：相邻轮次决策向量的差异
- 最后10轮平均汉明距离
- 最终稳定性 = 1 - (平均汉明距离 / n)

#### 场景C
**提前终止条件**（同时满足）：连续3轮
- |m_t - m_{t-1}| < 0.01
- anonymization策略不变
- 参与集合不变

**稳定性度量**：
- 参与率标准差（最后10轮）
- 参与稳定性 = 1 - std(参与率)
- 策略收敛：m是否趋于稳定

## 对比实验建议

### 1. FP vs 静态博弈

比较学习是否改善均衡质量：

```bash
# 静态博弈
python -m src.evaluators.evaluate_scenario_b --mode static --model deepseek-v3

# 虚拟博弈
python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3
```

**分析指标**：
- 最终分享集合与理论均衡的相似度
- 社会福利和平台利润
- 是否收敛到更好的均衡

### 2. 不同信念窗口

测试窗口大小对收敛的影响：

```bash
# 窗口5
python -m src.evaluators.evaluate_scenario_b --mode fp --belief_window 5

# 窗口10（默认）
python -m src.evaluators.evaluate_scenario_b --mode fp --belief_window 10

# 窗口20
python -m src.evaluators.evaluate_scenario_b --mode fp --belief_window 20
```

### 3. 多次运行稳定性

同一设置运行多次，观察随机性影响：

```bash
# Windows (PowerShell)
for ($i=1; $i -le 5; $i++) {
    python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3
}

# Linux/Mac (Bash)
for i in {1..5}; do
    python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3
done
```

所有结果将自动保存到 `evaluation_results/fp_deepseek-v3/` 子文件夹中，每次运行带不同时间戳。

## 代码结构

### 新增方法

- `simulate_fictitious_play()`: 主虚拟博弈流程
- `build_user_decision_prompt_fp()`: 构建包含历史的提示词
- `query_user_decision_fp()`: 查询用户决策（FP版本）
- `_compute_belief_probs()`: 计算信念概率（经验频率）
- `_check_convergence()`: 检查收敛
- `_analyze_convergence()`: 分析收敛性
- `_visualize_fictitious_play()`: 生成可视化图表（自动调用）
- `print_evaluation_summary_fp()`: 打印FP评估摘要

### 保留原有功能

- `simulate_static_game()`: 静态博弈（不变）
- `build_user_decision_prompt()`: 静态博弈提示词（不变）
- `query_user_decision()`: 静态博弈用户决策（不变）

## 成本估算

### 场景B
假设：
- 模型：deepseek-v3
- 用户数：n=10
- 轮数：50轮
- 每轮单次查询

**总查询次数**：50 × 10 = 500次

**预估成本**：根据deepseek-v3定价计算（通常比GPT-4便宜10-20倍）

### 场景C

假设：
- 模型：deepseek-v3.2
- 消费者数：N=100（但采样20个代表性消费者）
- 轮数：50轮

**配置B_FP**（理性中介 × LLM消费者）：
- 每轮：20次消费者决策
- **总查询次数**：50 × 20 = 1000次

**配置C_FP**（LLM中介 × 理性消费者）：
- 每轮：1次中介决策
- **总查询次数**：50 × 1 = 50次
- **最低成本**：仅为场景B的1/10

**配置D_FP**（LLM中介 × LLM消费者）：
- 每轮：1次中介决策 + 20次消费者决策
- **总查询次数**：50 × 21 = 1050次
- **最高成本**：约为场景B的2倍

**使用 `--fp_config all` 一次运行所有配置**：
- 总查询次数：50 + 1000 + 1050 = 2100次
- 相当于场景B的4倍
- 但可以一次性完整测试所有学习场景

**优化建议**：
- 减少采样消费者数（如10个）可减半成本
- 提前收敛通常在20-30轮
- 配置C最经济（仅测试中介学习）- 适合快速验证
- 配置B适中（仅测试消费者学习）
- 配置D最全面但成本最高（测试双方学习）
- 推荐使用 `--fp_config all` 一次运行完所有测试

## 常见问题

### Q1: 为什么不收敛？

可能原因：
1. LLM决策随机性较大
2. 信念窗口太小，对历史不敏感
3. 博弈本身没有纯策略Nash均衡
4. 提示词理解问题

**解决方案**：
- 检查历史轨迹，看是否有明显模式
- 增加信念窗口大小
- 多次运行验证

### Q2: 如何加速测试？

1. 减少轮数：`--max_rounds 20`
2. 使用更快的模型（如果可用）
3. 启用并行查询（代码已支持，自动并行同一轮内的用户决策）

### Q3: 可视化图表会自动生成吗？

是的！运行虚拟博弈后，会**自动生成**两个PNG图表：
- `{文件名}_share_rate.png`：分享率与收敛曲线
- `{文件名}_strategy_heatmap.png`：用户策略演化热力图

无需额外操作，图表会保存在JSON文件的同一目录中。

### Q4: 如何为已有的JSON结果生成可视化？

如果你已经有虚拟博弈的JSON结果，可以直接生成可视化而不需要重跑实验：

```bash
# 为单个文件生成可视化
python -m src.evaluators.evaluate_scenario_b --visualize evaluation_results/fp_deepseek-v3/eval_20260122_143530.json

# 为整个目录的所有JSON生成可视化
python -m src.evaluators.evaluate_scenario_b --visualize evaluation_results/fp_deepseek-v3/

# 为多个文件生成可视化
python -m src.evaluators.evaluate_scenario_b --visualize \
    evaluation_results/fp_deepseek-v3/eval_20260122_143530.json \
    evaluation_results/fp_deepseek-v3/eval_20260122_150015.json

# 使用通配符批量生成
python -m src.evaluators.evaluate_scenario_b --visualize "evaluation_results/fp_deepseek-v3/*.json"
```

注意：
- 只会为虚拟博弈（`game_type="fictitious_play"`）的结果生成可视化
- 如果图表已存在，会被覆盖
- 可视化会保存在JSON文件的同一目录中

如果需要自定义可视化，可以读取JSON文件：

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取结果
with open('evaluation_results/fp_deepseek-v3/eval_20260122_143530.json') as f:
    results = json.load(f)

# 提取数据
history = results['history']
conv_analysis = results['convergence_analysis']

# 自定义分析：例如计算每个用户的分享频率
n = len(history[0])
user_share_freq = {}
for user_id in range(n):
    total_shares = sum(round_dec[str(user_id)] for round_dec in history)
    user_share_freq[user_id] = total_shares / len(history)

# 绘制用户分享频率柱状图
plt.bar(user_share_freq.keys(), user_share_freq.values())
plt.xlabel('User ID')
plt.ylabel('Share Frequency')
plt.title('User Share Frequency in Fictitious Play')
plt.savefig('user_share_freq.png')
```

## 场景B vs 场景C的关键区别

| 维度 | 场景B（单层同时博弈） | 场景C（双层序贯博弈） |
|-----|-------------------|-------------------|
| **玩家结构** | n个对称用户 | 1个中介 + N个消费者 |
| **决策顺序** | 所有用户同时决策 | 中介先动，消费者后动 |
| **固定部分** | 价格向量（理论最优） | 无固定部分 |
| **策略空间** | 离散{分享,不分享} | 中介：连续m×{id,anon}<br>消费者：{参与,拒绝} |
| **信念对象** | 其他用户的分享概率 | 中介：参与率趋势<br>消费者：其他人参与概率 |
| **收敛条件** | 分享集合稳定 | 策略+参与集合稳定 |
| **外部性** | 推断外部性 | 社会数据外部性 |
| **FP配置** | 1种（所有用户学习） | 3种（B/C/D，测试不同学习组合） |
| **查询次数** | 50 × n ≈ 500次 | B: 50×20≈1000次<br>C: 50×1≈50次<br>D: 50×21≈1050次 |

## 理论背景

虚拟博弈（Fictitious Play）由Brown (1951)提出：

- 每个玩家假设对手的策略是**固定的**
- 玩家基于对手的**历史行动频率**形成信念
- 每轮选择**当前信念下的最优反应**
- 理论上在某些博弈中收敛到Nash均衡（如2人零和博弈、势博弈）

**本实现的特点**：
- 使用LLM作为玩家（而非理论最优反应）
- 场景B：测试LLM理解推断外部性的能力
- 场景C：测试LLM理解社会数据外部性的能力
- 隐式信念设计：不告诉LLM"这是虚拟博弈"
