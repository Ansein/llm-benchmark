# 场景A：个性化推荐系统使用指南

## 📖 场景描述

**场景A：Personalized Recommendation with Privacy Choice（推荐系统与隐私选择）**

消费者在数据市场中面临隐私-便利权衡：
- **分享数据** → 获得个性化推荐（搜索效率↑） vs 隐私成本↑
- **企业定价** → 考虑数据分享率和竞争
- **搜索决策** → 分享者按推荐顺序搜索，未分享者随机搜索

---

## 🗂️ 文件清单

| 文件 | 功能 | 行数 | 推荐度 |
|-----|------|------|--------|
| `src/scenarios/scenario_a_recommendation.py` | 理论求解器 | 479 | ⭐⭐⭐ |
| `src/evaluators/evaluate_scenario_a_full.py` | 完整LLM评估器 | 879 | ⭐⭐⭐⭐⭐ |
| `src/evaluators/evaluate_scenario_a_recommendation.py` | 简化评估器 | 638 | ⭐⭐⭐ |
| `scripts/run_scenario_a_sweep.py` | 批量实验脚本 | 142 | ⭐⭐⭐⭐ |

---

## 🚀 快速开始

### **步骤1：生成Ground Truth**

```bash
python -c "from src.scenarios.scenario_a_recommendation import generate_and_save_ground_truth; generate_and_save_ground_truth()"

# 输出：data/ground_truth/scenario_a_recommendation_result.json
```

### **步骤2：运行单次评估**

```bash
# 完全理性模式（Ground Truth基准）
python src/evaluators/evaluate_scenario_a_full.py \
  --rational-share --rational-price --rational-search \
  --n-consumers 10 --n-firms 5 --rounds 1

# 完全LLM模式
python src/evaluators/evaluate_scenario_a_full.py \
  --model deepseek-v3.2 \
  --n-consumers 10 --n-firms 5 --rounds 5

# 混合模式：LLM分享 + 理性市场
python src/evaluators/evaluate_scenario_a_full.py \
  --model gpt-5-mini-2025-08-07 \
  --rational-price --rational-search \
  --n-consumers 10 --n-firms 5 --rounds 5
```

### **步骤3：运行参数扫描（可选）**

```bash
# 企业数从1递增到10，对比多个模型
python scripts/run_scenario_a_sweep.py \
  --start-firm 1 --end-firm 10 \
  --n-consumers 10 \
  --models deepseek-v3.2 gpt-5-mini-2025-08-07 qwen-plus
```

---

## 📊 命令行参数

### **evaluate_scenario_a_full.py**

```bash
python src/evaluators/evaluate_scenario_a_full.py [options]

必需参数：
  无（全部有默认值）

常用参数：
  --model MODEL              LLM模型名称（默认：deepseek-v3.2）
  --rounds N                 模拟轮数（默认：5）
  --n-consumers N            消费者数量（默认：10）
  --n-firms N                企业数量（默认：5）
  --search-cost COST         搜索成本（默认：0.02）
  --seed SEED                随机种子（默认：42）

决策模式控制：
  --rational-share           使用理性分享决策
  --rational-price           使用理性定价决策
  --rational-search          使用理性搜索决策
  （不加参数默认为LLM模式）

输出控制：
  --output-dir DIR           输出目录（默认：evaluation_results/scenario_a）
```

### **run_scenario_a_sweep.py**

```bash
python scripts/run_scenario_a_sweep.py [options]

参数：
  --start-firm N             起始企业数（默认：1）
  --end-firm N               结束企业数（默认：10）
  --n-consumers N            消费者数量（默认：10）
  --search-cost COST         搜索成本（默认：0.02）
  --seed SEED                随机种子（默认：42）
  --models MODEL [MODEL ...] 模型列表（默认：deepseek-v3.2等）
```

---

## 🎯 混合模式组合

场景A支持8种混合模式（2³种组合）：

| 模式 | 分享 | 定价 | 搜索 | 命令行 | 用途 |
|-----|------|------|------|--------|------|
| 1 | 理性 | 理性 | 理性 | `--rational-share --rational-price --rational-search` | Ground Truth |
| 2 | LLM | 理性 | 理性 | `--rational-price --rational-search` | 测试LLM隐私决策 |
| 3 | 理性 | LLM | 理性 | `--rational-share --rational-search` | 测试LLM定价 |
| 4 | 理性 | 理性 | LLM | `--rational-share --rational-price` | 测试LLM搜索 |
| 5 | LLM | LLM | 理性 | `--rational-search` | LLM决策+理性搜索 |
| 6 | LLM | 理性 | LLM | `--rational-price` | LLM消费者行为 |
| 7 | 理性 | LLM | LLM | `--rational-share` | LLM市场行为 |
| 8 | LLM | LLM | LLM | 无参数 | 完全LLM |

---

## 📈 输出结果格式

### **JSON输出结构**

```json
{
  "model_name": "deepseek-v3.2",
  "scenario": "A_recommendation",
  "num_rounds": 5,
  "rational_share": false,
  "rational_price": false,
  "rational_search": false,
  "params": {
    "n_consumers": 10,
    "n_firms": 5,
    "search_cost": 0.02,
    "privacy_costs": [0.036, 0.053, ...],
    "v_dist": {"type": "uniform", "low": 0.0, "high": 1.0},
    "r_value": 0.8,
    "firm_cost": 0.0,
    "seed": 42
  },
  "delta": 0.057024,
  "all_rounds": [
    {
      "share_rate": 1.0,
      "avg_price": 0.6988,
      "consumer_surplus": 1.2988,
      "firm_profit": 5.5906,
      "social_welfare": 6.8894,
      "avg_search_cost": 0.0,
      "purchase_rate": 0.8,
      "consumers_data": [
        {
          "index": 0,
          "share": true,
          "share_reason": "...",
          "purchase_index": 2,
          "search_cost": 0.0,
          "utility": 0.1924,
          "privacy_cost": 0.0362
        },
        ...
      ],
      "firms_data": [
        {
          "index": 0,
          "price": 0.6988,
          "price_reason": "...",
          "sales_count": 2,
          "profit": 1.3976
        },
        ...
      ]
    },
    ...
  ],
  "averages": {
    "avg_share_rate": 0.95,
    "avg_price": 0.6950,
    "avg_consumer_surplus": 1.25,
    "avg_firm_profit": 5.50,
    "avg_social_welfare": 6.75,
    "avg_search_cost": 0.01,
    "avg_purchase_rate": 0.78
  }
}
```

### **关键指标说明**

| 指标 | 含义 | 典型值范围 |
|-----|------|----------|
| `share_rate` | 数据分享率 | 0.0 ~ 1.0 |
| `avg_price` | 平均企业价格 | 0.4 ~ 0.8 |
| `consumer_surplus` | 消费者总剩余 | -1.0 ~ 5.0 |
| `firm_profit` | 企业总利润 | 0.0 ~ 10.0 |
| `social_welfare` | 社会福利 | 0.0 ~ 15.0 |
| `avg_search_cost` | 平均搜索成本 | 0.0 ~ 0.05 |
| `purchase_rate` | 购买率 | 0.5 ~ 1.0 |

---

## 🔬 典型实验设计

### **实验1：企业数量对分享率的影响**

```bash
for n_firms in {1..10}; do
  python src/evaluators/evaluate_scenario_a_full.py \
    --rational-share --rational-price --rational-search \
    --n-firms $n_firms --rounds 1
done

# 预期观察：
# n_firms=1-2: share_rate=0.0（推荐价值低）
# n_firms=3: share_rate=0.2（部分分享）
# n_firms=4: share_rate=0.8（大部分分享）
# n_firms=5+: share_rate=1.0（全员分享）
```

### **实验2：LLM vs 理性决策对比**

```bash
# 理性基准
python src/evaluators/evaluate_scenario_a_full.py \
  --rational-share --rational-price --rational-search \
  --n-firms 5 --rounds 1

# LLM测试
python src/evaluators/evaluate_scenario_a_full.py \
  --model deepseek-v3.2 \
  --n-firms 5 --rounds 10

# 对比：
# - 分享率偏差
# - 定价偏差
# - 搜索行为差异
```

### **实验3：隐私成本异质性影响**

修改`scenario_a_recommendation.py`中的隐私成本分布：

```python
# 低隐私成本群体
privacy_costs = uniform(0.01, 0.03)  # → 高分享率

# 高隐私成本群体
privacy_costs = uniform(0.05, 0.08)  # → 低分享率
```

### **实验4：搜索成本影响**

```bash
# 低搜索成本
python src/evaluators/evaluate_scenario_a_full.py \
  --search-cost 0.01 --n-firms 5 --rounds 5

# 高搜索成本
python src/evaluators/evaluate_scenario_a_full.py \
  --search-cost 0.05 --n-firms 5 --rounds 5

# 预期：搜索成本越高，分享数据的价值越大
```

---

## 🐛 故障排除

### **问题1：LLM API调用失败**

```
错误：Connection timeout / API key invalid
解决：检查 configs/model_configs.json 中的API配置
```

### **问题2：JSON序列化错误**

```
错误：Object of type int64 is not JSON serializable
解决：已在save_results()中添加类型转换
```

### **问题3：搜索决策无响应**

```
错误：LLM返回无效的action
解决：
1. 检查prompt格式
2. 增加max_retries
3. 使用理性模式验证逻辑
```

---

## 📚 相关文档

- `docs/场景A重构说明.md` - 详细的重构过程和设计决策
- `docs/场景A原始实验参数分析.md` - 原始实验的参数推断
- `docs/场景A重构完成总结.md` - 完整的功能清单和验证结果

---

## 💡 最佳实践

### **1. 先运行理性基准**
```bash
# 建立准确的Ground Truth
python src/evaluators/evaluate_scenario_a_full.py \
  --rational-share --rational-price --rational-search \
  --n-consumers 10 --n-firms 5 --rounds 1
```

### **2. 然后测试LLM**
```bash
# 测试完全LLM模式
python src/evaluators/evaluate_scenario_a_full.py \
  --model deepseek-v3.2 \
  --n-consumers 10 --n-firms 5 --rounds 5
```

### **3. 分析偏差**
```python
import json

# 加载理性结果
with open("eval_A_full_deepseek-v3.2_rational_*.json") as f:
    rational = json.load(f)

# 加载LLM结果
with open("eval_A_full_deepseek-v3.2_*.json") as f:
    llm = json.load(f)

# 计算偏差
share_rate_error = abs(llm['averages']['avg_share_rate'] - 
                       rational['averages']['avg_share_rate'])
print(f"分享率偏差: {share_rate_error:.4f}")
```

### **4. 多模型对比**
```bash
# 批量运行多个模型
python scripts/run_scenario_a_sweep.py \
  --models deepseek-v3.2 gpt-5-mini-2025-08-07 qwen-plus \
  --start-firm 3 --end-firm 7

# 生成对比表格
```

---

## 🎓 核心概念

### **Delta（推荐效用增益）**
```
Delta = ∫_r^{v_high} [F_v - F_v^n] dv

物理含义：
- 通过推荐系统，消费者期望获得的额外效用
- 企业数n越大，Delta越大（推荐价值越高）
- 典型值：0.02 ~ 0.06
```

### **理性分享条件**
```
分享 ⟺ Delta + 搜索成本节省 >= 隐私成本

即：推荐带来的收益 > 隐私损失
```

### **最优停止规则（理性搜索）**
```
购买条件：v_i - p_i >= r - market_price

即：当前净收益 >= 期望后续净收益
```

### **贝叶斯纳什均衡（企业定价）**
```
max (p - c) * [σ * q_shared + (1-σ) * q_non_shared]

其中：
- σ: 数据分享率
- q_shared: 分享数据消费者的需求（推荐顺序）
- q_non_shared: 未分享消费者的需求（随机搜索）
```

---

## 📊 预期结果模式

### **企业数对分享率的影响（理性模式）**

| 企业数 | 分享率 | Delta | 说明 |
|-------|--------|-------|------|
| 1-2 | 0% | ~0.01 | 推荐价值太低 |
| 3 | 20% | ~0.024 | 低隐私成本者开始分享 |
| 4 | 80% | ~0.032 | 大部分消费者分享 |
| 5+ | 100% | ~0.057 | 全员分享 |

### **LLM vs 理性偏差（典型值）**

| 模型 | 分享率偏差 | 定价偏差 | 评级 |
|-----|-----------|---------|------|
| deepseek-v3.2 | +0.05 ~ +0.15 | ±0.03 | ✅ 良好 |
| gpt-5-mini | +0.10 ~ +0.20 | ±0.05 | ⚠️ 中等 |
| qwen-plus | +0.30 ~ +0.50 | +0.20 ~ +0.40 | ❌ 较差 |

---

## 🔗 整合到统一评估框架

### **在run_evaluation.py中添加场景A支持**

```python
# 在run_single_evaluation()中添加：

elif scenario == "A":
    # 场景A：推荐系统评估
    scenario_dir = Path(output_dir) / "scenario_a"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-u", "src/evaluators/evaluate_scenario_a_full.py",
        "--model", model_name,
        "--rounds", str(num_trials),
        "--n-consumers", "10",
        "--n-firms", "5",
        "--search-cost", "0.02",
        "--output-dir", str(scenario_dir)
    ]
    subprocess.run(cmd, check=True)
    
    # 加载结果
    result_files = sorted(scenario_dir.glob(f"eval_A_full_{model_name}*.json"))
    if result_files:
        with open(result_files[-1], 'r') as f:
            result = json.load(f)
        
        return {
            'scenario': 'A',
            'model': model_name,
            'share_rate': result['averages']['avg_share_rate'],
            'avg_price': result['averages']['avg_price'],
            'welfare': result['averages']['avg_social_welfare']
        }
```

---

## ✅ 验证清单

运行以下命令验证安装正确：

```bash
# 1. 测试理论求解器
python -c "from src.scenarios.scenario_a_recommendation import calculate_delta_sharing; print(calculate_delta_sharing({'low':0,'high':1}, 0.8, 5))"
# 预期输出：0.057024

# 2. 测试完整评估器（理性模式）
python src/evaluators/evaluate_scenario_a_full.py \
  --rational-share --rational-price --rational-search \
  --n-consumers 4 --n-firms 3 --rounds 1
# 预期：无错误，生成JSON文件

# 3. 测试批量脚本
python scripts/run_scenario_a_sweep.py \
  --start-firm 3 --end-firm 5 \
  --models deepseek-v3.2
# 预期：生成CSV汇总文件
```

---

## 🎓 技术细节

### **关键算法**

1. **Delta计算**：`scipy.integrate.quad` 数值积分
2. **价格优化**：`scipy.optimize.minimize_scalar` 有界优化
3. **固定点迭代**：简单迭代法求解分享率均衡
4. **最优停止**：阈值规则（v-p >= r-market_price）

### **数据流**

```
参数 → 生成消费者估值 → 分享决策 → 生成推荐序列
                                    ↓
                               企业定价 ← 分享率
                                    ↓
                         消费者搜索（按推荐/随机）
                                    ↓
                            购买决策（逐步判断）
                                    ↓
                          统计销量 → 计算利润/剩余
```

---

## 📞 支持

如有问题，请查阅：
1. `docs/场景A重构说明.md` - 详细技术文档
2. `docs/场景A原始实验参数分析.md` - 参数配置说明
3. 代码注释 - 函数文档字符串

或运行示例验证：
```bash
python src/evaluators/evaluate_scenario_a_full.py --help
```
