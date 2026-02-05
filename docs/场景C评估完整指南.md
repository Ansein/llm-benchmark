# 场景C评估完整指南

## 📋 前置条件

### ✅ 已完成
1. **Ground Truth生成** ✅
   - 文件：`data/ground_truth/scenario_c_common_preferences_optimal.json`
   - 最优策略：identified, m*=0.60 (统一补偿)
   - 参与率：r*=3.4%（低参与）

2. **代码修复** ✅
   - 支持向量形式的`m_star`
   - 修复评估器所有打印和比较逻辑
   - 修复度量计算函数
   - 4个文件共修改10+处

---

## 🚀 运行评估

### 方案1：虚拟博弈模式（推荐）

**优点**：
- 更快（无需真实多轮交互）
- 理论更严格（固定点均衡）
- 可独立运行B/C/D配置

#### 运行所有配置

```bash
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model gpt-4.1 \
    --rounds 50 \
    --belief_window 10
```

#### 单独运行配置

```bash
# 配置B_FP：理性中介 × LLM消费者（测试消费者学习）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config B \
    --model gpt-4.1 \
    --rounds 50

# 配置C_FP：LLM中介 × 理性消费者（测试中介学习）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config C \
    --model gpt-4.1 \
    --rounds 50

# 配置D_FP：LLM中介 × LLM消费者（测试双方学习）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config D \
    --model gpt-4.1 \
    --rounds 50
```

### 方案2：多轮迭代模式

**优点**：
- 真实多轮交互
- LLM从经验中学习

```bash
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model gpt-4.1 \
    --rounds 20
```

---

## ⏱️ 预计时间

### 虚拟博弈模式（50轮）
- **配置B_FP**: ~15-20分钟
- **配置C_FP**: ~15-20分钟  
- **配置D_FP**: ~25-30分钟
- **总计（all）**: ~55-70分钟

### 多轮迭代模式（20轮）
- **配置B+C+D**: ~60-90分钟

---

## 📊 输出文件

### 虚拟博弈模式

```
evaluation_results/scenario_c/
├── fp_configB_gpt-4.1/
│   ├── eval_20260201_123456.json          # 详细结果
│   ├── eval_20260201_123456_profit_rate.png      # 利润率曲线
│   └── eval_20260201_123456_strategy_evolution.png  # 策略演化
├── fp_configC_gpt-4.1/
│   └── ...
└── fp_configD_gpt-4.1/
    └── ...
```

### 多轮迭代模式

```
evaluation_results/scenario_c/
├── scenario_c_common_preferences_gpt-4.1_20260201_123456.csv
└── scenario_c_common_preferences_gpt-4.1_20260201_123456_detailed.json
```

---

## 📈 评估指标

### 1. 策略匹配度

```python
# 中介策略
{
  "m_llm": 0.58,              # LLM选择的补偿（均值）
  "m_theory": 0.60,           # GT最优补偿（均值）
  "m_absolute_error": 0.02,   # 绝对误差
  "m_relative_error": 0.033,  # 相对误差 (3.3%)
  
  "anon_llm": "identified",
  "anon_theory": "identified",
  "anon_match": 1,            # 策略匹配（0或1）
  
  "strategy_match": 1         # 完全匹配（0或1）
}
```

### 2. 利润表现

```python
{
  "profit_llm": 1.65,
  "profit_theory": 1.72,
  "profit_rate": 0.959,       # 相对GT的利润率 (95.9%)
  "profit_gap": 0.07
}
```

### 3. 参与率对比

```python
{
  "r_llm": 0.035,             # LLM策略下的参与率
  "r_theory": 0.034,          # GT参与率
  "participation_gap": 0.001  # 参与率偏差
}
```

### 4. 社会福利

```python
{
  "social_welfare_llm": 208.5,
  "social_welfare_theory": 209.4,
  "welfare_rate": 0.996       # 相对GT的福利 (99.6%)
}
```

---

## 🔍 分析关键问题

### 问题1：LLM能否找到identified策略？

**GT结果**：identified策略最优（意外！）

**分析重点**：
- 为什么identified比anonymized更优？
- 低参与率（3.4%）的原因
- 隐私成本τ=1.0 vs 补偿m=0.60

### 问题2：个性化补偿是否有价值？

**GT结果**：统一补偿最优（std≈0）

**理论解释**：
- Common Preferences + i.i.d. τ → 对称性
- 所有消费者"看起来"相同
- 个性化无优势

### 问题3：LLM学习曲线

**观察指标**：
- 前10轮：探索阶段
- 10-30轮：收敛阶段
- 30+轮：稳定阶段

---

## 📝 结果分析脚本

### 读取结果

```python
import json
import pandas as pd

# 虚拟博弈结果
with open('evaluation_results/scenario_c/fp_configD_gpt-4.1/eval_xxx.json') as f:
    results = json.load(f)

# 查看关键指标
print(f"最终策略: {results['final_intermediary_strategy']}")
print(f"利润率: {results['metrics']['profit_rate']:.2%}")
print(f"策略匹配: {results['metrics']['strategy_match']}")

# 多轮迭代结果
df = pd.read_csv('evaluation_results/scenario_c/scenario_c_xxx.csv')
print(df[['round', 'config', 'profit_rate', 'strategy_match']])
```

### 可视化

```python
import matplotlib.pyplot as plt

# 利润率趋势
rounds = results['round_history']
profit_rates = [r['metrics']['profit_rate'] for r in rounds]

plt.figure(figsize=(10, 6))
plt.plot(profit_rates, marker='o')
plt.axhline(y=1.0, color='r', linestyle='--', label='GT Baseline')
plt.xlabel('Round')
plt.ylabel('Profit Rate (vs GT)')
plt.title('LLM Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🐛 已知问题与解决

### ✅ 已修复

1. **GT文件缺少params_base** ✅
   - 添加了params_base字段

2. **向量m_star无法打印** ✅
   - 修改为显示均值和标准差

3. **向量m_star无法传递给消费者** ✅
   - 为每个消费者提取m[i]

4. **metrics计算无法处理向量** ✅
   - 使用均值进行对比

### ⚠️ 注意事项

1. **API配额**：
   - 每轮需要N+1次LLM调用
   - 50轮 × 21个调用 = 1050次
   - 确保API配额充足

2. **网络稳定性**：
   - 单次评估1-2小时
   - 建议在服务器上后台运行

3. **输出缓冲**：
   - Python默认行缓冲
   - 可能看不到实时输出
   - 添加`-u`标志：`python -u -m ...`

---

## 🎯 推荐工作流

### 第1步：快速测试（本地）

```bash
# 测试2轮确保代码正常
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model gpt-5.2 \
    --rounds 2
```

### 第2步：单配置测试（服务器）

```bash
# 测试配置D_FP（10轮）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config D \
    --model gpt-4.1 \
    --rounds 10
```

### 第3步：完整运行（服务器后台）

```bash
# 后台运行所有配置
nohup python -u -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model gpt-4.1 \
    --rounds 50 \
    > scenario_c_eval.log 2>&1 &

# 监控进度
tail -f scenario_c_eval.log
```

---

## 📚 参考资料

### 论文对应

- **配置A**：理论基准（Stackelberg均衡）
- **配置B**：消费者认知局限（Proposition 3讨论）
- **配置C**：中介策略学习（机制设计）
- **配置D**：双方学习（完整博弈）

### 相关文档

- `docs/场景C个性化补偿优化-完整流程.md`：优化算法详解
- `docs/场景C运行指南.md`：GT生成指南
- `src/evaluators/evaluate_scenario_c.py`：评估器源码

---

## ✅ 检查清单

评估前确认：

- [ ] GT文件存在且有效
- [ ] 模型配置正确（`configs/model_configs.json`）
- [ ] API密钥已设置
- [ ] 有足够的API配额
- [ ] 磁盘空间充足（结果+日志~100MB）
- [ ] 网络连接稳定

评估后检查：

- [ ] 所有配置成功完成
- [ ] 结果文件生成
- [ ] 可视化图表清晰
- [ ] 关键指标合理
- [ ] 日志无严重错误

---

**文档版本**: 1.0.0  
**更新日期**: 2026-02-01  
**状态**: 代码已修复，准备运行
