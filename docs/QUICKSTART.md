# 🚀 快速开始指南

## ✅ 系统已就绪

你现在有一个完整的LLM benchmark评估系统！

## 📁 文件清单

### 核心代码
- `llm_client.py` - LLM客户端封装
- `evaluate_scenario_a.py` - 场景A评估器（个性化定价）
- `evaluate_scenario_b.py` - 场景B评估器（推断外部性）
- `run_evaluation.py` - 主评估脚本

### 配置文件
- `model_configs.json` - LLM模型配置

### Ground Truth
- `scenario_a_result.json` - 场景A的理论基准
- `scenario_b_result.json` - 场景B的理论基准

### 测试脚本
- `test_evaluation.py` - 快速测试脚本

### 文档
- `README_evaluation.md` - 详细使用说明
- `QUICKSTART.md` - 本文件

## 🎯 运行命令

### 1. 快速测试（已通过✅）

```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"
python test_evaluation.py
```

### 2. 单个场景评估

```bash
# 场景A - 个性化定价与隐私选择
python run_evaluation.py --single --scenarios A --models gpt-4.1-mini

# 场景B - Too Much Data（推断外部性）
python run_evaluation.py --single --scenarios B --models gpt-4.1-mini
```

### 3. 批量评估（推荐）

```bash
# 评估所有场景和所有模型
python run_evaluation.py --scenarios A B --models gpt-4.1-mini deepseek-v3 grok-3-mini

# 或者单独指定
python run_evaluation.py --scenarios A B --models gpt-4.1-mini
```

### 4. 自定义参数

```bash
python run_evaluation.py \
  --scenarios A B \
  --models gpt-4.1-mini deepseek-v3 \
  --num-trials 5 \
  --max-iterations 15 \
  --output-dir my_evaluation_results
```

## 📊 测试结果示例

根据刚才的测试：

### 场景A（个性化定价）
- ✅ LLM找到了与理论均衡完全一致的披露集合
- ✅ MAE = 0（完美匹配！）
- ✅ 所有标签都正确

### 场景B（推断外部性）
- ⚠️ LLM选择不分享（过于谨慎）
- 理论均衡：3人分享
- MAE较大，说明LLM需要更好理解推断外部性

这正是benchmark的价值：**发现不同LLM在理解经济学概念上的差异**！

## 📈 预期输出

运行后会生成：

```
evaluation_results/
├── eval_scenario_A_gpt-4.1-mini.json
├── eval_scenario_B_gpt-4.1-mini.json
├── eval_scenario_A_deepseek-v3.json
├── eval_scenario_B_deepseek-v3.json
├── summary_report_20260113_154530.csv
└── all_results_20260113_154530.json
```

## 💡 下一步建议

1. **运行完整评估**（使用默认参数）：
   ```bash
   python run_evaluation.py --scenarios A B --models gpt-4.1-mini
   ```

2. **对比多个模型**：
   ```bash
   python run_evaluation.py --scenarios A B --models gpt-4.1-mini deepseek-v3 grok-3-mini
   ```

3. **增加决策稳定性测试**：
   ```bash
   python run_evaluation.py --scenarios A --models gpt-4.1-mini --num-trials 10
   ```

4. **分析结果**：
   - 查看`summary_report_*.csv`了解各模型对比
   - 查看单个JSON文件了解详细的收敛过程
   - 关注MAE指标（越小越好）和标签一致性（✅越多越好）

## ⚠️ 注意事项

1. **API调用成本**：
   - 每个场景约需100-300次LLM调用
   - 建议先用`--single`测试单个模型

2. **运行时间**：
   - 单个场景+单个模型：约5-15分钟
   - 2场景×3模型：约30-90分钟

3. **编码问题**（Windows）：
   - 在PowerShell中运行前先执行：`$env:PYTHONIOENCODING="utf-8"`
   - 或者创建一个批处理文件设置环境变量

## 🔧 故障排除

### 问题1：编码错误
```
UnicodeEncodeError: 'gbk' codec...
```
**解决**：在运行前执行 `$env:PYTHONIOENCODING="utf-8"`

### 问题2：API调用失败
```
❌ LLM调用失败...
```
**解决**：检查`model_configs.json`中的API密钥和base_url是否正确

### 问题3：Import错误
```
ImportError: cannot import name...
```
**解决**：确保所有依赖已安装：`pip install openai pandas numpy`

## 📖 更多信息

- 详细使用说明：`README_evaluation.md`
- 设计方案：`最终设计方案.md`
- Ground truth生成：`README_scenarios.md`

## 🎯 核心评估指标

1. **偏差指标（MAE）**：
   - 平台利润偏差
   - 消费者剩余偏差
   - 社会福利偏差
   - 披露/分享率偏差

2. **标签一致性**：
   - 披露率分桶（low/medium/high）
   - 过度披露/分享判断
   - 政策效果判断

3. **收敛性**：
   - 是否收敛到稳定均衡
   - 收敛迭代次数
   - 决策稳定性（多次试验的一致性）

---

**现在就开始评估吧！** 🚀
