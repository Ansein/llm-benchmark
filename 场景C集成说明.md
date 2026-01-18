# 场景C集成到run_evaluation.py完成说明

**完成时间**: 2026-01-16  
**状态**: ✅ **已完全集成**

---

## ✅ 已完成的集成工作

### 1. **导入模块**
```python
from src.evaluators.evaluate_scenario_c import ScenarioCEvaluator
```

### 2. **更新文档说明**
添加了场景C的使用示例：
```bash
# 场景C批量评估（社会数据外部性）
python run_evaluation.py --scenarios C --models gpt-4.1-mini --num-trials 3 --max-iterations 10

# 同时评估多个场景
python run_evaluation.py --scenarios A B C --models gpt-4.1-mini
```

### 3. **修改run_single_evaluation函数**
添加场景C的处理分支：
```python
elif scenario == "C":
    evaluator = ScenarioCEvaluator(llm_client)
    results = evaluator.evaluate(
        max_iterations=max_iterations,
        num_trials=num_trials
    )
```

### 4. **修改generate_summary_report函数**
添加场景C的指标输出：
```python
elif scenario == "C":
    row.update({
        "参与率_LLM": f"{metrics['llm']['participation_rate']:.2%}",
        "参与率_GT": f"{metrics['ground_truth']['participation_rate']:.2%}",
        "参与人数_LLM": f"{metrics['llm']['num_participants']}",
        "CS_MAE": f"{metrics['deviations']['consumer_surplus_mae']:.4f}",
        "利润MAE": f"{metrics['deviations']['producer_profit_mae']:.4f}",
        "福利MAE": f"{metrics['deviations']['social_welfare_mae']:.4f}",
        "Gini_MAE": f"{metrics['deviations']['gini_mae']:.4f}",
        "参与率分桶匹配": "[是]" if labels.get("bucket_match") else "[否]",
        "方向标签": labels.get("direction", "N/A")
    })
```

### 5. **更新命令行参数**
```python
choices=["A", "B", "C"],
help="要评估的场景列表 (A=个性化定价, B=推断外部性, C=社会数据)"
```

---

## 🚀 使用方法

### **方法1: 单个场景评估**
```bash
# 评估场景C（使用gpt-4.1-mini）
python run_evaluation.py --single --scenarios C --models gpt-4.1-mini --num-trials 3

# 快速测试（减少试验次数）
python run_evaluation.py --single --scenarios C --models gpt-4.1-mini --num-trials 1 --max-iterations 5
```

### **方法2: 批量评估（多个模型）**
```bash
# 评估场景C的多个模型
python run_evaluation.py --scenarios C --models gpt-4.1-mini deepseek-v3 gemini-2.5-flash --num-trials 3

# 同时评估所有场景
python run_evaluation.py --scenarios A B C --models gpt-4.1-mini --num-trials 1
```

### **方法3: 仅生成汇总报告**
```bash
# 使用已有的评估结果生成报告
python run_evaluation.py --summary-only
```

---

## 📊 输出文件

### **单次评估结果**
```
evaluation_results/eval_scenario_C_gpt-4.1-mini.json
```

**内容结构**：
```json
{
  "model_name": "gpt-4.1-mini",
  "scenario": "C",
  "converged": true,
  "iterations": 3,
  "llm_participation": [true, false, true, ...],
  "llm_participation_rate": 0.65,
  "gt_participation_rate": 0.70,
  "metrics": {
    "llm": {
      "participation_rate": 0.65,
      "num_participants": 13,
      "consumer_surplus": 12.34,
      "producer_profit": 8.56,
      "social_welfare": 20.90,
      ...
    },
    "ground_truth": {...},
    "deviations": {
      "participation_rate_mae": 0.05,
      "consumer_surplus_mae": 1.23,
      ...
    }
  },
  "labels": {
    "llm_participation_bucket": "medium",
    "gt_participation_bucket": "high",
    "bucket_match": false,
    "direction": "under_participation"
  }
}
```

### **汇总报告**
```
evaluation_results/summary_report_YYYYMMDD_HHMMSS.csv
evaluation_results/all_results_YYYYMMDD_HHMMSS.json
```

**CSV示例**：
```
场景 | 模型 | 收敛 | 迭代次数 | 参与率_LLM | 参与率_GT | CS_MAE | 利润MAE | 福利MAE | 参与率分桶匹配 | 方向标签
C    | gpt-4.1-mini | [是] | 3 | 65.00% | 70.00% | 1.2300 | 0.8900 | 2.1200 | [否] | under_participation
```

---

## 📋 完整的场景支持

现在`run_evaluation.py`支持三个场景：

| 场景 | 名称 | 类型 | 需要max_iterations |
|------|------|------|-------------------|
| **A** | 个性化定价与隐私选择 | 迭代博弈 | ✅ 是 |
| **B** | Too Much Data (推断外部性) | 静态博弈 | ❌ 否 |
| **C** | The Economics of Social Data | 迭代博弈 | ✅ 是 |

---

## 🔍 与其他场景的对比

### **场景A**
```bash
# 需要max-iterations（默认10）
python run_evaluation.py --scenarios A --models gpt-4.1-mini --num-trials 3 --max-iterations 15
```

**输出指标**：
- 披露率（LLM vs GT）
- 利润MAE, CS_MAE, 福利MAE
- 披露率分桶匹配
- 过度披露匹配

### **场景B**
```bash
# 不需要max-iterations（静态博弈）
python run_evaluation.py --scenarios B --models gpt-4.1-mini --num-trials 1
```

**输出指标**：
- 分享率（LLM vs GT）
- 分享集合相似度
- 利润MAE, 福利MAE, 泄露MAE
- 泄露分桶匹配
- 过度分享匹配

### **场景C** ✨ **新增**
```bash
# 需要max-iterations（默认10）
python run_evaluation.py --scenarios C --models gpt-4.1-mini --num-trials 3 --max-iterations 10
```

**输出指标**：
- 参与率（LLM vs GT）
- 参与人数
- CS_MAE, 利润MAE, 福利MAE
- Gini_MAE
- 参与率分桶匹配
- 方向标签（over/under/match）

---

## ⚠️ 注意事项

### **1. Ground Truth必须存在**
运行场景C评估前，确保已生成Ground Truth：
```bash
python generate_scenario_c_gt.py
# 或至少生成MVP配置：
python -m src.scenarios.scenario_c_social_data
```

**检查GT文件**：
```bash
ls data/ground_truth/scenario_c_result.json
```

### **2. API配置**
确保`configs/model_configs.json`已正确配置：
```json
{
  "gpt-4.1-mini": {
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini"
  }
}
```

### **3. 参数选择**
- **num_trials**: 建议3（平衡速度与稳定性）
- **max_iterations**: 场景C建议10-15（通常5-10次收敛）
- 快速测试可用: `--num-trials 1 --max-iterations 5`

### **4. 运行时间估计**
- 场景C单次评估（N=20, trials=3, max_iter=10）: 约5-10分钟
- 取决于：LLM响应速度、收敛速度、试验次数

---

## 🧪 测试验证

### **快速测试（不调用LLM）**
```bash
# 测试代码正确性（已完成）
python test_scenario_c.py
```

### **端到端测试（调用LLM）**
```bash
# 单次评估测试
python run_evaluation.py --single --scenarios C --models gpt-4.1-mini --num-trials 1 --max-iterations 5

# 检查输出文件
ls evaluation_results/eval_scenario_C_gpt-4.1-mini.json
```

---

## 📈 实验建议

### **Phase 1: MVP测试**
```bash
# 单个模型、默认配置
python run_evaluation.py --single --scenarios C --models gpt-4.1-mini --num-trials 3
```

### **Phase 2: 多模型对比**
```bash
# 评估多个LLM
python run_evaluation.py --scenarios C \
  --models gpt-4.1-mini deepseek-v3 gemini-2.5-flash grok-3-mini \
  --num-trials 3
```

### **Phase 3: 全场景评估**
```bash
# 评估所有场景
python run_evaluation.py --scenarios A B C --models gpt-4.1-mini --num-trials 1
```

---

## 🐛 故障排查

### **问题1: Ground Truth不存在**
```
错误: [Errno 2] No such file or directory: 'data/ground_truth/scenario_c_result.json'

解决: 
python generate_scenario_c_gt.py
```

### **问题2: 导入错误**
```
错误: No module named 'src.evaluators.evaluate_scenario_c'

解决:
检查文件是否存在: src/evaluators/evaluate_scenario_c.py
从项目根目录运行: cd d:\benchmark
```

### **问题3: LLM API错误**
```
错误: API authentication failed

解决:
检查 configs/model_configs.json 中的 api_key
测试API连接: python -m src.evaluators.llm_client
```

---

## ✅ 集成完成检查清单

- [x] 导入ScenarioCEvaluator
- [x] 更新文档字符串
- [x] 修改run_single_evaluation函数
- [x] 修改generate_summary_report函数
- [x] 更新命令行参数choices
- [x] 更新注释说明
- [x] 无linter错误
- [ ] 生成Ground Truth（待运行）
- [ ] 端到端测试（待运行）

---

## 🎯 下一步

1. **生成Ground Truth**
   ```bash
   python generate_scenario_c_gt.py
   ```

2. **运行LLM评估测试**
   ```bash
   python run_evaluation.py --single --scenarios C --models gpt-4.1-mini --num-trials 1
   ```

3. **批量评估实验**
   ```bash
   python run_evaluation.py --scenarios C --models gpt-4.1-mini deepseek-v3 --num-trials 3
   ```

---

**集成完成！** ✅  
场景C现在已经完全集成到主评估脚本中，可以像场景A和B一样使用！🎉
