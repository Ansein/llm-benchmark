# 使用已有结果生成汇总报告

## 功能说明

`run_evaluation.py` 现在支持 `--summary-only` 模式，可以直接使用已有的评估结果生成汇总报告，**无需重新运行LLM评估**。

## 使用方法

### 基本用法

```bash
python run_evaluation.py --summary-only
```

这会：
1. 扫描 `evaluation_results/` 目录下的所有评估结果文件
2. 加载所有 `eval_scenario_*.json` 文件
3. 生成汇总报告和CSV文件

### 指定输出目录

```bash
python run_evaluation.py --summary-only --output-dir custom_results
```

从 `custom_results/` 目录加载结果并生成报告。

## 文件命名规则

评估结果文件必须遵循以下命名格式：

```
eval_scenario_{场景}_{模型名}.json
```

示例：
- `eval_scenario_A_gpt-4.1-mini.json`
- `eval_scenario_B_deepseek-v3.json`
- `eval_scenario_B_grok-3-mini.json`

## 输出文件

汇总报告会生成以下文件：

1. **控制台输出**：格式化的表格显示所有结果
2. **CSV文件**：`evaluation_results/summary_report_{时间戳}.csv`
3. **Markdown文件**：`evaluation_results/summary_report_{时间戳}.md`

## 完整工作流程示例

### 1. 运行评估（一次性或分批）

```bash
# 评估多个模型
python run_evaluation.py --scenarios B --models gpt-4.1-mini deepseek-v3 --num-trials 1 --max-iterations 15
```

这会生成：
- `evaluation_results/eval_scenario_B_gpt-4.1-mini.json`
- `evaluation_results/eval_scenario_B_deepseek-v3.json`

### 2. 稍后添加更多模型

```bash
# 添加新模型评估
python run_evaluation.py --scenarios B --models deepseek-r1 --num-trials 1 --max-iterations 15
```

这会生成：
- `evaluation_results/eval_scenario_B_deepseek-r1.json`

### 3. 生成汇总报告（使用所有已有结果）

```bash
# 无需重新运行LLM，直接生成报告
python run_evaluation.py --summary-only
```

这会读取所有3个结果文件并生成汇总报告！

## 优势

✅ **节省时间**：无需重新运行耗时的LLM评估  
✅ **节省成本**：不消耗API额度  
✅ **灵活性**：可以随时添加新模型的评估，然后重新生成报告  
✅ **可重复性**：基于相同的评估结果文件，报告生成是确定性的

## 注意事项

1. **文件完整性**：确保评估结果文件包含所有必需字段（`metrics`、`labels`、`iterations`等）
2. **文件命名**：严格遵循命名规则，否则文件可能被跳过
3. **版本兼容性**：确保结果文件格式与当前代码版本兼容

## 故障排查

### 问题：没有找到评估结果文件

**原因**：输出目录为空或文件命名不符合规则

**解决**：
```bash
# 检查目录内容
ls evaluation_results/

# 确保文件名格式正确
# 正确: eval_scenario_B_gpt-4.1-mini.json
# 错误: result_B_gpt-4.1-mini.json
```

### 问题：KeyError 或字段缺失

**原因**：评估结果文件格式不完整或版本不匹配

**解决**：重新运行该模型的评估，生成最新格式的结果文件

### 问题：某些结果被跳过

**原因**：文件名解析失败或文件内容损坏

**解决**：查看控制台输出，找到被跳过的文件，检查其命名和内容

## 命令对比

| 命令 | 功能 | LLM调用 | 适用场景 |
|------|------|---------|---------|
| `python run_evaluation.py --scenarios B --models gpt-4.1-mini` | 运行评估并生成报告 | ✅ 是 | 首次评估或更新评估 |
| `python run_evaluation.py --summary-only` | 仅生成报告 | ❌ 否 | 基于已有结果生成报告 |
| `python run_evaluation.py --single --scenarios B --models gpt-4.1-mini` | 单次评估（测试） | ✅ 是 | 测试单个模型 |
