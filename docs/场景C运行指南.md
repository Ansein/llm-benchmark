# 场景C：运行指南

## 📋 概述

场景C是《The Economics of Social Data》的实现，包含以下部分：
1. **理论解生成**（Ground Truth）
2. **LLM评估**（4种配置：A/B/C/D）
3. **性能优化**（个性化补偿m_i + 利润约束R>0 + 关键词优化）

---

## 🎯 完整工作流程

```
第1步: 生成理论解GT
    ↓
第2步: 运行LLM评估
    ↓
第3步: 分析结果
```

---

## 📊 第1步：生成理论解（Ground Truth）

### 当前配置

理论解生成使用**混合优化方法**：
- **Grid Search**（粗搜索）：快速找到好的初始点
- **scipy L-BFGS-B**（精细优化）：从初始点进行连续优化
- **个性化补偿**：m变成N维向量（每个消费者可以不同）
- **利润约束**：强制R > 0，否则返回no_participation
- **CPU并行**：默认使用所有CPU核心加速Grid Search

### 运行命令

```bash
# 生成common_preferences的理论解
python -m src.scenarios.generate_scenario_c_gt

# 或者直接运行（Windows PowerShell）
python src/scenarios/generate_scenario_c_gt.py
```

### 关键参数

在`src/scenarios/generate_scenario_c_gt.py`中可以调整：

```python
# 优化参数
optimization_method='hybrid'        # 混合方法：Grid Search + L-BFGS-B
num_mc_samples=30                   # ΔU和m_0的MC样本数
max_iter=10                         # 固定点迭代最大轮数
grid_size=11                        # Grid Search的网格密度
n_jobs=-1                           # CPU并行数（-1=使用所有核心）

# 数据生成参数
N=20                                # 消费者数量
mu_theta=50.0                       # 偏好先验均值
sigma_theta=10.0                    # 偏好先验标准差
sigma=5.0                           # 信号噪声
tau_mean=2.0                        # 平均隐私成本
tau_std=1.0                         # 隐私成本标准差
```

### 输出文件

```
data/ground_truth/scenario_c_common_preferences_optimal.json
```

**内容结构**：
```json
{
  "timestamp": "2026-01-28 ...",
  "optimization_method": "hybrid",
  "optimal_strategy": {
    "m_star": [1.2, 1.3, 1.1, ...],  // N维个性化补偿
    "m_avg": 1.2,                     // 平均补偿
    "m_std": 0.1,                     // 补偿标准差
    "anonymization_star": "anonymized",
    "r_star": 0.65,                   // 均衡参与率
    "intermediary_profit_star": 0.5   // 中介利润
  },
  "equilibrium": {
    "consumer_surplus": 15.3,
    "producer_profit": 8.7,
    "intermediary_profit": 0.5,
    "social_welfare": 24.5,
    // ... 更多指标
  },
  "sample_data": [...],               // 样本数据（用于LLM评估）
  "optimization_details": {...}       // 优化过程详情
}
```

### 预期运行时间

**混合优化 + CPU并行（8核）**：
- Grid Search阶段：~30秒（11个网格点并行）
- L-BFGS-B优化：~2-3分钟（迭代20-30次）
- 均衡结果生成：~30秒
- **总计：约3-5分钟**

---

## 🤖 第2步：运行LLM评估

场景C支持**4种配置**的评估：

| 配置 | 中介 | 消费者 | 描述 |
|-----|------|--------|------|
| **A** | 理性 | 理性 | **理论基准**（Ground Truth） |
| **B** | 理性 | LLM | 测试**消费者**的理性程度 |
| **C** | LLM | 理性 | 测试**中介**的决策质量 |
| **D** | LLM | LLM | **完整LLM博弈**（多轮迭代） |

### 2.1 多轮迭代模式（默认）

**评估配置B、C、D**：

```bash
# 使用deepseek-v3.2模型
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2

# 使用gpt-4模型
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model gpt-4

# 自定义迭代轮数（默认20轮）
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2 \
    --rounds 30
```

**输出文件**：
```
evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_20260128_153045.csv
evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_20260128_153045_detailed.json
```

### 2.2 虚拟博弈模式（Fixed-Point）

**更快的收敛方法**：

```bash
# 运行所有三个配置（推荐）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model deepseek-v3.2

# 单独运行配置B_FP：理性中介 × LLM消费者
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config B \
    --model deepseek-v3.2

# 单独运行配置C_FP：LLM中介 × 理性消费者
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config C \
    --model deepseek-v3.2

# 单独运行配置D_FP：LLM中介 × LLM消费者
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config D \
    --model deepseek-v3.2

# 自定义参数
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model deepseek-v3.2 \
    --rounds 50 \
    --belief_window 10
```

### 2.3 测试评估器（使用模拟LLM）

**快速验证评估器功能**：

```bash
# 使用模拟的LLM代理进行测试
python test_scenario_c_evaluator.py
```

**输出**：
```
evaluation_results/scenario_c_test_report.csv
evaluation_results/scenario_c/scenario_c_test_detailed.json
```

---

## 🎨 第3步：使用中介提示词优化

### 3.1 关键词提取功能

**已实现的模块**：

1. **专家词表**（`src/scenarios/scenario_c_keywords_vocabulary.py`）
   - 933个关键词，86个类别
   - 基于论文理论设计

2. **关键词提取**（`src/scenarios/scenario_c_reason_keywords.py`）
   - 提取单条理由关键词
   - 批量总结迭代历史
   - 压缩比分析

3. **中介提示词生成**（`src/scenarios/scenario_c_intermediary_prompts.py`）
   - 完整的中介LLM提示词
   - 集成关键词优化
   - 自动压缩效果分析

### 3.2 测试关键词提取

```bash
# 测试关键词提取（使用专家词表）
python -c "import sys; sys.path.insert(0, 'src'); from scenarios.scenario_c_reason_keywords import example_usage; example_usage()"

# 测试中介提示词生成
python -c "import sys; sys.path.insert(0, 'src'); from scenarios.scenario_c_intermediary_prompts import example_intermediary_prompt; example_intermediary_prompt()"
```

### 3.3 集成到LLM评估中

**在配置C和D中使用关键词优化**：

修改你的LLM中介代理，使用`generate_intermediary_prompt_with_keywords()`：

```python
from src.scenarios.scenario_c_intermediary_prompts import (
    IntermediaryContext,
    generate_intermediary_prompt_with_keywords
)

def llm_intermediary_with_keywords(iteration_history, market_state):
    """使用关键词优化的LLM中介"""
    
    # 创建上下文
    context = IntermediaryContext(
        current_m=market_state['m'],
        current_anonymization=market_state['anonymization'],
        current_iteration=market_state['iteration'],
        current_participation_rate=market_state['r'],
        current_profit=market_state['profit'],
        iteration_history=iteration_history,
        N=market_state['N'],
        theta_prior_mean=50.0,
        theta_prior_std=10.0,
        tau_mean=2.0
    )
    
    # 生成优化后的提示词
    prompt = generate_intermediary_prompt_with_keywords(
        context,
        use_keywords=True,           # 使用关键词压缩
        max_keywords_per_category=5  # 每类保留top5
    )
    
    # 调用LLM
    response = call_llm(prompt, model='deepseek-v3.2')
    
    # 解析决策
    next_m, next_anonymization = parse_response(response)
    
    return next_m, next_anonymization
```

**预期效果**：
- **提示词长度**：减少60-80%（10轮迭代后）
- **Token成本**：节省60-80%
- **响应速度**：提升30-50%
- **决策质量**：保持95%+信息

---

## 📈 第4步：分析结果

### 4.1 查看CSV报告

```bash
# 使用pandas查看
python
>>> import pandas as pd
>>> df = pd.read_csv('evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_20260128_153045.csv')
>>> print(df)
```

**报告列**：
- `config`: 配置名称（A/B/C/D/B_FP/C_FP/D_FP）
- `m_decision`: 中介选择的补偿
- `anonymization_decision`: 匿名化策略
- `participation_rate`: 参与率
- `intermediary_profit`: 中介利润
- `consumer_surplus`: 消费者剩余
- `producer_profit`: 生产者利润
- `social_welfare`: 社会福利
- `regret_*`: 各项遗憾值（与GT对比）
- `decision_accuracy`: 决策准确度

### 4.2 查看详细JSON

```python
import json
with open('evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_20260128_153045_detailed.json', 'r', encoding='utf-8') as f:
    detailed = json.load(f)

# 查看配置D的迭代历史
print(detailed['config_D']['iteration_history'])

# 查看消费者理由
for record in detailed['config_D']['iteration_history']:
    print(f"消费者{record['consumer_id']}: {record['participation']} - {record['reason']}")
```

---

## 🚀 完整示例流程

### 示例1：完整运行（理论解 + LLM评估）

```bash
# Step 1: 生成理论解（约3-5分钟）
python -m src.scenarios.generate_scenario_c_gt

# Step 2: 运行LLM评估（约15-30分钟，取决于LLM速度）
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2 \
    --rounds 20

# Step 3: 查看结果
python -c "import pandas as pd; df = pd.read_csv('evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_*.csv', glob=True); print(df)"
```

### 示例2：快速测试（使用模拟LLM）

```bash
# 使用模拟LLM代理（秒级完成）
python test_scenario_c_evaluator.py
```

### 示例3：虚拟博弈模式（更快收敛）

```bash
# 生成理论解
python -m src.scenarios.generate_scenario_c_gt

# 运行虚拟博弈（更快）
python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model deepseek-v3.2 \
    --rounds 50
```

---

## ⚙️ 参数调优指南

### 理论解生成

**加快速度**：
```python
# 在 generate_scenario_c_gt.py 中修改：
num_mc_samples=20        # 减少MC样本（默认30）
max_iter=8               # 减少固定点迭代（默认10）
grid_size=7              # 减少网格密度（默认11）
n_jobs=-1                # 使用所有CPU核心（已经是最快）
```

**提高精度**：
```python
num_mc_samples=50        # 增加MC样本
max_iter=15              # 增加固定点迭代
grid_size=15             # 增加网格密度
```

### LLM评估

**调整迭代轮数**：
```bash
# 更少轮数（更快，但可能未收敛）
--rounds 10

# 更多轮数（更慢，但更稳定）
--rounds 30
```

**调整信念窗口**（虚拟博弈模式）：
```bash
# 更小窗口（更快收敛）
--belief_window 5

# 更大窗口（更稳定）
--belief_window 15
```

---

## 🔧 故障排查

### 问题1：`ModuleNotFoundError: No module named 'src'`

**解决方案**：
```bash
# 方法1：使用 -m 参数
python -m src.scenarios.generate_scenario_c_gt

# 方法2：设置PYTHONPATH（PowerShell）
$env:PYTHONPATH="D:\benchmark"
python src/scenarios/generate_scenario_c_gt.py

# 方法3：在脚本开头添加
import sys
sys.path.insert(0, 'D:/benchmark')
```

### 问题2：理论解生成太慢

**原因**：连续优化需要大量函数调用

**解决方案**：
1. 减少MC样本数：`num_mc_samples=20`
2. 减少固定点迭代：`max_iter=8`
3. 减少网格密度：`grid_size=7`
4. 确保CPU并行开启：`n_jobs=-1`

### 问题3：LLM评估失败

**可能原因**：
- API密钥未配置
- 模型名称错误
- 网络问题

**解决方案**：
```bash
# 1. 检查API配置
# 查看 configs/model_configs.json

# 2. 使用模拟LLM测试
python test_scenario_c_evaluator.py

# 3. 检查网络连接
curl https://api.openai.com/v1/models
```

### 问题4：关键词提取Unicode错误

**解决方案**：已修复，使用ASCII兼容的符号

如果仍有问题：
```python
# 在脚本开头添加
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

---

## 📚 相关文档

- **理论基础**：`papers/The Economics of Social Data.pdf`
- **代码实现**：
  - 核心求解器：`src/scenarios/scenario_c_social_data.py`
  - 连续优化：`src/scenarios/scenario_c_social_data_optimization.py`
  - GT生成：`src/scenarios/generate_scenario_c_gt.py`
  - LLM评估：`src/evaluators/evaluate_scenario_c.py`
- **优化文档**：
  - 中介提示词优化：`docs/场景C中介提示词优化-完整实现.md`
  - 关键词提取方案：`docs/场景C消费者理由关键词提取方案.md`

---

## 💡 最佳实践

### 1. 首次运行

```bash
# Step 1: 测试评估器（快速验证）
python test_scenario_c_evaluator.py

# Step 2: 生成理论解
python -m src.scenarios.generate_scenario_c_gt

# Step 3: 运行小规模LLM评估
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2 \
    --rounds 5
```

### 2. 正式实验

```bash
# 生成最新理论解
python -m src.scenarios.generate_scenario_c_gt

# 运行完整评估
python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2 \
    --rounds 20

# 备份结果
cp evaluation_results/scenario_c/scenario_c_common_preferences_*.csv backup/
```

### 3. 云端运行

```bash
# 云GPU环境（使用CPU并行加速）
export PYTHONPATH=/path/to/benchmark
nohup python -m src.scenarios.generate_scenario_c_gt > gt_generation.log 2>&1 &
nohup python -m src.evaluators.evaluate_scenario_c \
    --mode iterative \
    --model deepseek-v3.2 \
    --rounds 30 > evaluation.log 2>&1 &

# 监控进度
tail -f gt_generation.log
tail -f evaluation.log
```

---

**文档版本**: 1.0.0  
**最后更新**: 2026-01-28  
**作者**: AI Assistant

---

## 🎯 快速参考

### 常用命令

```bash
# 生成理论解
python -m src.scenarios.generate_scenario_c_gt

# 运行LLM评估
python -m src.evaluators.evaluate_scenario_c --mode iterative --model deepseek-v3.2

# 测试评估器
python test_scenario_c_evaluator.py

# 测试关键词提取
python -c "import sys; sys.path.insert(0, 'src'); from scenarios.scenario_c_reason_keywords import example_usage; example_usage()"

# 查看结果
python -c "import pandas as pd; df = pd.read_csv('evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_*.csv'); print(df)"
```

### 重要文件路径

```
理论解GT: data/ground_truth/scenario_c_common_preferences_optimal.json
评估结果: evaluation_results/scenario_c/
核心代码: src/scenarios/scenario_c_social_data.py
优化代码: src/scenarios/scenario_c_social_data_optimization.py
关键词词表: src/scenarios/scenario_c_keywords_vocabulary.py
中介提示词: src/scenarios/scenario_c_intermediary_prompts.py
```
