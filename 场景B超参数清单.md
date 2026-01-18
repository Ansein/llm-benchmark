# 场景B超参数清单

**文档日期**: 2026-01-15  
**适用代码**: 静态博弈版本

---

## 一、场景生成参数（Ground Truth）

位置：`src/scenarios/scenario_b_too_much_data.py` - `generate_instance()`

### 1.1 核心参数

| 参数名 | 默认值 | 含义 | 影响 | 建议取值范围 |
|--------|--------|------|------|--------------|
| `n` | 10 | 用户数量 | 计算复杂度O(2^n) | 8-12（论文建议） |
| `rho` | 0.6 | 类型相关系数 | 外部性强度 | {0, 0.3, 0.6, 0.9} |
| `sigma_noise_sq` | 0.1 | 观测噪声方差 | 推断精度 | 0.05-0.2 |
| `alpha` | 1.0 | 平台收益系数 | 规范化参数 | 固定为1.0 |
| `seed` | 42 | 随机种子 | 可复现性 | 任意整数 |

### 1.2 隐私偏好分布参数

| 参数名 | 默认值 | 含义 | 位置 |
|--------|--------|------|------|
| `v_min` | 0.3 | 隐私偏好下界 | `generate_instance()` L73 |
| `v_max` | 1.2 | 隐私偏好上界 | `generate_instance()` L73 |
| 分布类型 | Uniform | 当前是均匀分布 | 可改为两点分布 |

**公式**: `v ~ Uniform[0.3, 1.2]`

**替代方案**（论文建议）：
- **两点分布**: 50% v=0.2, 50% v=1.5
- **对数正态**: LogNormal(μ, σ)

---

## 二、评估器参数

位置：`src/evaluators/evaluate_scenario_b.py`

### 2.1 相关性压缩参数

| 参数名 | 默认值 | 含义 | 代码位置 |
|--------|--------|------|----------|
| `strong_corr_threshold` | 0.5 | "强相关"定义阈值 | L78 |
| `topk` | min(3, n-1) | 显示最强相关邻居数 | L79 |

**用途**：将n×n协方差矩阵压缩为每个用户的摘要信息

---

### 2.2 LLM查询参数

| 参数名 | 默认值 | 含义 | 代码位置 |
|--------|--------|------|----------|
| `num_trials` | 1 | 每个决策的重复查询次数 | `query_*()` 方法 |
| `max_retries` | 1 | 查询失败时的最大重试次数 | L375, L455 |

**说明**：
- `num_trials=1`: 节省成本，适合调试
- `num_trials=3`: 多数投票，提高鲁棒性
- `max_retries=1`: 失败时重试1次

---

### 2.3 默认值（容错机制）

| 场景 | 参数 | 默认值 | 代码位置 |
|------|------|--------|----------|
| 平台报价失败 | `uniform_price` | 0.5 | L408 |
| 平台报价失败 | `belief_share_rate` | 0.5 | L409 |
| 用户决策失败 | `share` | 0 | L481 |
| 用户决策失败 | `belief_share_rate` | 0.5 | L482 |
| JSON解析失败 | `belief_share_rate` | 0.5 | L391, L473 |

---

### 2.4 v值分组边界

用于决策模式分析（`print_evaluation_summary()`）

| 分组 | 条件 | 代码位置 |
|------|------|----------|
| 低v组 | v < 0.6 | L767 |
| 中v组 | 0.6 ≤ v < 0.9 | L768 |
| 高v组 | v ≥ 0.9 | L769 |

**用于Prompt的v相对位置判断**：

| 判断 | 条件 | 代码位置 |
|------|------|----------|
| 偏低 | v < v_mean - 0.2 | L254 |
| 中等 | v_mean - 0.2 ≤ v < v_mean + 0.2 | L257 |
| 偏高 | v ≥ v_mean + 0.2 | L259 |

其中 `v_mean = (v_min + v_max) / 2 = 0.75`

---

## 三、均衡质量评估参数

### 3.1 Jaccard相似度阈值

位置：`simulate_static_game()` L633-634

| 指标 | 阈值 | 含义 |
|------|------|------|
| `correct_equilibrium` | jaccard ≥ 0.6 | 判定为"正确均衡" |
| `equilibrium_type` | jaccard ≥ 0.6 → "good"<br>否则 → "bad" | 均衡质量分类 |

**公式**: `Jaccard(A, B) = |A ∩ B| / |A ∪ B|`

---

### 3.2 泄露量分桶阈值

位置：`src/scenarios/scenario_b_too_much_data.py` L297-302

| 分桶 | 条件 | 代码位置 |
|------|------|----------|
| "low" | leakage_ratio < 0.33 | L297 |
| "med" | 0.33 ≤ leakage_ratio < 0.67 | L299 |
| "high" | leakage_ratio ≥ 0.67 | L301 |

**公式**: `leakage_ratio = total_leakage / (n × 1.0)`

---

### 3.3 分享率分桶阈值

位置：`_bucket_share_rate()` L711-715

| 分桶 | 条件 |
|------|------|
| "low" | rate < 0.3 |
| "medium" | 0.3 ≤ rate < 0.7 |
| "high" | rate ≥ 0.7 |

---

## 四、批量评估参数

位置：`run_evaluation.py` - `main()`

| 参数名 | 默认值 | 含义 | 代码位置 |
|--------|--------|------|----------|
| `--scenarios` | ["A", "B"] | 评估场景列表 | L295 |
| `--models` | ["gpt-4.1-mini"] | 模型列表 | L304 |
| `--num-trials` | 3 | 每个决策重复次数 | L311 |
| `--max-iterations` | 10 | 最大迭代次数（场景B不用） | L318 |
| `--output-dir` | "evaluation_results" | 输出目录 | L325 |

**注意**: 场景B的静态博弈不使用`--max-iterations`参数

---

## 五、Prompt中的硬编码参数

### 5.1 隐私偏好边界（在Prompt中显示）

位置：多处（平台和用户Prompt）

```python
v_min, v_max = 0.3, 1.2  # L132, L241
```

**用途**：告知LLM隐私偏好的分布范围

### 5.2 相关性结构展示

位置：`build_user_decision_prompt()`

| 参数 | 值 | 说明 |
|------|-----|------|
| `neighbors_str` | TopK邻居 | 格式化显示最强相关邻居 |
| `strong_neighbors_count` | 阈值>0.5 | 强相关邻居数量 |

---

## 六、建议的参数扫描实验

### 6.1 相关性强度扫描

```python
rho_values = [0, 0.3, 0.6, 0.9]
```

**目的**：验证"相关性越强 → 外部性越强 → 过度分享更严重"

### 6.2 隐私偏好分布扫描

**方案1：均匀分布**
```python
v ~ Uniform[0.3, 1.2]  # 当前默认
```

**方案2：两点分布**
```python
v ∈ {0.2 (50%), 1.5 (50%)}
```

**目的**：研究异质性对均衡的影响

### 6.3 用户数量扫描

```python
n_values = [8, 10, 12]
```

**注意**：n>12时，枚举2^n个集合会很慢

### 6.4 噪声水平扫描

```python
sigma_noise_sq_values = [0.05, 0.1, 0.2]
```

**目的**：研究观测精度对外部性的影响

---

## 七、超参数调整建议

### 7.1 调试阶段

```python
n = 8                    # 小规模，快速验证
rho = 0.6               # 中等相关性
num_trials = 1          # 单次查询，节省成本
max_retries = 1         # 失败重试1次
```

### 7.2 正式实验

```python
n = 10                  # 论文推荐规模
rho ∈ {0, 0.3, 0.6, 0.9}  # 完整扫描
num_trials = 3          # 多数投票
max_retries = 1         # 保持容错
```

### 7.3 大规模分析

```python
n = 12                  # 最大可行规模（2^12=4096种集合）
# 注意：n=15时2^15=32768，求解会很慢
```

---

## 八、超参数修改位置索引

| 要修改的内容 | 文件 | 行号/函数 |
|-------------|------|-----------|
| 用户数n | `scenario_b_too_much_data.py` | L59 `generate_instance()` |
| 相关系数ρ | `scenario_b_too_much_data.py` | L59 `generate_instance()` |
| 隐私偏好范围 | `scenario_b_too_much_data.py` | L73 |
| 噪声方差 | `scenario_b_too_much_data.py` | L68 |
| 强相关阈值 | `evaluate_scenario_b.py` | L78 |
| TopK邻居数 | `evaluate_scenario_b.py` | L79 |
| 重试次数 | `evaluate_scenario_b.py` | L375, L455 |
| Jaccard阈值 | `evaluate_scenario_b.py` | L633 |
| 泄露分桶阈值 | `scenario_b_too_much_data.py` | L297-302 |
| v分组边界 | `evaluate_scenario_b.py` | L767-769 |
| 默认重复次数 | `run_evaluation.py` | L311 |

---

## 九、参数依赖关系

```
n (用户数)
  ↓
  ├─→ topk = min(3, n-1)
  ├─→ max_possible_leakage = n × 1.0
  └─→ 计算复杂度 O(2^n)

rho (相关系数)
  ↓
  └─→ 外部性强度
      └─→ 均衡分享率

v_min, v_max
  ↓
  ├─→ v_mean = (v_min + v_max) / 2
  ├─→ Prompt中的分布说明
  └─→ v分组边界判断
```

---

## 十、重要注意事项

### 10.1 不应修改的"参数"

以下是理论推导的结果，**不建议修改**：

- `alpha = 1.0`（平台收益系数，用于规范化）
- 泄露计算公式（基于贝叶斯后验方差）
- 边际泄露定义（I_i(分享) - I_i(不分享)）

### 10.2 需要同步修改的参数

如果修改了`v_min`或`v_max`，需要同步修改：
1. `scenario_b_too_much_data.py` L73
2. `evaluate_scenario_b.py` L132（平台Prompt）
3. `evaluate_scenario_b.py` L241（用户Prompt）
4. v分组边界可能需要调整（L767-769）

### 10.3 计算复杂度限制

| n | 枚举集合数 | 求解时间（参考） |
|---|-----------|-----------------|
| 8 | 256 | <1秒 |
| 10 | 1024 | 1-2秒 |
| 12 | 4096 | 5-10秒 |
| 15 | 32768 | 30-60秒 |
| 20 | 1048576 | 不可行 |

**建议**：n ≤ 12

---

## 十一、快速修改指南

### 场景1：测试新的相关性水平

```python
# 修改: src/scenarios/scenario_b_too_much_data.py L340
params = generate_instance(n=10, rho=0.9, seed=13645)  # 改rho
```

### 场景2：改用两点分布

```python
# 修改: src/scenarios/scenario_b_too_much_data.py L73-74
# v = np.random.uniform(0.3, 1.2, size=n).tolist()  # 注释掉
v = [0.2 if i < n//2 else 1.5 for i in range(n)]  # 新增
```

### 场景3：增加决策稳定性

```bash
# 命令行调用
python run_evaluation.py --scenarios B --models gpt-4o-mini --num-trials 5
```

### 场景4：放宽均衡判定标准

```python
# 修改: src/evaluators/evaluate_scenario_b.py L633
"correct_equilibrium": 1 if jaccard_sim >= 0.5 else 0,  # 从0.6改为0.5
```

---

**文档版本**: v1.0  
**最后更新**: 2026-01-15  
**维护者**: AI Assistant
