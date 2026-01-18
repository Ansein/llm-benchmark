# 场景B重构摘要（静态博弈版本）

**重构日期**: 2026-01-15  
**重构范围**: `src/evaluators/evaluate_scenario_b.py`, `run_evaluation.py`

---

## 一、重构动机

### 旧设计的根本性问题

原有的多轮迭代+广播机制存在以下逻辑错误：

1. **多轮博弈语义不清**：
   - 用户在第t轮分享数据后，第t+1轮分享的是什么？
   - 同一数据重复分享无意义
   - 新数据则意味着每轮都是新用户，不应追求"收敛"

2. **与论文不符**：
   - TMD论文描述的是**静态博弈**：平台报价 → 用户同时决策
   - 不存在"上一轮广播"和"动态调整"

3. **信息结构错误**：
   - 用户看到"上一轮分享集合"违反了静态博弈的信息结构
   - 应该是：用户基于**理性预期**和**公共先验**决策

---

## 二、新设计：两阶段静态博弈

### 博弈时序

```
阶段0 (抽样)：
  - 生成相关结构 Σ（公共知识）
  - 生成隐私偏好 v_i（私有信息）

阶段1 (平台报价)：
  - 平台LLM输出统一价格 p（或个性化价格 p_i）
  - 平台基于先验分布 F_v 形成对分享率的预期

阶段2 (用户同时决策)：
  - 所有用户同时决策（share ∈ {0,1}）
  - 每个用户只知道：自己的 v_i, 报价 p_i, 公共知识
  - 用户基于理性预期形成对"其他人分享比例"的信念

阶段3 (结算)：
  - 计算泄露信息量 I_i(S)
  - 计算效用、利润、福利
```

### 关键改动

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| **主方法** | `simulate_llm_equilibrium()` | `simulate_static_game()` |
| **迭代次数** | 15轮收敛 | 无迭代（单次博弈） |
| **用户信息** | 看到"上一轮广播" | 只知道自己的v和p |
| **用户决策** | `query_llm_sharing_decision(last_round_broadcast)` | `query_user_decision(price)` |
| **平台报价** | 固定计算公式 | `query_platform_pricing()` LLM决策 |
| **输出** | `belief_share_rate`（新增） | 显式要求LLM输出信念 |

---

## 三、核心实现细节

### 3.1 相关性结构压缩

**问题**：协方差矩阵 Σ (n×n) 无法直接喂给LLM

**解决**：为每个用户计算压缩摘要
```python
{
  "mean_corr": 0.62,  # 与其他人的平均相关系数
  "topk_neighbors": [(2, 0.85), (5, 0.78), (1, 0.71)],  # 最强相关邻居
  "strong_neighbors_count": 4  # 相关系数 > 0.5 的邻居数量
}
```

### 3.2 平台报价Prompt（统一价P1版本）

**给平台的信息**：
- 用户数 n
- 隐私偏好分布 F_v（例如：v ~ Uniform[0.3, 1.2]）
- 相关性结构摘要（总体ρ，每个用户的mean_corr等）

**要求输出**：
```json
{
  "uniform_price": 0.5,
  "belief_share_rate": 0.45,
  "reason": "基于次模性，设定中等价格..."
}
```

### 3.3 用户决策Prompt

**给用户i的信息**：
- **私有**：v_i, p_i
- **公共**：n, ρ, F_v, 用户i的相关性摘要

**关键说明**：
1. "即使不分享，别人分享也会泄露你的信息"（推断外部性）
2. "分享的成本是**边际泄露**，而非总泄露"
3. "基于先验推测他人行为"（理性预期）

**要求输出**：
```json
{
  "share": 1,
  "belief_share_rate": 0.4,
  "reason": "我的v偏低(0.35)，预期40%的人分享..."
}
```

### 3.4 信念一致性分析

新增分析模块，评估：
- 实际分享率 vs 平均信念分享率
- 信念误差（mean, max, std）
- 用于诊断LLM是否真正理解博弈结构

---

## 四、评估指标变化

### 删除的指标
- ❌ `converged`（是否收敛）
- ❌ `rounds`/`iterations`（迭代次数）
- ❌ `cycle_detected`（2-cycle检测）
- ❌ `convergence_history`（收敛轨迹）

### 新增的指标
- ✅ `pricing_mode`（定价模式：uniform/personalized）
- ✅ `platform.belief_share_rate`（平台预期分享率）
- ✅ `users.beliefs`（每个用户的信念）
- ✅ `belief_consistency`（信念一致性分析）

### 保留的指标
- ✅ `equilibrium_quality`（均衡质量：Jaccard相似度等）
- ✅ `metrics`（利润、福利、泄露等）
- ✅ `labels`（过度分享、泄露分桶等）

---

## 五、代码文件变更

### 5.1 `src/evaluators/evaluate_scenario_b.py`

**新增方法**：
- `_compute_correlation_summaries()`: 计算相关性摘要
- `build_platform_pricing_prompt()`: 平台报价提示词
- `build_user_decision_prompt()`: 用户决策提示词（静态博弈版）
- `query_platform_pricing()`: 查询平台报价
- `query_user_decision()`: 查询用户决策
- `simulate_static_game()`: 主评估方法（静态博弈）
- `_analyze_belief_consistency()`: 信念一致性分析

**删除方法**：
- `build_sharing_prompt()`: 旧的用户提示词（含广播信息）
- `query_llm_sharing_decision()`: 旧的用户查询方法
- `simulate_llm_equilibrium()`: 旧的多轮迭代方法

**修改方法**：
- `__init__()`: 新增相关性摘要缓存
- `print_evaluation_summary()`: 更新打印格式（新增平台、信念分析）

### 5.2 `run_evaluation.py`

**修改**：
```python
# 场景B的评估调用
elif scenario == "B":
    evaluator = ScenarioBEvaluator(llm_client)
    # 旧: results = evaluator.simulate_llm_equilibrium(num_trials, max_rounds)
    # 新: 
    results = evaluator.simulate_static_game(
        pricing_mode="uniform",
        num_trials=num_trials
    )
```

**修改**：`generate_summary_report()`中对场景B的处理，删除"收敛"和"迭代次数"列

---

## 六、使用示例

### 单独运行评估器

```bash
cd d:\benchmark
python src/evaluators/evaluate_scenario_b.py
```

### 批量评估

```bash
python run_evaluation.py --scenarios B --models gpt-4o-mini deepseek-v3 --num-trials 1
```

**注意**：`--max-iterations`参数在场景B中已不再使用

---

## 七、与Ground Truth的对齐

Ground Truth求解器（`src/scenarios/scenario_b_too_much_data.py`）已经是正确的静态均衡求解：
- 枚举所有可能的分享集合 S
- 找到平台利润最大的 S*（即均衡）
- 计算社会最优 S_fb（福利最大）

LLM的评估结果与GT对比：
- **行为匹配**：Jaccard(LLM_S, GT_S)
- **福利匹配**：|W_LLM - W_GT|
- **信念校准**：用户信念 vs 实际结果

---

## 八、后续扩展方向

### 8.1 个性化价格（P2版本）
- 平台观察每个用户的"可观察特征"
- 设定不同的 p_i
- 研究"信息不对称"与"歧视性定价"

### 8.2 参数扫描实验
按照新方案建议，扫描以下维度：
1. **相关性强度**：ρ ∈ {0, 0.3, 0.6, 0.9}
2. **隐私偏好分布**：均匀分布 vs 两点分布
3. **平台定价能力**：P1（统一价）vs P2（个性化）

### 8.3 诊断性分析
- 按v值分组看决策模式
- 检测"支配策略错误"
- 分析信念形成机制

---

## 九、验证清单

在投入大规模实验前，请验证：

- [x] 代码可以运行（无语法错误）
- [ ] 单个评估（`python src/evaluators/evaluate_scenario_b.py`）能正常完成
- [ ] 批量评估（`python run_evaluation.py --scenarios B --models gpt-4o-mini`）能正常完成
- [ ] 输出的JSON结果结构正确
- [ ] 汇总报告中场景B的列正确显示
- [ ] LLM的`belief_share_rate`字段有合理值（不是全0.5）
- [ ] 信念一致性分析有合理结果

---

## 十、理论对齐声明

重构后的代码现在严格对齐TMD论文的以下核心设定：

1. ✅ **静态博弈**：一次性决策，无迭代
2. ✅ **信息结构**：用户只知道自己的v和p，基于先验推测他人
3. ✅ **推断外部性**：I_i(S)在效用函数中，体现他人决策的影响
4. ✅ **理性预期**：用户形成对a_{-i}的主观信念
5. ✅ **平台报价**：平台基于F_v与外部性结构设定价格

**不再存在的非对齐问题**：
- ❌ 多轮调整/收敛
- ❌ 上一轮广播信息
- ❌ 同一数据重复分享的语义矛盾

---

**重构完成时间**: 2026-01-15  
**验证状态**: 等待测试运行
