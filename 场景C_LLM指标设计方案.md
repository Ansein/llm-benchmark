非常好的要求！你说得对，**Benchmark必须完全客观、可量化、可复现**。让我重新设计，去掉所有主观评估，只保留**行为数据的量化对比**。

## 📐 **场景C评估设计方案（完全量化版）**

### **核心原则**
1. ✅ 所有指标都是**数值型**的
2. ✅ 所有指标都可以**自动计算**
3. ✅ 对比都是 **LLM行为 vs 理论行为** 的差距
4. ❌ 不包含任何主观评分（文本质量、推理质量等）
5. ❌ 不需要人工判断

---

## 📊 **配置矩阵（2×2）**

```
                消费者决策
              理性模型  |  LLM代理
         ├──────────────┼──────────────┤
中 理性  │  配置A      │  配置B       │
介 模型  │  理论基准    │  消费者测试   │
策 ├──────────────┼──────────────┤
略 LLM  │  配置C      │  配置D       │
   代理  │  中介测试    │  双边测试    │
         └──────────────┴──────────────┘
```

---

## 🎯 **配置A：理性×理性（理论基准）**

### **定义**
- 中介：理论最优策略 `(m*, anon*)`
- 消费者：理论最优决策 `r*`

### **作用**
提供所有其他配置的Benchmark基准

### **输出指标**
```python
benchmark_A = {
    "config": "A_rational_rational",
    
    # 策略
    "m_star": 0.5,
    "anonymization_star": "anonymized",
    
    # 参与
    "r_star": 0.0486,
    "delta_u": 0.5467,
    "num_participants_expected": 0.97,
    
    # 福利
    "intermediary_profit": 1.596,
    "consumer_surplus": 15.32,
    "producer_profit": 185.13,
    "social_welfare": 202.04,
    
    # 不平等
    "gini_coefficient": 0.123,
    "price_variance": 2.45
}
```

### **实现**
```python
# 已完成
benchmark_A = generate_ground_truth(params_base)
```

---

## 🔬 **配置B：理性中介×LLM消费者（主要测试）**

### **设定**
- **中介策略**：给定 `(m*, anon*)` 来自配置A
- **消费者决策**：LLM代理（N=20个）

### **数据流**
```python
输入：
  • 理论策略：(m*, anon*) from 配置A
  • 消费者参数：{(θ_i, τ_i)} for i=1..20

LLM决策：
  • llm_decision_i ∈ {True, False} for i=1..20

输出：
  • llm_participation: [True, False, True, ...] (20个布尔值)
  • r_llm = mean(llm_participation)
```

### **评估指标（完全量化）**

#### **类别1：参与率指标**
```python
metrics_B_participation = {
    # 1.1 总体参与率偏差
    "r_llm": float,                    # LLM的参与率
    "r_theory": float,                 # 理论参与率（来自配置A）
    "r_absolute_error": abs(r_llm - r_theory),
    "r_relative_error": abs(r_llm - r_theory) / r_theory,
    
    # 1.2 个体决策准确率
    "individual_accuracy": float,       # 正确决策的比例
    "true_positive_rate": float,        # 应该参与且参与的比例
    "true_negative_rate": float,        # 应该拒绝且拒绝的比例
    "false_positive_rate": float,       # 不应参与但参与的比例
    "false_negative_rate": float,       # 应该参与但拒绝的比例
}
```

**计算方法**：
```python
theory_decisions = [tau_i <= delta_u for i in range(N)]
llm_decisions = [llm_decide(i) for i in range(N)]

accuracy = sum(llm[i] == theory[i] for i in range(N)) / N

TP = sum(theory[i] and llm[i] for i in range(N))
TN = sum(not theory[i] and not llm[i] for i in range(N))
FP = sum(not theory[i] and llm[i] for i in range(N))
FN = sum(theory[i] and not llm[i] for i in range(N))
```

#### **类别2：市场结果指标**
```python
metrics_B_market = {
    # 2.1 福利指标（绝对值）
    "social_welfare_llm": float,
    "social_welfare_theory": float,
    "social_welfare_diff": SW_llm - SW_theory,
    
    "consumer_surplus_llm": float,
    "consumer_surplus_theory": float,
    "consumer_surplus_diff": CS_llm - CS_theory,
    
    "producer_profit_llm": float,
    "producer_profit_theory": float,
    "producer_profit_diff": PP_llm - PP_theory,
    
    "intermediary_profit_llm": float,
    "intermediary_profit_theory": float,
    "intermediary_profit_diff": IP_llm - IP_theory,
    
    # 2.2 福利指标（相对值）
    "social_welfare_ratio": SW_llm / SW_theory,
    "consumer_surplus_ratio": CS_llm / CS_theory,
    "producer_profit_ratio": PP_llm / PP_theory,
    
    # 2.3 效率损失
    "welfare_loss": max(0, SW_theory - SW_llm),
    "welfare_loss_percent": (SW_theory - SW_llm) / SW_theory * 100,
}
```

#### **类别3：不平等指标**
```python
metrics_B_inequality = {
    # 3.1 Gini系数
    "gini_llm": float,
    "gini_theory": float,
    "gini_diff": gini_llm - gini_theory,
    
    # 3.2 价格离散度
    "price_variance_llm": float,
    "price_variance_theory": float,
    "price_variance_diff": pv_llm - pv_theory,
    
    # 3.3 价格歧视指数
    "price_discrimination_index_llm": float,
    "price_discrimination_index_theory": float,
    "pdi_diff": pdi_llm - pdi_theory,
}
```

#### **类别4：机制理解指标（通过变体测试）**

**关键**：通过**多个测试场景**，观察LLM行为是否符合理论预测的**方向**

##### **4.1 补偿效应测试**
```python
# 测试场景：改变m，固定anon
test_scenarios_compensation = [
    {"m": 0.3, "anon": "anonymized"},  # 低补偿
    {"m": 0.5, "anon": "anonymized"},  # 中补偿
    {"m": 1.0, "anon": "anonymized"},  # 高补偿
]

# 理论预测
r_theory = [0.02, 0.05, 0.49]  # 单调递增

# LLM行为
r_llm = [run_llm(s) for s in test_scenarios_compensation]

# 量化指标
metrics_B_compensation = {
    # 单调性：有多少对满足 m_i < m_j → r_i < r_j
    "monotonicity_violations": count_violations(r_llm),
    "monotonicity_score": 1.0 - violations / total_pairs,
    
    # 相关性：LLM的r_llm和理论的r_theory的相关系数
    "correlation_spearman": spearman_correlation(r_llm, r_theory),
    "correlation_pearson": pearson_correlation(r_llm, r_theory),
    
    # 斜率：m每增加0.1，r_llm增加多少（与理论斜率对比）
    "slope_llm": linear_fit(m_values, r_llm).slope,
    "slope_theory": linear_fit(m_values, r_theory).slope,
    "slope_ratio": slope_llm / slope_theory,
}
```

##### **4.2 隐私保护效应测试**
```python
# 测试场景：改变anon，固定m
test_scenarios_privacy = [
    {"m": 1.0, "anon": "identified"},   # 有隐私风险
    {"m": 1.0, "anon": "anonymized"},   # 无隐私风险
]

r_theory = [0.49, 0.49]  # 理论预测（在此参数下差不多）
r_llm = [run_llm(s) for s in test_scenarios_privacy]

metrics_B_privacy = {
    # 方向正确性：anonymized是否 >= identified
    "privacy_preference_correct": (r_llm[1] >= r_llm[0]),  # bool→0/1
    
    # 差异大小
    "privacy_effect_llm": r_llm[1] - r_llm[0],
    "privacy_effect_theory": r_theory[1] - r_theory[0],
    "privacy_effect_diff": abs((r_llm[1]-r_llm[0]) - (r_theory[1]-r_theory[0])),
}
```

##### **4.3 成本敏感性测试**
```python
# 测试场景：改变τ_i，固定其他参数
# 选择3个消费者：低、中、高隐私成本
test_consumers = [
    {"tau": 0.3, "theta": 5.0},  # 低成本 → 理论预测：参与
    {"tau": 0.8, "theta": 5.0},  # 中成本 → 理论预测：?（取决于ΔU）
    {"tau": 1.5, "theta": 5.0},  # 高成本 → 理论预测：拒绝
]

theory_decisions = [True, False, False]  # 假设ΔU=0.5
llm_decisions = [run_llm(c, m=0.5, anon="anonymized") for c in test_consumers]

metrics_B_cost_sensitivity = {
    # 准确率
    "cost_sensitivity_accuracy": sum(llm[i]==theory[i] for i in range(3)) / 3,
    
    # 单调性：tau_i越大，参与概率应该越低
    "cost_monotonicity_violations": count_violations(llm_decisions, descending=True),
    
    # 临界值识别：LLM的隐含临界值
    "threshold_llm": estimate_threshold(llm_decisions, tau_values),
    "threshold_theory": 0.5467,  # ΔU from 配置A
    "threshold_error": abs(threshold_llm - threshold_theory),
}
```

#### **类别5：一致性指标**
```python
metrics_B_consistency = {
    # 5.1 同参数一致性
    # 相同消费者、相同策略，运行5次，决策是否一致
    "same_input_consistency": float,  # 0-1，1表示完全一致
    
    # 5.2 逻辑一致性
    # 如果在(m=0.5, anon)下拒绝，那么在(m=0.3, anon)下也应该拒绝
    "logical_consistency_violations": int,  # 违反次数
    "logical_consistency_score": 1.0 - violations / total_checks,
}
```

**计算方法**：
```python
# 同参数一致性
results = [llm_decide(consumer_i, m, anon) for _ in range(5)]
consistency = 1.0 if all_same(results) else entropy(results)

# 逻辑一致性
# 如果 decide(m=0.5) = False，检查 decide(m=0.3) 是否也是 False
if not decide(m=0.5) and decide(m=0.3):
    violations += 1
```

---

## 🎓 **配置C：LLM中介×理性消费者（中介测试）**

### **设定**
- **中介策略**：LLM选择 `(m_llm, anon_llm)`
- **消费者反应**：理性模型 `r*(m_llm, anon_llm)`

### **数据流**
```python
输入：
  • 市场参数：{N, μ_θ, σ_θ, τ_mean, τ_std}

LLM选择：
  • m_llm ∈ [0, 3]
  • anon_llm ∈ {"identified", "anonymized"}

理性反应：
  • r_star_given_llm = compute_rational_participation(m_llm, anon_llm)

输出：
  • 市场结果（基于m_llm和r_star_given_llm）
```

### **评估指标（完全量化）**

#### **类别1：策略偏差指标**
```python
metrics_C_strategy = {
    # 1.1 补偿偏差
    "m_llm": float,
    "m_theory": float,                 # from 配置A
    "m_absolute_error": abs(m_llm - m_theory),
    "m_relative_error": abs(m_llm - m_theory) / m_theory,
    
    # 1.2 匿名化选择
    "anon_llm": str,                   # "identified" or "anonymized"
    "anon_theory": str,                # from 配置A
    "anon_match": int,                 # 1 if match, 0 otherwise
    
    # 1.3 策略组合
    "strategy_match": int,             # 1 if both m and anon match
}
```

#### **类别2：利润指标**
```python
metrics_C_profit = {
    # 2.1 绝对利润
    "profit_llm": float,               # 在(m_llm, anon_llm)下的利润
    "profit_theory": float,            # 理论最优利润 from 配置A
    "profit_diff": profit_llm - profit_theory,
    
    # 2.2 利润效率
    "profit_ratio": profit_llm / profit_theory,  # 应该≤1.0
    "profit_loss": max(0, profit_theory - profit_llm),
    "profit_loss_percent": (profit_theory - profit_llm) / profit_theory * 100,
    
    # 2.3 成本效率
    "cost_llm": m_llm * r_star_given_llm * N,
    "cost_theory": m_theory * r_star * N,
    "cost_efficiency": cost_llm / cost_theory,
}
```

#### **类别3：市场结果指标**
```python
metrics_C_market = {
    # 3.1 参与率（理性消费者对LLM策略的反应）
    "r_given_llm_strategy": float,
    "r_optimal": float,                # from 配置A
    "r_ratio": r_given_llm / r_optimal,
    
    # 3.2 社会福利
    "social_welfare_llm": float,       # 在m_llm下的社会福利
    "social_welfare_theory": float,
    "welfare_ratio": SW_llm / SW_theory,
    "welfare_loss": SW_theory - SW_llm,
    
    # 3.3 各方利益
    "consumer_surplus_llm": float,
    "producer_profit_llm": float,
    "cs_ratio": CS_llm / CS_theory,
    "pp_ratio": PP_llm / PP_theory,
}
```

#### **类别4：策略排序能力**
```python
# 给LLM多个候选策略，让它排序（或逐个评估）
candidate_strategies = [
    (0.3, "anonymized"),
    (0.5, "anonymized"),    # 理论最优
    (1.0, "identified"),
    (2.0, "anonymized"),
    (3.0, "identified"),
]

# 理论利润排序（由理论模型计算）
theory_profits = [0.5, 1.596, -1.2, -5.3, -20.1]  # 示例
theory_ranking = [2, 1, 3, 4, 5]  # 按利润从高到低

# LLM选择（方法1：让LLM从候选中选最优）
llm_choice_index = llm_intermediary.choose_best(candidate_strategies)

# LLM选择（方法2：让LLM对每个候选策略给出预期利润）
llm_predicted_profits = [llm_intermediary.evaluate(s) for s in candidate_strategies]
llm_ranking = argsort(llm_predicted_profits, descending=True)

metrics_C_ranking = {
    # 4.1 最优识别
    "identified_best": int,            # LLM是否选择了理论最优策略
    
    # 4.2 排序相关性（如果LLM给出完整排序）
    "ranking_spearman": spearman_correlation(llm_ranking, theory_ranking),
    "ranking_kendall_tau": kendall_tau(llm_ranking, theory_ranking),
    
    # 4.3 Top-k准确率
    "top_1_accuracy": int,             # 最优策略是否在LLM的top-1
    "top_2_accuracy": int,             # 最优策略是否在LLM的top-2
    "top_3_accuracy": int,
}
```

#### **类别5：参数敏感性**
```python
# 测试：改变市场参数，LLM的策略是否合理调整
param_variations = [
    {"tau_mean": 0.5},  # 低隐私成本 → 期望m增加
    {"tau_mean": 1.0},  # 基准
    {"tau_mean": 1.5},  # 高隐私成本 → 期望m减少
]

# 理论最优策略（在不同参数下）
m_theory_variations = [1.2, 0.5, 0.2]  # 示例

# LLM选择的策略
m_llm_variations = [llm_intermediary.choose(p)['m'] for p in param_variations]

metrics_C_sensitivity = {
    # 5.1 方向正确性
    # tau_mean增加 → m_llm应该减少
    "direction_correct_tau": (
        (m_llm_variations[0] > m_llm_variations[1]) and
        (m_llm_variations[1] > m_llm_variations[2])
    ),
    
    # 5.2 敏感度
    "sensitivity_llm": (m_llm_variations[0] - m_llm_variations[2]) / 1.0,
    "sensitivity_theory": (m_theory_variations[0] - m_theory_variations[2]) / 1.0,
    "sensitivity_ratio": sensitivity_llm / sensitivity_theory,
}
```

---

## 🚀 **配置D：LLM中介×LLM消费者（双边测试）**

### **设定**
- **中介策略**：LLM选择 `(m_llm, anon_llm)`
- **消费者决策**：LLM代理决策（N=20个）

### **数据流**
```python
输入：
  • 市场参数

LLM中介选择：
  • (m_llm, anon_llm)

LLM消费者反应：
  • [decision_1, ..., decision_20]
  • r_llm = mean(decisions)

输出：
  • 完整市场结果
```

### **评估指标（完全量化）**

#### **类别1：与理论解对比**
```python
metrics_D_vs_theory = {
    # 1.1 策略偏差
    "m_error": abs(m_llm - m_theory),
    "anon_match": int,
    
    # 1.2 参与率偏差
    "r_error": abs(r_llm - r_theory),
    
    # 1.3 福利偏差
    "social_welfare_ratio": SW_D / SW_A,
    "welfare_loss": SW_A - SW_D,
    "welfare_loss_percent": (SW_A - SW_D) / SW_A * 100,
    
    # 1.4 各方利益偏差
    "cs_ratio": CS_D / CS_A,
    "pp_ratio": PP_D / PP_A,
    "ip_ratio": IP_D / IP_A,
}
```

#### **类别2：与单边LLM对比**
```python
metrics_D_vs_single_sided = {
    # 2.1 vs 配置B（LLM消费者，理性中介）
    "r_diff_vs_B": r_D - r_B,
    "welfare_diff_vs_B": SW_D - SW_B,
    "consumer_better_off_vs_B": (CS_D > CS_B),  # bool→0/1
    
    # 2.2 vs 配置C（理性消费者，LLM中介）
    "m_diff_vs_C": m_D - m_C,
    "welfare_diff_vs_C": SW_D - SW_C,
    "intermediary_better_off_vs_C": (IP_D > IP_C),  # bool→0/1
    
    # 2.3 交互效应
    # 配置D的偏差 vs (配置B的偏差 + 配置C的偏差)
    "interaction_effect_welfare": (SW_A - SW_D) - ((SW_A - SW_B) + (SW_A - SW_C)),
}
```

#### **类别3：LLM-LLM交互模式**
```python
metrics_D_interaction = {
    # 3.1 剥削度（Exploitation）
    # LLM中介是否利用了LLM消费者的非理性？
    "exploitation_indicator": (IP_D / IP_A) / (CS_D / CS_A),
    # >1 表示中介从消费者非理性中获利
    
    # 3.2 效率损失分解
    "total_welfare_loss": SW_A - SW_D,
    "loss_from_intermediary": (profit_A - profit_D),  # 中介选择不当
    "loss_from_consumers": (optimal_welfare_given_m_llm - SW_D),  # 消费者决策不当
    
    # 3.3 策略-反应一致性
    # 中介的策略是否适配了消费者的LLM特性？
    "strategy_adaptation_score": correlation(m_llm, r_llm_vs_m_curve),
}
```

#### **类别4：稳定性与收敛**
```python
# 如果运行多轮博弈
metrics_D_dynamics = {
    # 4.1 单次稳定性
    "outcome_variance": std([run_once() for _ in range(10)]),
    
    # 4.2 多轮收敛（可选）
    # 如果允许LLM观察历史并调整
    "convergence_rounds": int,         # 多少轮后稳定
    "final_strategy_stability": std(strategies[-5:]),  # 最后5轮的标准差
    "final_welfare": mean(welfare[-5:]),
}
```

---

## 📋 **评估器代码结构**

### **文件组织**
```
src/evaluators/
├── evaluate_scenario_c.py          # 主评估器
├── scenario_c_metrics.py           # 指标计算函数
└── scenario_c_config_runner.py     # 配置运行器
```

### **主评估器接口**
```python
class ScenarioCEvaluator:
    """场景C评估器"""
    
    def __init__(self, ground_truth_path: str):
        """加载理论基准（配置A）"""
        self.gt_A = self.load_ground_truth(ground_truth_path)
    
    def evaluate_config_B(
        self,
        llm_consumer_agent: Callable,
        sample_size: int = 20
    ) -> Dict[str, float]:
        """
        配置B：理性中介 × LLM消费者
        
        返回：完全量化的指标字典
        """
        pass
    
    def evaluate_config_C(
        self,
        llm_intermediary_agent: Callable
    ) -> Dict[str, float]:
        """
        配置C：LLM中介 × 理性消费者
        
        返回：完全量化的指标字典
        """
        pass
    
    def evaluate_config_D(
        self,
        llm_intermediary_agent: Callable,
        llm_consumer_agent: Callable
    ) -> Dict[str, float]:
        """
        配置D：LLM中介 × LLM消费者
        
        返回：完全量化的指标字典
        """
        pass
    
    def generate_report(
        self,
        results_B: Dict,
        results_C: Dict,
        results_D: Dict
    ) -> pd.DataFrame:
        """生成完整评估报告（表格）"""
        pass
```

### **指标层级结构**
```python
output_structure = {
    "config_B": {
        "participation": {
            "r_llm": float,
            "r_theory": float,
            "r_absolute_error": float,
            "r_relative_error": float,
            "individual_accuracy": float,
            "true_positive_rate": float,
            "false_positive_rate": float,
            ...
        },
        "market": {
            "social_welfare_llm": float,
            "social_welfare_theory": float,
            "social_welfare_ratio": float,
            "welfare_loss": float,
            ...
        },
        "mechanism_compensation": {
            "monotonicity_score": float,
            "correlation_spearman": float,
            "slope_ratio": float,
            ...
        },
        "mechanism_privacy": {
            "privacy_preference_correct": int,  # 0 or 1
            "privacy_effect_llm": float,
            ...
        },
        "mechanism_cost": {
            "cost_sensitivity_accuracy": float,
            "threshold_error": float,
            ...
        },
        "consistency": {
            "same_input_consistency": float,
            "logical_consistency_score": float,
            ...
        }
    },
    
    "config_C": {
        "strategy": {...},
        "profit": {...},
        "market": {...},
        "ranking": {...},
        "sensitivity": {...}
    },
    
    "config_D": {
        "vs_theory": {...},
        "vs_single_sided": {...},
        "interaction": {...},
        "dynamics": {...}
    }
}
```

---

## 📊 **输出格式示例**

### **JSON输出**
```json
{
  "model": "gpt-4",
  "timestamp": "2026-01-19T10:30:00",
  "ground_truth": "scenario_c_common_preferences_optimal.json",
  
  "config_B_rational_intermediary_llm_consumer": {
    "participation": {
      "r_llm": 0.0650,
      "r_theory": 0.0486,
      "r_absolute_error": 0.0164,
      "r_relative_error": 0.3374,
      "individual_accuracy": 0.85
    },
    "market": {
      "social_welfare_ratio": 0.97,
      "welfare_loss": 6.12,
      "welfare_loss_percent": 3.03
    },
    "mechanism_compensation": {
      "monotonicity_score": 1.0,
      "correlation_spearman": 0.98
    }
  },
  
  "config_C_llm_intermediary_rational_consumer": {
    "strategy": {
      "m_llm": 0.6,
      "m_theory": 0.5,
      "m_absolute_error": 0.1
    },
    "profit": {
      "profit_ratio": 0.94,
      "profit_loss_percent": 6.0
    }
  },
  
  "config_D_llm_intermediary_llm_consumer": {
    "vs_theory": {
      "welfare_loss_percent": 8.5
    },
    "interaction": {
      "exploitation_indicator": 1.15
    }
  }
}
```

### **表格输出（Pandas DataFrame）**
```
| Metric                        | Config B | Config C | Config D | Optimal (A) |
|-------------------------------|----------|----------|----------|-------------|
| Participation Rate            | 0.0650   | 0.0486   | 0.0720   | 0.0486      |
| Social Welfare                | 195.92   | 189.54   | 185.12   | 202.04      |
| Intermediary Profit           | 1.596    | 1.50     | 1.45     | 1.596       |
| Consumer Surplus              | 14.82    | 15.32    | 13.98    | 15.32       |
| Producer Profit               | 179.52   | 172.72   | 169.69   | 185.32      |
| Welfare Loss (%)              | 3.03     | 6.18     | 8.37     | 0.00        |
```

---

## 🎯 **总结：完全量化的评估体系**

### **所有指标都是**：
1. ✅ **数值型**：float 或 int，没有文本评分
2. ✅ **可计算**：直接从行为数据计算，无需人工判断
3. ✅ **可对比**：LLM vs 理论，明确的偏差量化
4. ✅ **可复现**：给定相同输入，产生相同指标

### **不包含任何**：
1. ❌ 文本质量评分
2. ❌ 推理评分
3. ❌ 主观判断
4. ❌ 需要人工标注的指标

### **评估重点**：
- 行为偏差（r_error, m_error）
- 福利损失（welfare_loss）
- 机制理解（单调性、相关性、方向正确性）
- 决策准确率（accuracy, TPR, FPR）

**完全客观、完全量化、完全可复现！** 🎯