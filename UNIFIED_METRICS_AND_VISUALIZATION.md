# Benchmark 统一指标与可视化方案

> **版本**: 1.0  
> **更新日期**: 2026-01-26  
> **适用场景**: 场景A（个性化定价）、场景B（推断外部性）、场景C（社会数据外部性）

---

## 📋 目录

1. [统一指标体系](#统一指标体系)
2. [场景特定指标](#场景特定指标)
3. [可视化方案](#可视化方案)
4. [数据结构设计](#数据结构设计)
5. [实施步骤](#实施步骤)

---

## 统一指标体系

### 1. 核心统一指标（跨场景可比）

#### 1.1 决策质量维度 (Decision Quality)

| 指标名称 | 计算公式 | 适用场景 | 取值范围 | 说明 |
|---------|---------|---------|---------|------|
| **集合相似度** | `Jaccard(LLM_set, GT_set)` | A, B, C | [0, 1] | 决策集合与理论最优的相似度 |
| **参与率偏差** | `|LLM_rate - GT_rate|` | A, B, C | [0, 1] | 参与/分享/披露率的绝对误差 |
| **决策稳定性** | `1 - std(decisions) / mean(decisions)` | A, B, C | [0, 1] | 多次试验的一致性（变异系数的倒数） |

**集合定义对应关系**：
- **场景A**: `disclosure_set` - 披露个人数据的消费者集合
- **场景B**: `share_set` - 分享数据的用户集合
- **场景C**: `participation_set` - 参与数据收集的消费者集合

**参与率定义**：
- **场景A**: 披露率 = `|disclosure_set| / n`
- **场景B**: 分享率 = `|share_set| / n`
- **场景C**: 参与率 = `|participation_set| / N`

#### 1.2 经济效率维度 (Economic Efficiency)

| 指标名称 | 计算公式 | 适用场景 | 取值范围 | 说明 |
|---------|---------|---------|---------|------|
| **社会福利偏差率** | `|LLM_welfare - GT_welfare| / GT_welfare` | A, B, C | [0, ∞) | 相对误差百分比 |
| **利润偏差率** | `|LLM_profit - GT_profit| / GT_profit` | A, B, C | [0, ∞) | 平台/中介利润相对误差 |
| **效率损失** | `(GT_welfare - LLM_welfare) / GT_welfare` | A, B, C | (-∞, 1] | 负值表示LLM超过理论 |
| **利润绝对误差** | `|LLM_profit - GT_profit|` | A, B, C | [0, ∞) | MAE（绝对误差） |

**利润主体对应关系**：
- **场景A**: 平台利润（所有企业利润之和）
- **场景B**: 平台利润
- **场景C**: 中介利润

#### 1.3 收敛性能维度 (Convergence Performance)

| 指标名称 | 计算公式 | 适用场景 | 取值范围 | 说明 |
|---------|---------|---------|---------|------|
| **收敛标志** | `converged: bool` | A, B_FP, C_FP | {0, 1} | 是否达到稳定均衡 |
| **收敛速度** | `iterations_to_converge` | A, B_FP, C_FP | [1, max_iter] | 达到收敛所需轮数 |
| **收敛质量** | `final_similarity` | A, B_FP, C_FP | [0, 1] | 最终状态与理论的相似度 |
| **收敛稳定性** | `stable_rounds / total_rounds` | A, B_FP, C_FP | [0, 1] | 稳定轮数占比 |

**注意事项**：
- 场景B的静态博弈模式不涉及收敛指标
- 场景C的FP模式（B_FP、C_FP、D_FP）均适用收敛指标

---

## 场景特定指标

### 场景A：个性化定价与隐私选择

#### 核心指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **披露率分桶匹配** | LLM和理论的披露率是否在同一区间 | `llm_bucket == gt_bucket` (low/medium/high) |
| **过度披露标签** | 是否存在过度披露现象 | `|disclosure_set| > |gt_disclosure_set|` |
| **消费者剩余MAE** | 消费者剩余的绝对误差 | `|llm_CS - gt_CS|` |
| **搜索成本偏差** | 平均搜索成本的相对误差 | `(llm_search_cost - gt_search_cost) / gt_search_cost` |

#### 特有维度
- **隐私-效用权衡**：披露决策的理性程度
- **推荐系统效率**：匹配质量、搜索成本节省

---

### 场景B：推断外部性

#### 核心指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **泄露量MAE** | 总信息泄露的绝对误差 | `|llm_leakage - gt_leakage|` |
| **泄露分桶匹配** | LLM和理论的泄露量是否在同一区间 | `llm_bucket == gt_bucket` |
| **过度分享标签** | 是否存在过度分享现象 | `|share_set| > |gt_share_set|` |
| **信念一致性** | 用户信念与实际结果的偏差 | `mean(|belief_i - actual_rate|)` |

#### FP模式额外指标（B_FP）

| 指标 | 说明 |
|------|------|
| **信念收敛速度** | 用户信念达到稳定所需轮数 |
| **信念准确度** | 最终信念与真实分享率的误差 |
| **策略振荡** | 连续轮次决策变化的频率 |

#### 特有维度
- **推断外部性理解**：相关性参数的影响
- **信息泄露控制**：次模性、外部性的理解程度

---

### 场景C：社会数据外部性

#### 核心指标（所有配置通用）

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **策略参数误差(m)** | 补偿金额的绝对误差 | `|llm_m - gt_m|` |
| **匿名化策略匹配** | 匿名化选择是否正确 | `llm_anon == gt_anon` |
| **参与率偏差** | 参与率的绝对误差 | `|llm_rate - gt_rate|` |
| **中介利润偏差率** | 中介利润的相对误差 | `|llm_profit - gt_profit| / |gt_profit|` |

#### 三种配置对比

| 配置 | 中介 | 消费者 | 核心评估点 |
|------|------|---------|-----------|
| **B_FP** | 理性（理论最优） | LLM | LLM消费者的学习收敛能力 |
| **C_FP** | LLM | 理性 | LLM中介的策略学习能力 |
| **D_FP** | LLM | LLM | 双方学习下的均衡收敛性 |

#### FP模式特有指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **策略演化轨迹** | (m, anon)在策略空间的路径 | 轨迹可视化 |
| **利润达成率** | 最终利润占理论最优的比例 | `final_profit / optimal_profit` |
| **策略振荡幅度** | m值变化的标准差 | `std([m_1, m_2, ..., m_T])` |
| **提前收敛轮次** | 比最大轮数提前收敛的轮数 | `max_rounds - actual_rounds` |

#### 双维数据结构对比

| 数据结构 | 说明 | 预期差异 |
|---------|------|---------|
| **共同偏好** | 消费者有相似的theta分布 | 参与率可能更高 |
| **共同经历** | 消费者有相似的tau分布 | 匿名化策略影响更大 |

---

## 可视化方案

### 方案总览

```
可视化体系（三层级）
├── 第一层：跨场景对比（Cross-Scenario）
│   ├── 综合性能雷达图
│   ├── 模型-场景热力图
│   └── 效率前沿散点图
├── 第二层：场景内对比（Within-Scenario）
│   ├── 场景A & B：决策集合可视化
│   ├── 场景C：三配置对比
│   └── 收敛曲线分析
└── 第三层：细节分析（Deep-Dive）
    ├── 误差诊断
    ├── 时序分析
    └── 失败模式分析
```

---

### 第一层：跨场景对比

#### 1.1 综合性能雷达图

**目的**：对比不同模型在多维度的综合表现

**维度**（6-8个）：
1. 集合相似度（Decision Similarity）
2. 经济效率（Economic Efficiency）：1 - 福利偏差率
3. 收敛速度（Convergence Speed）：归一化的收敛轮数倒数
4. 决策稳定性（Stability）
5. 利润准确性（Profit Accuracy）：1 - 利润偏差率
6. 参与率准确性（Rate Accuracy）：1 - 参与率偏差

**可视化类型**：雷达图（Radar Chart）

**实现要点**：
```python
# 归一化处理，所有指标转换为[0, 1]，1表示最优
normalized_metrics = {
    'decision_similarity': jaccard_score,  # 已经在[0,1]
    'economic_efficiency': 1 - min(welfare_deviation, 1),  # 截断在1
    'convergence_speed': 1 - (iterations / max_iterations),
    'stability': stability_score,  # 已经在[0,1]
    'profit_accuracy': 1 - min(profit_deviation, 1),
    'rate_accuracy': 1 - abs(rate_deviation)
}
```

**分组展示**：
- 按场景分组：每个场景一个雷达图，多个模型叠加
- 按模型分组：每个模型一个雷达图，多个场景叠加

---

#### 1.2 模型-场景性能热力图

**目的**：一眼看出哪个模型在哪个场景表现最好

**矩阵结构**：
```
行：模型（deepseek-v3.2, gpt-5-mini, gemini-3-flash, qwen-plus）
列：场景配置（A, B_static, B_FP, C_B_FP, C_C_FP, C_D_FP）
值：综合得分（0-100分）
```

**综合得分计算**：
```python
综合得分 = (
    0.30 * 决策相似度 * 100 +
    0.25 * 经济效率 * 100 +
    0.20 * 利润准确性 * 100 +
    0.15 * 收敛性能 * 100 +
    0.10 * 稳定性 * 100
)
```

**颜色映射**：
- 0-60分：红色系（差）
- 60-75分：黄色系（中等）
- 75-85分：浅绿色（良好）
- 85-100分：深绿色（优秀）

---

#### 1.3 效率前沿图

**目的**：成本-性能权衡分析

**坐标系**：
- X轴：API调用成本（对数尺度）
- Y轴：综合性能得分（0-100）
- 气泡大小：参数量/模型规模
- 颜色：场景类型

**实现要点**：
```python
# 成本估算（基于API调用次数）
cost_estimate = {
    '场景A': iterations * n_consumers * unit_cost,
    '场景B_static': n_users * unit_cost,
    '场景B_FP': max_rounds * n_users * unit_cost,
    '场景C_B_FP': max_rounds * N_consumers * unit_cost,
    '场景C_C_FP': max_rounds * 1 * unit_cost,
    '场景C_D_FP': max_rounds * (N_consumers + 1) * unit_cost
}
```

---

### 第二层：场景内对比

#### 2.1 场景A & B：决策集合可视化

##### 方案1：维恩图（Venn Diagram）

**适用场景**：直观展示集合交集和差异

**元素**：
- 圆A：LLM决策集合
- 圆B：理论最优集合
- 交集：正确预测的用户
- 差集：误判的用户

**标注信息**：
```
LLM集合: [0, 1, 3, 5, 8, ...]
GT集合:  [0, 2, 3, 4, 5, ...]
Jaccard相似度: 0.75
精确率: 0.80
召回率: 0.83
```

##### 方案2：混淆矩阵热力图

**适用场景**：二分类决策的详细分析

```
              GT: 不参与    GT: 参与
LLM: 不参与      TN           FN
LLM: 参与        FP           TP
```

**衍生指标**：
- 精确率 = TP / (TP + FP)
- 召回率 = TP / (TP + FN)
- F1-score = 2 * (精确率 * 召回率) / (精确率 + 召回率)

##### 方案3：个体决策热力图

**适用场景**：展示每个用户的决策模式

**矩阵结构**：
```
行：用户ID
列：特征（隐私偏好v, 报价p, 决策LLM, 决策GT, 一致性）
颜色：数值大小 / 布尔值
```

---

#### 2.2 场景A & B：福利分解图

**目的**：展示社会福利的各组成部分

**可视化类型**：堆叠柱状图（Stacked Bar Chart）

**组成部分**：
- **场景A**：
  - 消费者剩余（Consumer Surplus）
  - 平台利润（Platform Profit）
  - 隐私成本（Privacy Cost，负值）
  
- **场景B**：
  - 平台利润（Platform Profit）
  - 用户效用（User Utility）
  - 信息泄露成本（Leakage Cost，负值）

**对比展示**：
```
[LLM柱] vs [GT柱]
每个柱子包含：正向收益（上部）+ 负向成本（下部）
```

---

#### 2.3 场景C：三配置对比

##### 方案1：利润收敛对比图

**可视化类型**：多线图（Multi-line Chart）

**坐标系**：
- X轴：迭代轮次
- Y轮：中介利润
- 线条：B_FP（绿色）、C_FP（蓝色）、D_FP（红色）
- 参考线：理论最优利润（虚线）

**标注**：
- 收敛点标记
- 最终利润标签
- 达成率百分比

##### 方案2：策略空间轨迹图

**可视化类型**：2D轨迹图 + 散点

**坐标系**：
- X轴：补偿金额 m
- Y轴：参与率
- 颜色：匿名化策略（蓝色=匿名，红色=实名）
- 轨迹：箭头连接各轮决策点
- 目标：理论最优点（★标记）

**实现要点**：
```python
# 绘制轨迹
for i in range(len(history)-1):
    plt.arrow(
        x=history[i]['m'], 
        y=history[i]['participation_rate'],
        dx=history[i+1]['m'] - history[i]['m'],
        dy=history[i+1]['participation_rate'] - history[i]['participation_rate'],
        color='blue' if history[i]['anonymization']=='anonymized' else 'red',
        alpha=0.5
    )
```

##### 方案3：三配置综合对比表

**可视化类型**：表格 + 热力图

| 指标 | B_FP | C_FP | D_FP | 最优 |
|------|------|------|------|------|
| 收敛轮次 | 15 | 23 | 31 | - |
| 最终利润 | 0.089 | 0.075 | 0.062 | 0.100 |
| 利润达成率 | 89% | 75% | 62% | 100% |
| 参与率偏差 | 0.05 | 0.12 | 0.18 | 0.00 |
| m参数误差 | 0.00 | 0.15 | 0.23 | 0.00 |
| 策略匹配 | ✓ | ✓ | ✗ | ✓ |

**颜色编码**：
- 绿色：表现优秀
- 黄色：表现中等
- 红色：表现较差

---

#### 2.4 收敛曲线分析

**适用场景**：A, B_FP, C_FP (所有迭代模式)

##### 核心指标收敛曲线

**可视化类型**：多子图面板

**子图1：参与率收敛**
- Y轴：参与率
- 参考线：理论最优参与率
- 阴影区域：±5%容忍带

**子图2：利润收敛**
- Y轴：利润
- 参考线：理论最优利润

**子图3：集合相似度演化**
- Y轴：Jaccard相似度
- 参考线：y=1（完美匹配）

**子图4：决策变化率**
- Y轴：连续两轮的Hamming距离
- 说明：接近0表示收敛

---

### 第三层：细节分析

#### 3.1 误差诊断图

##### 方案1：小提琴图（Violin Plot）

**目的**：展示各指标误差的分布形态

**X轴分组**：
- 社会福利MAE
- 利润MAE
- 参与率偏差
- 集合相似度

**分组对比**：
- 按场景分组
- 按模型分组

##### 方案2：误差散点图

**目的**：发现误差的系统性偏差

**坐标系**：
- X轴：理论值（Ground Truth）
- Y轴：LLM预测值
- 对角线：y=x（完美预测）
- 回归线：实际拟合线

**标注**：
- 相关系数 R²
- RMSE
- 偏差方向（高估/低估）

---

#### 3.2 时序分析图

##### 方案1：决策热力矩阵

**适用场景**：迭代博弈的个体决策演化

**矩阵结构**：
```
行：用户ID（0-19）
列：迭代轮次（1-50）
颜色：决策（不参与=白色，参与=蓝色）
```

**实现要点**：
```python
# 构建决策矩阵
matrix = np.zeros((n_users, n_rounds))
for round_idx, decisions in enumerate(history):
    for user_id in decisions['participation_set']:
        matrix[user_id, round_idx] = 1

sns.heatmap(matrix, cmap='Blues', cbar_kws={'label': '参与=1, 不参与=0'})
```

##### 方案2：累积变化曲线

**目的**：展示系统总体变化趋势

**指标**：
- 累积决策变化次数
- 累积福利改进
- 累积参与率方差

---

#### 3.3 失败模式分析

##### 方案1：失败案例聚类

**步骤**：
1. 识别失败案例（Jaccard < 0.5 或 未收敛）
2. 提取特征向量（参与率偏差、利润偏差、收敛轮次等）
3. K-means聚类
4. 可视化：2D PCA投影 + 聚类中心

##### 方案2：决策树可视化

**目的**：解释失败原因的层级结构

```
失败案例 (N=25)
├── 未收敛 (N=10)
│   ├── 策略振荡 (N=6)
│   └── 收敛过慢 (N=4)
└── 收敛但错误 (N=15)
    ├── 过度参与 (N=8)
    └── 不足参与 (N=7)
```

---

## 数据结构设计

### 统一结果数据格式

```json
{
  "metadata": {
    "benchmark_name": "LLM Strategic Reasoning Benchmark",
    "version": "1.0.0",
    "timestamp": "2026-01-26T12:00:00Z",
    "description": "评估LLM在经济学博弈场景下的战略推理能力"
  },
  
  "scenarios": {
    "A": {
      "scenario_name": "个性化定价与隐私选择",
      "scenario_type": "iterative_game",
      "models": {
        "deepseek-v3.2": {
          "runs": [
            {
              "run_id": "A_deepseek-v3.2_20260125_190313",
              "timestamp": "2026-01-25T19:03:13Z",
              
              "unified_metrics": {
                "decision_quality": {
                  "set_similarity": 0.75,
                  "participation_rate_llm": 0.60,
                  "participation_rate_gt": 0.50,
                  "participation_rate_deviation": 0.10,
                  "decision_stability": 0.85
                },
                "economic_efficiency": {
                  "welfare_deviation_ratio": 0.08,
                  "profit_deviation_ratio": 0.12,
                  "efficiency_loss": 0.05,
                  "welfare_mae": 0.45,
                  "profit_mae": 0.32
                },
                "convergence_performance": {
                  "converged": true,
                  "iterations_to_converge": 8,
                  "convergence_quality": 0.75,
                  "stable_rounds": 3
                }
              },
              
              "scenario_specific_metrics": {
                "disclosure_rate_bucket_match": true,
                "over_disclosure": false,
                "consumer_surplus_mae": 0.28,
                "search_cost_deviation": -0.15
              },
              
              "raw_results": {
                "llm_disclosure_set": [0, 1, 2, 5, 6, 9],
                "gt_disclosure_set": [0, 1, 3, 4, 6],
                "iterations": 8,
                "history": [...]
              }
            }
          ],
          
          "aggregated_statistics": {
            "mean_set_similarity": 0.73,
            "std_set_similarity": 0.05,
            "success_rate": 0.80
          }
        }
      }
    },
    
    "B": {
      "scenario_name": "推断外部性",
      "scenario_type": "static_game",
      "models": {
        "deepseek-v3.2": {
          "runs": [
            {
              "run_id": "B_deepseek-v3.2_20260125_120000",
              
              "unified_metrics": {
                "decision_quality": {
                  "set_similarity": 0.82,
                  "participation_rate_llm": 0.75,
                  "participation_rate_gt": 0.80,
                  "participation_rate_deviation": 0.05,
                  "decision_stability": 0.90
                },
                "economic_efficiency": {
                  "welfare_deviation_ratio": 0.05,
                  "profit_deviation_ratio": 0.07,
                  "efficiency_loss": 0.03,
                  "welfare_mae": 0.25,
                  "profit_mae": 0.18
                },
                "convergence_performance": null
              },
              
              "scenario_specific_metrics": {
                "leakage_mae": 0.12,
                "leakage_bucket_match": true,
                "over_sharing": false,
                "belief_consistency": 0.88
              },
              
              "raw_results": {
                "llm_share_set": [0, 3, 4, 5, 6, 8, 10, 13, 14, 15, 16, 17, 18, 19],
                "gt_share_set": [0, 2, 3, 4, 5, 6, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19]
              }
            }
          ]
        }
      }
    },
    
    "B_FP": {
      "scenario_name": "推断外部性（虚拟博弈）",
      "scenario_type": "fictitious_play",
      "models": {
        "deepseek-v3.2": {
          "runs": [
            {
              "run_id": "B_FP_deepseek-v3.2_20260126_100000",
              
              "unified_metrics": {
                "decision_quality": {
                  "set_similarity": 0.78,
                  "participation_rate_llm": 0.70,
                  "participation_rate_gt": 0.80,
                  "participation_rate_deviation": 0.10,
                  "decision_stability": 0.85
                },
                "economic_efficiency": {
                  "welfare_deviation_ratio": 0.09,
                  "profit_deviation_ratio": 0.11,
                  "efficiency_loss": 0.06,
                  "welfare_mae": 0.38,
                  "profit_mae": 0.28
                },
                "convergence_performance": {
                  "converged": true,
                  "iterations_to_converge": 23,
                  "convergence_quality": 0.78,
                  "stable_rounds": 3
                }
              },
              
              "scenario_specific_metrics": {
                "belief_convergence_speed": 18,
                "belief_accuracy": 0.92,
                "strategy_oscillation": 0.15
              }
            }
          ]
        }
      }
    },
    
    "C_B_FP": {
      "scenario_name": "社会数据外部性（配置B_FP: 理性中介 × LLM消费者）",
      "scenario_type": "fictitious_play",
      "data_structures": ["common_preferences", "common_experience"],
      
      "models": {
        "deepseek-v3.2": {
          "common_preferences": {
            "runs": [
              {
                "run_id": "C_B_FP_common_preferences_deepseek-v3.2_20260126_184624",
                
                "unified_metrics": {
                  "decision_quality": {
                    "set_similarity": 0.85,
                    "participation_rate_llm": 0.55,
                    "participation_rate_gt": 0.50,
                    "participation_rate_deviation": 0.05,
                    "decision_stability": 0.90
                  },
                  "economic_efficiency": {
                    "welfare_deviation_ratio": null,
                    "profit_deviation_ratio": 0.08,
                    "efficiency_loss": null,
                    "welfare_mae": null,
                    "profit_mae": 0.015
                  },
                  "convergence_performance": {
                    "converged": true,
                    "iterations_to_converge": 15,
                    "convergence_quality": 0.85,
                    "stable_rounds": 3
                  }
                },
                
                "scenario_specific_metrics": {
                  "strategy_m_error": 0.00,
                  "strategy_anon_match": true,
                  "profit_achievement_rate": 0.92,
                  "strategy_oscillation_amplitude": 0.12,
                  "early_convergence_rounds": 35
                },
                
                "raw_results": {
                  "final_m": 0.0,
                  "final_anonymization": "anonymized",
                  "final_participation_set": [1, 2, 3, 5, 7, 8, 9, 10, 11],
                  "history": [...]
                }
              }
            ]
          },
          
          "common_experience": {
            "runs": [...]
          }
        }
      }
    },
    
    "C_C_FP": {
      "scenario_name": "社会数据外部性（配置C_FP: LLM中介 × 理性消费者）",
      "scenario_type": "fictitious_play",
      "data_structures": ["common_preferences", "common_experience"],
      "models": {...}
    },
    
    "C_D_FP": {
      "scenario_name": "社会数据外部性（配置D_FP: LLM中介 × LLM消费者）",
      "scenario_type": "fictitious_play",
      "data_structures": ["common_preferences", "common_experience"],
      "models": {...}
    }
  },
  
  "aggregated_results": {
    "cross_scenario_rankings": [
      {"model": "deepseek-v3.2", "overall_score": 82.5, "rank": 1},
      {"model": "gpt-5-mini-2025-08-07", "overall_score": 78.3, "rank": 2},
      {"model": "gemini-3-flash-preview", "overall_score": 75.1, "rank": 3},
      {"model": "qwen-plus", "overall_score": 72.8, "rank": 4}
    ],
    
    "model_comparison": {
      "deepseek-v3.2": {
        "strengths": ["收敛速度快", "决策稳定性高"],
        "weaknesses": ["场景C_D_FP表现不佳"],
        "avg_set_similarity": 0.79,
        "avg_economic_efficiency": 0.91,
        "convergence_success_rate": 0.85
      }
    },
    
    "scenario_difficulty_ranking": [
      {"scenario": "C_D_FP", "avg_score": 65.2, "difficulty": "hard"},
      {"scenario": "C_C_FP", "avg_score": 72.5, "difficulty": "medium"},
      {"scenario": "B_FP", "avg_score": 75.8, "difficulty": "medium"},
      {"scenario": "A", "avg_score": 78.3, "difficulty": "easy"},
      {"scenario": "B", "avg_score": 82.1, "difficulty": "easy"}
    ]
  }
}
```

---

## 实施步骤

### 步骤1：数据标准化处理（1-2天）

#### 任务清单

- [ ] **创建统一数据解析器** (`src/analysis/data_parser.py`)
  - 解析场景A的JSON结果
  - 解析场景B的JSON结果
  - 解析场景C的JSON结果（含FP模式）
  - 提取原始数据和决策集合

- [ ] **实现统一指标计算器** (`src/analysis/unified_metrics.py`)
  - 决策质量指标计算
  - 经济效率指标计算
  - 收敛性能指标计算

- [ ] **生成标准化数据** (`evaluation_results/unified_results.json`)
  - 按照上述数据结构组织
  - 验证数据完整性

**示例代码框架**：

```python
# src/analysis/unified_metrics.py

import numpy as np
from typing import Dict, List, Set, Any

class UnifiedMetricsCalculator:
    """统一指标计算器"""
    
    @staticmethod
    def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
        """计算Jaccard相似度"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_decision_quality(
        llm_set: Set[int],
        gt_set: Set[int],
        total_population: int,
        stability_scores: List[float] = None
    ) -> Dict[str, float]:
        """计算决策质量指标"""
        
        set_similarity = UnifiedMetricsCalculator.jaccard_similarity(llm_set, gt_set)
        
        llm_rate = len(llm_set) / total_population
        gt_rate = len(gt_set) / total_population
        rate_deviation = abs(llm_rate - gt_rate)
        
        if stability_scores:
            decision_stability = np.mean(stability_scores)
        else:
            decision_stability = None
        
        return {
            'set_similarity': set_similarity,
            'participation_rate_llm': llm_rate,
            'participation_rate_gt': gt_rate,
            'participation_rate_deviation': rate_deviation,
            'decision_stability': decision_stability
        }
    
    @staticmethod
    def compute_economic_efficiency(
        llm_welfare: float,
        gt_welfare: float,
        llm_profit: float,
        gt_profit: float
    ) -> Dict[str, float]:
        """计算经济效率指标"""
        
        welfare_deviation_ratio = abs(llm_welfare - gt_welfare) / abs(gt_welfare) if gt_welfare != 0 else None
        profit_deviation_ratio = abs(llm_profit - gt_profit) / abs(gt_profit) if gt_profit != 0 else None
        efficiency_loss = (gt_welfare - llm_welfare) / gt_welfare if gt_welfare != 0 else None
        
        return {
            'welfare_deviation_ratio': welfare_deviation_ratio,
            'profit_deviation_ratio': profit_deviation_ratio,
            'efficiency_loss': efficiency_loss,
            'welfare_mae': abs(llm_welfare - gt_welfare),
            'profit_mae': abs(llm_profit - gt_profit)
        }
    
    @staticmethod
    def compute_convergence_performance(
        converged: bool,
        iterations: int,
        max_iterations: int,
        final_similarity: float,
        stable_rounds: int
    ) -> Dict[str, Any]:
        """计算收敛性能指标"""
        
        return {
            'converged': converged,
            'iterations_to_converge': iterations if converged else max_iterations,
            'convergence_quality': final_similarity,
            'stable_rounds': stable_rounds,
            'convergence_speed_normalized': 1 - (iterations / max_iterations)
        }
```

---

### 步骤2：基础可视化实现（2-3天）

#### 优先级1：跨场景对比（必须）

- [ ] **综合性能雷达图** (`src/visualization/radar_chart.py`)
  - 6维雷达图
  - 多模型叠加
  - 归一化处理

- [ ] **模型-场景热力图** (`src/visualization/heatmap.py`)
  - 性能矩阵
  - 颜色编码
  - 数值标注

#### 优先级2：场景内对比（重要）

- [ ] **决策集合可视化** (`src/visualization/set_visualization.py`)
  - 维恩图
  - 混淆矩阵

- [ ] **收敛曲线** (`src/visualization/convergence_plot.py`)
  - 多指标面板
  - 参考线标注

- [ ] **场景C三配置对比** (`src/visualization/scenario_c_comparison.py`)
  - 利润收敛曲线
  - 策略空间轨迹

#### 优先级3：细节分析（可选）

- [ ] **误差诊断** (`src/visualization/error_diagnostics.py`)
  - 小提琴图
  - 散点图 + 回归线

---

### 步骤3：生成报告系统（1-2天）

- [ ] **HTML报告生成器** (`src/reporting/html_generator.py`)
  - 自动嵌入图表
  - 响应式布局
  - 交互式表格

- [ ] **Markdown报告生成器** (`src/reporting/md_generator.py`)
  - 论文级格式
  - 图表引用
  - 统计摘要

**报告结构**：

```
benchmark_report_2026-01-26/
├── index.html                          # 主报告
├── assets/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── interactive.js
│   └── figures/
│       ├── radar_chart_all_models.png
│       ├── heatmap_model_scenario.png
│       ├── scenario_a_convergence.png
│       ├── scenario_b_venn_diagram.png
│       ├── scenario_c_comparison.png
│       └── ...
├── data/
│   ├── unified_results.json
│   └── summary_statistics.csv
└── README.md
```

---

### 步骤4：自动化流程（1天）

- [ ] **创建主脚本** (`scripts/generate_unified_analysis.py`)
  - 一键生成所有统一指标
  - 一键生成所有可视化
  - 一键生成完整报告

**使用示例**：

```bash
# 完整流程
python scripts/generate_unified_analysis.py \
  --input-dir evaluation_results \
  --output-dir unified_analysis \
  --scenarios A B C \
  --models deepseek-v3.2 gpt-5-mini-2025-08-07 gemini-3-flash-preview qwen-plus \
  --report-format html

# 仅更新指标
python scripts/generate_unified_analysis.py --metrics-only

# 仅更新可视化
python scripts/generate_unified_analysis.py --visualizations-only
```

---

### 步骤5：验证与优化（1天）

- [ ] **验证数据正确性**
  - 抽查样本数据
  - 对比原始结果
  - 验证指标计算

- [ ] **优化可视化美观度**
  - 调整颜色方案
  - 优化字体大小
  - 改进布局

- [ ] **性能优化**
  - 缓存中间结果
  - 并行处理
  - 渐进式加载

---

## 技术栈建议

### Python库

| 功能 | 推荐库 | 说明 |
|------|--------|------|
| **数据处理** | pandas, numpy | 数据清洗和计算 |
| **静态图表** | matplotlib, seaborn | 论文级质量图表 |
| **交互式图表** | plotly | 网页交互式可视化 |
| **网络图** | networkx | 决策树、关系图 |
| **报告生成** | jinja2, markdown | HTML/MD模板渲染 |
| **统计分析** | scipy, sklearn | 显著性检验、聚类 |

### 可视化主题配置

```python
# 统一的可视化风格
import matplotlib.pyplot as plt
import seaborn as sns

# 设置论文级风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

# 设置颜色主题
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'scenario_a': '#4CAF50',
    'scenario_b': '#2196F3',
    'scenario_c': '#FF9800'
}

# 模型颜色映射
MODEL_COLORS = {
    'deepseek-v3.2': '#8E24AA',
    'gpt-5-mini-2025-08-07': '#00ACC1',
    'gemini-3-flash-preview': '#FB8C00',
    'qwen-plus': '#43A047'
}
```

---

## 预期输出示例

### 综合性能雷达图

![Radar Chart Example](https://via.placeholder.com/600x400?text=Radar+Chart+Example)

**说明**：
- 6个维度：决策相似度、经济效率、收敛速度、稳定性、利润准确性、参与率准确性
- 4个模型叠加对比
- 每个场景一个子图

---

### 模型-场景性能热力图

![Heatmap Example](https://via.placeholder.com/800x400?text=Heatmap+Example)

**说明**：
- 行：4个模型
- 列：6个场景配置（A, B, B_FP, C_B_FP, C_C_FP, C_D_FP）
- 颜色：综合得分（0-100）

---

### 场景C策略空间轨迹图

![Trajectory Example](https://via.placeholder.com/600x500?text=Strategy+Trajectory+Example)

**说明**：
- X轴：补偿金额 m
- Y轴：参与率
- 箭头轨迹：策略演化路径
- ★标记：理论最优点

---

## 附录

### A. 指标优先级

**核心指标**（必须计算）：
1. 集合相似度
2. 参与率偏差
3. 利润偏差率
4. 收敛标志

**次要指标**（重要但非必须）：
5. 社会福利偏差率
6. 决策稳定性
7. 收敛速度

**补充指标**（场景特定）：
8. 场景A: 消费者剩余MAE
9. 场景B: 信息泄露MAE
10. 场景C: 策略参数误差

---

### B. 场景配置命名规范

| 简称 | 完整名称 | 说明 |
|------|---------|------|
| **A** | Scenario A | 个性化定价（迭代博弈） |
| **B** | Scenario B (Static) | 推断外部性（静态博弈） |
| **B_FP** | Scenario B (Fictitious Play) | 推断外部性（虚拟博弈） |
| **C_B_FP** | Scenario C Config B (FP) | 理性中介 × LLM消费者 |
| **C_C_FP** | Scenario C Config C (FP) | LLM中介 × 理性消费者 |
| **C_D_FP** | Scenario C Config D (FP) | LLM中介 × LLM消费者 |

---

### C. 常见问题

**Q1: 场景C为什么没有社会福利指标？**

A: 场景C是双层序贯博弈，主要关注中介利润和消费者参与决策，社会福利不是主要评估目标。如需要，可以通过消费者效用总和 + 中介利润计算。

**Q2: 如何处理未收敛的情况？**

A: 未收敛的情况收敛轮次记为`max_iterations`，收敛质量记为最后一轮的集合相似度，在可视化时用特殊标记（如虚线）表示。

**Q3: 不同场景的指标是否可以直接加总计算综合得分？**

A: 不建议直接加总。应该先在每个场景内归一化（转换为0-1或0-100），然后使用加权平均。权重可以根据场景难度或重要性调整。

**Q4: 如何比较场景C的三种配置？**

A: 三种配置测试的是不同能力：
- B_FP: LLM作为消费者的学习能力
- C_FP: LLM作为中介的策略制定能力
- D_FP: LLM双方博弈的协调能力

不建议直接对比得分，而应该看各自在目标任务上的表现。

---

### D. 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-01-26 | 1.0.0 | 初始版本，包含完整的统一指标体系和可视化方案 |

---

## 联系方式

如有疑问或建议，请联系：
- Email: [your-email@example.com]
- GitHub Issue: [repo-url/issues]

---

**文档结束**
