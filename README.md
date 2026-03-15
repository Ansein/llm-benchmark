# LLM隐私外部性Benchmark系统

## 📋 系统概述

**以机制为导向的隐私外部性大模型 Benchmark 框架**

数据市场中的决策过程通常涉及多主体互动、跨阶段反馈以及复杂的因果结构，因此为评估大语言模型（LLMs）的决策能力提供了天然而严格的测试场景。本项目提出一个**以机制为导向的隐私外部性大模型 benchmark 框架**，用于系统评估 LLM 理解能力与推断能力：

- **三类可计算的外部性机制**：围绕价格传导、推断外部性、社会数据外部性构建三个完全不同的经济学场景
- **可复现的理论求解器**：为每个场景实现可验证的理论求解器作为 Ground Truth
- **LLM嵌入博弈过程**：将 LLM 作为决策主体嵌入多轮博弈，观察其决策行为
- **统一诊断指标体系**：同时度量终局偏差、策略结构一致性与动态收敛过程
- **多轮虚拟博弈**：考察模型是否在机制反馈下显著提升其收敛性与均衡一致性

我们的 benchmark 将**"终局正确"与"机制正确"显式分离**，为研究者提供一套具备计算理论解支撑的诊断性评测工具，用以刻画并比较不同 LLM 的机制理解、稳健性与动态稳定性。

### 完整 Pipeline

```
第1步                    第2步                    第3步                    第4步
理论求解器           →   LLM嵌入博弈         →   诊断指标计算        →   鲁棒性验证
(Ground Truth)          (静态/迭代/虚拟博弈)       (MAE+标签+收敛)         (敏感度+虚拟博弈)
                                                                          
• 枚举均衡求解            • 将LLM作为决策主体      • 终局偏差（MAE）        • 参数网格扫描
• 可复现性验证            • 提示词信息完整度实验    • 策略结构一致性         • 虚拟博弈学习能力
• 多场景覆盖              • 多配置对比评估          • 动态收敛检测           • 12款模型对比
```

**各场景评估策略差异**：

| 场景 | 核心评估 | 鲁棒性验证 | 研究问题 |
|------|---------|------------|---------|
| A | 迭代博弈（10轮） | — | 能否理解价格传导？ |
| B | 提示词实验（6版本×12模型） | 虚拟博弈（表现最差模型） + 敏感度分析（3×3网格） | 信息完整度如何影响决策？是真懂还是碰巧？ |
| C | 迭代博弈（4配置×12模型） | 14张可视化 + 综合分析 | 能否在Stackelberg博弈中盈利？ |

## 🗂️ 项目目录结构

```
benchmark/
├── src/
│   ├── scenarios/                         # 【第1步】理论求解器（Ground Truth生成）
│   │   ├── scenario_a_personalization.py      # 场景A：个性化定价（价格传导外部性）
│   │   ├── scenario_b_too_much_data.py        # 场景B：Too Much Data（推断外部性）
│   │   ├── scenario_c_social_data.py          # 场景C：社会数据（社会数据外部性）
│   │   ├── scenario_c_social_data_optimization.py  # 场景C优化求解器
│   │   └── generate_scenario_c_gt.py          # 生成场景C Ground Truth
│   │
│   ├── evaluators/                        # 【第2步】LLM评估器（嵌入博弈过程）
│   │   ├── llm_client.py                      # LLM客户端（OpenAI兼容）
│   │   ├── evaluate_scenario_a.py             # 场景A评估器（迭代博弈）
│   │   ├── evaluate_scenario_b.py             # 场景B评估器（静态博弈+提示词实验）
│   │   ├── evaluate_scenario_c.py             # 场景C评估器（4配置迭代博弈）⭐
│   │   └── scenario_c_metrics.py              # 场景C指标计算
│   │
│   └── utils/
│       └── extract_pdf_text.py                # 论文文本提取工具
│
├── configs/
│   └── model_configs.json                 # 模型配置（12款LLM的API配置）
│
├── data/
│   └── ground_truth/                      # 【第1步输出】理论基准数据
│       ├── scenario_a_*.json
│       ├── scenario_b_*.json
│       ├── scenario_c_*.json
│       └── sensitivity_b/                     # 场景B敏感度分析GT（3×3网格）
│
├── evaluation_results/                    # 【第2-3步输出】评估结果与指标
│   ├── scenario_a/                            # 场景A结果
│   ├── scenario_b/                            # 场景B结果
│   ├── scenario_c/                            # 场景C结果（12个模型）⭐
│   │   ├── scenario_c_*_<model>_*.csv             # 单模型评估CSV
│   │   ├── scenario_c_*_<model>_*_detailed.json   # 单模型详细JSON
│   │   └── visualizations/                        # 可视化图表（14张）⭐
│   │       ├── 1-8: 基础图表（利润、准确率、热力图等）
│   │       ├── advanced/                          # 高级分析图表
│   │       │   └── 1-6: 帕累托前沿、策略空间等
│   │       ├── 场景C评估结果分析报告.md            # 完整分析报告
│   │       └── model_ranking.csv                  # 模型排名数据
│   ├── fp_gpt-5/                              # 场景B虚拟博弈（GPT-5学习验证）
│   ├── fp_deepseek-r1/                        # 场景B虚拟博弈（DeepSeek-r1学习验证）
│   ├── prompt_experiments_b/                  # 场景B提示词实验（6版本×12模型）
│   └── summary_report_*.csv                   # 跨场景汇总报告
│
├── sensitivity_results/                   # 【第4步输出】敏感度分析结果
│   └── scenario_b/                            # 参数网格扫描（ρ×v）
│       ├── sensitivity_3x3_*/                 # 多模型×多参数实验
│       └── gpt_family_comparison_plots/       # GPT家族对比可视化
│
├── scripts/                               # 【辅助脚本】
│   ├── generate_sensitivity_b_gt.py           # 生成场景B敏感度GT
│   ├── generate_scenario_c_sensitivity_gt.py  # 生成场景C敏感度GT
│   ├── run_scenario_a_sweep.py                # 场景A参数扫描
│   └── summarize_scenario_a_results.py        # 场景A结果汇总
│
├── run_evaluation.py                      # 【主入口】批量评估（A+B+C）
├── run_sensitivity_b.py                   # 【敏感度实验】场景B多模型网格扫描
├── run_prompt_experiments.py              # 【提示词实验】场景B提示词版本对比
├── compare_models_sensitivity.py          # 【结果分析】多模型敏感度对比
├── compare_gpt_models_sensitivity.py      # 【结果分析】GPT家族演进分析
├── visualize_scenario_c_results.py        # 【可视化】场景C基础图表（8张）
├── visualize_scenario_c_advanced.py       # 【可视化】场景C高级图表（6张）
│
└── docs/                                  # 【文档】
    ├── design/                                # 设计文档
    ├── README_scenario_c_evaluator.md         # 场景C评估器使用说明 ⭐
    ├── README_evaluation.md                   # 评估系统说明
    ├── README_scenarios.md                    # 场景与GT生成说明
    ├── FICTITIOUS_PLAY_GUIDE.md               # 虚拟博弈实现指南
    └── LLM_LOGGING_GUIDE.md                   # LLM日志记录说明
```

## 📦 模块说明

### src/evaluators/ — 评估器（第2步：LLM嵌入博弈）
- `llm_client.py`: 封装OpenAI兼容的API接口（含超时/重试机制）
- `evaluate_scenario_a.py`: 场景A（个性化定价）评估器
- `evaluate_scenario_b.py`: 场景B（推断外部性）评估器（提示词实验+虚拟博弈）
- `evaluate_scenario_c.py`: 场景C（社会数据外部性）评估器（4配置迭代博弈）⭐
- `scenario_c_metrics.py`: 场景C指标计算（利润/福利/公平性）

### src/scenarios/ — 理论求解器（第1步：Ground Truth）
- `scenario_a_personalization.py`: 场景A理论求解器
- `scenario_b_too_much_data.py`: 场景B理论求解器
- `scenario_c_social_data.py`: 场景C核心模拟（贝叶斯更新+固定点迭代）
- `scenario_c_social_data_optimization.py`: 场景C个性化补偿优化（混合优化+利润约束）
- `generate_scenario_c_gt.py`: 场景C Ground Truth生成入口

### configs/
- `model_configs.json`: 12+款LLM的API配置（OpenAI兼容接口）

### data/ground_truth/
- 各场景的理论基准数据（由求解器生成）

### evaluation_results/
- 评估结果输出（按场景分类），含可视化与分析报告

### papers/
- 论文PDF及提取的文本

## 🎯 三个评估场景：可计算的外部性机制

### 场景A：Personalization & Privacy Choice（价格传导外部性）

**论文来源**：Rhodes & Zhou (2019) "Personalization and Privacy Choice"

**外部性机制**：
- **价格传导**：消费者披露数据 → 平台对其个性化定价 → 影响统一价格 → 损害未披露者福利
- **过度披露**：个体决策未内生化对他人的负外部性 → Nash均衡披露率 > 社会最优
- **机制关键**：个性化定价的价格歧视效应通过统一价格传导给其他消费者

**理论求解器**（第1步）：
- 枚举所有可能的披露集合 D ⊆ {1, ..., N}
- 对每个 D，计算平台最优统一价格（针对未披露者）
- 计算 Nash 均衡：无消费者单边偏离能增益
- 计算社会最优：最大化 W(D) = CS + Platform Profit

**LLM任务**（第2步）：
- 每个消费者决定是否披露数据（迭代博弈，最多10轮）
- 需要权衡：隐私成本 vs 可能的购买效用
- **核心挑战**：理解披露对未披露者价格的影响

**评估指标**（第3步）：
- **终局偏差**：披露率、平台利润、消费者剩余、社会福利的MAE
- **策略结构**：披露率分桶（low/medium/high）匹配度
- **方向标签**：是否识别出"过度披露"现象
- **动态收敛**：迭代轮数、是否收敛到稳定均衡

### 场景B：Too Much Data（推断外部性/信息外部性）

**论文来源**：Acemoglu et al. (2022) "Too Much Data: Prices and Inefficiencies in Data Markets"

**外部性机制**：
- **推断外部性**：用户类型相关 → 他人分享数据会"连带泄露"你的信息（即使你不分享）
- **边际信息次模性**：更多数据 → 边际信息价值递减 → 平台压低补偿价格
- **过度分享**：平台利用信息外部性压价 → 均衡分享率可能 > 社会最优

**理论求解器**（第1步）：
- 用户类型服从多元高斯分布 N(0, Σ)，Σ[i,j] = ρ（相关系数）
- 枚举所有分享集合 S ⊆ {1, ..., N}
- 对每个 S，计算贝叶斯后验协方差（Kalman filtering）
- 计算信息泄露量：I(S) = Σ_i [log det(Σ_prior) - log det(Σ_posterior[i])]
- 平台选择最大化利润的 S：max α·I(S) - Σ_{i∈S} p_i

**LLM评估**（三阶段递进）：

- **第2步：提示词实验（12模型×6版本）**
  - 6个提示词版本（b.v1-b.v6）：从纯文字描述到显式公式+算例，逐步增加信息完整度
  - **核心问题**：信息完整度如何影响LLM对推断外部性的理解？
  - 12个模型全部参与，分析哪些模型在低信息条件下也能正确决策
  
- **第3步：虚拟博弈（表现最差模型的学习能力验证）**
  - 从提示词实验中筛选表现最差的模型（GPT-5、DeepSeek-r1）
  - 使用v4版本提示词，运行Fictitious Play多轮迭代博弈
  - **核心问题**：表现差的模型能否通过多轮反馈学习改善决策？
  
- **第4步：敏感度分析（3×3参数网格）**
  - 参数空间：ρ ∈ {0.3, 0.6, 0.9} × v ∈ 3个区间
  - **核心问题**：LLM是真的理解隐私外部性机制，还是碰巧蒙对了？

**评估指标**：
- **终局偏差**：分享率、平台利润、社会福利、总泄露量的MAE
- **集合相似度**：Jaccard(S_LLM, S_GT)
- **策略结构**：泄露量分桶匹配、过度分享判断
- **学习能力**（虚拟博弈）：策略收敛速度、分享率演化趋势
- **鲁棒性**（敏感度分析）：跨参数方差、参数敏感度

### 场景C：The Economics of Social Data（社会数据外部性）

**论文来源**：Bergemann & Bonatti (2022) "The Economics of Social Data"

**外部性机制**：
- **社会数据特性**：个人数据对预测他人有价值（相关性 → 信息外溢）
- **搭便车问题**：拒绝者仍能从参与者数据中学习改善决策 → 削弱参与激励
- **匿名化机制**：阻止个性化定价 → 保护消费者免受价格歧视
- **Stackelberg博弈**：中介先动设定补偿 → 消费者后动决定参与

**理论求解器**（第1步）：
- **两种数据结构**：
  - Common Preferences（共同偏好）：w_i = θ + σε_i（独立噪声）
  - Common Experience（共同经历）：s_i = w_i + σε（共同冲击）
- **贝叶斯后验更新**：消费者和生产者根据数据库X更新信念
- **关键区别**：匿名化下生产者无法个性化定价
- **个性化补偿m_i**：中介对每个消费者设定差异化补偿，最大化利润 R = m_0 - Σ m_i·p_i（约束R>0）

**LLM任务**（第2步）：
- **四种配置**（逐步增加LLM参与度，隔离各角色的影响）：
  - A: 理性中介 × 理性消费者（Ground Truth基准）
  - B: 理性中介 × LLM消费者（测试消费者理解能力）
  - C: LLM中介 × 理性消费者（测试中介定价能力）
  - D: LLM中介 × LLM消费者（完整博弈，真正的试金石）
- **迭代学习**：20轮，中介根据利润反馈调整补偿，消费者观察参与结果

**评估指标**（第3步）：
- **终局偏差**：参与率、消费者剩余、生产者利润、社会福利的MAE
- **策略结构**：中介补偿偏差、参与率匹配度
- **利润分析**：中介利润、利润损失率、个性化补偿成本
- **福利分析**：Consumer Surplus Gap（正=消费者受益）、Gini Consumer Surplus（公平性）
- **配置间一致性**：Cross-Config Consistency（配置B/C/D跨配置鲁棒性）

## 📊 完整评估 Pipeline（论文参考）

> 以下内容按论文写作所需的详细程度组织，覆盖每个步骤的数学描述、实验设计动机、具体参数与输出格式。

---

### 第1步：理论求解器（Ground Truth生成）

**目标**：为每个场景生成可复现、可验证的理论基准，作为评估LLM决策质量的参照。

#### 1.1 场景A — Nash均衡枚举

**模型设定**：N个消费者，每人持有私有估值 \(v_i \sim F\) 和隐私成本 \(c_i\)，决定是否向平台披露数据。平台观察到披露集合 D 后，对披露者个性化定价，对未披露者收取统一价格 \(p(D)\)。

**求解方法**：
1. 枚举所有可能的披露集合 \(D \subseteq \{1, ..., N\}\)（\(2^N\) 种）
2. 对每个 D，计算平台最优统一价格 \(p^*(D)\) — 最大化对未披露者的利润
3. **Nash均衡**：找到 \(D^*\) 使得没有消费者能通过单边改变披露决策获益
4. **社会最优**：找到 \(D^{SO}\) 使社会福利 \(W(D) = CS(D) + \Pi(D)\) 最大化

**输出**：`data/ground_truth/scenario_a_params_n10_seed42.json`

#### 1.2 场景B — 推断外部性下的最优报价

**模型设定**：N个用户，类型向量 \(\mathbf{v} \sim \mathcal{N}(0, \Sigma)\)，其中 \(\Sigma_{ij} = \rho\)（用户间相关系数）。每个用户有隐私偏好 \(v_i \sim \text{Uniform}[v_{lo}, v_{hi}]\)。平台向用户报价购买数据。

**求解方法**：
1. 枚举所有分享集合 \(S \subseteq \{1, ..., N\}\)
2. 对每个 S，计算贝叶斯后验协方差（Kalman filtering）
3. 计算信息泄露量：\(I(S) = \sum_i [\log\det(\Sigma_{prior}) - \log\det(\Sigma_{posterior}^{(i)})]\)
4. 平台最优：\(\max_S \alpha \cdot I(S) - \sum_{i \in S} p_i\)

**默认参数**：N=8, ρ=0.6, v∈[0.3, 1.2], α=1.0, seed=42

**敏感度GT**：额外生成3×3参数网格的GT（ρ∈{0.3, 0.6, 0.9} × v∈{低, 中, 高}）

**输出**：
- `data/ground_truth/scenario_b_rho0.6_v0.3-1.2.json`（默认参数）
- `data/ground_truth/sensitivity_b/`（3×3网格，9组GT）

#### 1.3 场景C — Stackelberg均衡与个性化补偿优化

**模型设定**：N=20个消费者，数据结构为Common Preferences（\(w_i = \theta + \sigma\varepsilon_i\)）。中介（data intermediary）先动设定补偿向量 \(\mathbf{m} = (m_1, ..., m_N)\)，消费者后动决定是否参与。

**核心参数**：

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| 消费者数量 | N | 20 | 市场规模 |
| 偏好均值 | \(\mu_\theta\) | 5.0 | 消费者平均偏好强度 |
| 偏好方差 | \(\sigma_\theta\) | 1.0 | 偏好异质性 |
| 噪声标准差 | \(\sigma\) | 1.0 | 信号噪声 |
| 隐私成本均值 | \(\bar{\tau}\) | 1.0 | 平均参与代价 |
| 隐私成本方差 | \(\sigma_\tau\) | 0.3 | 隐私偏好异质性 |
| 直接成本 | c | 0.0 | 数据处理成本 |
| 随机种子 | seed | 42 | 可复现性 |

**求解方法**（混合优化）：
1. **网格搜索初始化**：在 \(m \in [0, 3]\) 上均匀采样（grid_size=11），评估每个均匀 m 对应的中介利润
2. **连续优化**：以网格搜索最优解为初始值，使用 `scipy.optimize.minimize`（L-BFGS-B）优化 N 维补偿向量 \(\mathbf{m}^*\)
3. **利润约束**：\(R = m_0 - \sum_i m_i \cdot p_i > 0\)（中介利润必须为正，否则不购买数据）
4. **固定点迭代**：对每个 m 向量，内层通过固定点迭代求解均衡参与率 \(r^*\) 和效用差 \(\Delta U_i\)
5. **对 anonymized 和 identified 两种策略分别优化**，选择利润最高的策略

**最终输出**（`data/ground_truth/scenario_c_common_preferences_optimal.json`）：

| 字段 | 值 | 说明 |
|------|-----|------|
| \(m^*\) | [0.6, 0.6, ..., 0.6] (20维) | 最优个性化补偿向量 |
| anonymization | identified | 最优匿名化策略 |
| \(R^*\) | 1.716 | 中介最优利润 |
| \(r^*\) | 3.42% | 均衡参与率 |
| \(\Delta U^*\) | 0.453 | 参与的效用提升 |
| \(m_0^*\) | 1.710 | 中介从生产者获得的收入 |
| CS | 76.55 | 消费者剩余 |
| \(\Pi_{producer}\) | 131.18 | 生产者利润 |
| SW | 209.05 | 社会福利 |

```bash
# 场景A：自动生成（内置于evaluate_scenario_a.py）
# 场景B：python scripts/generate_sensitivity_b_gt.py
# 场景C：python -m src.scenarios.generate_scenario_c_gt
```

---

### 第2步：LLM嵌入博弈过程

**目标**：将LLM作为决策主体嵌入博弈，观察其在不同信息条件和角色配置下的决策行为。

#### 2.1 场景A — 迭代博弈

**设计**：LLM扮演消费者，在最多10轮迭代中决定是否披露数据。

```
输入: 当前披露集合 S_t, 消费者i的参数(v_i, c_i), 机制说明
              ↓
         LLM决策: action_i ∈ {披露, 不披露}
              ↓
         更新: S_{t+1}
              ↓
         收敛检测: S_{t+1} == S_t → 终止
```

**每个决策重复 num_trials=3 次，多数投票确定最终行动。**

#### 2.2 场景B — 提示词信息完整度实验

**实验设计动机**：推断外部性是三个场景中LLM最难理解的机制。我们设计了6个提示词版本（b.v1-b.v6），逐步增加提供给LLM的信息量，以回答：**LLM需要多少信息才能正确理解"即使不分享也会被推断"这一机制？**

**6个提示词版本的信息完整度递进**：

| 版本 | 个人参数 | 市场参数 | 参数解释 | 外部性机制 | 次模性 | 决策框架 | 数学公式 |
|------|---------|---------|---------|-----------|--------|---------|---------|
| b.v1 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| b.v2 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| b.v3 | ✅ | ✅ | ✅ | 基础 | ❌ | ❌ | ❌ |
| b.v4 | ✅ | ✅ | ✅ | 完整 | ✅ | 基础 | ❌ |
| b.v5 | ✅ | ✅ | ✅ | 完整 | ✅ | 结构化 | ❌ |
| b.v6 | ✅ | ✅ | ✅ | 完整 | ✅ | 理性期望框架 | ✅ |

**各版本的关键差异**：
- **b.v1**（纯原始数据）：仅提供 v_i, p_i, N, ρ, σ² 等数值，不做任何解释
- **b.v2**（+参数解释）：解释 ρ 表示用户间相关性（0-1标度）、σ² 表示噪声程度
- **b.v3**（+基础机制）：引入"推断外部性"概念 — 他人分享数据会影响你的隐私
- **b.v4**（+完整机制）：解释次模性（更多人分享 → 边际泄露递减）、补偿定价逻辑
- **b.v5**（+结构化呈现）：清晰的章节划分、明确的"你的任务"指引
- **b.v6**（+数学公式）：提供完整理性期望框架，含期望效用公式 E[u_i|share] vs E[u_i|not share]

**实验矩阵**：12个模型 × 6个版本 × num_trials=3 = 216 次实验

```
for 每个模型 M ∈ {12款LLM}:
  for 每个版本 V ∈ {b.v1, ..., b.v6}:
    平台: 使用理论求解器计算最优报价 {p_i}
    for 每个用户 i:
      提示词 = 构建(V, p_i, ρ, v_i, ...)  // 按版本决定包含哪些信息
      LLM决策: share / not_share (重复3次, 多数投票)
    → 形成 S_LLM
    → 计算: Jaccard(S_LLM, S_GT), 分享率MAE, 利润MAE, ...
```

#### 2.3 场景B — 虚拟博弈（Fictitious Play）

**实验设计动机**：从提示词实验中筛选出表现最差的模型（GPT-5、DeepSeek-r1），验证它们能否通过多轮反馈学习改善理解。

**选择逻辑**：
- **GPT-5**：在v4提示词下Jaccard相似度低，分享率严重偏离GT
- **DeepSeek-r1**：推理型模型，静态博弈中过于保守，几乎不分享

**实验参数**：
- 提示词版本：b.v4（信息完整度中等 — 测试学习能力而非信息依赖）
- 最大轮数：50轮
- 信念窗口：10轮（基于最近10轮历史形成信念）
- 平台策略：理性报价（使用GT最优报价，保持不变）

```
初始化: beliefs = 均匀分布
for t = 1, 2, ..., 50:
  平台: p_i = GT最优报价（固定不变）
  for 每个用户 i:
    观察: 最近10轮的分享频率分布 {freq_j}
    提示词: b.v4 + 历史分享频率信息
    LLM决策: share / not_share
  更新: beliefs_{t+1} = moving_avg(history[t-10:t])
  记录: 分享率, Jaccard, 策略热力图
  收敛检测: 若分享集合连续3轮不变 → 提前终止
```

**输出**：
- 分享率演化曲线（share_rate.png）— 观察是否逐轮向GT收敛
- 策略热力图（strategy_heatmap.png）— 每轮每用户的决策可视化
- 结果JSON — 最终均衡与GT的偏差

#### 2.4 场景C — 四配置迭代博弈

**实验设计动机**：通过逐步增加LLM参与度的4种配置，**隔离消费者理解能力与中介定价能力的各自贡献**。

**四种配置**：

| 配置 | 中介角色 | 消费者角色 | 测试目标 | 迭代轮数 |
|------|---------|-----------|---------|---------|
| A | 理性（GT最优策略） | 理性（τ_i vs ΔU_i） | 基准 | — |
| B | 理性（GT最优m*） | **LLM** | 消费者能否理解参与收益？ | 1轮 |
| C | **LLM** | 理性 | 中介能否学会定价？ | 20轮（迭代） |
| D | **LLM** | **LLM** | 完整博弈：双方都是LLM时能否达到均衡？ | 20轮（迭代） |

**配置B详细流程**（单轮，测试消费者理解力）：

```
中介: 使用GT最优策略 (m* = 0.6, identified)
for 每个消费者 i:
  提示词包含:
    - 个人参数: θ_i, τ_i, s_i
    - 补偿: m_i = 0.6
    - 匿名化策略: identified
    - 市场背景: Common Preferences, N=20
  LLM决策: participate / reject (重复3次, 多数投票)
→ 计算: individual_accuracy = (TP+TN) / N
```

**配置C详细流程**（迭代，测试中介学习能力）：

```
for t = 1, 2, ..., 20:
  LLM中介接收信息:
    - 市场参数: N=20, μ_θ=5, σ_θ=1, τ̄=1, σ_τ=0.3, data_structure
    - 反馈（t>1时）: 上轮利润, 参与率, m_0, 中介成本
    - 消费者理由: 参与者列出"τ_i ≤ ΔU_i", 拒绝者列出"τ_i > ΔU_i"
  LLM中介决策: 补偿m, 匿名化策略
  →
  理性消费者: 根据(m, 匿名化)计算ΔU_i, 若τ_i ≤ ΔU_i则参与
  →
  计算: 利润, 参与率, 福利指标
  记录: 每轮结果, 选取最高利润轮作为最终结果
```

**配置D详细流程**（完整博弈，双方迭代学习）：

```
for t = 1, 2, ..., 20:
  LLM中介: 同配置C（接收上轮反馈 → 调整m和匿名化策略）
  →
  for 每个LLM消费者 i:
    提示词包含: θ_i, τ_i, s_i, 本轮的m_i, 匿名化策略
    LLM决策: participate / reject
  →
  计算: 利润, 参与率, 消费者剩余, 社会福利
  记录: 所有指标, 选取最高利润轮作为最终结果
```

**稳定性评估**：
- 每个消费者决策重复 `num_trials=3` 次，多数投票确定最终行动
- 记录决策分布与一致性（confidence）

---

### 第3步：诊断指标体系

**核心思想**：将"终局正确"（结果准确性）与"机制正确"（决策逻辑正确性）**显式分离**，避免模型"蒙对结果但不理解机制"。

#### 3.1 通用指标（所有场景）

**A. 终局偏差（Outcome Deviation）**

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| 参与率/披露率/分享率 MAE | \(\|r_{LLM} - r_{GT}\|\) | A, B, C |
| 平台/中介利润 MAE | \(\|\Pi_{LLM} - \Pi_{GT}\|\) | A, B, C |
| 消费者剩余 MAE | \(\|CS_{LLM} - CS_{GT}\|\) | A, C |
| 社会福利 MAE | \(\|SW_{LLM} - SW_{GT}\|\) | A, B, C |
| 信息泄露量 MAE | \(\|I_{LLM} - I_{GT}\|\) | B |

**B. 策略结构一致性（Structural Consistency）**

```python
# 分桶标签（将连续值离散化，检测结构性理解）
if rate < 0.33: bucket = "low"
elif rate < 0.67: bucket = "medium"
else: bucket = "high"
label_match = (bucket_LLM == bucket_GT)  # bool

# 方向标签（是否识别出过度披露/过度分享）
over_participation = (rate_LLM > rate_optimal)
direction_match = (over_LLM == over_GT)  # bool
```

**C. 集合相似度（Set-level Metrics）**

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| Jaccard | \(J = \frac{\|S_{LLM} \cap S_{GT}\|}{\|S_{LLM} \cup S_{GT}\|}\) | [0,1] | 集合重叠度 |
| Hamming | \(H = \frac{\|S_{LLM} \triangle S_{GT}\|}{N}\) | [0,1] | 归一化对称差 |
| F1-score | \(F1 = \frac{2 \cdot P \cdot R}{P + R}\) | [0,1] | 精确率×召回率 |

#### 3.2 场景B专用指标

| 指标 | 说明 | 意义 |
|------|------|------|
| 泄露量分桶匹配 | 总泄露量离散化后是否与GT一致 | 信息量级理解 |
| 过度分享判断 | \(r_{LLM} > r_{GT}\) | 是否识别过度分享现象 |
| 关停更优判断 | 是否识别出关停平台对社会更优的情况 | 深层机制理解 |
| 分享率演化趋势（FP） | 50轮中分享率的时间序列 | 学习动态 |

#### 3.3 场景C专用指标

**参与与准确率**：

| 指标 | 公式 | 说明 |
|------|------|------|
| Individual Accuracy | \(\frac{TP + TN}{N}\) | 消费者决策正确率 |
| 参与率绝对误差 | \(\|r_{LLM} - r_{GT}\|\) | 均衡偏离度 |
| 参与率相对误差 | \(\frac{\|r_{LLM} - r_{GT}\|}{r_{GT}}\) | 相对偏离度 |

**策略与利润**：

| 指标 | 公式 | 说明 |
|------|------|------|
| 补偿偏差 | \(\|m_{LLM} - m_{GT}\|\) | 中介定价准确度 |
| 匿名化策略匹配 | \(\mathbb{1}[anon_{LLM} = anon_{GT}]\) | 策略方向正确性 |
| 利润损失率 | \(\frac{\Pi_{GT} - \Pi_{LLM}}{\Pi_{GT}} \times 100\%\) | 中介效率损失 |

**福利与公平性**：

| 指标 | 公式 | 说明 |
|------|------|------|
| Consumer Surplus Gap | \(CS_{LLM} - CS_{GT}\) | **正值=消费者受益，负值=消费者受损** |
| Gini Consumer Surplus | Gini系数 | 消费者间的福利分配公平性 |
| Welfare Loss | \(SW_{GT} - SW_{LLM}\) | 社会福利效率损失 |

**跨配置分析**：

| 指标 | 公式 | 说明 |
|------|------|------|
| Cross-Config Consistency | 变异系数的倒数或盈利配置占比 | 跨B/C/D配置的利润稳定性 |
| Interaction Effect | \((SW_A - SW_D) - [(SW_A - SW_B) + (SW_A - SW_C)]\) | 双方LLM交互的额外福利损失 |
| Exploitation Indicator | \(\frac{\Pi_D / \Pi_A}{CS_D / CS_A}\) | >1 表示中介利润增长快于消费者 |

---

### 第4步：鲁棒性验证（场景B）

**目标**：验证LLM在提示词实验中的表现是源于真正的机制理解还是偶然/过拟合。

#### 4.1 虚拟博弈 — 学习能力验证

**实验设计动机**：如果模型在静态博弈中表现差，但能通过多轮反馈逐步改善，说明模型具备学习能力但缺乏先验知识。如果多轮后仍无改善，说明模型根本无法理解该机制。

| 参数 | 值 | 设计理由 |
|------|-----|---------|
| 被测模型 | GPT-5, DeepSeek-r1 | 提示词实验中表现最差 |
| 提示词版本 | b.v4 | 中等信息量 — 测试学习能力而非信息依赖 |
| 最大轮数 | 50 | 足够观察收敛趋势 |
| 信念窗口 | 10 | 平衡历史依赖与适应性 |
| 平台策略 | GT最优报价（固定） | 控制变量：仅观察用户端学习 |
| 收敛条件 | 分享集合连续3轮不变 | 提前终止条件 |

**分析维度**：
- 分享率时间序列 — 是否单调趋近GT？
- 策略热力图 — 哪些用户最先/最后改变决策？
- 最终Jaccard — 学习50轮后的均衡质量

#### 4.2 敏感度分析 — 机制理解验证

**实验设计动机**：如果LLM在默认参数下表现好，但参数变化后表现大幅波动，说明模型可能是对特定数值模式匹配而非理解底层机制。真正理解机制的模型应在不同参数下保持一致的决策逻辑。

**3×3参数网格**：

| | v∈[0.3,0.6]（低隐私） | v∈[0.6,0.9]（中隐私） | v∈[0.9,1.2]（高隐私） |
|---|---|---|---|
| **ρ=0.3（弱相关）** | GT-1 | GT-2 | GT-3 |
| **ρ=0.6（中相关）** | GT-4 | GT-5（默认） | GT-6 |
| **ρ=0.9（强相关）** | GT-7 | GT-8 | GT-9 |

**参数语义**：
- **ρ变大**：用户间关联增强 → 推断外部性加剧 → 理性分享率应降低
- **v变大**：隐私偏好增强 → 参与代价更高 → 理性分享率应降低

**核心检验**：LLM的分享率是否随ρ和v变化而展现出与理论一致的单调趋势？

**实验规模**：9组参数 × 12模型 × num_trials=3 = 324次实验

**分析维度**：
- **跨参数一致性**：不同ρ×v下LLM表现的方差
- **趋势正确性**：分享率是否随ρ↑/v↑而↓（与理论方向一致）
- **GPT家族演进**：GPT-5 → 5.1 → 5.2 跨参数方差是否逐代减小

---

### 配置参数汇总

| 参数 | 默认值 | 场景A | 场景B-提示词 | 场景B-FP | 场景B-敏感度 | 场景C |
|------|--------|-------|-------------|---------|-------------|------|
| N（代理数） | — | 10 | 8 | 8 | 8 | 20 |
| `num_trials` | 3 | ✓ | ✓ | ✓ | ✓ | ✓ |
| `max_iterations` | — | 10 | 1 | 50 | 1 | 20 |
| `prompt_version` | — | — | b.v1-b.v6 | b.v4 | b.v4 | — |
| `belief_window` | — | — | — | 10 | — | — |
| ρ | — | — | 0.6 | 0.6 | {0.3,0.6,0.9} | — |
| v | — | — | [0.3,1.2] | [0.3,1.2] | 3个区间 | — |
| 模型数 | 12 | 12 | 12 | 2(gpt-5,r1) | 12 | 12 |
| 总实验数 | — | ~12 | 216 | ~2 | 324 | ~48 |

## 🚀 快速开始

### 0. 安装依赖

```bash
pip install openai httpx numpy pandas scipy matplotlib seaborn
```

Windows 用户建议设置编码：
```powershell
$env:PYTHONIOENCODING="utf-8"
```

---

### 完整 Pipeline 示例

#### 【场景A】个性化定价（价格传导外部性）

```bash
# 第1步：GT自动生成（内置于评估器）
# 第2步：LLM迭代博弈（最多10轮）
# 第3步：诊断指标计算
python run_evaluation.py --single --scenarios A --models deepseek-v3.2 --max-iterations 10

# 输出：evaluation_results/scenario_a/eval_scenario_A_deepseek-v3.2.json
```

#### 【场景B】Too Much Data（推断外部性）— 完整三阶段Pipeline

```bash
# 第1步：生成GT（含敏感度分析用3×3网格）
python scripts/generate_sensitivity_b_gt.py

# 第2步：提示词实验（12模型×6版本 → 分析信息完整度的影响）
python run_prompt_experiments.py \
  --versions b.v1 b.v2 b.v3 b.v4 b.v5 b.v6 \
  --model deepseek-v3.2 \
  --num-trials 3
# 对所有12个模型重复上述命令
# 输出：evaluation_results/prompt_experiments_b/
# 可视化：python plot_prompt_comparison_academic.py

# 第3步：虚拟博弈（表现最差模型的学习能力验证）
# 使用v4提示词，对gpt-5和deepseek-r1运行Fictitious Play
python run_evaluation.py --scenarios B --models gpt-5 --mode fp --rounds 50
python run_evaluation.py --scenarios B --models deepseek-r1 --mode fp --rounds 50
# 输出：evaluation_results/fp_gpt-5/, evaluation_results/fp_deepseek-r1/

# 第4步：敏感度分析（3×3参数网格 → 验证是真懂还是蒙对）
python run_sensitivity_b.py \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3
# 结果分析与可视化
python compare_models_sensitivity.py
python compare_gpt_models_sensitivity.py
# 输出：sensitivity_results/scenario_b/
```

#### 【场景C】社会数据外部性（完整评估已完成）

```bash
# 第1步：生成Ground Truth（个性化补偿，混合优化）
python -m src.scenarios.generate_scenario_c_gt
# 输出：data/ground_truth/scenario_c_common_preferences_optimal.json

# 第2步：批量评估（12个模型，4配置：A/B/C/D）
python run_evaluation.py --scenarios C --models gpt-5.2 gpt-5.1 deepseek-r1 qwen3-max-2026-01-23
# 输出：evaluation_results/scenario_c/scenario_c_*_<model>_*.csv

# 第2步（单模型）：直接运行评估器
python src/evaluators/evaluate_scenario_c.py
# 在脚本中修改 TARGET_MODEL = "deepseek-v3.2" 选择模型

# 第3步：生成可视化与分析报告（14张图表）
python visualize_scenario_c_results.py      # 8张基础图表
python visualize_scenario_c_advanced.py     # 6张高级图表
# 输出：evaluation_results/scenario_c/visualizations/
```

---

### 批量评估与模型对比

#### 跨场景评估

```bash
# 评估所有场景（默认模型）
python run_evaluation.py --scenarios A B C

# 评估多个模型
python run_evaluation.py \
  --scenarios A B C \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3 \
  --max-iterations 15

# 仅生成汇总报告（不重新运行）
python run_evaluation.py --summary-only
```

#### GPT家族演进分析（场景B敏感度）

```bash
# 评估GPT-5、GPT-5.1、GPT-5.2三代模型
python run_sensitivity_b.py \
  --models gpt-5 gpt-5.1 gpt-5.2 \
  --prompt-version b.v4 \
  --num-trials 3

# 生成演进分析报告
python compare_gpt_models_sensitivity.py
# 输出：sensitivity_results/scenario_b/gpt_family_comparison_plots/
#   - gpt_evolution_trends.png        # 三代演进趋势
#   - gpt_improvement_matrix.png      # 逐代改进矩阵
#   - gpt_parameter_impact.png        # 参数敏感度
#   - gpt_radar_profiles.png          # 多维雷达图
```

## 📈 测试结果（示例）

### 场景A测试结果

```
【披露集合比较】
  LLM均衡: []
  理论均衡: []
  收敛情况: ✅ 已收敛 (迭代1次)

【关键指标】
  平台利润:     LLM=3.500  |  GT=3.500  |  MAE=0.000
  消费者剩余:   LLM=1.400  |  GT=1.400  |  MAE=0.000
  社会福利:     LLM=4.900  |  GT=4.900  |  MAE=0.000
  披露率:       LLM=0.00%  |  GT=0.00%  |  MAE=0.00%

【标签一致性】
  披露率分桶:   LLM=low  |  GT=low  |  ✅
  过度披露:     LLM=0  |  GT=0  |  ✅
```

**解读**：GPT-4.1-mini在场景A中表现完美，完全理解了个性化定价的外部性机制。

### 场景B测试结果

```
【分享集合比较】
  LLM均衡: []
  理论均衡: [4, 5, 6]
  收敛情况: ✅ 已收敛 (迭代1次)

【关键指标】
  平台利润:     LLM=0.0000  |  GT=4.5620  |  MAE=4.5620
  社会福利:     LLM=0.0000  |  GT=1.8161  |  MAE=1.8161
  总泄露量:     LLM=0.0000  |  GT=5.6940  |  MAE=5.6940
  分享率:       LLM=0.00%  |  GT=37.50%  |  MAE=37.50%

【标签一致性】
  泄露量分桶:   LLM=low  |  GT=high  |  ❌
  过度分享:     LLM=0  |  GT=0  |  ✅
  关停更优:     LLM=0  |  GT=0  |  ✅
```

**解读**：GPT-4.1-mini在场景B中过于谨慎，没有理解"即使不分享也会被推断"的关键机制，导致所有用户都选择不分享。

## 💡 关键发现与研究价值

### 核心创新

1. **"终局正确" vs "机制正确"的显式分离**
   - 传统评估：只看最终结果（如分享率、利润）
   - 本框架：同时评估结果偏差、策略结构、动态过程
   - 价值：区分模型行为是源于机制理解，还是语言启发式/模式匹配

2. **可计算的外部性机制**
   - 三类完全不同的经济学场景（价格传导、推断外部性、社会数据外部性）
   - 每个场景都有可验证的理论求解器作为 Ground Truth
   - 为研究者提供具备理论支撑的诊断工具

3. **多维度诊断指标体系**
   - 终局偏差（MAE）：度量结果准确性
   - 策略结构一致性：度量决策逻辑
   - 动态收敛过程：度量学习能力与稳定性
   - 集合相似度：度量均衡质量

### 评估发现

#### 场景C完整评估结果（12款LLM，2026-02-05）

场景C是目前评估最完整的场景，已对12个模型完成了全部4种配置的评估，并生成了14张专业可视化图表。

**综合排名 Top 5**:

| 排名 | 模型 | 配置D利润 | 配置D准确率 | 核心优势 |
|------|------|-----------|-------------|---------|
| 1 | **qwen3-max-2026-01-23** | 2.346 (+79%) | 100% | 利润最高，策略创新 |
| 2 | **gpt-5.1** | 1.858 (+42%) | 95% | 鲁棒性强，跨配置稳定 |
| 3 | **gpt-5.2** | 1.680 (+28%) | 75% | **唯一Win-Win**（利润+消费者受益） |
| 4 | deepseek-v3.2 | 1.177 (-10%) | 10% | DeepSeek唯一盈利 |
| 5 | gpt-5.1-2025-11-13 | 0.561 (-57%) | 45% | 中等表现 |

> 理论最优利润 = 1.311，补偿 m* = 0.6，参与率 r* = 3.42%

**核心发现**:

1. **配置D是真正的试金石**：仅5/12（42%）模型在完全LLM环境下盈利
2. **超越理论的可能性**：qwen3-max-2026-01-23利润超理论79%，证明LLM可探索理论未覆盖的策略空间
3. **帕累托改进极为稀缺**：仅gpt-5.2同时实现利润增长（+28%）与消费者受益（CS Gap +0.61）
4. **DeepSeek-r1的推理-行动脱节**：消费者角色100%准确率（顶级），但中介角色零利润（过于保守）
5. **系统性认知偏差**：LLM展现损失厌恶（参与率偏低）、框架效应（匿名化偏好反理论）、协调失败（58%零利润）

**详细分析报告**: `evaluation_results/scenario_c/visualizations/场景C评估结果分析报告.md`

#### 模型家族对比

- **GPT家族**（gpt-5 → gpt-5.1 → gpt-5.2）：
  - V字演进：gpt-5在配置D亏损，gpt-5.1/5.2大幅改进
  - gpt-5.2是唯一Win-Win模型（帕累托最优）
  - 场景B中Jaccard相似度逐代提升（0.45 → 0.62 → 0.71）
  
- **DeepSeek家族**（v3-0324 → v3.1 → v3.2 → r1）：
  - 配置C稳定（1.607-1.714），配置D仅v3.2盈利
  - r1推理能力最强（100%消费者准确率），但策略执行保守
  - 价值取向演进：从消费者友好（v3-0324）到利润导向（v3.2）
  
- **Qwen家族**（qwen-plus → qwen3-max-2026-01-23）：
  - 跨代飞跃：qwen3-max-2026-01-23综合排名第1
  - 极低补偿策略（m=0.4）打破传统智慧
  - qwen-plus系列配置C/D双重失败（补偿过低）

#### 场景难度排序
1. **场景A（最简单）**：多数模型能理解价格传导机制，MAE < 0.1
2. **场景C配置B/C（中等）**：消费者准确率≥90%，中介利润损失-20~-50%
3. **场景C配置D（困难）**：58%模型零利润，42%模型亏损或零利润
4. **场景B（最难）**：推断外部性理解最弱，集合相似度普遍 < 0.5

#### 提示词敏感度（场景B）
- 版本b.v1-b.v3（无公式）：Jaccard < 0.3
- 版本b.v4-b.v6（显式公式+算例）：Jaccard提升至0.4-0.6
- 结论：机制解释方式对理解至关重要

### 场景C可视化体系（14张图表）

场景C的评估结果已生成完整的两层可视化分析，所有图表使用Times New Roman字体、英文标签、300 DPI高分辨率。

**基础可视化**（8张，`evaluation_results/scenario_c/visualizations/`）：

| 图表 | 文件名 | 内容 |
|------|--------|------|
| 1 | `1_config_D_profit_comparison.png` | 配置D中介利润对比 |
| 2 | `2_config_C_intermediary_strategy.png` | LLM中介策略质量 |
| 3 | `3_config_B_consumer_accuracy.png` | LLM消费者决策准确率 |
| 4 | `4_multi_config_heatmap.png` | 三配置利润损失热力图 |
| 5 | `5_config_D_strategy_analysis.png` | 补偿/参与率 vs 利润 |
| 6 | `6_consumer_welfare_analysis.png` | 消费者福利+基尼系数 |
| 7 | `7_top5_models_radar.png` | Top 5模型5维雷达图 |
| 8 | `8_comprehensive_ranking.png` | 综合排名表 |

**高级可视化**（6张，`evaluation_results/scenario_c/visualizations/advanced/`）：

| 图表 | 文件名 | 科学价值 |
|------|--------|---------|
| 1 | `1_pareto_frontier_analysis.png` | 利润-福利权衡（帕累托前沿） |
| 2 | `2_strategy_space_map.png` | 战略定位分析（气泡图） |
| 3 | `3_model_series_evolution.png` | 模型系列演进（折线图） |
| 4 | `4_metric_correlation_heatmap.png` | 指标相关性（Pearson） |
| 5 | `5_parallel_coordinates_profit.png` | 配置间鲁棒性（平行坐标） |
| 6 | `6_efficiency_frontier_analysis.png` | 效率前沿（归一化比较） |

```bash
# 生成基础图表
python visualize_scenario_c_results.py

# 生成高级图表
python visualize_scenario_c_advanced.py
```

### Benchmark 价值

1. **研究工具**：
   - 为LLM机制理解能力提供定量评估框架
   - 支持跨模型、跨版本、跨参数的系统对比
   - 可扩展到更多经济学/博弈论场景

2. **诊断能力**：
   - 识别模型的系统性弱点（如推断外部性理解）
   - 评估提示词工程的有效性
   - 测试多轮学习与收敛能力

3. **未来方向**：
   - 引入异质部署（不同代理使用不同LLM）
   - 扩展到动态博弈（长期重复博弈）
   - 添加解释题评估（why/how questions）

## 📊 输出文件

### 输出目录结构

评估结果按场景分类保存：

```
evaluation_results/
├── scenario_a/                    # 场景A评估结果
│   ├── eval_scenario_A_deepseek-v3.2.json
│   └── ...
├── scenario_b/                    # 场景B评估结果
│   ├── eval_scenario_B_deepseek-v3.2.json
│   └── ...
├── scenario_c/                    # 场景C评估结果（12个模型）⭐
│   ├── scenario_c_common_preferences_<model>_<timestamp>.csv
│   ├── scenario_c_common_preferences_<model>_<timestamp>_detailed.json
│   ├── visualizations/            # 可视化与分析 ⭐
│   │   ├── 1-8: 基础可视化（8张PNG）
│   │   ├── advanced/              # 高级可视化（6张PNG）
│   │   │   ├── 1_pareto_frontier_analysis.png
│   │   │   └── ...
│   │   ├── model_ranking.csv      # 模型排名数据
│   │   └── 场景C评估结果分析报告.md # 完整分析报告
│   └── ...
├── fp_gpt-5/                      # 场景B虚拟博弈（GPT-5）
├── fp_deepseek-r1/                # 场景B虚拟博弈（DeepSeek-r1）
├── prompt_experiments_b/          # 场景B提示词实验（6版本×12模型）
└── summary_report_*.csv           # 跨场景汇总报告
```

### 单个评估结果

```json
{
  "model_name": "deepseek-v3.2",
  "llm_disclosure_set": [0, 2, 5],
  "gt_disclosure_set": [5],
  "converged": true,
  "iterations": 3,
  "metrics": {
    "llm": {...},
    "ground_truth": {...},
    "deviations": {
      "profit_mae": 0.123,
      "cs_mae": 0.456,
      ...
    }
  },
  "labels": {...}
}
```

### 汇总报告（CSV）

| 场景 | 模型 | 收敛 | 迭代次数 | 利润MAE | CS_MAE | 福利MAE | 标签匹配 |
|------|------|------|----------|---------|--------|---------|----------|
| A | deepseek-v3.2 | ✅ | 1 | 0.000 | 0.000 | 0.000 | ✅ |
| B | deepseek-v3.2 | ✅ | 1 | 4.562 | - | 1.816 | ❌ |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 🔧 技术细节

### LLM提示设计

每个提示包含：
1. **场景描述**：清晰解释外部性机制
2. **个性化信息**：当前代理的参数
3. **当前状态**：其他代理的决策
4. **计算辅助**：预先计算关键数值
5. **决策要求**：JSON格式输出

### 均衡求解策略

- **固定点迭代**：重复决策直到收敛
- **随机顺序**：避免顺序偏差
- **多次试验**：评估稳定性
- **收敛检测**：无代理改变决策

### 评分方法

- **MAE计算**：使用理论基准作为ground truth
- **标签评分**：结构/方向/桶化匹配
- **置信区间**：多次运行计算统计显著性（可选）

## 📚 理论基础与相关论文

### 本项目实现的三篇核心论文

1. **场景A：Personalization and Privacy Choice**
   - 作者：Rhodes & Zhou (2019)
   - 期刊：未发表工作稿
   - 机制：价格传导外部性 → 过度披露
   - 实现：`src/scenarios/scenario_a_personalization.py`

2. **场景B：Too Much Data: Prices and Inefficiencies in Data Markets**
   - 作者：Acemoglu, Makhdoumi, Malekian, Ozdaglar (2022)
   - 期刊：American Economic Review
   - 机制：推断外部性 + 边际信息次模性 → 过度分享
   - 实现：`src/scenarios/scenario_b_too_much_data.py`

3. **场景C：The Economics of Social Data**
   - 作者：Bergemann, Bonatti, Gan (2022)
   - 期刊：RAND Journal of Economics
   - 机制：社会数据外部性 + 搭便车 + 匿名化保护
   - 实现：`src/scenarios/scenario_c_social_data.py`

### 其他相关工作

4. **Ichihashi (2020)**: "Data-enabled Learning, Network Effects, and Competitive Advantage"
   - 机制：网络效应与竞争优势
   - 未来扩展方向（场景D）

5. **LLM in Economics (Horton 2023, Aher et al. 2023)**
   - LLM作为经济学实验对象的探索
   - 本项目：聚焦机制理解而非行为模拟

### 论文文本提取

论文文本已提取并存放在 `papers/` 目录，用于设计提示词时参考原文表述：
```
papers/
├── acemoglu2022_too_much_data.txt         # 场景B
├── bergemann2022_social_data.txt          # 场景C
├── rhodes2019_personalization.txt         # 场景A
└── ichihashi2020_data_learning.txt        # 场景D（未来）
```

## 🎯 项目完成度与未来工作

### 已完成 ✅

#### 核心框架
- [x] **场景A**：理论求解器 + 迭代博弈评估器 + 诊断指标
- [x] **场景B**：理论求解器 + 提示词实验（6版本×12模型）+ 虚拟博弈（gpt-5/r1）+ 敏感度分析（3×3网格）
- [x] **场景C**：理论求解器（个性化补偿m_i）+ 4配置迭代博弈评估（12模型）+ 完整诊断指标
- [x] LLM客户端封装（OpenAI兼容，含超时重试机制）
- [x] 统一诊断指标体系（MAE + 标签 + 收敛 + 集合相似度）

#### 场景B完整评估
- [x] **提示词实验**：6个版本（b.v1-b.v6）× 12模型，分析信息完整度影响
- [x] **虚拟博弈**：gpt-5和deepseek-r1的Fictitious Play学习能力验证
- [x] **敏感度分析**：3×3参数网格（ρ×v），验证机制理解的鲁棒性
- [x] **GPT家族演进分析**：GPT-5 → 5.1 → 5.2 逐代对比
- [x] **可视化**：学术风格对比图、演进趋势图、雷达图

#### 场景C完整评估（2026-02-05）
- [x] **12个模型全量评估**：GPT×4 + DeepSeek×4 + Qwen×4
- [x] **4种配置评估**：A(理论) / B(理性中介×LLM消费者) / C(LLM中介×理性消费者) / D(完全LLM)
- [x] **个性化补偿优化**：m_i为N维向量，混合优化（网格搜索+L-BFGS-B）+ 利润约束R>0
- [x] **14张专业可视化图表**：8基础+6高级（帕累托前沿、效率前沿等）
- [x] **综合分析报告**：13000+字论文级中文分析报告

#### 批量评估与汇总
- [x] 统一入口：`run_evaluation.py`（场景A+B+C）
- [x] 敏感度实验：`run_sensitivity_b.py`
- [x] 提示词实验：`run_prompt_experiments.py`
- [x] 结果对比：`compare_models_sensitivity.py`, `compare_gpt_models_sensitivity.py`
- [x] 场景C可视化：`visualize_scenario_c_results.py`, `visualize_scenario_c_advanced.py`
- [x] 自动生成CSV汇总报告

### 进行中 🚧

- [ ] 场景A、C的敏感度分析（参数网格扫描）
- [ ] 场景C的离散类型优化（K=3 types for m）

### 未来扩展 🔮

#### 新场景
- [ ] **场景D**：Data-enabled Learning（Ichihashi 2020）
  - 机制：网络效应 + 学习外部性
  - 挑战：动态博弈 + 竞争优势演化

#### 评估增强
- [ ] **异质部署**：不同代理使用不同LLM，测试模型协作
- [ ] **长期博弈**：重复博弈、声誉效应、路径依赖
- [ ] **解释题评估**：关键点抽取、因素排序、机制理解深度

#### 工具改进
- [ ] 交互式可视化仪表板（Plotly/Dash）
- [ ] 分布式实验调度（加速大规模评估）
- [ ] 置信区间计算（多次运行统计）

## 🤖 评估的大语言模型

本框架评估了来自**3个模型家族**的**12款**大语言模型（场景C完整评估）：

### GPT家族（OpenAI）— 4个版本
| 编号 | 模型 | config_name | 说明 |
|------|------|-------------|------|
| 1 | GPT-5 | `gpt-5` | 基础版 |
| 2 | GPT-5.1 (2025-11-13) | `gpt-5.1-2025-11-13` | 早期版本 |
| 3 | GPT-5.1 | `gpt-5.1` | 稳定版 |
| 4 | GPT-5.2 | `gpt-5.2` | 最新版本 |

### DeepSeek家族（DeepSeek AI）— 4个版本
| 编号 | 模型 | config_name | 说明 |
|------|------|-------------|------|
| 5 | DeepSeek-v3-0324 | `deepseek-v3-0324` | 早期版本 |
| 6 | DeepSeek-v3.1 | `deepseek-v3.1` | 中间版本 |
| 7 | DeepSeek-v3.2 | `deepseek-v3.2` | 最新版本 |
| 8 | DeepSeek-r1 | `deepseek-r1` | 推理模型 |

### Qwen家族（阿里云）— 4个版本
| 编号 | 模型 | config_name | 说明 |
|------|------|-------------|------|
| 9 | Qwen-Plus | `qwen-plus` | 基础版 |
| 10 | Qwen-Plus (2025-12-01) | `qwen-plus-2025-12-01` | 指定版本 |
| 11 | Qwen3-Max | `qwen3-max` | 最新系列 |
| 12 | Qwen3-Max (2026-01-23) | `qwen3-max-2026-01-23` | 最新版本 |

> 另有 `gpt-4.1-mini` 用于早期场景A/B测试。

**配置文件**：所有模型的API配置在 `configs/model_configs.json` 中，支持OpenAI兼容接口。

---

## ⚠️ 注意事项

### 成本与时间
1. **API成本**：
   - 场景A单模型：约30-100次LLM调用
   - 场景B提示词实验（单模型×6版本）：约150次调用
   - 场景B虚拟博弈（50轮）：约200-400次调用
   - 场景B敏感度分析（3×3网格）：约270次调用/模型
   - 场景C单模型（4配置×20轮）：约300-500次调用
   - 建议使用支持批量定价的API

2. **运行时间**：
   - 场景A单模型：5-10分钟
   - 场景B提示词实验（单模型）：10-20分钟
   - 场景B虚拟博弈（单模型）：20-40分钟
   - 场景B敏感度分析（单模型）：30-60分钟
   - 场景C单模型：15-30分钟
   - 12模型完整评估（B+C）：约8-12小时

### 技术要求
3. **编码问题**（Windows）：
   ```powershell
   # PowerShell
   $env:PYTHONIOENCODING="utf-8"
   
   # CMD
   set PYTHONIOENCODING=utf-8
   ```

4. **API密钥安全**：
   - ⚠️ **不要在公开仓库提交 `configs/model_configs.json` 明文密钥**
   - 建议使用环境变量：`export OPENAI_API_KEY=your_key`
   - 或使用 `.env` 文件并加入 `.gitignore`

5. **内存要求**：
   - 场景A、B：< 2GB
   - 场景B虚拟博弈（50轮）：< 4GB
   - 场景C（12模型）：< 4GB
   - 敏感度分析（保存所有历史）：< 8GB

6. **Python版本**：
   - 推荐：Python 3.10+
   - 必需包：`openai`, `httpx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

### 可复现性
7. **随机种子**：
   - 理论求解器使用固定种子（如 `seed=42`）保证可复现
   - LLM调用使用 `temperature=0.7` 引入适度随机性
   - 虚拟博弈使用 `num_trials=3` 评估稳定性

8. **版本控制**：
   - 模型版本会影响结果（如 GPT-5 vs GPT-5.2）
   - 记录每次实验的模型版本号
   - 提示词版本记录在文件名中（如 `b.v4`）

## 📞 文档与帮助

### 详细文档

- **场景C评估器使用说明**：`docs/README_scenario_c_evaluator.md` ⭐ 推荐优先阅读
- **场景C评估结果分析报告**：`evaluation_results/scenario_c/visualizations/场景C评估结果分析报告.md` ⭐ 12模型完整分析
- **场景C可视化指南**：`evaluation_results/scenario_c/visualizations/VISUALIZATION_GUIDE.md`（英文）
- **评估系统说明**：`docs/README_evaluation.md`
- **场景与GT生成说明**：`docs/README_scenarios.md`
- **虚拟博弈实现指南**：`FICTITIOUS_PLAY_GUIDE.md`
- **LLM日志记录说明**：`LLM_LOGGING_GUIDE.md`

### 设计文档

- **场景设计**：`docs/design/场景.md`
- **场景A对照分析**：`场景A论文对照分析.md`
- **场景B超参数清单**：`场景B超参数清单.md`
- **场景C实现总结**：`场景C实现总结报告.md`
- **场景C个性化补偿优化**：`docs/场景C个性化补偿优化-完整流程.md`
- **场景C评估完整指南**：`docs/场景C评估完整指南.md`
- **场景C敏感度分析方案**：`docs/场景C敏感度分析方案.md`

### 常见问题

**Q1: 如何添加新模型？**
```json
// 在 configs/model_configs.json 中添加
{
  "my-model": {
    "config_name": "my-model",
    "model_name": "my-provider/my-model-name",
    "api_key": "your_api_key",
    "base_url": "https://api.provider.com/v1",
    "generate_args": {
      "temperature": 0.7,
      "max_tokens": 1500
    }
  }
}
```

**Q2: 评估结果保存在哪里？**
- 场景A：`evaluation_results/scenario_a/`
- 场景B提示词实验：`evaluation_results/prompt_experiments_b/`
- 场景B虚拟博弈：`evaluation_results/fp_gpt-5/`, `evaluation_results/fp_deepseek-r1/`
- 场景B敏感度分析：`sensitivity_results/scenario_b/`
- 场景C：`evaluation_results/scenario_c/`（含visualizations子目录）

**Q3: 如何复现论文结果？**
```bash
# 1. 生成所有GT
python scripts/generate_sensitivity_b_gt.py          # 场景B（含敏感度3×3网格）
python -m src.scenarios.generate_scenario_c_gt        # 场景C

# 2. 场景A
python run_evaluation.py --scenarios A --models deepseek-v3.2 gpt-5.1 qwen-plus

# 3. 场景B - 提示词实验（12模型×6版本）
python run_prompt_experiments.py --versions b.v1 b.v2 b.v3 b.v4 b.v5 b.v6 --model <每个模型>
python plot_prompt_comparison_academic.py              # 可视化

# 4. 场景B - 虚拟博弈（表现最差模型）
python run_evaluation.py --scenarios B --models gpt-5 deepseek-r1 --mode fp --rounds 50

# 5. 场景B - 敏感度分析
python run_sensitivity_b.py --models deepseek-v3.2 gpt-5.1 qwen-plus --num-trials 3
python compare_models_sensitivity.py
python compare_gpt_models_sensitivity.py

# 6. 场景C（12模型×4配置）
python run_evaluation.py --scenarios C \
  --models gpt-5.2 gpt-5.1 gpt-5.1-2025-11-13 gpt-5 \
           deepseek-v3.2 deepseek-v3.1 deepseek-v3-0324 deepseek-r1 \
           qwen3-max-2026-01-23 qwen3-max qwen-plus-2025-12-01 qwen-plus
python visualize_scenario_c_results.py                # 基础图表
python visualize_scenario_c_advanced.py               # 高级图表
```

**Q4: 如何只测试一个场景？**
```bash
# 场景A（快速，迭代博弈）
python run_evaluation.py --single --scenarios A --models deepseek-v3.2

# 场景B提示词实验（推荐先跑v4版本）
python run_prompt_experiments.py --versions b.v4 --model deepseek-v3.2 --num-trials 3

# 场景C（4配置迭代博弈，推荐）
python run_evaluation.py --scenarios C --models deepseek-v3.2
```

---

## 📝 如何引用

如果本框架对您的研究有帮助，请引用：

```bibtex
@article{llm-privacy-externality-benchmark,
  title={以机制为导向的隐私外部性大模型Benchmark框架},
  author={[作者姓名]},
  year={2026},
  note={LLM隐私外部性Benchmark系统}
}
```

**论文摘要**：

数据市场中的决策过程通常涉及多主体互动、跨阶段反馈以及复杂的因果结构，因此为评估大语言模型（LLMs）的决策能力提供了天然而严格的测试场景。本文提出一个以机制为导向的隐私外部性大模型 benchmark 框架，用于系统评估 LLM 理解能力与推断能力：围绕三类可计算的外部性机制构建三个完全不同的经济学场景，为每个场景实现可复现的理论求解器作为 ground truth，并将 LLM 嵌入博弈过程，配合统一的诊断指标体系同时度量终局偏差、策略结构一致性与动态收敛过程。我们评估了来自3个模型家族的12款大语言模型，并在部分原本表现较弱的模型上引入多轮虚拟博弈。我们的 benchmark 将"终局正确"与"机制正确"显式分离，为研究者提供一套具备计算理论解支撑的诊断性评测工具。

---

## 🙏 致谢

本项目基于以下经典经济学论文的理论框架：
- Rhodes & Zhou (2019): Personalization and Privacy Choice
- Acemoglu et al. (2022): Too Much Data
- Bergemann & Bonatti (2022): The Economics of Social Data

感谢所有参与测试和提供反馈的研究者。

---

**祝评估顺利！** 🎉

如有任何问题或建议，欢迎提issue或联系项目维护者。
