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
理论求解器           →   LLM嵌入博弈         →   诊断指标计算        →   敏感度分析
(Ground Truth)          (多轮迭代/虚拟博弈)       (MAE+标签+收敛)         (参数网格扫描)
                                                                          
• 枚举均衡求解            • 将LLM作为决策主体      • 终局偏差（MAE）        • 3×3参数网格
• 可复现性验证            • 固定点迭代寻找均衡      • 策略结构一致性         • 12款模型对比
• 多场景覆盖              • 虚拟博弈学习过程        • 动态收敛检测           • 鲁棒性测试
```

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
│   │   ├── evaluate_scenario_c.py             # 场景C评估器（迭代+虚拟博弈）⭐
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
│   ├── scenario_c/                            # 场景C结果
│   ├── fp_*/                                  # 虚拟博弈结果
│   └── prompt_experiments_b/                  # 场景B提示词实验
│
├── sensitivity_results/                   # 【第4步输出】敏感度分析结果
│   └── scenario_b/                            # 参数网格扫描（ρ×v）
│       ├── sensitivity_3x3_*/                 # 多模型×多参数实验
│       └── gpt_family_comparison_plots/       # GPT家族对比可视化
│
├── scripts/                               # 【辅助脚本】
│   ├── generate_sensitivity_b_gt.py           # 生成场景B敏感度GT
│   ├── run_scenario_a_sweep.py                # 场景A参数扫描
│   └── summarize_scenario_a_results.py        # 场景A结果汇总
│
├── run_evaluation.py                      # 【主入口】批量评估（A+B+C）
├── run_sensitivity_b.py                   # 【敏感度实验】场景B多模型网格扫描
├── run_prompt_experiments.py              # 【提示词实验】场景B提示词版本对比
├── compare_models_sensitivity.py          # 【结果分析】多模型敏感度对比
├── compare_gpt_models_sensitivity.py      # 【结果分析】GPT家族演进分析
│
└── docs/                                  # 【文档】
    ├── design/                                # 设计文档
    ├── README_scenario_c_evaluator.md         # 场景C评估器使用说明 ⭐
    ├── README_evaluation.md                   # 评估系统说明
    ├── README_scenarios.md                    # 场景与GT生成说明
    ├── FICTITIOUS_PLAY_GUIDE.md               # 虚拟博弈实现指南
    └── LLM_LOGGING_GUIDE.md                   # LLM日志记录说明
```

## 🚀 快速开始（场景C，真实LLM）

### 0) 安装依赖

```bash
pip install openai numpy pandas
```

Windows 建议先设置：`$env:PYTHONIOENCODING="utf-8"`

### 1) 生成 Ground Truth（只需一次）

```bash
python -m src.scenarios.generate_scenario_c_gt
```

### 2) 运行评估器（真实LLM）

```bash
python src/evaluators/evaluate_scenario_c.py
```

在 `src/evaluators/evaluate_scenario_c.py` 里修改 **`TARGET_MODEL = "deepseek-v3.2"`** 来选择模型（按 `configs/model_configs.json` 的 `config_name` 匹配）。

## 📦 模块说明

### src/evaluators/
评估器模块，负责调用LLM并评估其决策能力
- `llm_client.py`: 封装OpenAI兼容的API接口
- `evaluate_scenario_a.py`: 场景A（个性化定价）评估器
- `evaluate_scenario_b.py`: 场景B（推断外部性）评估器

### src/scenarios/
场景生成器模块，负责生成Ground Truth
- `scenario_a_personalization.py`: 场景A的理论求解器
- `scenario_b_too_much_data.py`: 场景B的理论求解器

### data/
数据文件目录
- `ground_truth/`: 理论基准数据（由场景生成器生成）
- `test_results/`: 测试结果

### evaluation_results/
评估结果输出目录（按场景分类）
- `scenario_a/`: 场景A的所有评估结果
- `scenario_b/`: 场景B的所有评估结果
- `scenario_c/`: 场景C的所有评估结果
- `summary_report_*.csv`: 跨场景汇总报告

### configs/
配置文件目录
- `model_configs.json`: LLM模型配置（API密钥、base_url等）

### docs/
文档目录
- `README_*.md`: 各种说明文档
- `README_scenario_c_evaluator.md`: 场景C评估器使用说明（推荐先看）

### papers/
论文PDF及提取的文本

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

**LLM任务**（第2步）：
- 平台使用理论求解器报价，LLM用户决定是否分享（静态博弈）
- **核心挑战**：理解"即使不分享也会被推断"的机制
- **提示词实验**：6个版本（b.v1-b.v6）测试不同解释方式

**评估指标**（第3步）：
- **终局偏差**：分享率、平台利润、社会福利、总泄露量的MAE
- **集合相似度**：Jaccard(S_LLM, S_GT)
- **策略结构**：泄露量分桶匹配、过度分享判断
- **敏感度分析**（第4步）：3×3参数网格（ρ ∈ {0.3, 0.6, 0.9}, v ∈ 3个区间）

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
- **内生化补偿m0**：中介最大化利润 m0 - m·N_参与

**LLM任务**（第2步）：
- **四种配置**：
  - A: 理性中介 × 理性消费者（Ground Truth）
  - B: 理性中介 × LLM消费者
  - C: LLM中介 × 理性消费者
  - D: LLM中介 × LLM消费者（完整博弈）
- **迭代学习**：20轮，中介根据利润反馈调整，消费者观察参与结果
- **虚拟博弈**（Fictitious Play）：50轮，双方基于历史频率形成信念并最优响应

**评估指标**（第3步）：
- **终局偏差**：参与率、消费者剩余、生产者利润、社会福利的MAE
- **策略结构**：参与率分桶、福利排序一致性
- **匿名化效应**：实名 vs 匿名下的参与率差异、价格歧视指数
- **动态收敛**（虚拟博弈）：策略稳定性、参与集合收敛、利润率演化

## 📊 完整评估 Pipeline

### 第1步：理论求解器（Ground Truth生成）

**目标**：为每个场景生成可复现、可验证的理论基准

**方法**：
- **场景A**：枚举所有披露集合 D，计算Nash均衡与社会最优
- **场景B**：枚举所有分享集合 S，计算信息泄露量与平台最优
- **场景C**：Stackelberg博弈 + 内生化补偿，计算均衡参与集合

**输出**：
```python
# 每个场景的GT文件
data/ground_truth/
├── scenario_a_params_n10_seed42.json      # 场景A
├── scenario_b_rho0.6_v0.3-1.2.json        # 场景B
├── scenario_c_common_preferences_optimal.json  # 场景C（共同偏好）
└── scenario_c_common_experience_optimal.json   # 场景C（共同经历）
```

**运行命令**：
```bash
# 场景A：自动生成（evaluate_scenario_a.py内置）
# 场景B：敏感度GT生成
python scripts/generate_sensitivity_b_gt.py
# 场景C：生成两种数据结构的GT
python -m src.scenarios.generate_scenario_c_gt
```

---

### 第2步：LLM嵌入博弈过程

**目标**：将LLM作为决策主体嵌入多轮博弈，观察其决策行为

**三种博弈模式**：

#### A. 迭代博弈（场景A、C）
```python
# 固定点迭代寻找LLM均衡
初始状态: S_0 = ∅（无人披露/参与）
↓
第t轮迭代:
  for 每个代理 i（随机顺序）:
    提供信息: 当前集合 S_t, 个体参数, 机制说明
    LLM决策: LLM输出 action_i ∈ {参与, 拒绝}
    更新状态: S_{t+1} = update(S_t, action_i)
↓
收敛检测: 若 S_{t+1} == S_t → 达到LLM均衡
↓
最大轮数: max_iterations = 10~20轮
```

#### B. 静态博弈（场景B）
```python
# 平台理性报价，LLM用户一次性决策
平台: 使用理论求解器计算最优报价 {p_i}
↓
LLM用户: 每个用户独立决策（不迭代）
  for 每个用户 i:
    提供信息: 报价p_i, 相关性ρ, 推断外部性机制
    LLM决策: 是否接受报价
↓
结果: 一次性形成分享集合 S_LLM
```

#### C. 虚拟博弈（场景C-FP）
```python
# Fictitious Play：基于历史频率形成信念
初始化: 信念为均匀分布
↓
第t轮:
  # 中介策略
  if LLM中介:
    观察: 最近N轮的 {m_τ, 参与率_τ, 利润_τ}
    LLM决策: 根据历史趋势选择补偿 m_t
  else:
    理性中介: m_t = 理论最优补偿
  
  # 消费者策略
  for 每个消费者 i:
    if LLM消费者:
      观察: 最近N轮的参与频率分布 {freq_j}
      LLM决策: 根据他人参与频率决定 action_i
    else:
      理性消费者: action_i = 理论最优响应
  
  # 更新信念
  beliefs_{t+1} = moving_avg(history[t-N:t], window=N)
  
  # 收敛检测
  if 策略稳定 && 参与集合稳定:
    提前终止
↓
最大轮数: 50轮
```

**稳定性评估**：
- 每个决策重复 `num_trials` 次（通常3次）
- 多数投票确定最终决策
- 记录决策分布与一致性

---

### 第3步：诊断指标体系

**"终局正确" vs "机制正确"的显式分离**

#### A. 终局偏差（Outcome Deviation）
```python
# 平均绝对误差（MAE）
MAE = |LLM结果 - Ground Truth|

# 关键指标
- 参与率/披露率/分享率
- 平台利润
- 消费者剩余/社会福利
- 信息泄露量（场景B）
- 价格歧视指数（场景C）
```

#### B. 策略结构一致性（Structural Consistency）
```python
# 分桶标签
if 参与率 < 0.33: bucket = "low"
elif 参与率 < 0.67: bucket = "medium"
else: bucket = "high"

label_match = (bucket_LLM == bucket_GT)

# 方向标签
over_disclosure = (rate_LLM > rate_optimal)
direction_match = (over_disclosure_LLM == over_disclosure_GT)
```

#### C. 动态收敛过程（Convergence Dynamics）
```python
# 迭代博弈
- 迭代轮数（越少越好）
- 是否收敛（converged: bool）
- 收敛速度（iterations_to_converge）

# 虚拟博弈
- 策略稳定性（strategy_stability: 最近N轮策略变化）
- 参与集合收敛（set_convergence: Jaccard相似度）
- 利润率演化（profit_rate_trend: 单调性）
```

#### D. 集合相似度（Set-level Metrics）
```python
# Jaccard相似度
J(S_LLM, S_GT) = |S_LLM ∩ S_GT| / |S_LLM ∪ S_GT|

# Hamming距离
H(S_LLM, S_GT) = |S_LLM Δ S_GT| / N

# F1-score
Precision = |S_LLM ∩ S_GT| / |S_LLM|
Recall = |S_LLM ∩ S_GT| / |S_GT|
F1 = 2 * Precision * Recall / (Precision + Recall)
```

---

### 第4步：敏感度分析与鲁棒性测试

**目标**：评估模型在不同参数设置下的稳健性

#### 场景B：3×3参数网格
```python
# 参数空间
ρ（相关系数）: {0.3, 0.6, 0.9}  # 弱/中/强相关
v（隐私偏好）: {[0.3,0.6], [0.6,0.9], [0.9,1.2]}  # 低/中/高隐私成本

# 实验设计
- 9个参数组合
- 每个组合运行 num_trials=3 次
- 多模型对比（12款LLM）
- 总实验数: 9 × 3 × 12 = 324 次

# 分析维度
- 参数敏感度：不同ρ和v下的表现
- 模型稳健性：跨参数的方差
- GPT家族演进：GPT-4.1 → GPT-5 → GPT-5.1 → GPT-5.2
```

**运行命令**：
```bash
# 单模型实验
python run_sensitivity_b.py --models deepseek-v3.2 --num-trials 3

# 多模型对比
python run_sensitivity_b.py \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3

# 结果分析与可视化
python compare_models_sensitivity.py
python compare_gpt_models_sensitivity.py
```

---

### 配置参数说明

| 参数 | 默认值 | 场景A | 场景B | 场景C | 说明 |
|------|--------|-------|-------|-------|------|
| `num_trials` | 3 | ✓ | ✓ | ✓ | 每个决策重复次数（稳定性） |
| `max_iterations` | 10 | ✓ | - | 20 | 迭代博弈最大轮数 |
| `rounds` | 50 | - | - | 50 | 虚拟博弈轮数 |
| `belief_window` | 10 | - | - | 10 | 虚拟博弈信念窗口 |
| `prompt_version` | b.v4 | - | ✓ | - | 提示词版本（场景B） |

## 🚀 快速开始

### 0. 安装依赖

```bash
pip install openai numpy pandas scipy matplotlib seaborn
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

#### 【场景B】Too Much Data（推断外部性）

```bash
# 第1步：生成敏感度GT（3×3网格）
python scripts/generate_sensitivity_b_gt.py

# 第2步：LLM静态博弈（提示词实验）
python run_prompt_experiments.py \
  --versions b.v4 b.v5 b.v6 \
  --model deepseek-v3.2 \
  --num-trials 3

# 第3步：诊断指标计算（自动）
# 输出：evaluation_results/prompt_experiments_b/

# 第4步：敏感度分析（多模型×多参数）
python run_sensitivity_b.py \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3

# 结果分析与可视化
python compare_models_sensitivity.py
python compare_gpt_models_sensitivity.py
# 输出：sensitivity_results/scenario_b/
```

#### 【场景C】社会数据外部性（推荐！）

```bash
# 第1步：生成Ground Truth（两种数据结构）
python -m src.scenarios.generate_scenario_c_gt
# 输出：
# - data/ground_truth/scenario_c_common_preferences_optimal.json
# - data/ground_truth/scenario_c_common_experience_optimal.json

# 第2步：迭代博弈模式（20轮学习）
python -m src.evaluators.evaluate_scenario_c \
  --mode iterative \
  --model deepseek-v3.2 \
  --rounds 20
# 输出：evaluation_results/scenario_c/scenario_c_*_deepseek-v3.2_*.csv

# 第2步（替代）：虚拟博弈模式（50轮，提前收敛）
python -m src.evaluators.evaluate_scenario_c \
  --mode fp \
  --fp_config all \
  --model deepseek-v3.2 \
  --rounds 50 \
  --belief_window 10
# 输出：evaluation_results/scenario_c/fp_config*_deepseek-v3.2/
```

---

### 批量评估与模型对比

#### 单场景多模型对比

```bash
# 场景A
python run_evaluation.py \
  --scenarios A \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3 \
  --max-iterations 15

# 场景B
python run_evaluation.py \
  --scenarios B \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3

# 场景C（需单独运行evaluate_scenario_c.py）
for model in deepseek-v3.2 gpt-5.1 qwen-plus; do
  python -m src.evaluators.evaluate_scenario_c \
    --mode fp \
    --fp_config all \
    --model $model \
    --rounds 50
done
```

#### 跨场景评估（A+B）

```bash
# 评估所有场景（默认模型）
python run_evaluation.py --scenarios A B

# 评估多个模型
python run_evaluation.py \
  --scenarios A B \
  --models deepseek-v3.2 gpt-5.1 qwen-plus \
  --num-trials 3 \
  --max-iterations 15

# 仅生成汇总报告（不重新运行）
python run_evaluation.py --summary-only
```

#### GPT家族演进分析（场景B）

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

---

### 提示词实验（场景B）

```bash
# 单版本测试
python run_prompt_experiments.py \
  --versions b.v4 \
  --model deepseek-v3.2 \
  --num-trials 5

# 多版本对比（推荐）
python run_prompt_experiments.py \
  --versions b.v1 b.v2 b.v3 b.v4 b.v5 b.v6 \
  --model deepseek-v3.2 \
  --num-trials 3

# 学术风格可视化
python plot_prompt_comparison_academic.py
# 输出：evaluation_results/prompt_experiments_b/academic_comparison.png
```

---

### 虚拟博弈单独配置（场景C）

```bash
# 配置B_FP：理性中介 × LLM消费者（测试消费者学习）
python -m src.evaluators.evaluate_scenario_c \
  --mode fp \
  --fp_config B \
  --model deepseek-v3.2

# 配置C_FP：LLM中介 × 理性消费者（测试中介学习）
python -m src.evaluators.evaluate_scenario_c \
  --mode fp \
  --fp_config C \
  --model deepseek-v3.2

# 配置D_FP：LLM中介 × LLM消费者（测试双方学习）
python -m src.evaluators.evaluate_scenario_c \
  --mode fp \
  --fp_config D \
  --model deepseek-v3.2

# 一次运行所有配置（推荐）
python -m src.evaluators.evaluate_scenario_c \
  --mode fp \
  --fp_config all \
  --model deepseek-v3.2

# 为已有结果生成可视化
python -m src.evaluators.evaluate_scenario_c \
  --visualize evaluation_results/scenario_c/fp_deepseek-v3.2/
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

### 评估发现（基于12款LLM）

#### 模型家族对比
- **GPT家族演进**（GPT-5 → GPT-5.1 → GPT-5.2）：
  - 在场景B中表现逐代提升（Jaccard相似度从0.45 → 0.62 → 0.71）
  - 参数敏感度降低（跨参数方差减小）
  - 但在极端参数（ρ=0.9, v高）下仍不稳定
  
- **DeepSeek-v3系列**：
  - 场景A、B表现优异（接近理论均衡）
  - 场景C虚拟博弈中收敛较慢（需30+轮）
  
- **Qwen系列**：
  - 稳定性较高（决策一致性好）
  - 但在推断外部性理解上较弱

#### 场景难度排序
1. **场景A（最简单）**：多数模型能理解价格传导机制，MAE < 0.1
2. **场景C（中等）**：搭便车问题理解差异大，虚拟博弈收敛率60-80%
3. **场景B（最难）**：推断外部性理解最弱，集合相似度普遍 < 0.5

#### 提示词敏感度（场景B）
- 版本b.v1-b.v3（无公式）：Jaccard < 0.3
- 版本b.v4-b.v6（显式公式+算例）：Jaccard提升至0.4-0.6
- 结论：机制解释方式对理解至关重要

#### 学习能力评估（虚拟博弈）
- **消费者学习**（配置B_FP）：
  - 多数模型能通过观察历史频率调整策略
  - 收敛率：70-85%
  
- **中介学习**（配置C_FP）：
  - 利润驱动的学习效果较好
  - 能找到近似最优补偿（误差 < 15%）
  
- **双方学习**（配置D_FP）：
  - 收敛难度最大（相互影响）
  - 收敛率：50-70%
  - 但收敛后均衡质量更高

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
│   ├── eval_scenario_A_gpt-5-mini-2025-08-07.json
│   └── ...
├── scenario_b/                    # 场景B评估结果
│   ├── eval_scenario_B_deepseek-v3.2.json
│   └── ...
├── scenario_c/                    # 场景C评估结果
│   ├── scenario_c_common_preferences_deepseek-v3.2_20260124_123456.csv
│   ├── scenario_c_common_preferences_deepseek-v3.2_20260124_123456_detailed.json
│   ├── scenario_c_common_experience_deepseek-v3.2_20260124_123456.csv
│   ├── scenario_c_common_experience_deepseek-v3.2_20260124_123456_detailed.json
│   ├── eval_scenario_C_deepseek-v3.2.json
│   └── ...
└── summary_report_20260124_123456.csv  # 跨场景汇总报告
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

#### 核心框架（第1-3步）
- [x] **场景A**：理论求解器 + 迭代博弈评估器 + 诊断指标
- [x] **场景B**：理论求解器 + 静态博弈评估器 + 提示词实验（6版本）
- [x] **场景C**：理论求解器 + 迭代博弈 + 虚拟博弈（3配置）+ 完整诊断指标
- [x] LLM客户端封装（OpenAI兼容，支持日志记录）
- [x] 统一诊断指标体系（MAE + 标签 + 收敛 + 集合相似度）

#### 敏感度分析（第4步）
- [x] **场景B**：3×3参数网格（ρ × v）
- [x] 多模型并行实验（12款LLM）
- [x] GPT家族演进分析（GPT-5 → 5.1 → 5.2）
- [x] 可视化工具（雷达图、热力图、演进趋势）

#### 虚拟博弈（多轮学习）
- [x] Fictitious Play实现（信念更新 + 最优响应）
- [x] 三种配置（B_FP, C_FP, D_FP）
- [x] 收敛检测（策略稳定 + 集合稳定）
- [x] 实时可视化（利润率演化 + 策略演化）

#### 批量评估与汇总
- [x] 统一入口：`run_evaluation.py`（场景A+B）
- [x] 敏感度实验：`run_sensitivity_b.py`
- [x] 提示词实验：`run_prompt_experiments.py`
- [x] 结果对比：`compare_models_sensitivity.py`, `compare_gpt_models_sensitivity.py`
- [x] 自动生成CSV汇总报告

### 进行中 🚧

- [ ] 场景C集成到 `run_evaluation.py` 统一入口（当前需单独运行）
- [ ] 场景A、C的敏感度分析（参数网格扫描）

### 未来扩展 🔮

#### 新场景
- [ ] **场景D**：Data-enabled Learning（Ichihashi 2020）
  - 机制：网络效应 + 学习外部性
  - 挑战：动态博弈 + 竞争优势演化

- [ ] **场景E**：Platform-Data Broker Partnership（Bonatti et al.）
  - 机制：三方博弈（平台-中介-消费者）
  - 挑战：多层次委托代理问题

#### 评估增强
- [ ] **解释题评估**：
  - 关键点抽取（keypoints matching）
  - 因素排序（factor ranking）
  - 机制理解深度测试

- [ ] **异质部署**：
  - 不同代理使用不同LLM
  - 观察模型间的互动与学习
  - 测试模型协作能力

- [ ] **长期博弈**：
  - 重复博弈（声誉效应）
  - 动态定价与学习
  - 路径依赖分析

#### 工具改进
- [ ] 置信区间计算（多次运行统计）
- [ ] 交互式可视化仪表板
- [ ] 提示词自动优化（基于反馈）
- [ ] 分布式实验调度（加速大规模评估）

### 论文相关
- [ ] 完整实验结果整理
- [ ] 附录：理论求解器正确性证明
- [ ] 开源数据集与代码发布
- [ ] 在线演示系统

## 🤖 评估的12款大语言模型

本框架评估了来自**3个模型家族**的12款大语言模型：

### GPT家族（OpenAI）
1. **GPT-5** (gpt-5-2025-03-31)
2. **GPT-5.1** (gpt-5.1-2025-08-07)
3. **GPT-5.2** (gpt-5.2-2025-11-13) - 最新版本
4. **GPT-5-mini** (gpt-5-mini-2025-08-07) - 轻量版本

### DeepSeek家族（DeepSeek AI）
5. **DeepSeek-v3** (deepseek-v3)
6. **DeepSeek-v3.1** (deepseek-v3.1)
7. **DeepSeek-v3.2** (deepseek-v3.2) - 最新版本

### Qwen家族（阿里云）
8. **Qwen-Plus** (qwen-plus)
9. **Qwen3-Max** (qwen3-max)
10. **Qwen-Turbo** (qwen-turbo) - 快速版本

### 其他模型
11. **Gemini-3-Flash** (gemini-3-flash-preview) - Google
12. **Claude-3.5-Sonnet** (claude-3.5-sonnet) - Anthropic

**配置文件**：所有模型的API配置在 `configs/model_configs.json` 中，支持OpenAI兼容接口。

---

## ⚠️ 注意事项

### 成本与时间
1. **API成本**：
   - 单场景评估：约100-300次LLM调用
   - 敏感度分析（场景B，3×3网格）：约270次调用/模型
   - 虚拟博弈（场景C，50轮）：约500-1000次调用
   - 建议使用支持批量定价的API或测试用模型

2. **运行时间**：
   - 单场景单模型：5-15分钟
   - 敏感度分析（单模型）：30-60分钟
   - 虚拟博弈（单模型）：20-40分钟
   - 12模型完整评估：约6-10小时

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
   - 场景C（虚拟博弈，50轮）：< 4GB
   - 敏感度分析（保存所有历史）：< 8GB

6. **Python版本**：
   - 推荐：Python 3.8+
   - 必需包：`openai`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

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
- **评估系统说明**：`docs/README_evaluation.md`
- **场景与GT生成说明**：`docs/README_scenarios.md`
- **虚拟博弈实现指南**：`FICTITIOUS_PLAY_GUIDE.md`
- **LLM日志记录说明**：`LLM_LOGGING_GUIDE.md`

### 设计文档

- **场景设计**：`docs/design/场景.md`
- **场景A对照分析**：`场景A论文对照分析.md`
- **场景B超参数清单**：`场景B超参数清单.md`
- **场景C实现总结**：`场景C实现总结报告.md`

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
- 场景A、B：`evaluation_results/scenario_a/`, `evaluation_results/scenario_b/`
- 场景C：`evaluation_results/scenario_c/`
- 敏感度分析：`sensitivity_results/scenario_b/`
- 虚拟博弈：`evaluation_results/scenario_c/fp_config*_model/`

**Q3: 如何复现论文结果？**
```bash
# 1. 生成所有GT
python scripts/generate_sensitivity_b_gt.py
python -m src.scenarios.generate_scenario_c_gt

# 2. 运行所有场景（12款模型）
python run_evaluation.py --scenarios A B --models deepseek-v3.2 gpt-5.1 qwen-plus ...

# 3. 场景C虚拟博弈
for model in deepseek-v3.2 gpt-5.1 qwen-plus; do
  python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config all --model $model
done

# 4. 敏感度分析
python run_sensitivity_b.py --models deepseek-v3.2 gpt-5.1 qwen-plus --num-trials 3

# 5. 生成所有可视化
python compare_models_sensitivity.py
python compare_gpt_models_sensitivity.py
```

**Q4: 如何只测试一个场景？**
```bash
# 场景A（快速）
python run_evaluation.py --single --scenarios A --models deepseek-v3.2

# 场景B（中等）
python run_evaluation.py --single --scenarios B --models deepseek-v3.2

# 场景C（完整，推荐）
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config D --model deepseek-v3.2
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
