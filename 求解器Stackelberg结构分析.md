# 场景C求解器完整结构梳理

**文件**: `src/scenarios/scenario_c_social_data.py` (2717行)  
**作用**: 场景C的核心理论求解器，实现论文"The Economics of Social Data"的完整三层Stackelberg博弈框架  
**日期**: 2026-01-18

---

## 📋 目录

1. [整体架构](#整体架构)
2. [数据结构定义](#数据结构定义)
3. [核心功能模块](#核心功能模块)
4. [三层博弈框架](#三层博弈框架)
5. [函数索引](#函数索引)
6. [代码流程图](#代码流程图)

---

## 🏗️ 整体架构

### 文件组织结构

```
scenario_c_social_data.py (2717行)
│
├── 📦 数据结构定义 (第264-464行)
│   ├── ScenarioCParams: 参数配置类
│   ├── ConsumerData: 消费者数据
│   └── MarketOutcome: 市场结果
│
├── 🎲 数据生成 (第466-507行)
│   └── generate_consumer_data(): 生成(w, s)
│
├── 🧮 贝叶斯后验估计 (第509-654行)
│   ├── _compute_ce_posterior_approx(): CE后验近似
│   └── compute_posterior_mean_consumer(): 消费者后验
│
├── 💰 生产者定价 (第656-964行)
│   ├── compute_optimal_price_personalized(): 个性化定价
│   ├── compute_optimal_price_uniform(): 统一定价（数值优化）
│   ├── compute_optimal_price_uniform_piecewise(): 统一定价（分段枚举）
│   └── compute_optimal_price_uniform_efficient_DEPRECATED(): 已废弃
│
├── 🔍 生产者后验估计 (第966-1112行)
│   └── compute_producer_posterior(): 生产者信息集
│
├── 🎯 市场均衡模拟 (第1114-1521行)
│   ├── simulate_market_outcome(): 完整市场均衡
│   └── compute_gini(): 基尼系数
│
├── 👥 消费者参与决策 (第1523-1857行)
│   ├── compute_expected_utility_ex_ante(): Ex Ante期望效用
│   ├── compute_expected_utility_given_participation(): Ex Post期望效用
│   ├── compute_rational_participation_rate_ex_ante(): Ex Ante固定点
│   ├── compute_rational_participation_rate_ex_post(): Ex Post固定点
│   ├── compute_rational_participation_rate(): 统一接口
│   └── generate_participation_from_tau(): 基于τ_i生成参与
│
├── 📊 Ground Truth生成 (第1910-2136行)
│   └── generate_ground_truth(): 主生成函数
│
└── 🎲 中介优化（Stackelberg外层）(第2164-2717行)
    ├── simulate_market_outcome_no_data(): 无数据baseline
    ├── evaluate_intermediary_strategy(): 评估单个策略
    ├── optimize_intermediary_policy(): 求解最优策略
    ├── verify_proposition_2(): 验证论文命题
    ├── analyze_optimal_compensation_curve(): 分析补偿曲线
    └── export_optimization_results(): 导出结果
```

---

## 📦 数据结构定义

### 1. `ScenarioCParams` (第264-417行)

**作用**: 场景C的完整参数配置

**参数分类**:

```python
@dataclass
class ScenarioCParams:
    # 基础参数
    N: int                    # 消费者数量
    data_structure: str       # "common_preferences" or "common_experience"
    anonymization: str        # "identified" or "anonymized"
    
    # 数据生成参数（对应论文Section 3）
    mu_theta: float           # 先验均值
    sigma_theta: float        # 先验标准差
    sigma: float              # 噪声水平
    
    # 支付参数（对应论文Section 5）
    m: float                  # 中介→消费者补偿
    m_0: float = 0.0          # 生产者→中介支付
    c: float = 0.0            # 边际成本
    
    # 异质性参数（我们的扩展）
    tau_mean: float = 0.5     # 隐私成本均值
    tau_std: float = 0.3      # 隐私成本标准差
    tau_dist: str = "none"    # "normal", "uniform", or "none"
    
    # 时序模式（学术关键）
    participation_timing: str = "ex_ante"  # "ex_ante" or "ex_post"
    
    # 算法参数
    posterior_method: str = "approx"  # "exact" or "approx"
    seed: int = 42
```

**对应论文**: Section 2-4的模型设定

---

### 2. `ConsumerData` (第419-427行)

**作用**: 存储生成的消费者数据

```python
@dataclass
class ConsumerData:
    w: np.ndarray           # (N,) 真实支付意愿
    s: np.ndarray           # (N,) 观测信号
    e: np.ndarray           # (N,) 噪声成分
    theta: Optional[float]  # 共同偏好（仅CP）
    epsilon: Optional[float]  # 共同噪声（仅CE）
```

---

### 3. `MarketOutcome` (第429-464行)

**作用**: 完整的市场结果和福利指标

```python
@dataclass
class MarketOutcome:
    # 参与情况
    participation: np.ndarray
    participation_rate: float
    num_participants: int
    
    # 价格与数量
    prices: np.ndarray
    quantities: np.ndarray
    
    # 后验估计
    mu_consumers: np.ndarray
    mu_producer: np.ndarray
    
    # 福利指标
    utilities: np.ndarray
    consumer_surplus: float
    producer_profit: float
    intermediary_profit: float
    social_welfare: float
    
    # 学习质量
    learning_quality_participants: float
    learning_quality_rejecters: float
    
    # 不平等指标
    gini_coefficient: float
    acceptor_avg_utility: float
    rejecter_avg_utility: float
    
    # 价格歧视指标
    price_variance: float
    price_discrimination_index: float
```

---

## 🧩 核心功能模块

### 模块1: 数据生成 (第466-507行)

#### `generate_consumer_data(params) -> ConsumerData`

**作用**: 根据数据结构生成真实偏好和信号

**Common Preferences** (论文式3.1):
```python
θ ~ N(μ_θ, σ_θ²)
w_i = θ  for all i
e_i ~ N(0, 1) i.i.d.
s_i = θ + σ·e_i
```

**Common Experience** (论文式3.2):
```python
w_i ~ N(μ_θ, σ_θ²) i.i.d.
ε ~ N(0, 1)
e_i = ε  for all i
s_i = w_i + σ·ε
```

**对应论文**: Section 3.1-3.2

---

### 模块2: 贝叶斯后验估计 (第509-654行)

#### 核心函数

**1. `compute_posterior_mean_consumer(s_i, X, params) -> μ_i`**

**作用**: 计算消费者i的后验期望 E[w_i | s_i, X]

**关键特性**:
- ✅ **P0-1修正**: 必须包含私人信号s_i（论文信息集I_i={s_i}∪X）
- ✅ 避免double counting（如果s_i在X中）

**Common Preferences实现**:
```python
# 精度（precision = 1/variance）
τ_0 = 1 / σ_θ²     # 先验精度
τ_s = 1 / σ²       # 信号精度

# 后验精度 = 先验 + 自己信号 + 他人信号
τ_post = τ_0 + τ_s + len(X_others) * τ_s

# 后验均值（加权平均）
μ_i = (τ_0·μ_θ + τ_s·s_i + τ_s·Σ(X_others)) / τ_post
```

**对应论文**: Section 3.3, 式(3.3)

---

**2. `_compute_ce_posterior_approx(s_i, X, params) -> μ_i`**

**作用**: Common Experience的近似后验估计

**步骤**:
1. 估计共同噪声: `ε̂ ≈ f(mean(X) - μ_θ)`
2. 过滤噪声: `ŝ_i = s_i - σ·ε̂`
3. 结合先验: `μ_i = g(ŝ_i, μ_θ)`

**对应论文**: 论文附录A

---

### 模块3: 生产者定价 (第656-964行)

#### 函数1: 个性化定价

**`compute_optimal_price_personalized(μ_i, c) -> p_i*`**

**理论**: 线性-二次模型下的闭式解

```python
需求函数: q_i(p) = max(μ_i - p, 0)
利润函数: π_i(p) = (p - c) · q_i
一阶条件: μ_i - 2p + c = 0
最优价格: p_i* = (μ_i + c) / 2
```

**对应论文**: Section 2.2, 式(2.3)

---

#### 函数2: 统一定价

**`compute_optimal_price_uniform(μ_list, c) -> (p*, π*)`**

**方法**: 数值优化（推荐）

```python
目标函数: Π(p) = Σ_i (p - c) · max(μ_i - p, 0)
优化方法: scipy.optimize.minimize_scalar
搜索区间: [c, max(μ)]
```

**为什么不能用 μ_i/2 候选集？**
- ❌ 错误: 混淆了个性化定价和统一定价
- ✅ 正确: 统一定价是N消费者耦合优化，无简单闭式解

**对应论文**: Section 4, 匿名化下的定价

---

**`compute_optimal_price_uniform_piecewise(μ_list, c) -> p*`**

**方法**: 分段枚举（高效且精确）

**核心思想**:
- 利润函数Π(p)是分段线性的
- 最优价格必在某个分段的内点或边界
- 候选价格: `p_k = (μ̄_{1:k} + c) / 2` for k=1,...,N
- 复杂度: O(N log N)

---

### 模块4: 生产者后验估计 (第966-1112行)

#### `compute_producer_posterior(data, participation, X, params) -> μ_producer`

**作用**: 计算生产者对每个消费者的后验期望（匿名化机制的核心！）

**实名（Identified）**:
```python
生产者信息集: Y_0 = {(i, s_i) : i ∈ participants}

对参与者i:
  μ_producer[i] = E[w_i | s_i, X]  # 个体后验
  
对拒绝者j:
  # ⚠️ P0-2修正: 不能固定为先验！
  Common Preferences: μ_producer[j] = E[θ | X]
  Common Experience: μ_producer[j] = E[w | X] (利用ε估计)
```

**匿名（Anonymized）**:
```python
生产者信息集: Y_0 = {s_i : i ∈ participants}（无身份）

对所有人:
  # ⚠️ P0-3修正: 仍可学习！
  Common Preferences: μ_producer[:] = E[θ | X]
  Common Experience: μ_producer[:] = E[w | X] (利用ε估计)
```

**关键区别**:
- 实名: μ_producer异质 → 可个性化定价
- 匿名: μ_producer同质 → 必须统一定价

**对应论文**: Section 4, 式(4.1)-(4.2), Proposition 2核心机制

---

### 模块5: 市场均衡模拟 (第1114-1521行)

#### `simulate_market_outcome(data, participation, params) -> MarketOutcome`

**作用**: 给定参与决策，模拟完整市场均衡

**步骤流程**:

```
1️⃣ 数据收集与匿名化处理
   - 收集参与者信号 → X
   - 如果anonymized: shuffle(X)（破坏身份映射）

2️⃣ 消费者后验估计（贝叶斯学习）
   - 消费者i信息集: I_i = {s_i, X}
   - μ_consumers[i] = E[w_i | s_i, X]
   - 关键: 拒绝者也能学习（搭便车）

3️⃣ 生产者后验估计（匿名化关键）
   - μ_producer = compute_producer_posterior()
   - 实名 vs 匿名的核心区别

4️⃣ 生产者定价
   - Identified: p_i = (μ_producer[i] + c) / 2
   - Anonymized: p = uniform_price(μ_producer)

5️⃣ 消费者购买决策
   - q_i = max(μ_consumers[i] - p_i, 0)

6️⃣ 效用实现（用真实w_i结算）
   - u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
   - 参与者: u_i += m（补偿）

7️⃣ 福利指标计算
   - CS = Σu_i
   - PS = Σ(p_i - c)·q_i
   - IS = m_0 - m·N_participants
   - SW = CS + PS + IS

8️⃣ 学习质量与不平等指标
   - 学习误差: |μ_i - w_i|
   - Gini系数
   - 价格歧视指数: max(p) - min(p)
```

**对应论文**: Section 2-4的完整博弈序列（论文Figure 1）

---

### 模块6: 消费者参与决策 (第1523-1857行)

#### 核心权衡（论文式5.1）:

```
ΔU_i = E[u_i | 参与, r] - E[u_i | 拒绝, r] + m - τ_i

消费者参与 ⟺ ΔU_i ≥ 0
```

---

#### 函数1: Ex Ante参与（学术正确）

**`compute_rational_participation_rate_ex_ante(params) -> (r*, history, ΔU)`**

**时序** (对应论文Section 5.1):
1. 中介发布合约(m, 匿名化)
2. **消费者在不知道(w, s)实现时决策** ← Ex Ante
3. 信号实现，数据流动
4. 生产者定价，消费者购买

**期望效用计算** (两层Monte Carlo):
```python
外层: 遍历世界状态(w, s)
内层: 遍历参与者集合

E[u_i | a_i, r] = E_{w,s} E_{a_{-i}|r} [u_i(w, s, a, 信息流)]
```

**固定点方程** (有异质性):
```python
r* = P(τ_i ≤ ΔU(r*)) = F_τ(ΔU(r*))

其中F_τ是隐私成本的CDF:
- tau_dist="normal": Φ((ΔU - μ_τ) / σ_τ)
- tau_dist="uniform": (ΔU - a) / (b - a)
- tau_dist="none": 1 if ΔU>0 else 0
```

**对应论文**: Section 5.1, Ex Ante合约时序

---

#### 函数2: Ex Post参与（鲁棒性）

**`compute_rational_participation_rate_ex_post(data, params) -> (r*, history)`**

**时序**:
1. (w, s)实现
2. **消费者观察到s_i后决策** ← Ex Post
3. 数据流动，定价，购买

**注意**: 与论文时序不一致，仅用于对比分析

---

#### 函数3: 统一接口

**`compute_rational_participation_rate(params, data, ...) -> (r*, history, ΔU)`**

根据`params.participation_timing`自动选择Ex Ante或Ex Post

---

#### 函数4: 基于τ_i生成参与

**`generate_participation_from_tau(ΔU, params, seed) -> participation`**

**经济学microfoundation** (⚠️ P2-2修正):
```python
每个消费者i有隐私成本 τ_i ~ F_τ
消费者i参与 ⟺ ΔU ≥ τ_i

这比独立Bernoulli(r*)更符合理论结构
```

---

### 模块7: Ground Truth生成 (第1910-2136行)

#### `generate_ground_truth(params, ...) -> result_dict`

**完整流程** (⚠️ P1-1修正):

```
第1步: 计算理性参与率r*（固定点）
  - Ex Ante: 对所有随机性取平均
  - Ex Post: 基于realized (w, s)

第2步: 计算期望outcome（MC平均，理论基准）
  - 重复num_outcome_samples次:
    * 生成数据
    * 基于τ_i生成participation（P2-2修正）
    * 模拟市场结果
    * 累加
  - 平均得到期望指标（不受抽样波动影响）

第3步: 生成示例outcome（单次抽样，用于LLM评估）
  - 生成一次数据
  - 基于τ_i生成participation
  - 模拟市场结果

第4步: 输出两套指标
  - 理论指标: r*, E[outcome | r*]
  - 示例指标: sample_data, sample_outcome
```

**输出结构**:
```python
{
  "params": {...},
  "rational_participation_rate": r*,  # 固定点
  
  # 理论基准（MC平均）
  "expected_outcome": {
    "participation_rate_realized": ...,
    "consumer_surplus": ...,
    "producer_profit": ...,
    "social_welfare": ...,
    ...
  },
  
  # 示例数据（LLM评估用）
  "sample_data": {w, s, theta, epsilon},
  "sample_participation": [...],
  "sample_outcome": {...},
  "sample_detailed_results": {prices, quantities, utilities, ...}
}
```

**对应论文**: 完整的理论求解流程

---

## 🎮 三层博弈框架

### Stackelberg结构（逆向归纳）

```
┌─────────────────────────────────────────────────────────────┐
│  外层: 中介优化（Stackelberg Leader）                        │
│  optimize_intermediary_policy()                             │
│                                                             │
│  中介选择策略: (m*, anonymization*)                         │
│  目标: max R = m_0 - m·N_参与                               │
│  预测: 消费者和生产者的反应                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │ 给定 (m, anonymization)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  内层-1: 消费者均衡（Nash Equilibrium）                      │
│  compute_rational_participation_rate()                      │
│                                                             │
│  消费者同时独立决策: a_i ∈ {0, 1}                          │
│  均衡条件: r* = F_τ(ΔU(r*))                                 │
│  权衡: 补偿+学习 vs 价格歧视+隐私成本                        │
└─────────────────┬───────────────────────────────────────────┘
                  │ 给定 r*, 形成数据库X
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  内层-2: 生产者优化（Profit Maximization）                   │
│  simulate_market_outcome()                                  │
│                                                             │
│  生产者观察X，定价:                                          │
│  - Identified: p_i* = (μ_i^prod + c) / 2                   │
│  - Anonymized: p* = argmax Σ(p-c)·max(μ_i-p, 0)            │
│                                                             │
│  消费者购买: q_i = max(μ_i^cons - p_i, 0)                  │
└─────────────────────────────────────────────────────────────┘
```

---

### 模块8: 中介优化实现 (第2164-2717行)

#### 函数1: 无数据baseline

**`simulate_market_outcome_no_data(data, params) -> MarketOutcome`**

**场景**: 中介不存在，生产者只能依赖先验

```python
信息结构:
  - 生产者: μ_producer[i] = μ_θ for all i（无学习）
  - 消费者: μ_consumer[i] = E[w_i | s_i]（只用自己信号）
  
定价:
  - 必然统一定价（无法区分）
  - p* = argmax Σ(p-c)·max(μ_θ-p, 0)
  
用途:
  - 计算生产者利润增益
  - 确定m_0（生产者支付意愿）
```

---

#### 函数2: 策略评估

**`evaluate_intermediary_strategy(m, anonymization, params_base, ...) -> IntermediaryOptimizationResult`**

**执行逆向归纳**:

```python
第1步: 内层 - 求解消费者均衡
  r*, delta_u = compute_rational_participation_rate(
    params=(m, anonymization, ...)
  )

第2步: 生成市场实现
  data = generate_consumer_data()
  participation = generate_participation_from_tau(delta_u)

第3步: 中层 - 计算生产者利润
  outcome_with_data = simulate_market_outcome(data, participation)
  producer_profit_with_data = outcome_with_data.producer_profit

第4步: Baseline - 计算无数据利润
  outcome_no_data = simulate_market_outcome_no_data(data)
  producer_profit_no_data = outcome_no_data.producer_profit

第5步: 外层 - 计算中介利润
  # 生产者支付意愿 = 数据带来的利润增量
  m_0 = max(0, producer_profit_with_data - producer_profit_no_data)
  
  # 中介净利润
  intermediary_cost = m * num_participants
  intermediary_profit = m_0 - intermediary_cost

返回: IntermediaryOptimizationResult{
  m, anonymization, r*, delta_u, num_participants,
  producer_profit_with_data, producer_profit_no_data,
  m_0, intermediary_cost, intermediary_profit,
  consumer_surplus, social_welfare, gini, price_discrimination
}
```

**对应论文**: Section 5.2-5.3, 中介最优信息设计

---

#### 函数3: 最优策略求解

**`optimize_intermediary_policy(params_base, m_grid, policies, ...) -> OptimalPolicy`**

**网格搜索**:
```python
候选策略空间:
  - m ∈ [0, 3.0]（31个点）
  - anonymization ∈ {'identified', 'anonymized'}
  - 总计: 31 × 2 = 62个候选

遍历所有候选:
  for m in m_grid:
    for policy in ['identified', 'anonymized']:
      result = evaluate_intermediary_strategy(m, policy, ...)
      all_results.append(result)

找到最优:
  optimal_result = max(all_results, key=lambda x: x.intermediary_profit)

返回: OptimalPolicy{
  optimal_m,
  optimal_anonymization,
  optimal_result,
  all_results,
  optimization_summary
}
```

**对应论文**: Section 5.2-5.3, Theorem 1

---

#### 函数4: 验证论文命题

**`verify_proposition_2(params_base, N_values, ...) -> results_dict`**

**命题2**: N足够大时，anonymized最优

**验证逻辑**:
```python
for N in [10, 20, 50, 100]:
  # 计算两种策略下的中介利润
  R_identified = evaluate_strategy(m, 'identified', N=N)
  R_anonymized = evaluate_strategy(m, 'anonymized', N=N)
  
  # 对比
  if R_anonymized > R_identified:
    print(f"✅ N={N}: anonymized占优")
  else:
    print(f"❌ N={N}: identified占优")

理论预期:
  - N小: identified可能占优（个性化定价价值高）
  - N大: anonymized占优（聚合数据仍准确，参与率更高）
```

**对应论文**: Section 5.2, Proposition 2

---

#### 函数5: 分析补偿曲线

**`analyze_optimal_compensation_curve(optimal_policy, ...) -> curve_data`**

**可视化中介trade-off**:
```python
对每个policy ∈ {'identified', 'anonymized'}:
  提取所有m对应的结果:
    - r*(m): 参与率曲线
    - m_0(m): 收入曲线（随r*增加）
    - cost(m) = m·r*·N: 成本曲线
    - R(m) = m_0(m) - cost(m): 利润曲线
    
  找到最优点:
    m* = argmax R(m)
```

**理论洞察**:
- 提高m → 提高r* → 提高m_0（收益↑）
- 但成本也增加: m·r*·N（成本↑）
- 最优m*在边际收益 = 边际成本处

**对应论文**: Theorem 1, 一阶条件

---

## 📑 函数索引（按功能分类）

### 数据与参数

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `ScenarioCParams` | 264-417 | 参数配置类 |
| `ConsumerData` | 419-427 | 消费者数据结构 |
| `MarketOutcome` | 429-464 | 市场结果结构 |
| `generate_consumer_data()` | 466-507 | 生成(w, s)数据 |

### 贝叶斯估计

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `_compute_ce_posterior_approx()` | 509-553 | CE后验近似 |
| `compute_posterior_mean_consumer()` | 555-654 | 消费者后验 |
| `compute_producer_posterior()` | 966-1112 | 生产者后验 |

### 定价

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `compute_optimal_price_personalized()` | 656-699 | 个性化定价 |
| `compute_optimal_price_uniform()` | 701-809 | 统一定价（数值） |
| `compute_optimal_price_uniform_piecewise()` | 811-919 | 统一定价（分段） |
| `compute_optimal_price_uniform_efficient_DEPRECATED()` | 924-964 | 已废弃（错误） |

### 市场均衡

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `simulate_market_outcome()` | 1114-1485 | 完整市场均衡 |
| `compute_gini()` | 1487-1521 | 基尼系数 |
| `simulate_market_outcome_no_data()` | 2228-2323 | 无数据baseline |

### 参与决策

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `compute_expected_utility_ex_ante()` | 1523-1590 | Ex Ante期望效用 |
| `compute_expected_utility_given_participation()` | 1592-1638 | Ex Post期望效用 |
| `compute_rational_participation_rate_ex_ante()` | 1640-1726 | Ex Ante固定点 |
| `compute_rational_participation_rate_ex_post()` | 1728-1802 | Ex Post固定点 |
| `compute_rational_participation_rate()` | 1804-1857 | 统一接口 |
| `generate_participation_from_tau()` | 1859-1908 | 基于τ_i生成参与 |

### Ground Truth

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `generate_ground_truth()` | 1910-2136 | 主生成函数 |

### 中介优化

| 函数名 | 行数 | 作用 |
|--------|------|------|
| `IntermediaryOptimizationResult` | 2176-2207 | 策略评估结果 |
| `OptimalPolicy` | 2209-2226 | 最优策略结果 |
| `evaluate_intermediary_strategy()` | 2325-2438 | 评估单个策略 |
| `optimize_intermediary_policy()` | 2440-2557 | 求解最优策略 |
| `verify_proposition_2()` | 2559-2637 | 验证命题2 |
| `analyze_optimal_compensation_curve()` | 2639-2692 | 分析补偿曲线 |
| `export_optimization_results()` | 2694-2717 | 导出结果 |

---

## 🔄 代码流程图

### Ground Truth生成完整流程

```
用户调用
│
▼
generate_ground_truth(params)
│
├─► 计算理性参与率r*（固定点）
│   │
│   ├─ Ex Ante模式:
│   │  └─► compute_rational_participation_rate_ex_ante()
│   │      │
│   │      ├─► compute_expected_utility_ex_ante() [参与]
│   │      │   │
│   │      │   └─► 两层MC循环:
│   │      │       ├─ 外层: 遍历(w, s)
│   │      │       └─ 内层: 遍历参与者集合
│   │      │           └─► simulate_market_outcome()
│   │      │
│   │      ├─► compute_expected_utility_ex_ante() [拒绝]
│   │      │
│   │      ├─► delta_u = E[u|参与] - E[u|拒绝]
│   │      │
│   │      └─► r_new = F_τ(delta_u)  # 固定点更新
│   │
│   └─ Ex Post模式:
│      └─► compute_rational_participation_rate_ex_post(data)
│
├─► 计算期望outcome（MC平均）
│   │
│   └─ For i in range(num_outcome_samples):
│       ├─► generate_consumer_data()
│       ├─► generate_participation_from_tau(delta_u)
│       └─► simulate_market_outcome()
│
├─► 生成示例outcome（单次）
│   │
│   ├─► generate_participation_from_tau(delta_u)
│   └─► simulate_market_outcome()
│
└─► 输出两套指标
    ├─ 理论指标: r*, E[outcome]
    └─ 示例指标: sample_data, sample_outcome
```

---

### simulate_market_outcome()内部流程

```
simulate_market_outcome(data, participation, params)
│
├─► 1. 数据收集与匿名化
│   ├─ X = s[participation]
│   └─ if anonymized: shuffle(X)
│
├─► 2. 消费者后验估计
│   └─ for i in range(N):
│       └─► compute_posterior_mean_consumer(s[i], X)
│
├─► 3. 生产者后验估计
│   └─► compute_producer_posterior(data, participation, X)
│       │
│       ├─ Identified:
│       │  ├─ 参与者: E[w_i | s_i, X]
│       │  └─ 拒绝者: E[w_i | X]（P0-2修正）
│       │
│       └─ Anonymized:
│          └─ 所有人: E[θ | X] or E[w | X]（P0-3修正）
│
├─► 4. 生产者定价
│   │
│   ├─ Identified:
│   │  └─ for i: p[i] = (μ_producer[i] + c) / 2
│   │
│   └─ Anonymized:
│      └─► p = compute_optimal_price_uniform(μ_producer)
│
├─► 5. 消费者购买
│   └─ q[i] = max(μ_consumer[i] - p[i], 0)
│
├─► 6. 效用实现
│   ├─ u[i] = w[i]·q[i] - p[i]·q[i] - 0.5·q[i]²
│   └─ u[participation] += m
│
├─► 7. 福利计算
│   ├─ CS = Σu[i]
│   ├─ PS = Σ(p[i] - c)·q[i]
│   ├─ IS = m_0 - m·N_参与
│   └─ SW = CS + PS + IS
│
└─► 8. 不平等指标
    ├─► compute_gini(utilities)
    └─ price_discrimination = max(p) - min(p)
```

---

### 中介优化完整流程

```
optimize_intermediary_policy(params_base, m_grid, policies)
│
└─► For each (m, policy) in 候选空间:
    │
    └─► evaluate_intermediary_strategy(m, policy)
        │
        ├─► 1. 内层 - 消费者均衡
        │   └─► compute_rational_participation_rate()
        │       └─ 返回: r*, delta_u
        │
        ├─► 2. 市场实现
        │   ├─► generate_consumer_data()
        │   └─► generate_participation_from_tau(delta_u)
        │
        ├─► 3. 中层 - 生产者利润
        │   ├─► simulate_market_outcome()  # 有数据
        │   └─► simulate_market_outcome_no_data()  # 无数据
        │
        └─► 4. 外层 - 中介利润
            ├─ m_0 = producer_profit_with - producer_profit_no
            ├─ cost = m * num_participants
            └─ R = m_0 - cost
│
├─► 找到最优策略
│   optimal_result = max(all_results, key=lambda x: x.intermediary_profit)
│
└─► 返回 OptimalPolicy{
      optimal_m,
      optimal_anonymization,
      optimal_result,
      all_results
    }
```

---

## 🎯 关键修正历史

### P0级（必须修复的机制错误）

| 修正 | 行数 | 问题 | 解决方案 |
|------|------|------|----------|
| **P0-1** | 555-654 | 消费者后验未包含s_i | 必须包含私人信号（论文I_i={s_i}∪X） |
| **P0-2** | 966-1112 | Identified下拒绝者后验固定为先验 | 用X更新后验（社会数据外部性） |
| **P0-3** | 1060-1109 | Anonymized+CE下无学习 | 用X估计ε改善预测 |

### P1级（重要的学术问题）

| 修正 | 行数 | 问题 | 解决方案 |
|------|------|------|----------|
| **P1-1** | 1910-2136 | r*与realization混淆 | 区分理论和示例两套指标 |
| **P1-2** | 1720, 1796 | 固定点未收敛仍返回 | 未收敛raise RuntimeError |
| **P1-4** | 1640-1726 | 时序不一致（Ex Post） | 实现Ex Ante（对所有随机性取平均） |
| **P1-5** | 1487-1521 | Gini系数负值不稳健 | 平移到正区间，clip到[0,1] |
| **P1-6** | 409, 643-650 | 后验方法hardcoded | 添加posterior_method参数 |

### P2级（工程质量）

| 修正 | 行数 | 问题 | 解决方案 |
|------|------|------|----------|
| **P2-1** | 359-376 | 无异质性（总是0/1） | 添加tau_dist支持 |
| **P2-2** | 1859-1908 | Bernoulli(r*)抽样 | 基于τ_i阈值生成participation |
| **P2-7** | 333, 1369 | 缺少中介利润 | 添加m_0和intermediary_profit |

---

## 📚 与论文的对应关系

| 论文章节 | 对应代码 | 行数 | 说明 |
|---------|---------|------|------|
| Section 2.1 | `simulate_market_outcome()` | 1114-1485 | 产品市场均衡 |
| Section 2.2 | `compute_optimal_price_*()` | 656-919 | 生产者定价 |
| Section 3.1-3.2 | `generate_consumer_data()` | 466-507 | 数据生成 |
| Section 3.3 | `compute_posterior_mean_consumer()` | 555-654 | 贝叶斯更新 |
| Section 4 | `compute_producer_posterior()` | 966-1112 | 匿名化机制 |
| Section 5.1 | `compute_rational_participation_rate_ex_ante()` | 1640-1726 | 参与决策 |
| Section 5.2-5.3 | `optimize_intermediary_policy()` | 2440-2557 | 中介优化 |
| Proposition 1 | P0-2, P0-3修正 | 966-1112 | 社会数据外部性 |
| Proposition 2 | `verify_proposition_2()` | 2559-2637 | 匿名化最优性 |
| Theorem 1 | `analyze_optimal_compensation_curve()` | 2639-2692 | 最优补偿 |

---

## 🔍 代码质量特点

### ✅ 优点

1. **理论严谨**: 严格对齐论文机制，Ex Ante时序符合学术标准
2. **注释详尽**: 每个函数都有详细的理论说明和论文对应
3. **模块化好**: 清晰的功能分层，易于维护和扩展
4. **鲁棒性强**: 处理边界情况（如无参与者、未收敛等）
5. **可配置**: 支持多种模式（Ex Ante/Post, 不同tau分布等）

### 📌 特色

1. **三层Stackelberg完整实现**: 从中介优化到消费者均衡
2. **Ex Ante + 异质性**: 产生有意义的内点参与率
3. **区分理论和示例**: Ground Truth输出两套指标
4. **完整的福利分解**: CS, PS, IS, SW, Gini, 价格歧视等

### 🎯 适用场景

- ✅ 学术研究：理论求解器，生成Ground Truth
- ✅ LLM Benchmark：评估LLM决策偏差
- ✅ 政策分析：对比不同匿名化政策的福利效应
- ✅ 市场设计：优化中介的补偿和信息披露策略

---

## 📝 使用示例

### 基础使用

```python
from src.scenarios.scenario_c_social_data import (
    ScenarioCParams, generate_ground_truth
)

# 创建参数
params = ScenarioCParams(
    N=20,
    data_structure="common_preferences",
    anonymization="identified",
    mu_theta=5.0,
    sigma_theta=1.0,
    sigma=1.0,
    m=1.0,
    tau_dist="normal",  # 启用异质性
    tau_mean=0.5,
    tau_std=0.3,
    participation_timing="ex_ante",  # Ex Ante时序
    seed=42
)

# 生成Ground Truth
gt = generate_ground_truth(params, max_iter=30, num_mc_samples=50)

# 输出
print(f"理性参与率 r* = {gt['rational_participation_rate']:.2%}")
print(f"期望社会福利 = {gt['expected_outcome']['social_welfare']:.2f}")
```

### 中介优化

```python
from src.scenarios.scenario_c_social_data import optimize_intermediary_policy

# 基础参数（不含m和anonymization）
params_base = {
    'N': 20,
    'data_structure': 'common_preferences',
    'mu_theta': 5.0,
    'sigma_theta': 1.0,
    'sigma': 1.0,
    'tau_dist': 'normal',
    'tau_mean': 0.5,
    'tau_std': 0.3,
    'participation_timing': 'ex_ante',
    'seed': 42
}

# 求解最优策略
optimal = optimize_intermediary_policy(
    params_base,
    m_grid=np.linspace(0, 3, 31),
    policies=['identified', 'anonymized'],
    num_mc_samples=50
)

print(f"最优补偿 m* = {optimal.optimal_m:.2f}")
print(f"最优策略 = {optimal.optimal_anonymization}")
print(f"最大利润 R* = {optimal.optimal_result.intermediary_profit:.2f}")
```

---

## 📊 性能考虑

### 计算复杂度

| 函数 | 复杂度 | 瓶颈 |
|------|--------|------|
| `generate_consumer_data()` | O(N) | 快 |
| `compute_posterior_mean_consumer()` | O(N) | 快 |
| `compute_optimal_price_uniform()` | O(M·N) | 中（M≈20-50） |
| `simulate_market_outcome()` | O(N) | 快 |
| `compute_rational_participation_rate_ex_ante()` | O(iter·samples·N) | 慢 |
| `optimize_intermediary_policy()` | O(grid_size·上述所有) | 非常慢 |

### 优化建议

1. **并行化**: 中介优化的候选策略评估可并行
2. **缓存**: 相同参数的后验计算可缓存
3. **早停**: 固定点迭代可添加早停机制
4. **采样数**: 根据精度需求调整MC样本数

---

**文档版本**: v2.0  
**创建日期**: 2026-01-18  
**最后更新**: 2026-01-18  
**作者**: AI Assistant  
**用途**: 场景C求解器完整技术文档
