# 场景C Ground Truth输出结构完整解析

**文档版本**: v1.0  
**创建日期**: 2026-01-18  
**文件路径**: `data/ground_truth/scenario_c_*.json`  
**生成函数**: `src/scenarios/scenario_c_social_data.py::generate_ground_truth()`

---

## 📋 目录

1. [输出概述](#输出概述)
2. [完整数据结构](#完整数据结构)
3. [字段详细说明](#字段详细说明)
4. [不同配置的输出格式](#不同配置的输出格式)
5. [使用指南](#使用指南)
6. [常见问题解答](#常见问题解答)

---

## 🎯 输出概述

### Ground Truth的双重性质

场景C的Ground Truth输出具有**双重性质**（P1-1修正后）：

```
┌─────────────────────────────────────────────────────────┐
│  理论指标 (Theoretical Metrics)                         │
│  - 固定点收敛值 r*                                       │
│  - 期望市场结果 E[outcome | r*]（MC平均，不受抽样波动） │
│  - 用于验证理论结论                                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  示例指标 (Sample Metrics)                              │
│  - 具体的数据实现 (w, s, θ, ε)                         │
│  - 单次参与抽样 participation                           │
│  - 对应的市场结果（包含所有细节）                        │
│  - 用于LLM评估和Benchmark任务                           │
└─────────────────────────────────────────────────────────┘
```

**为什么需要两套指标？**

| 用途 | 理论指标 | 示例指标 |
|------|---------|---------|
| **学术验证** | ✅ 不受抽样波动，精确 | ❌ 单次抽样，有噪声 |
| **LLM评估** | ❌ 太抽象，无具体数据 | ✅ 有具体(w,s,p,q)，可解释 |
| **理论对比** | ✅ 直接对应论文结论 | ⚠️ 需注意抽样偏差 |
| **算法调试** | ✅ 收敛轨迹，诊断性强 | ✅ 细节丰富，易发现问题 |

---

## 📦 完整数据结构

### 顶层结构（8个一级字段）

```json
{
  "params": {...},                    // 1️⃣ 完整参数配置
  "rational_participation_rate": 0.84, // 2️⃣ 固定点r*
  "r_history": [...],                 // 3️⃣ 收敛历史
  
  "expected_outcome": {...},          // 4️⃣ 期望市场结果（理论）
  
  "sample_data": {...},               // 5️⃣ 示例数据(w,s,θ,ε)
  "sample_participation": [...],      // 6️⃣ 示例参与决策
  "sample_outcome": {...},            // 7️⃣ 示例市场结果
  "sample_detailed_results": {...},   // 8️⃣ 示例细节(p,q,u,μ)
  
  // 向后兼容字段（指向expected_outcome）
  "outcome": {...},                   // ⚠️ 兼容旧版本
  "data": {...},                      // ⚠️ 兼容旧版本
  "rational_participation": [...]     // ⚠️ 兼容旧版本
}
```

---

## 🔍 字段详细说明

### 1️⃣ `params` - 参数配置

**作用**: 记录生成此Ground Truth使用的所有参数

**字段列表**:

```json
{
  "params": {
    // 基础参数
    "N": 20,                          // 消费者数量
    "data_structure": "common_preferences",  // 数据结构
    "anonymization": "identified",    // 匿名化策略
    
    // 数据生成参数（对应论文Section 3）
    "mu_theta": 5.0,                  // 先验均值 μ_θ
    "sigma_theta": 1.0,               // 先验标准差 σ_θ
    "sigma": 1.0,                     // 噪声水平 σ
    
    // 支付参数（对应论文Section 5）
    "m": 1.0,                         // 消费者补偿
    "m_0": 0.0,                       // 生产者支付（扩展）
    "c": 0.0,                         // 边际成本
    
    // 异质性参数（我们的扩展）
    "tau_mean": 0.5,                  // 隐私成本均值 μ_τ
    "tau_std": 0.5,                   // 隐私成本标准差 σ_τ
    "tau_dist": "normal",             // 隐私成本分布
    
    // 时序模式（学术关键）
    "participation_timing": "ex_ante", // Ex Ante/Ex Post
    
    // 算法参数
    "posterior_method": "approx",     // 后验估计方法
    "seed": 42                        // 随机种子
  }
}
```

**参数详解**:

#### 基础参数

| 参数 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| `N` | int | 10-100 | 消费者数量，影响数据外部性强度 |
| `data_structure` | str | `"common_preferences"` \| `"common_experience"` | 数据结构类型（论文Section 3.1-3.2） |
| `anonymization` | str | `"identified"` \| `"anonymized"` | 匿名化策略（论文Section 4核心） |

**数据结构对比**:

```
Common Preferences（共同偏好）:
  w_i = θ  for all i     （所有人相同）
  s_i = θ + σ·e_i        （噪声独立）
  特点: 多人数据可通过平均滤掉噪声
  
Common Experience（共同经历）:
  w_i ~ N(μ_θ, σ_θ²)     （每人不同）
  s_i = w_i + σ·ε        （噪声共同）
  特点: 多人数据可识别并过滤共同噪声
```

**匿名化对比**:

```
Identified（实名）:
  生产者信息: Y_0 = {(i, s_i)}  （有身份映射）
  定价能力: 个性化定价 p_i = (μ_i + c) / 2
  
Anonymized（匿名）:
  生产者信息: Y_0 = {s_i}  （无身份映射）
  定价能力: 统一定价 p = argmax Π(p)
```

#### 数据生成参数

| 参数 | 典型值 | 说明 | 影响 |
|------|--------|------|------|
| `mu_theta` | 5.0 | 先验均值，表示平均支付意愿 | 影响价格和需求水平 |
| `sigma_theta` | 1.0 | 先验标准差，表示偏好不确定性 | 影响学习价值 |
| `sigma` | 1.0 | 噪声水平，表示信号质量 | σ越大，多人数据越重要 |

**信噪比**: `SNR = σ_θ / σ`，越大表示学习价值越高

#### 支付参数

| 参数 | 典型值 | 说明 | 影响 |
|------|--------|------|------|
| `m` | 0.5-2.0 | 中介→消费者补偿 | m越高，r*越高 |
| `m_0` | 0.0 | 生产者→中介支付 | 影响中介利润 |
| `c` | 0.0 | 边际成本 | 影响定价 |

**权衡**: 提高m → 提高r* → 提高数据质量 vs 提高成本

#### 异质性参数（产生内点参与率的关键）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_mean` | 0.5 | 平均而言，消费者需要多少补偿才愿意承担隐私风险 |
| `tau_std` | 0.3 | 消费者隐私偏好的异质性程度 |
| `tau_dist` | `"normal"` | `"normal"` \| `"uniform"` \| `"none"` |

**作用**:
- `tau_dist="none"`: r* ∈ {0, 1}（角点解，难以衡量偏差）
- `tau_dist="normal"`: r* ∈ (0, 1)（内点解，Benchmark友好）

**参与决策逻辑**:
```python
消费者i参与 ⟺ ΔU ≥ τ_i

其中:
  ΔU = E[u|参与] - E[u|拒绝] + m  （对所有人相同）
  τ_i ~ F_τ  （个体隐私成本）

均衡参与率:
  r* = P(τ_i ≤ ΔU) = F_τ(ΔU)
```

#### 时序模式（影响学术可信度）

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `"ex_ante"` | 消费者在不知道(w,s)实现时决策 | **主结果**，对应论文时序 |
| `"ex_post"` | 消费者看到realized (w,s)后决策 | 鲁棒性检验 |

---

### 2️⃣ `rational_participation_rate` - 理性参与率

**类型**: `float`  
**范围**: [0, 1]  
**含义**: 固定点收敛值 r*

**理论意义**:

```
固定点方程:
  r* = F_τ(ΔU(r*))

其中:
  ΔU(r*) = E[u | 参与, r*] - E[u | 拒绝, r*] + m

均衡性质:
  - 给定r*，没有消费者愿意改变决策
  - r*同时反映了消费者的集体理性预期
```

**示例值解读**:

```json
"rational_participation_rate": 0.8374674897276418
```

解读:
- 在均衡状态下，约83.7%的消费者会选择参与
- 这是考虑了所有随机性（信号、偏好、他人决策）后的期望参与率
- 不是单次实现（单次可能是16/20=80%或17/20=85%）

---

### 3️⃣ `r_history` - 收敛历史

**类型**: `List[float]`  
**含义**: 固定点迭代的收敛轨迹

**示例**:

```json
"r_history": [
  0.5,                    // 初始值
  0.834434967219911,      // 第1次迭代
  0.8365514229378355,     // 第2次迭代
  0.8371483014641599,     // ...
  0.8373018051083823,
  0.8373848727735698,
  0.8374373847260552,
  0.8374470338640199,
  0.8374674897276418      // 收敛值
]
```

**诊断用途**:

1. **收敛性检查**:
   ```python
   # 是否收敛？
   converged = abs(r_history[-1] - r_history[-2]) < tol
   
   # 收敛速度？
   num_iterations = len(r_history) - 1
   ```

2. **震荡检测**:
   ```python
   # 是否震荡？
   oscillating = any(
     abs(r_history[i] - r_history[i-2]) < abs(r_history[i] - r_history[i-1])
     for i in range(2, len(r_history))
   )
   ```

3. **单调性**:
   ```python
   # 是否单调？
   monotonic_increasing = all(
     r_history[i] >= r_history[i-1]
     for i in range(1, len(r_history))
   )
   ```

---

### 4️⃣ `expected_outcome` - 期望市场结果（理论基准）

**作用**: MC平均的市场结果，不受单次抽样波动影响

**字段列表**:

```json
{
  "expected_outcome": {
    "participation_rate_realized": 0.7925,  // 实际参与率的期望
    "consumer_surplus": 74.487,             // 期望消费者剩余
    "producer_profit": 118.984,             // 期望生产者利润
    "intermediary_profit": -15.85,          // 期望中介利润
    "social_welfare": 177.621,              // 期望社会福利
    "gini_coefficient": 0.045,              // 期望基尼系数
    "price_discrimination_index": 3.99e-16  // 期望价格歧视指数
  }
}
```

#### 字段详解

**1. `participation_rate_realized` - 实际参与率**

```
定义: 单次抽样中实际参与的人数比例

与r*的关系:
  - r* = 固定点理论值（均衡预期）
  - participation_rate_realized = E[实际参与率]（MC平均）
  
为什么不同？
  - r*是连续值，实际参与是离散的（整数人数）
  - 有抽样随机性（基于τ_i阈值）
  
示例:
  r* = 0.8375
  participation_rate_realized = 0.7925 ≈ 15.85/20
```

**2. `consumer_surplus` - 消费者剩余**

```
定义: CS = Σ_i u_i（包含补偿）

组成:
  CS = 产品消费效用 + 数据补偿 - 支付成本
     = Σ (w_i·q_i - 0.5·q_i²) - Σ p_i·q_i + m·N_参与

典型范围: [50, 150]（N=20）
```

**3. `producer_profit` - 生产者利润**

```
定义: PS = Σ_i (p_i - c) · q_i

特点:
  - 不包含向中介支付的m_0（转移支付）
  - 只反映产品销售利润
  
影响因素:
  - 定价策略（个性化 vs 统一）
  - 信息质量（参与率越高，信息越准确）
  
典型范围: [100, 150]（N=20）
```

**4. `intermediary_profit` - 中介利润**

```
定义: IS = m_0 - m · N_参与

默认情况（m_0=0）:
  IS < 0（中介纯支出）
  IS = -m · N_参与
  
示例:
  m = 1.0, N_参与 = 15.85（期望）
  IS = 0 - 1.0 × 15.85 = -15.85
  
扩展情况（m_0>0）:
  IS可能为正（中介盈利）
```

**5. `social_welfare` - 社会福利**

```
定义: SW = CS + PS + IS

性质:
  - 补偿m是转移支付，不影响总福利（如果m_0=0）
  - 如果m_0 = m·N_参与，则SW = CS + PS（完全转移）
  
理论意义:
  - 衡量匿名化政策的效率
  - 对比不同m下的社会总剩余
  
典型范围: [170, 180]（N=20）
```

**6. `gini_coefficient` - 基尼系数**

```
定义: Gini ∈ [0, 1]，衡量效用分配的不平等程度

解释:
  - 0: 完全平等（所有人效用相同）
  - 1: 完全不平等（一人获得所有效用）
  
影响因素:
  - 个性化定价（Identified）→ Gini更高
  - 统一定价（Anonymized）→ Gini更低
  - 参与者 vs 拒绝者的效用差
  
典型值:
  - Anonymized: 0.03-0.05（低不平等）
  - Identified + Common Experience: 0.10-0.15（中等不平等）
```

**7. `price_discrimination_index` - 价格歧视指数**

```
定义: PDI = max(p) - min(p)

解释:
  - 0: 统一定价（无歧视）
  - >0: 有价格歧视
  
理论预期:
  - Anonymized: PDI ≈ 0（必然统一定价）
  - Identified + Common Preferences: PDI ≈ 0（后验相近）
  - Identified + Common Experience: PDI > 0（后验异质）
  
示例值:
  3.99e-16 → 实际为0（浮点误差）
```

---

### 5️⃣ `sample_data` - 示例数据

**作用**: 一次具体的数据实现，用于LLM评估

**字段列表**:

```json
{
  "sample_data": {
    "w": [5.497, 5.497, ...],    // (N,) 真实支付意愿
    "s": [5.358, 6.144, ...],    // (N,) 观测信号
    "theta": 5.497,              // 共同偏好（仅CP）
    "epsilon": null              // 共同噪声（仅CE）
  }
}
```

#### 数据结构差异

**Common Preferences**:
```json
{
  "w": [θ, θ, θ, ...],           // 所有人相同
  "s": [θ+σe₁, θ+σe₂, ...],      // 独立噪声
  "theta": θ,                    // 记录真实值
  "epsilon": null                // 无共同噪声
}
```

**Common Experience**:
```json
{
  "w": [w₁, w₂, w₃, ...],        // 每人不同
  "s": [w₁+σε, w₂+σε, ...],      // 共同噪声
  "theta": null,                 // 无共同偏好
  "epsilon": ε                   // 记录共同噪声
}
```

#### 使用示例

```python
import json

# 读取Ground Truth
with open("scenario_c_result.json") as f:
    gt = json.load(f)

# 提取数据
w = gt["sample_data"]["w"]        # 真实偏好
s = gt["sample_data"]["s"]        # 观测信号
theta = gt["sample_data"]["theta"] # 共同偏好（如果有）

# 检查数据结构
data_structure = gt["params"]["data_structure"]
if data_structure == "common_preferences":
    assert all(w_i == w[0] for w_i in w), "CP下所有w应相同"
    print(f"共同偏好 θ = {theta:.3f}")
elif data_structure == "common_experience":
    epsilon = gt["sample_data"]["epsilon"]
    print(f"共同噪声 ε = {epsilon:.3f}")
    
# 验证信号生成
sigma = gt["params"]["sigma"]
if data_structure == "common_preferences":
    # s_i = θ + σ·e_i
    e_reconstructed = [(s[i] - theta) / sigma for i in range(len(s))]
    print(f"噪声样本: {e_reconstructed[:3]}")
```

---

### 6️⃣ `sample_participation` - 示例参与决策

**类型**: `List[bool]`  
**长度**: N  
**含义**: 每个消费者的参与决策（true=参与，false=拒绝）

**示例**:

```json
"sample_participation": [
  true,   // 消费者0参与
  true,   // 消费者1参与
  true,   // ...
  true,
  false,  // 消费者4拒绝
  true,
  // ...
]
```

**生成机制**（P2-2修正）:

```python
# 旧方法（已废弃）: 独立Bernoulli抽样
participation[i] ~ Bernoulli(r*)  # ❌ 结构不对

# 新方法（P2-2）: 基于τ_i阈值
τ_i ~ F_τ（隐私成本分布）
participation[i] = (ΔU ≥ τ_i)  # ✅ 经济学microfoundation

其中:
  ΔU = E[u|参与] - E[u|拒绝] + m（对所有人相同）
```

**统计检查**:

```python
# 实际参与率
actual_rate = sum(sample_participation) / len(sample_participation)

# 与r*的关系
# actual_rate ≈ r*（随机波动）
# 如果N足够大且tau_dist不是"none"
```

---

### 7️⃣ `sample_outcome` - 示例市场结果

**作用**: 给定`sample_participation`的完整市场结果

**字段列表**:

```json
{
  "sample_outcome": {
    "participation_rate": 0.95,              // 此次参与率（19/20）
    "num_participants": 19,                  // 参与人数
    "consumer_surplus": 98.133,              // 消费者剩余
    "producer_profit": 143.675,              // 生产者利润
    "intermediary_profit": -19.0,            // 中介利润
    "social_welfare": 222.807,               // 社会福利
    "gini_coefficient": 0.0097,              // 基尼系数
    "price_variance": 8.58e-31,              // 价格方差
    "price_discrimination_index": 8.88e-16,  // 价格歧视指数
    "acceptor_avg_utility": 4.957,           // 参与者平均效用
    "rejecter_avg_utility": 3.956,           // 拒绝者平均效用
    "learning_quality_participants": 0.136,  // 参与者学习质量
    "learning_quality_rejecters": 0.141      // 拒绝者学习质量
  }
}
```

#### 与`expected_outcome`的区别

| 指标 | `expected_outcome` | `sample_outcome` |
|------|-------------------|------------------|
| **性质** | MC平均（理论基准） | 单次实现 |
| **用途** | 验证理论结论 | LLM评估，可解释性 |
| **波动** | 无（平滑） | 有（随机） |
| **参与率** | 期望值（连续） | 实际值（离散，如19/20） |
| **细节** | 较少 | 丰富（含学习质量、分组统计） |

#### 额外字段说明

**1. `acceptor_avg_utility` / `rejecter_avg_utility`**

```
定义:
  - acceptor_avg = mean(u_i | participation[i]=True)
  - rejecter_avg = mean(u_i | participation[i]=False)

理论预期:
  - 如果m足够大: acceptor_avg > rejecter_avg
    （参与是有利可图的）
  
  - 如果m太小: acceptor_avg < rejecter_avg
    （拒绝者搭便车成功）
  
  - 均衡时（r*固定点）:
    边际参与者无差异，但由于τ_i异质性，
    平均而言acceptor_avg ≈ rejecter_avg + E[τ_i | 参与]
```

**2. `learning_quality_participants` / `rejecter_avg_utility`**

```
定义: 学习误差 = mean(|μ_i - w_i|)

解释:
  - 越小表示后验估计越准确
  - 衡量数据外部性的学习效果

理论预期:
  - 参与率↑ → |X|↑ → 学习质量↑（误差↓）
  - Common Preferences: 参与者与拒绝者学习质量相近
  - Common Experience: 参与者可能略好（s_i在X中）
  
典型值:
  - 0.10-0.20（N=20, σ=1.0）
```

---

### 8️⃣ `sample_detailed_results` - 示例细节

**作用**: 每个消费者的具体价格、需求、效用、后验

**字段列表**:

```json
{
  "sample_detailed_results": {
    "prices": [2.680, 2.680, ...],      // (N,) 每个消费者的价格
    "quantities": [2.680, 2.680, ...],  // (N,) 每个消费者的购买量
    "utilities": [4.957, 4.957, ...],   // (N,) 每个消费者的效用
    "mu_consumers": [5.361, 5.361, ...]  // (N,) 消费者后验估计
  }
}
```

#### 使用示例

**1. 验证定价逻辑**

```python
# Common Preferences + Identified
# 理论: 所有人后验相近 → 价格相近
mu_consumers = gt["sample_detailed_results"]["mu_consumers"]
prices = gt["sample_detailed_results"]["prices"]

assert max(mu_consumers) - min(mu_consumers) < 0.01, "CP下后验应相近"
assert max(prices) - min(prices) < 0.01, "因此价格应相近"

# 验证定价公式: p_i = (μ_i + c) / 2
c = gt["params"]["c"]
for i in range(len(prices)):
    expected_price = (mu_consumers[i] + c) / 2
    assert abs(prices[i] - expected_price) < 1e-6
```

**2. 验证需求公式**

```python
# 需求: q_i = max(μ_i - p_i, 0)
quantities = gt["sample_detailed_results"]["quantities"]

for i in range(len(quantities)):
    expected_quantity = max(mu_consumers[i] - prices[i], 0)
    assert abs(quantities[i] - expected_quantity) < 1e-6
```

**3. 验证效用计算**

```python
# 效用: u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
# 参与者: u_i += m
w = gt["sample_data"]["w"]
utilities = gt["sample_detailed_results"]["utilities"]
participation = gt["sample_participation"]
m = gt["params"]["m"]

for i in range(len(utilities)):
    base_utility = (
        w[i] * quantities[i] - 
        prices[i] * quantities[i] - 
        0.5 * quantities[i]**2
    )
    if participation[i]:
        base_utility += m
    
    assert abs(utilities[i] - base_utility) < 1e-6
```

**4. 分析价格歧视**

```python
# 识别被歧视的消费者
if gt["params"]["anonymization"] == "identified":
    # 找到高价消费者
    avg_price = sum(prices) / len(prices)
    high_price_consumers = [
        i for i in range(len(prices))
        if prices[i] > avg_price
    ]
    
    # 分析是否参与
    high_price_participation = [
        participation[i] for i in high_price_consumers
    ]
    
    print(f"高价消费者参与率: {sum(high_price_participation) / len(high_price_participation):.2%}")
```

**5. 可视化效用分布**

```python
import matplotlib.pyplot as plt

# 按参与状态分组
acceptor_utilities = [
    utilities[i] for i in range(len(utilities))
    if participation[i]
]
rejecter_utilities = [
    utilities[i] for i in range(len(utilities))
    if not participation[i]
]

plt.figure(figsize=(10, 6))
plt.hist(acceptor_utilities, alpha=0.5, label='Participants', bins=10)
plt.hist(rejecter_utilities, alpha=0.5, label='Non-participants', bins=10)
plt.xlabel('Utility')
plt.ylabel('Frequency')
plt.legend()
plt.title('Utility Distribution by Participation Status')
plt.show()
```

---

## 📂 不同配置的输出格式

### 1. 单配置输出

**文件**: `scenario_c_common_preferences_identified.json`

**格式**: 单个JSON对象（如前述完整结构）

**使用场景**:
- MVP配置验证
- 核心对比实验（2×2矩阵）

**命名规则**:
```
scenario_c_{data_structure}_{anonymization}.json

例如:
- scenario_c_common_preferences_identified.json
- scenario_c_common_preferences_anonymized.json
- scenario_c_common_experience_identified.json
- scenario_c_common_experience_anonymized.json
```

---

### 2. 补偿扫描输出

**文件**: `scenario_c_payment_sweep.json`

**格式**: JSON数组，每个元素对应一个m值

```json
[
  {
    "m": 0.0,
    "participation_rate": 0.136,
    "consumer_surplus": 55.135,
    "producer_profit": 121.577,
    "social_welfare": 176.712
  },
  {
    "m": 0.5,
    "participation_rate": 0.488,
    "consumer_surplus": 62.098,
    "producer_profit": 120.309,
    "social_welfare": 177.932
  },
  {
    "m": 1.0,
    "participation_rate": 0.837,
    "consumer_surplus": 74.487,
    "producer_profit": 118.984,
    "social_welfare": 177.621
  },
  // ...
]
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `m` | float | 补偿水平 |
| `participation_rate` | float | r*（固定点） |
| `consumer_surplus` | float | 期望CS |
| `producer_profit` | float | 期望PS |
| `social_welfare` | float | 期望SW |

**注意**: 不包含详细数据（w, s, p, q），仅关键指标

**使用场景**:
- 分析m对r*的影响曲线
- 找到最优补偿m*
- 验证论文Theorem 1

**示例代码**:

```python
import json
import matplotlib.pyplot as plt

# 读取补偿扫描结果
with open("scenario_c_payment_sweep.json") as f:
    sweep = json.load(f)

# 提取数据
m_values = [item["m"] for item in sweep]
r_values = [item["participation_rate"] for item in sweep]
sw_values = [item["social_welfare"] for item in sweep]

# 绘制曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 参与率曲线
ax1.plot(m_values, r_values, 'o-')
ax1.set_xlabel('Compensation (m)')
ax1.set_ylabel('Participation Rate (r*)')
ax1.set_title('r*(m) Curve')
ax1.grid(True)

# 社会福利曲线
ax2.plot(m_values, sw_values, 's-', color='green')
ax2.set_xlabel('Compensation (m)')
ax2.set_ylabel('Social Welfare')
ax2.set_title('SW(m) Curve')
ax2.grid(True)

plt.tight_layout()
plt.show()

# 找到最优补偿
optimal_idx = sw_values.index(max(sw_values))
optimal_m = m_values[optimal_idx]
print(f"最优补偿 m* = {optimal_m:.2f}")
print(f"最大社会福利 SW* = {sw_values[optimal_idx]:.2f}")
```

---

## 📖 使用指南

### 场景1: 验证理论结论

**目标**: 验证"匿名化降低价格歧视"（论文Proposition 2）

```python
import json

# 读取两个配置
with open("scenario_c_common_experience_identified.json") as f:
    gt_id = json.load(f)

with open("scenario_c_common_experience_anonymized.json") as f:
    gt_anon = json.load(f)

# 比较价格歧视指数
pdi_id = gt_id["expected_outcome"]["price_discrimination_index"]
pdi_anon = gt_anon["expected_outcome"]["price_discrimination_index"]

print(f"Identified PDI: {pdi_id:.4f}")
print(f"Anonymized PDI: {pdi_anon:.4f}")

assert pdi_anon < pdi_id, "匿名化应降低价格歧视"
print("✅ 理论预测验证成功")
```

---

### 场景2: LLM Benchmark评估

**目标**: 评估LLM的参与决策是否理性

```python
import json

# 读取Ground Truth
with open("scenario_c_result.json") as f:
    gt = json.load(f)

# LLM决策（假设已运行评估）
llm_participation_rate = 0.65  # LLM决策的参与率

# 理性基准
rational_rate = gt["rational_participation_rate"]

# 计算偏差
deviation = abs(llm_participation_rate - rational_rate)
relative_deviation = deviation / rational_rate

print(f"理性参与率: {rational_rate:.2%}")
print(f"LLM参与率: {llm_participation_rate:.2%}")
print(f"绝对偏差: {deviation:.2%}")
print(f"相对偏差: {relative_deviation:.1%}")

# 判断
if relative_deviation < 0.1:
    print("✅ LLM表现接近理性")
elif relative_deviation < 0.3:
    print("⚠️ LLM有一定偏差")
else:
    print("❌ LLM严重偏离理性")
```

---

### 场景3: 福利分析

**目标**: 分解福利来源，理解不同策略的影响

```python
import json
import pandas as pd

# 读取所有配置
configs = [
    "common_preferences_identified",
    "common_preferences_anonymized",
    "common_experience_identified",
    "common_experience_anonymized"
]

results = []
for config in configs:
    with open(f"scenario_c_{config}.json") as f:
        gt = json.load(f)
        
    results.append({
        "config": config,
        "r*": gt["rational_participation_rate"],
        "CS": gt["expected_outcome"]["consumer_surplus"],
        "PS": gt["expected_outcome"]["producer_profit"],
        "SW": gt["expected_outcome"]["social_welfare"],
        "Gini": gt["expected_outcome"]["gini_coefficient"],
        "PDI": gt["expected_outcome"]["price_discrimination_index"]
    })

# 创建DataFrame
df = pd.DataFrame(results)
print(df.to_string(index=False))

# 分析
print("\n关键发现:")
print(f"1. 最高SW: {df.loc[df['SW'].idxmax(), 'config']}")
print(f"2. 最高r*: {df.loc[df['r*'].idxmax(), 'config']}")
print(f"3. 最低Gini: {df.loc[df['Gini'].idxmin(), 'config']}")
```

**输出示例**:

```
                           config       r*      CS      PS      SW   Gini    PDI
  common_preferences_identified   0.837  74.49  118.98  177.62  0.045  0.000
common_preferences_anonymized   0.846  75.12  118.45  177.72  0.042  0.000
  common_experience_identified   0.725  68.23  125.34  178.82  0.087  1.234
 common_experience_anonymized   0.812  73.45  119.87  178.57  0.053  0.000

关键发现:
1. 最高SW: common_experience_identified
2. 最高r*: common_preferences_anonymized
3. 最低Gini: common_preferences_anonymized
```

---

### 场景4: 诊断参与率异常

**问题**: 理论r*=100%，但实际参与率=0%

**原因排查**:

```python
import json

with open("scenario_c_result.json") as f:
    gt = json.load(f)

# 检查收敛性
r_history = gt["r_history"]
print(f"收敛历史: {r_history}")

if len(r_history) < 5:
    print("❌ 迭代次数过少，可能未收敛")

if abs(r_history[-1] - r_history[-2]) > 0.01:
    print("❌ 未收敛到固定点")

# 检查参数合理性
params = gt["params"]
m = params["m"]
tau_mean = params["tau_mean"]

if m < tau_mean:
    print(f"⚠️ 补偿m={m}小于平均隐私成本τ_mean={tau_mean}")
    print("   期望较低参与率")

# 检查异质性设置
if params["tau_dist"] == "none":
    print("⚠️ 无异质性，r*必为0或1")
    print("   建议设置tau_dist='normal'")

# 检查sample vs expected
sample_rate = gt["sample_outcome"]["participation_rate"]
expected_rate = gt["expected_outcome"]["participation_rate_realized"]
rational_rate = gt["rational_participation_rate"]

print(f"\n参与率对比:")
print(f"  理论r*: {rational_rate:.2%}")
print(f"  期望实现: {expected_rate:.2%}")
print(f"  单次抽样: {sample_rate:.2%}")

if abs(sample_rate - rational_rate) > 0.3:
    print("⚠️ 单次抽样与r*偏差较大（正常，N较小时常见）")
```

---

## ❓ 常见问题解答

### Q1: 为什么`rational_participation_rate`和`expected_outcome.participation_rate_realized`不同？

**A**: 
- `rational_participation_rate` (r*): 固定点理论值，连续值
- `participation_rate_realized`: 单次抽样的期望，考虑了离散性

**示例**:
```
r* = 0.8375（固定点）
实际参与人数 ~ Binomial(N=20, p=0.8375)
期望人数 = 20 × 0.8375 = 16.75
但实际只能是整数（16或17）
因此E[实际参与率] ≈ 16.75/20 = 0.8375

但由于基于τ_i的生成机制，实际期望可能略低
```

---

### Q2: `sample_outcome`和`expected_outcome`哪个更准确？

**A**: 
- **理论验证**: 用`expected_outcome`（不受抽样波动）
- **LLM评估**: 用`sample_outcome`（有具体数据）
- **算法调试**: 都用（对比发现问题）

---

### Q3: 如何判断Ground Truth质量？

**检查清单**:

```python
✅ r_history收敛（最后几项变化<1e-3）
✅ expected_outcome与sample_outcome不要相差太大
✅ 价格歧视指数符合预期:
   - Anonymized → PDI ≈ 0
   - Identified + CP → PDI ≈ 0
   - Identified + CE → PDI > 0
✅ Gini系数在合理范围[0, 0.2]
✅ 社会福利 = CS + PS + IS（数值检验）
```

---

### Q4: 为什么`outcome`字段很多都是0.0？

**A**: `outcome`字段是向后兼容的旧格式，指向`expected_outcome`

但某些字段（如`acceptor_avg_utility`）在期望化时难以计算，因此设为0。

**建议**: 新代码使用`expected_outcome`和`sample_outcome`

---

### Q5: Common Preferences下价格为何相同？

**A**: 这是理论预测！

```
数据结构: w_i = θ for all i（所有人相同）

后验估计:
  消费者: μ_i^cons = E[θ | s_i, X]（包含私人信号s_i）
  生产者: μ_i^prod = E[θ | X]（参与者）或E[θ]（拒绝者）

关键洞察:
  虽然消费者有私人信号s_i，但由于大家估计的是同一个θ
  当参与者足够多时，X包含的信息主导
  因此μ_i^cons ≈ μ_j^cons for all i,j
  
定价:
  Identified: p_i = (μ_i^prod + c) / 2 ≈ 相同
  Anonymized: p统一

结论:
  CP下即使实名制也无法有效歧视（后验相近）
  这是论文的核心发现之一！
```

---

### Q6: 如何生成自定义配置的GT？

```python
from src.scenarios.scenario_c_social_data import (
    ScenarioCParams, generate_ground_truth
)
import json

# 自定义参数
params = ScenarioCParams(
    N=50,                           # 增加消费者数
    data_structure="common_experience",
    anonymization="identified",
    mu_theta=10.0,                  # 更高的平均支付意愿
    sigma_theta=2.0,                # 更大的不确定性
    sigma=1.5,                      # 更大的噪声
    m=2.5,                          # 更高的补偿
    tau_dist="normal",
    tau_mean=1.0,
    tau_std=0.5,
    participation_timing="ex_ante",
    seed=123                        # 不同的种子
)

# 生成GT
gt = generate_ground_truth(
    params,
    max_iter=50,
    num_mc_samples=100,
    num_outcome_samples=30
)

# 保存
with open("custom_gt.json", "w") as f:
    json.dump(gt, f, indent=2)

print(f"✅ 自定义GT已生成: r* = {gt['rational_participation_rate']:.2%}")
```

---

## 📊 输出指标速查表

### 核心指标（必看）

| 指标 | 字段路径 | 类型 | 范围 | 说明 |
|------|---------|------|------|------|
| **理性参与率** | `rational_participation_rate` | float | [0,1] | 固定点r* |
| **消费者剩余** | `expected_outcome.consumer_surplus` | float | ℝ | 期望CS |
| **生产者利润** | `expected_outcome.producer_profit` | float | ℝ+ | 期望PS |
| **社会福利** | `expected_outcome.social_welfare` | float | ℝ | 期望SW=CS+PS+IS |
| **基尼系数** | `expected_outcome.gini_coefficient` | float | [0,1] | 不平等程度 |
| **价格歧视** | `expected_outcome.price_discrimination_index` | float | ℝ+ | max(p)-min(p) |

### 诊断指标

| 指标 | 字段路径 | 用途 |
|------|---------|------|
| **收敛历史** | `r_history` | 检查固定点收敛 |
| **学习质量** | `sample_outcome.learning_quality_*` | 衡量数据外部性 |
| **分组效用** | `sample_outcome.acceptor_avg_utility` | 分析参与激励 |

### 细节数据（调试用）

| 数据 | 字段路径 | 维度 | 说明 |
|------|---------|------|------|
| **价格** | `sample_detailed_results.prices` | (N,) | 每人价格 |
| **需求** | `sample_detailed_results.quantities` | (N,) | 每人购买量 |
| **效用** | `sample_detailed_results.utilities` | (N,) | 每人效用 |
| **后验** | `sample_detailed_results.mu_consumers` | (N,) | 每人后验估计 |

---

## 🔗 相关文档

- **求解器结构**: `求解器Stackelberg结构分析.md`
- **参数配置**: `场景C配置参数说明.md`
- **评估器使用**: `docs/README_scenario_c.md`
- **论文解析**: `docs/论文解析_The_Economics_of_Social_Data.md`

---

## 📝 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-01-18 | 初始版本，完整解析GT输出结构 |

---

**文档作者**: AI Assistant  
**维护状态**: 活跃  
**反馈**: 如有疑问或建议，请提issue
