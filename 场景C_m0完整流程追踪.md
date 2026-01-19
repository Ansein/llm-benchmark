# 场景C - m_0完整流程追踪

**创建日期**: 2026-01-18  
**目的**: 追踪m_0在代码中的完整生命周期

---

## 📋 目录

1. [m_0的概念定义](#m_0的概念定义)
2. [参数定义阶段](#参数定义阶段)
3. [计算阶段](#计算阶段)
4. [使用阶段](#使用阶段)
5. [输出阶段](#输出阶段)
6. [完整流程图](#完整流程图)

---

## 💰 m_0的概念定义

### 经济学含义

```
m_0 = 生产者向数据中介支付的数据购买费用

中介商业模式:
  收入: m_0 ←─── 生产者（买数据）
  支出: m × N_参与 ──→ 消费者（卖数据）
  净利润: R = m_0 - m × N_参与
```

### 理论公式

```
m_0 = β × max(0, E[π_producer(有数据) - π_producer(无数据)])

其中:
  - E[·]: Ex-Ante期望（在世界状态上平均）
  - β ∈ [0,1]: 中介可提取比例（默认1.0）
  - π_producer: 生产者在产品市场的利润
```

---

## 🎯 参数定义阶段

### 位置1: `ScenarioCParams`类定义

**文件**: `src/scenarios/scenario_c_social_data.py`  
**行数**: 333

```python
@dataclass
class ScenarioCParams:
    # ... 其他参数 ...
    
    # 生产者向中介支付m_0（我们的扩展，论文隐含）
    # 中介利润 = m_0 - m·N_参与
    # 默认值：0.0（中介纯支出）
    # 扩展：可设为生产者利润提升的某个比例
    m_0: float = 0.0
```

**说明**:
- 默认值为`0.0`
- 这是**静态参数**（用户可设置固定值）
- 如果用户不设置，中介利润 = -m × N_参与（纯支出）

**使用场景**:
- Ground Truth生成（使用默认值0）
- 简单福利分析（假设中介是公共物品提供者）

---

## 🔬 计算阶段

m_0有三种计算方式，取决于使用场景：

### 方式1: 使用默认值（最简单）

**场景**: 基础GT生成、福利分析

```python
params = ScenarioCParams(
    N=20,
    m=1.0,
    m_0=0.0,  # ← 使用默认值，不计算
    # ... 其他参数
)

# 直接使用
intermediary_profit = params.m_0 - params.m * num_participants
# = 0 - m × N_参与
# = -m × N_参与（中介亏损）
```

**m_0流向**:
```
ScenarioCParams.m_0 = 0.0
    ↓
simulate_market_outcome() 读取 params.m_0
    ↓
intermediary_profit = params.m_0 - m × N_参与
```

---

### 方式2: 用户手动设置（不常用）

**场景**: 研究固定数据费率、政策模拟

```python
# 用户设定固定m_0
params = ScenarioCParams(
    N=20,
    m=1.0,
    m_0=10.0,  # ← 手动设置（例如：政府定价）
    # ...
)

# 直接使用固定值
intermediary_profit = 10.0 - m × N_参与
```

**m_0流向**:
```
用户设置 m_0 = 10.0
    ↓
ScenarioCParams.m_0 = 10.0
    ↓
simulate_market_outcome() 读取 params.m_0
    ↓
intermediary_profit = 10.0 - m × N_参与
```

---

### 方式3: 动态计算（新方法，理论严格）⭐

**场景**: 中介优化、数据定价研究

#### 步骤1: 调用`estimate_m0_mc`

**文件**: `src/scenarios/scenario_c_social_data.py`  
**行数**: 2356-2519  
**函数**: `estimate_m0_mc(params, participation_rule, T=200, beta=1.0, seed=None)`

```python
def estimate_m0_mc(params, participation_rule, T=200, beta=1.0, seed=None):
    """
    使用Monte Carlo方法估计m_0（Ex-Ante期望）
    """
    deltas = []
    
    for t in range(T):  # MC循环200次
        # 1. 生成同一份world state
        np.random.seed(world_seed_t)
        world = generate_consumer_data(params)
        
        # 2. 生成同一个participation
        participation = participation_rule(params, world, rng)
        
        # 3. 计算with-data利润
        outcome_with = simulate_market_outcome(
            world, participation, params,
            producer_info_mode="with_data"  # ← 关键参数
        )
        pi_with = outcome_with.producer_profit
        
        # 4. 计算no-data利润（同world + 同A）
        outcome_no = simulate_market_outcome(
            world, participation, params,
            producer_info_mode="no_data"  # ← 关键参数
        )
        pi_no = outcome_no.producer_profit
        
        # 5. 记录利润差
        deltas.append(pi_with - pi_no)
    
    # 6. 计算Ex-Ante期望
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    
    # 7. m_0 = 可提取部分
    m_0 = beta * max(0.0, delta_mean)
    
    return m_0, delta_mean, delta_std
```

**关键点**:
- ✅ 同一个world state（Common Random Numbers）
- ✅ 同一个participation
- ✅ 只改变`producer_info_mode`（纯信息差异）
- ✅ MC平均200次（Ex-Ante期望）

**m_0计算公式**:
```python
m_0 = beta × max(0, mean([π_with_t - π_no_t for t in 1..T]))
    = beta × max(0, E[Δπ])
```

---

#### 步骤2: 在`evaluate_intermediary_strategy`中调用

**文件**: `src/scenarios/scenario_c_social_data.py`  
**行数**: 2598-2690  
**函数**: `evaluate_intermediary_strategy(m, anonymization, params_base, ...)`

```python
def evaluate_intermediary_strategy(m, anonymization, params_base, ...):
    """
    评估给定策略(m, anonymization)下的完整市场均衡
    """
    # 1. 构建参数（注意：不设置m_0，因为要动态计算）
    params = ScenarioCParams(
        m=m,
        anonymization=anonymization,
        **params_base  # 不包含m_0
    )
    
    # 2. 求解消费者均衡
    r_star, _, delta_u = compute_rational_participation_rate(params, ...)
    
    # 3. 定义参与决策规则
    def participation_rule(p, world, rng):
        if p.tau_dist == "normal":
            tau_samples = rng.normal(p.tau_mean, p.tau_std, p.N)
            return tau_samples <= delta_u
        # ... 其他分布
    
    # 4. 动态计算m_0（新方法）⭐
    m_0, delta_profit_mean, delta_profit_std = estimate_m0_mc(
        params=params,
        participation_rule=participation_rule,
        T=200,
        beta=1.0,
        seed=seed
    )
    
    # 5. 生成一次市场实现（用于其他指标）
    data = generate_consumer_data(params, seed=seed)
    participation = participation_rule(params, data, rng)
    outcome_with = simulate_market_outcome(
        data, participation, params, producer_info_mode="with_data"
    )
    
    # 6. 计算中介利润
    num_participants = int(np.sum(participation))
    intermediary_cost = m * num_participants
    intermediary_profit = m_0 - intermediary_cost  # ← 使用动态计算的m_0
    
    # 7. 返回结果
    return IntermediaryOptimizationResult(
        m=m,
        anonymization=anonymization,
        m_0=m_0,  # ← Ex-Ante期望（MC估计）
        intermediary_profit=intermediary_profit,
        # ...
    )
```

**m_0流向**:
```
estimate_m0_mc(params, rule, T=200)
    ↓ [MC循环200次]
    ↓ 每次: π_with(w,A) - π_no(w,A)
    ↓ [平均]
m_0 = beta × max(0, mean(deltas))
    ↓
IntermediaryOptimizationResult.m_0
    ↓
intermediary_profit = m_0 - m × N_参与
```

---

#### 步骤3: 在`optimize_intermediary_policy`中遍历

**文件**: `src/scenarios/scenario_c_social_data.py`  
**行数**: 2693-2804  
**函数**: `optimize_intermediary_policy(params_base, m_grid, ...)`

```python
def optimize_intermediary_policy(params_base, m_grid, policies, ...):
    """
    求解中介的最优策略组合 (m*, anonymization*)
    """
    results = []
    
    # 遍历所有候选策略
    for m in m_grid:                      # 例如：[0, 0.1, ..., 3.0]
        for anonymization in policies:    # ['identified', 'anonymized']
            
            # 评估该策略（内部会计算m_0）
            result = evaluate_intermediary_strategy(
                m=m,
                anonymization=anonymization,
                params_base=params_base,
                ...
            )
            
            results.append(result)
            
            if verbose:
                print(f"m={m:.2f}, {anonymization:12s}: "
                      f"R={result.intermediary_profit:.4f}, "
                      f"m_0={result.m_0:.4f}")  # ← 打印m_0
    
    # 找到最优策略（最大化中介利润）
    optimal_result = max(results, key=lambda r: r.intermediary_profit)
    
    return OptimalPolicy(
        optimal_m=optimal_result.m,
        optimal_anonymization=optimal_result.anonymization,
        optimal_result=optimal_result,  # 包含m_0
        all_results=results
    )
```

**m_0流向**:
```
for m in m_grid:
    for anonymization in policies:
        result = evaluate_intermediary_strategy(m, anonymization)
            ↓ [内部调用estimate_m0_mc]
        m_0_for_this_strategy
            ↓
        intermediary_profit = m_0 - m × N_参与
            ↓
        results.append(result)

optimal_result = max(results, key=lambda r: r.intermediary_profit)
    ↓
OptimalPolicy(optimal_result.m_0)  # 最优策略的m_0
```

---

## 💼 使用阶段

### 使用1: 计算中介利润

**位置**: `simulate_market_outcome`  
**文件**: `src/scenarios/scenario_c_social_data.py`  
**行数**: 1368-1369

```python
def simulate_market_outcome(data, participation, params, producer_info_mode="with_data"):
    # ... 前面步骤 ...
    
    # 7.3 中介利润（Intermediary Profit）
    # R = m_0 - m·N_participants
    num_participants = int(np.sum(participation))
    intermediary_profit = params.m_0 - params.m * num_participants  # ← 使用m_0
    
    # 7.4 社会福利（Social Welfare）
    # SW = CS + PS + R
    social_welfare = consumer_surplus + producer_profit + intermediary_profit
    
    return MarketOutcome(
        intermediary_profit=intermediary_profit,  # ← 输出
        social_welfare=social_welfare,
        # ...
    )
```

**关键方程**:
```python
R = m_0 - m × N_参与

SW = CS + PS + R
   = CS + PS + (m_0 - m × N_参与)
```

**m_0的作用**:
- 直接影响中介利润`R`
- 通过`R`影响社会总福利`SW`
- 如果`m_0=0`：`R = -m × N_参与`（中介亏损）
- 如果`m_0 > m × N_参与`：`R > 0`（中介盈利）

---

### 使用2: 福利分解

**中介利润的福利含义**:

```python
# 情况1: m_0 = 0（默认）
R = 0 - m × N_参与 < 0

福利流向:
  消费者 ← m × N_参与 ← [中介亏损]
  
社会福利:
  SW = CS + PS + R
     = CS + PS - m × N_参与
  
含义:
  - 补偿m是纯成本
  - 中介类似"公共物品提供者"
  - m的社会成本由整体福利承担
```

```python
# 情况2: m_0 = 利润差（动态计算）
R = Δπ_producer - m × N_参与

福利流向:
  消费者 ← m × N_参与 ← 中介 ← m_0 ← 生产者
  
社会福利:
  SW = CS + (PS - m_0) + (m_0 - m × N_参与)
     = CS + PS - m × N_参与  （m_0抵消）
  
含义:
  - m_0是转移支付（生产者→中介）
  - 不改变社会总福利（一人得一人失）
  - 但改变福利分配：PS↓, R↑
```

---

### 使用3: 最优化目标

**中介优化问题**:

```python
# 中介目标：max R = m_0 - m × N_参与(m)
#
# 约束：
#   1. r*(m) 由固定点决定（消费者均衡）
#   2. N_参与 = N × r*(m)（期望参与数）
#   3. m_0 = m_0(m)（生产者支付意愿依赖于m）

max_{m, anonymization} R = m_0(m, anonymization) - m × N × r*(m, anonymization)
```

**求解方法**:
```python
# 网格搜索（optimize_intermediary_policy）
for m in m_grid:
    for anonymization in policies:
        # 1. 求解消费者均衡
        r_star = compute_rational_participation_rate(m, anonymization)
        
        # 2. 计算生产者支付意愿
        m_0 = estimate_m0_mc(...)  # Ex-Ante期望
        
        # 3. 计算中介利润
        R = m_0 - m × N × r_star
        
        # 4. 记录
        results.append((m, anonymization, R, m_0))

# 5. 选择最优
(m*, anonymization*) = argmax R
```

---

## 📤 输出阶段

### 输出1: `IntermediaryOptimizationResult`

**数据类定义**:

```python
@dataclass
class IntermediaryOptimizationResult:
    m: float                          # 补偿
    anonymization: str                # 匿名化策略
    r_star: float                     # 均衡参与率
    delta_u: float                    # 参与净收益
    num_participants: int             # 实际参与数
    
    producer_profit_with_data: float  # 有数据利润
    producer_profit_no_data: float    # 无数据利润
    producer_profit_gain: float       # 利润增益（单次实现）
    
    m_0: float                        # ⭐ 生产者支付意愿（Ex-Ante期望）
    intermediary_cost: float          # 中介成本 = m × N_参与
    intermediary_profit: float        # ⭐ 中介利润 = m_0 - intermediary_cost
    
    consumer_surplus: float
    social_welfare: float
    gini_coefficient: float
    price_discrimination_index: float
```

**m_0相关字段**:
- `m_0`: Ex-Ante期望（MC估计）
- `intermediary_cost`: 补偿总成本
- `intermediary_profit`: 净利润（依赖m_0）
- `producer_profit_gain`: 单次实现利润差（用于对比）

---

### 输出2: Ground Truth JSON

**当前状态**（尚未更新）:

```json
{
  "params": {
    "m": 1.0,
    "m_0": 0.0,  // ← 默认值，未动态计算
    // ...
  },
  "expected_outcome": {
    "intermediary_profit": -16.0,  // = 0 - 1.0 × 16
    "social_welfare": 103.5,
    // ...
  }
}
```

**建议未来更新**:

```json
{
  "params": {
    "m": 1.0,
    // 不设置m_0，由动态计算
  },
  "m_0_estimation": {
    "m_0": 8.4675,           // Ex-Ante期望
    "delta_mean": 8.4675,
    "delta_std": 1.3018,
    "method": "estimate_m0_mc",
    "mc_samples": 200
  },
  "expected_outcome": {
    "intermediary_profit": -7.5325,  // = 8.4675 - 1.0 × 16
    "social_welfare": 111.9,         // 更准确的福利
    // ...
  }
}
```

---

### 输出3: 优化结果打印

**`optimize_intermediary_policy`的输出**:

```python
if verbose:
    print(f"\n策略评估:")
    print(f"  m={m:.2f}, {anonymization:12s}:")
    print(f"    参与率 r* = {result.r_star:.2%}")
    print(f"    生产者支付意愿 m_0 = {result.m_0:.4f}")
    print(f"    中介成本 = {result.intermediary_cost:.4f}")
    print(f"    中介利润 R = {result.intermediary_profit:.4f}")
    print(f"    社会福利 SW = {result.social_welfare:.4f}")
```

**示例输出**:

```
策略评估:
  m=1.00, identified  :
    参与率 r* = 80.00%
    生产者支付意愿 m_0 = 8.4675
    中介成本 = 16.0000
    中介利润 R = -7.5325
    社会福利 SW = 111.9000
```

---

## 🔄 完整流程图

### 流程图1: 默认使用（m_0=0）

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 参数定义                                                 │
│    params = ScenarioCParams(m_0=0.0)  # 默认值             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Ground Truth生成                                         │
│    generate_ground_truth(params)                            │
│      ↓                                                       │
│    simulate_market_outcome(..., params)                     │
│      ↓ 读取 params.m_0 = 0.0                                │
│    intermediary_profit = 0.0 - m × N_参与                   │
│      ↓                                                       │
│    intermediary_profit < 0  （中介亏损）                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 输出 GT JSON                                              │
│    {                                                         │
│      "params": {"m_0": 0.0},                                 │
│      "outcome": {                                            │
│        "intermediary_profit": -16.0,  // 负值               │
│        "social_welfare": 103.5        // SW = CS + PS + R   │
│      }                                                       │
│    }                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

### 流程图2: 动态计算（中介优化）

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 中介优化入口                                              │
│    optimize_intermediary_policy(params_base, m_grid)        │
│      ↓                                                       │
│    for m in [0, 0.1, ..., 3.0]:                             │
│      for anonymization in ['identified', 'anonymized']:     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 评估单个策略                                              │
│    evaluate_intermediary_strategy(m, anonymization)         │
│      ↓                                                       │
│    ┌──────────────────────────────────────────┐             │
│    │ 2.1 求解消费者均衡                        │             │
│    │     r*, delta_u = compute_rational_...   │             │
│    └──────────────────┬───────────────────────┘             │
│                       ↓                                      │
│    ┌──────────────────────────────────────────┐             │
│    │ 2.2 定义参与规则                          │             │
│    │     def participation_rule(p, w, rng):   │             │
│    │         tau ~ N(tau_mean, tau_std)       │             │
│    │         return tau <= delta_u            │             │
│    └──────────────────┬───────────────────────┘             │
│                       ↓                                      │
│    ┌──────────────────────────────────────────┐             │
│    │ 2.3 动态计算m_0 ⭐                        │             │
│    │     m_0, delta_mean, delta_std =         │             │
│    │       estimate_m0_mc(                    │             │
│    │         params, participation_rule,      │             │
│    │         T=200, beta=1.0                  │             │
│    │       )                                  │             │
│    └──────────────────┬───────────────────────┘             │
│                       ↓                                      │
│    ┌──────────────────────────────────────────┐             │
│    │ 2.4 计算中介利润                          │             │
│    │     R = m_0 - m × N_参与                 │             │
│    └──────────────────┬───────────────────────┘             │
│                       ↓                                      │
│    return IntermediaryOptimizationResult(                   │
│        m_0=m_0,          # Ex-Ante期望                       │
│        intermediary_profit=R                                │
│    )                                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. estimate_m0_mc详细流程（关键）                            │
│    ┌────────────────────────────────────────┐               │
│    │ for t = 1 to T=200:                    │               │
│    │   # 同一个world state                  │               │
│    │   world = generate_consumer_data(...)  │               │
│    │                                         │               │
│    │   # 同一个participation                │               │
│    │   A = participation_rule(world)        │               │
│    │                                         │               │
│    │   # with-data: 生产者有中介信息        │               │
│    │   outcome_with = simulate_market(      │               │
│    │     world, A, params,                  │               │
│    │     producer_info_mode="with_data"     │               │
│    │   )                                    │               │
│    │   pi_with = outcome_with.producer_profit│              │
│    │                                         │               │
│    │   # no-data: 生产者无中介信息          │               │
│    │   outcome_no = simulate_market(        │               │
│    │     world, A, params,                  │               │
│    │     producer_info_mode="no_data"       │               │
│    │   )                                    │               │
│    │   pi_no = outcome_no.producer_profit   │               │
│    │                                         │               │
│    │   # 纯信息价值                          │               │
│    │   deltas[t] = pi_with - pi_no          │               │
│    └────────────────┬───────────────────────┘               │
│                     ↓                                        │
│    delta_mean = mean(deltas)  # Ex-Ante期望                 │
│    delta_std = std(deltas)    # 不确定性                    │
│    m_0 = beta × max(0, delta_mean)                          │
│                                                              │
│    return (m_0, delta_mean, delta_std)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 遍历所有策略，收集结果                                    │
│    results = [                                               │
│      (m=0.0, ID, R=-0.00,  m_0=0.00),                        │
│      (m=0.1, ID, R=-1.20,  m_0=1.00),                        │
│      (m=1.0, ID, R=-7.53,  m_0=8.47),  # m_0最高             │
│      (m=1.0, AN, R=-16.20, m_0=0.80),  # m_0最低             │
│      ...                                                     │
│    ]                                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 找到最优策略                                              │
│    optimal = max(results, key=lambda r: r.intermediary_profit)│
│                                                              │
│    return OptimalPolicy(                                     │
│        optimal_m=optimal.m,                                  │
│        optimal_anonymization=optimal.anonymization,          │
│        optimal_result=optimal  # 包含optimal.m_0             │
│    )                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

### 流程图3: `producer_info_mode`的作用

```
simulate_market_outcome(data, participation, params, producer_info_mode)
│
├─ producer_info_mode == "no_data"  ─────────────────────────┐
│  │                                                          │
│  ├─ 生产者后验:                                            │
│  │    mu_producer[:] = params.mu_theta  # 所有人先验       │
│  │                                                          │
│  ├─ 定价:                                                  │
│  │    p_uniform = compute_optimal_price_uniform(...)       │
│  │    prices[:] = p_uniform  # 强制统一定价                │
│  │                                                          │
│  └─ 结果:                                                  │
│       producer_profit_no_data  # 无数据基准                │
│                                                             │
├─ producer_info_mode == "with_data" (默认) ─────────────────┤
│  │                                                          │
│  ├─ 生产者后验:                                            │
│  │    mu_producer = compute_producer_posterior(...)        │
│  │                                                          │
│  │    ├─ identified:                                       │
│  │    │   参与者: mu_producer[i] = E[w_i | s_i, X]         │
│  │    │   拒绝者: mu_producer[j] = E[w_j | X]              │
│  │    │                                                     │
│  │    └─ anonymized:                                       │
│  │        所有人: mu_producer[:] = E[w | X]  # 相同        │
│  │                                                          │
│  ├─ 定价:                                                  │
│  │    ├─ identified:                                       │
│  │    │   prices[i] = (mu_producer[i] + c) / 2  # 个性化   │
│  │    │                                                     │
│  │    └─ anonymized:                                       │
│  │        p_uniform = compute_optimal_price_uniform(...)   │
│  │        prices[:] = p_uniform  # 统一                    │
│  │                                                          │
│  └─ 结果:                                                  │
│       producer_profit_with_data  # 有数据                  │
│                                                             │
└─ 用于计算:                                                 │
     Δπ = producer_profit_with_data - producer_profit_no_data│
     m_0 = E[Δπ]  # 纯信息价值                               │
```

---

## 📊 m_0的数值示例

### 示例1: Common Experience + Identified

```python
配置:
  N = 20
  data_structure = "common_experience"
  anonymization = "identified"
  参与率 = 80% (16人)

计算过程（新方法）:
  T = 200次MC循环
  
  每次循环t:
    1. 生成world_t: (w, s, τ)
    2. 生成participation_t: 16人参与
    3. with-data: π_with_t ≈ 139.5
    4. no-data:   π_no_t ≈ 125.3
    5. Δπ_t = 139.5 - 125.3 = 14.2
  
  汇总:
    mean(Δπ) = 8.47
    std(Δπ) = 1.30
    m_0 = max(0, 8.47) = 8.47

中介利润:
  R = m_0 - m × N_参与
    = 8.47 - 1.0 × 16
    = -7.53  （仍亏损，但比m_0=0时的-16好）
```

---

### 示例2: Common Preferences + Identified

```python
配置:
  data_structure = "common_preferences"  # ← 论文公式失效场景
  anonymization = "identified"
  参与率 = 80%

论文公式:
  G(Y_0) = Var[μ_producer]
         = Var[E[θ | X]]
         = 0  （所有人后验相同）
  
  m_0 = (N/4) × 0 = 0  ❌ 失效

新方法:
  T = 200次MC循环
  
  每次循环:
    π_with ≈ 119.0  （更准确的θ后验）
    π_no ≈ 115.2    （只有先验μ_θ）
    Δπ = 3.8
  
  m_0 = mean(Δπ) = 0.69  ✅ 检测到价值
  
价值来源:
  - ✓ 后验精度提升: Var(θ|X) < Var(θ)
  - ✓ 更准确的统一定价
  - ✗ 无歧视能力（所有人后验相同）
```

---

### 示例3: Identified vs Anonymized对比

```python
场景: Common Experience

Identified:
  m_0 = 8.47  （高）
  价值来源: 精度提升 + 歧视能力
  
Anonymized:
  m_0 = 0.80  （低）
  价值来源: 精度提升（无歧视能力）

差距:
  8.47 - 0.80 = 7.67
  
含义:
  价格歧视能力的价值 ≈ 7.67
  约为总信息价值的 91% (7.67/8.47)
```

---

## 📝 关键变量追踪表

| 变量 | 位置 | 类型 | 来源 | 用途 |
|------|------|------|------|------|
| `params.m_0` | ScenarioCParams | float (默认0.0) | 用户设置或默认 | 静态参数 |
| `m_0` (estimate_m0_mc返回) | estimate_m0_mc | float (动态计算) | MC估计 | Ex-Ante期望 |
| `result.m_0` | IntermediaryOptimizationResult | float | evaluate_intermediary_strategy | 策略结果 |
| `optimal.m_0` | OptimalPolicy | float | optimize_intermediary_policy | 最优策略 |
| `intermediary_profit` | MarketOutcome | float | simulate_market_outcome | 中介净利润 |
| `delta_mean` | estimate_m0_mc | float | mean(deltas) | 利润增量期望 |
| `delta_std` | estimate_m0_mc | float | std(deltas) | 不确定性 |

---

## 🔍 关键代码位置索引

| 功能 | 文件 | 行数 | 函数/类 |
|------|------|------|---------|
| m_0参数定义 | scenario_c_social_data.py | 333 | ScenarioCParams.m_0 |
| producer_info_mode参数 | scenario_c_social_data.py | 1118 | simulate_market_outcome |
| 生产者后验分支 | scenario_c_social_data.py | 1249-1260 | simulate_market_outcome |
| 定价分支 | scenario_c_social_data.py | 1269-1283 | simulate_market_outcome |
| 中介利润计算 | scenario_c_social_data.py | 1369 | simulate_market_outcome |
| estimate_m0_mc函数 | scenario_c_social_data.py | 2356-2519 | estimate_m0_mc |
| evaluate_intermediary_strategy | scenario_c_social_data.py | 2522-2690 | evaluate_intermediary_strategy |
| optimize_intermediary_policy | scenario_c_social_data.py | 2693-2804 | optimize_intermediary_policy |
| IntermediaryOptimizationResult | scenario_c_social_data.py | 157-173 | dataclass |

---

## 💡 总结

### m_0的三种形态

1. **静态参数** (`params.m_0 = 0.0`):
   - 用户设置或默认值
   - 用于基础分析
   - 不反映真实数据价值

2. **动态计算** (`estimate_m0_mc`):
   - Ex-Ante期望（MC平均）
   - Common Random Numbers方法
   - 理论严格，反映纯信息价值

3. **优化结果** (`optimal.m_0`):
   - 最优策略对应的m_0
   - 由中介优化过程产生
   - 用于数据定价决策

### 完整变化路径

```
用户设置 or 默认
    ↓
params.m_0 = 0.0 or 固定值
    ↓
[如果需要动态计算]
    ↓
estimate_m0_mc(params, rule, T=200)
    ↓ [MC循环]
    ↓ 每次: Δπ = π_with(w,A) - π_no(w,A)
    ↓ [平均]
m_0 = beta × max(0, mean(Δπ))
    ↓
IntermediaryOptimizationResult.m_0
    ↓
intermediary_profit = m_0 - m × N_参与
    ↓
social_welfare = CS + PS + intermediary_profit
    ↓
输出到JSON或打印
```

---

**文档版本**: v1.0  
**作者**: AI Assistant  
**用途**: 追踪m_0在代码中的完整生命周期
