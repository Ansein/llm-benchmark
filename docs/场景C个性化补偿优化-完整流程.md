# 场景C：个性化补偿优化完整流程

## 📋 目录

1. [优化目标](#优化目标)
2. [优化架构](#优化架构)
3. [详细流程](#详细流程)
4. [关键函数说明](#关键函数说明)
5. [数学原理](#数学原理)
6. [算法细节](#算法细节)

---

## 🎯 优化目标

### 总体目标

求解中介的最优策略 `(m*, π*)`，使得中介利润最大化：

```
max_{m, π} R(m, π) = m_0(m, π) - E[Σ m_i · a_i]
s.t. R > 0  (利润约束)
```

其中：
- `m = (m_1, ..., m_N)`：对N个消费者的个性化补偿（**决策变量**）
- `π ∈ {identified, anonymized}`：匿名化策略（**离散选择**）
- `m_0`：生产者对数据的支付意愿（**内生变量**，依赖于m和π）
- `a_i ∈ {0,1}`：消费者i的参与决策（**内生变量**，依赖于m）
- `R`：中介利润

### 优化难点

1. **高维连续优化**：N=20维的m向量
2. **内生变量**：m_0和a_i都依赖于m，需要嵌套求解
3. **固定点迭代**：a_i的分布通过固定点方程求解
4. **MC估计**：m_0通过蒙特卡洛模拟估计
5. **离散选择**：需要对比identified和anonymized策略

---

## 🏗️ 优化架构

```
优化入口: optimize_intermediary_policy_personalized()
    │
    ├─► 策略循环: for π in [identified, anonymized]
    │       │
    │       └─► 混合优化: optimize_m_vector_scipy_hybrid()
    │               │
    │               ├─► 第1步：网格搜索（粗搜索）
    │               │       └─► evaluate_m_vector_profit(m_uniform)
    │               │               └─► 计算R(m_uniform, π)
    │               │
    │               └─► 第2步：L-BFGS-B（精细优化）
    │                       └─► evaluate_m_vector_profit(m_vector)
    │                               └─► 计算R(m_vector, π)
    │
    └─► 策略选择: max_{π} R(m*, π)
            └─► 应用利润约束: R > 0
```

---

## 🔄 详细流程

### 第0步：初始化

**输入参数**：
```python
params_base = {
    'N': 20,                    # 消费者数量
    'preference_structure': 'common_preferences',
    'tau_dist': 'uniform',      # 隐私成本分布
    'tau_mean': 0.75,
    'tau_std': 0.15,
    # ... 其他参数
}
```

**优化配置**：
```python
optimization_method = 'hybrid'       # 混合优化
m_bounds = (0.0, 3.0)               # m_i的取值范围
grid_size = 21                      # 网格搜索点数
num_mc_samples = 30                 # MC采样数
max_iter = 5                        # 固定点迭代上限
```

---

### 第1步：策略循环

**目的**：对比identified和anonymized两种策略

```python
for policy in ['identified', 'anonymized']:
    # 为当前策略优化m向量
    m_star[policy], R_star[policy] = optimize_m_vector(policy)
```

**每个策略独立优化**，因为：
- 不同的π会影响消费者的效用
- identified：消费者担心隐私泄露
- anonymized：消费者隐私得到保护，效用更高

---

### 第2步：混合优化 - 网格搜索（初始化）

**目的**：快速找到一个好的起始点，避免L-BFGS-B陷入局部最优

**方法**：在统一补偿空间粗搜索
```python
m_grid = np.linspace(m_bounds[0], m_bounds[1], grid_size)
# 例如：[0.0, 0.15, 0.30, ..., 2.85, 3.0]

best_m_uniform = None
best_profit = -∞

for m_val in m_grid:
    m_vector = np.full(N, m_val)  # 所有人相同的补偿
    profit = evaluate_m_vector_profit(m_vector, policy)
    
    if profit > best_profit:
        best_profit = profit
        best_m_uniform = m_val
```

**输出**：
- `best_m_uniform`：网格搜索找到的最优统一补偿值
- 例如：`m_uniform = 0.75`

**性能优化**：
- 尝试并行计算（Windows下可能失败，fallback到串行）
- 21个点，每个点需要1次`evaluate_m_vector_profit`调用

---

### 第3步：混合优化 - L-BFGS-B（精细优化）

**目的**：从统一补偿出发，探索个性化补偿的可能性

**初始点**：
```python
m_init = np.full(N, best_m_uniform)
# 例如：[0.75, 0.75, ..., 0.75]
```

**优化器**：
```python
from scipy.optimize import minimize

result = minimize(
    fun=objective,               # 目标函数（负利润）
    x0=m_init,                   # 初始点
    method='L-BFGS-B',           # 拟牛顿法
    bounds=[(m_bounds[0], m_bounds[1])] * N,  # 每个m_i的范围
    options={'maxiter': 100}
)
```

**目标函数**：
```python
def objective(m_vector):
    profit = evaluate_m_vector_profit(m_vector, policy)
    return -profit  # 最大化profit = 最小化-profit
```

**优化过程**：
- L-BFGS-B通过数值梯度评估每个m_i的边际收益
- 如果增加m_i能带来净收益（更高的m_0减去更高的成本），则增加m_i
- 如果个性化没有优势，m_vector会保持接近统一
- 迭代直到收敛（梯度接近0）

**输出**：
- `m_star_vector`：最优补偿向量
- `profit_star`：对应的最大利润

---

### 第4步：利润评估 `evaluate_m_vector_profit(m_vector, policy)`

**这是优化的核心！** 每次优化器尝试新的m_vector时，都会调用此函数评估利润。

#### 4.1 构建参数

```python
params = params_base.copy()
params['m'] = m_vector              # 设置当前尝试的补偿向量
params['anonymization'] = policy    # 设置当前策略
params = ScenarioCParams(**params)  # 构建参数对象
```

#### 4.2 求解消费者均衡（固定点迭代）

```python
r_star, _, _, delta_u_vector, p_vector = compute_rational_participation_rate(
    params,
    max_iter=5,
    tol=1e-3,
    num_mc_samples=30,
    compute_per_consumer=True  # ✅ 关键：为每个消费者单独计算
)
```

**这一步在优化什么？**
- **不是优化**，而是**求解均衡**
- 给定m向量，求解理性消费者的参与决策
- 返回：
  - `delta_u_vector[i]`：消费者i的效用差 ΔU_i = U_i(参与) - U_i(不参与)
  - `p_vector[i]`：消费者i的参与概率 p_i = P(τ_i < ΔU_i)
  - `r_star`：总体参与率 r* = mean(p_vector)

**详见第5步**：固定点迭代的详细过程

#### 4.3 定义参与决策规则

```python
def participation_rule(params, world, rng):
    # 基于ΔU_i和τ_i的分布，决定每个消费者是否参与
    if params.tau_dist == "uniform":
        tau_low = params.tau_mean - sqrt(3) * params.tau_std
        tau_high = params.tau_mean + sqrt(3) * params.tau_std
        tau_samples = rng.uniform(tau_low, tau_high, N)
        return tau_samples <= delta_u_vector
    # ... 其他分布
```

#### 4.4 估计生产者支付意愿 m_0

```python
m_0, _, _, e_num_participants = estimate_m0_mc(
    params=params,
    participation_rule=participation_rule,
    T=100,              # MC采样次数
    beta=1.0,           # 折扣因子
    seed=42
)
```

**这一步在计算什么？**
- 生产者对数据的支付意愿
- 通过蒙特卡洛模拟估计：
  ```
  m_0 = β × E[π_with_data - π_no_data]
  ```
- π_with_data：使用消费者数据后的利润
- π_no_data：不使用数据的利润

**详见第6步**：MC估计的详细过程

#### 4.5 计算中介成本

```python
# ✅ 使用个性化参与概率
intermediary_cost = np.sum(m_vector * p_vector)
# = m_1*p_1 + m_2*p_2 + ... + m_N*p_N
```

**为什么这样计算？**
- E[Σ m_i · a_i] = Σ m_i · E[a_i] = Σ m_i · p_i
- 每个消费者的期望成本 = 补偿 × 参与概率

#### 4.6 计算利润

```python
intermediary_profit = m_0 - intermediary_cost
return intermediary_profit
```

**返回给优化器**：
- 如果返回值高，优化器会朝这个方向继续
- 如果返回值低，优化器会调整m_vector

---

### 第5步：固定点迭代 `compute_rational_participation_rate()`

**目的**：求解理性参与率 r* 和个性化效用差 ΔU_i

**为什么需要固定点迭代？**
- 消费者的效用依赖于**其他人的参与率** r
- 但r又依赖于**每个消费者的决策**
- 形成循环依赖，需要迭代求解

#### 5.1 初始化

```python
r = 0.5  # 初始参与率
delta_u_vector = np.zeros(N)
p_vector = np.zeros(N)
```

#### 5.2 迭代计算（最多5次）

```python
for iteration in range(max_iter=5):
    # 为每个消费者i单独计算ΔU_i
    for i in range(N):
        # 计算"参与"的期望效用
        U_accept_i = compute_expected_utility_ex_ante(
            consumer_id=i,
            participates=True,
            others_participation_rate=r,  # 假设其他人参与率为r
            params=params
        )
        
        # 计算"不参与"的期望效用
        U_reject_i = compute_expected_utility_ex_ante(
            consumer_id=i,
            participates=False,
            others_participation_rate=r,
            params=params
        )
        
        # 效用差（包含m_i的补偿）
        delta_u_vector[i] = U_accept_i - U_reject_i
    
    # 根据ΔU_i和τ分布，计算每个消费者的参与概率
    if params.tau_dist == "uniform":
        tau_low = tau_mean - sqrt(3) * tau_std
        tau_high = tau_mean + sqrt(3) * tau_std
        # p_i = P(τ_i < ΔU_i)，τ_i ~ Uniform[tau_low, tau_high]
        p_vector = np.clip((delta_u_vector - tau_low) / (tau_high - tau_low), 0, 1)
    
    # 新的参与率
    r_new = np.mean(p_vector)
    
    # 检查收敛
    if abs(r_new - r) < tol:
        return r_new, history, mean(delta_u_vector), delta_u_vector, p_vector
    
    # 平滑更新（避免震荡）
    r = 0.6 * r_new + 0.4 * r
```

#### 5.3 期望效用计算（MC估计）

```python
def compute_expected_utility_ex_ante(consumer_id, participates, others_participation_rate, params):
    """
    估计消费者i的期望效用（对所有随机性取期望）
    """
    total_utility = 0.0
    
    # 采样多个世界状态
    for _ in range(num_world_samples):
        world = sample_world_state()  # 采样 (w, s)
        
        # 采样多个市场参与配置
        for _ in range(num_market_samples):
            # 其他消费者的参与决策（根据r采样）
            others_participation = sample_participation(N-1, others_participation_rate)
            
            # 构建完整的参与配置
            participation = construct_participation(consumer_id, participates, others_participation)
            
            # 模拟市场结果
            outcome = simulate_market_outcome(world, participation, params)
            
            # 消费者i的效用
            utility_i = outcome['utilities'][consumer_id]
            total_utility += utility_i
    
    # 返回平均效用
    return total_utility / (num_world_samples * num_market_samples)
```

**关键点**：
- 每个消费者i的ΔU_i(m_i)是**单独计算**的
- 考虑了m_i对消费者i效用的影响
- 考虑了其他人参与率r对i的外部性影响

---

### 第6步：估计m_0 `estimate_m0_mc()`

**目的**：计算生产者对数据的支付意愿

**方法**：蒙特卡洛估计
```
m_0 = β × E[π_with_data - π_no_data]
```

#### 6.1 采样循环（T=100次）

```python
profit_diff_samples = []

for t in range(T=100):
    # 1. 采样世界状态
    world = sample_world_state()  # (w, s, τ)
    
    # 2. 采样消费者参与决策（根据participation_rule）
    participation = participation_rule(params, world, rng)
    # 例如：[True, False, True, ..., True]（N个布尔值）
    
    # 3. 模拟"有数据"情况
    outcome_with_data = simulate_market_outcome(
        world, 
        participation, 
        params,
        producer_has_data=True   # 生产者能观察到参与者的信号
    )
    profit_with_data = outcome_with_data['producer_profit']
    
    # 4. 模拟"无数据"情况（使用相同的world和participation）
    outcome_no_data = simulate_market_outcome(
        world, 
        participation, 
        params,
        producer_has_data=False  # 生产者没有数据
    )
    profit_no_data = outcome_no_data['producer_profit']
    
    # 5. 记录利润差
    profit_diff = profit_with_data - profit_no_data
    profit_diff_samples.append(profit_diff)

# 6. 计算m_0
m_0 = beta * np.mean(profit_diff_samples)
```

**关键**：
- 使用Common Random Numbers：同一个world，同一个participation
- 只改变生产者的信息集Y_0
- 确保m_0度量的是**纯信息价值**

#### 6.2 期望参与人数

```python
e_num_participants = sum(p_vector)
# = p_1 + p_2 + ... + p_N
```

---

### 第7步：策略选择

**目的**：从identified和anonymized中选择最优策略

```python
# 过滤亏损策略
profitable_policies = {
    policy: result 
    for policy, result in results_by_policy.items()
    if result['profit'] > 0.0
}

if not profitable_policies:
    # 所有策略都亏损，中介不参与
    return {
        'm_star_vector': np.zeros(N),
        'anonymization_star': 'no_participation',
        'profit_star': 0.0
    }

# 选择利润最大的策略
best_policy = max(profitable_policies.keys(), 
                 key=lambda p: profitable_policies[p]['profit'])

return {
    'm_star_vector': results[best_policy]['m_vector'],
    'anonymization_star': best_policy,
    'profit_star': results[best_policy]['profit'],
    'r_star': results[best_policy]['info']['r_star'],
    'delta_u': results[best_policy]['info']['delta_u']
}
```

---

## 🔑 关键函数说明

### 函数调用层次

```
optimize_intermediary_policy_personalized()
    └─► optimize_m_vector_scipy_hybrid()
            ├─► evaluate_m_vector_profit()
            │       ├─► compute_rational_participation_rate()
            │       │       └─► compute_expected_utility_ex_ante()
            │       │               └─► simulate_market_outcome()
            │       └─► estimate_m0_mc()
            │               └─► simulate_market_outcome()
            └─► scipy.optimize.minimize()
```

### 函数功能表

| 函数 | 输入 | 输出 | 作用 |
|-----|------|------|------|
| `optimize_intermediary_policy_personalized` | params_base | (m*, π*, R*) | 总入口，对比策略 |
| `optimize_m_vector_scipy_hybrid` | policy | (m*, R*) | 混合优化m向量 |
| `evaluate_m_vector_profit` | m_vector, policy | R | 评估利润（被优化器反复调用） |
| `compute_rational_participation_rate` | m_vector | (r*, ΔU, p) | 求解参与均衡（固定点） |
| `compute_expected_utility_ex_ante` | i, m_i | U_i | 计算消费者i的期望效用 |
| `estimate_m0_mc` | m, π, r* | m_0 | 估计生产者支付意愿 |
| `simulate_market_outcome` | world, A | outcome | 模拟一次市场结果 |

---

## 📐 数学原理

### Stackelberg博弈结构

```
第1阶段（中介）：选择 (m, π)
    ↓
第2阶段（消费者）：观察m，决策a_i ∈ {0,1}
    ↓
第3阶段（生产者）：观察A，决策价格p
    ↓
第4阶段（消费者）：观察p，决策购买q_i
```

### 反向归纳求解

#### 第2阶段：消费者参与决策

消费者i参与当且仅当：
```
ΔU_i(m_i, r) = E[U_i(a_i=1 | m_i, r)] - E[U_i(a_i=0 | r)] > τ_i
```

其中：
- `m_i`：中介给i的补偿
- `r`：其他人的参与率（外部性）
- `τ_i ~ F_τ`：隐私成本（异质性）

**均衡条件**（固定点）：
```
r* = E[1{τ_i < ΔU_i(m_i, r*)}] = ∫ 1{ΔU_i(m_i, r*) > τ} dF_τ(τ)
```

如果 τ ~ Uniform[a, b]：
```
r* = P(τ < ΔU(r*)) = (ΔU(r*) - a) / (b - a)
```

#### 个性化版本

```
p_i = P(τ_i < ΔU_i(m_i, r))  # 每个人不同
r = (1/N) Σ p_i              # 总体参与率
```

#### 第1阶段：中介优化

```
max_{m, π} R = m_0(m, π) - E[Σ m_i · a_i]
           = m_0(m, π) - Σ m_i · p_i
s.t. R > 0
```

其中：
```
m_0(m, π) = β × E[π_producer(data) - π_producer(no data)]
```

---

## ⚙️ 算法细节

### L-BFGS-B算法

**特点**：
- 拟牛顿法，适合大规模优化
- 有界约束（box constraints）
- 只需函数值，不需显式梯度（数值梯度）

**数值梯度**：
```
∂R/∂m_i ≈ [R(m + ε·e_i) - R(m)] / ε
```

每次迭代需要N+1次函数评估（1次当前点 + N次扰动点）

### 网格搜索

**空间复杂度**：
- 如果直接在N维空间网格搜索：21^20 ≈ 10^26 个点（不可行）
- 我们只在1维统一补偿空间搜索：21个点（可行）

**trade-off**：
- 网格搜索：全局，但只能找统一补偿
- L-BFGS-B：局部，但可以探索个性化

**混合策略**：
- 网格找到好的统一补偿作为起点
- L-BFGS-B从这个起点探索个性化的可能性

### 固定点迭代收敛性

**收敛条件**（Banach不动点定理）：
如果映射 `r → T(r)` 是压缩映射（contraction），则存在唯一不动点。

**实践**：
- 使用平滑更新避免震荡：`r_new = 0.6*T(r) + 0.4*r`
- 通常3-5次迭代收敛

### 蒙特卡洛方差

**m_0的估计方差**：
```
Var[m̂_0] = Var[π_diff] / T
```

- T=100：标准误差 ≈ 0.1 × std(π_diff)
- 足够用于优化（不需要非常精确）

---

## 🚀 性能分析

### 计算复杂度

**网格搜索**：
- 21个点 × (固定点5次 + MC100次) ≈ 2100次市场模拟
- 时间：~2分钟（串行）

**L-BFGS-B**：
- 每次迭代：N+1次函数评估 = 21次`evaluate_m_vector_profit`
- 每次`evaluate_m_vector_profit`：
  - 固定点5次 × 每次N个消费者 × 每个消费者MC30×20 = 30,000次市场模拟
- 假设20次迭代收敛：20 × 21 × 30,000 = 12,600,000次市场模拟
- 时间：~5-10分钟

**总计**：~7-12分钟

### 关键性能瓶颈

1. **个性化计算**：N倍增加（20倍）
2. **MC采样**：T=100次
3. **固定点迭代**：5次
4. **优化迭代**：20次

### 优化建议（如需更快）

1. **减少MC采样**：T=100 → T=50（2倍加速，精度略降）
2. **减少固定点迭代**：max_iter=5 → max_iter=3（1.7倍加速）
3. **使用进化算法**：全局优化，但收敛慢
4. **GPU加速**：并行MC采样
5. **代理模型**：用神经网络近似`evaluate_m_vector_profit`

---

## 📊 验证结果

### 预期观察

**如果个性化有优势**：
```json
{
  "m_star": [0.6, 0.65, 0.7, ..., 0.85],
  "m_star_std": 0.08,
  "profit_star": 1.85
}
```

**如果统一补偿最优**：
```json
{
  "m_star": [0.75, 0.75, 0.75, ..., 0.75],
  "m_star_std": 0.0001,
  "profit_star": 1.78
}
```

### 理论预期

**在Common Preferences场景下**：
- 所有消费者w_i相同（偏好同质）
- τ_i独立同分布（隐私成本异质但不可观测）
- **对称性** → 统一补偿可能是最优的

**如果要看到个性化优势，需要**：
- 异质的数据质量（有些人信号更准确）
- 可观测的异质性（中介能区分不同类型）
- Common Experience场景（经验异质性）

---

## 🎓 总结

### 优化流程总结

1. **外层循环**：对比identified vs anonymized
2. **中层循环**：L-BFGS-B迭代优化m向量
3. **内层循环**：固定点迭代求解r*
4. **最内层**：MC估计期望效用和m_0

### 关键创新

1. ✅ **真正的个性化计算**：为每个消费者单独计算ΔU_i(m_i)
2. ✅ **混合优化策略**：网格+梯度，兼顾全局和局部
3. ✅ **利润约束**：R>0，避免亏损策略
4. ✅ **内生m_0**：通过MC估计，理论严格

### 理论贡献

这个优化过程严格对应论文的Stackelberg博弈框架，实现了：
- 中介作为Stackelberg leader优化策略
- 消费者作为follower理性响应
- 生产者内生支付意愿
- 完整的均衡求解

---

**文档版本**: 1.0.0  
**创建日期**: 2026-01-29  
**作者**: AI Assistant  
**状态**: 完整实现
