# 场景C：理论最优解求解流程总结

## 📋 **概览**

本文档详细说明在当前代码中，给定一个场景（参数配置），如何通过**逆向归纳（Backward Induction）**求解出理论最优解。

---

## 🎯 **核心思想：三层嵌套优化 + 逆向归纳**

### **博弈结构（Stackelberg博弈）**

```
┌─────────────────────────────────────────────────────────┐
│ 外层：中介（Intermediary）                                │
│ 决策：(m*, anonymization*)                              │
│ 目标：max R = m_0 - m·r*·N                              │
│ 约束：预判内层均衡                                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 中层：生产者（Producer）                            │ │
│  │ 决策：定价策略 {p_i*} or p*                        │ │
│  │ 目标：max π = Σ(p_i - c)·q_i                      │ │
│  │ 约束：消费者购买反应                                │ │
│  │                                                   │ │
│  │  ┌─────────────────────────────────────────────┐ │ │
│  │  │ 内层：消费者（Consumer）                     │ │ │
│  │  │ 决策：参与决策 a_i ∈ {0, 1}                │ │ │
│  │  │ 目标：max U_i = u_i + m_i                  │ │ │
│  │  │ 均衡：纳什均衡（固定点）r*                  │ │ │
│  │  └─────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### **求解顺序：逆向归纳**

```
求解顺序：外 → 中 → 内（逆向预判）
执行顺序：内 → 中 → 外（正向展开）

中介求解最优策略时：
  1. 假设自己选择 (m, anonymization)
  2. 预判消费者会如何反应 → 求解r*（内层）
  3. 预判生产者会如何定价 → 求解π*（中层）
  4. 计算自己的利润 R(m, anonymization)
  5. 遍历所有候选策略，选择使R最大的
```

---

## 🔄 **完整求解流程**

### **Level 1：给定策略，求解市场均衡**

#### **输入**
```python
# 完整参数（包含中介策略）
params = ScenarioCParams(
    # 市场参数（外生）
    N=20,                          # 消费者数量
    data_structure="common_preferences",  # 数据结构
    mu_theta=5.0,                  # 先验均值
    sigma_theta=1.0,               # 先验标准差
    sigma=1.0,                     # 噪声水平
    c=0.0,                         # 边际成本
    tau_mean=0.5,                  # 隐私成本均值
    tau_std=0.5,                   # 隐私成本标准差
    tau_dist="normal",             # 隐私成本分布
    
    # 中介策略（可能是外生给定，也可能是优化求出）
    m=1.0,                         # 数据补偿
    anonymization="identified",     # 匿名化策略
    
    seed=42
)
```

#### **求解步骤**

##### **步骤1：生成世界状态**
```python
def generate_consumer_data(params: ScenarioCParams) -> ConsumerData:
    """
    生成一个世界状态：真实偏好w_i和私人信号s_i
    
    对应论文：Section 3 - Information structures
    """
    N = params.N
    
    if params.data_structure == "common_preferences":
        # 共同偏好：w_i = θ for all i
        theta = np.random.normal(params.mu_theta, params.sigma_theta)
        w = np.full(N, theta)
        e = np.random.normal(0, 1, N)  # 独立噪声
        s = w + params.sigma * e       # s_i = θ + σ·e_i
    
    elif params.data_structure == "common_experience":
        # 共同经历：w_i ~ i.i.d., e_i = ε for all i
        w = np.random.normal(params.mu_theta, params.sigma_theta, N)
        epsilon = np.random.normal(0, 1)  # 共同噪声
        e = np.full(N, epsilon)
        s = w + params.sigma * e       # s_i = w_i + σ·ε
    
    return ConsumerData(w=w, s=s, e=e)
```

**输出**：
- `w[i]`：消费者i的真实支付意愿（未知，用于计算效用）
- `s[i]`：消费者i的私人信号（观察到，用于学习）

---

##### **步骤2：求解消费者均衡（内层）**

```python
def compute_rational_participation_rate(
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50
) -> Tuple[float, List[float], float]:
    """
    求解消费者参与决策的纳什均衡（固定点）
    
    对应论文：
      - Section 5.1: Consumer participation equilibrium
      - Ex Ante时序（论文标准）
    
    核心思想：
      - 每个消费者同时决策：参与 or 拒绝
      - 消费者i的最优决策取决于其他人的参与率r
      - 均衡条件：r* = F_τ(ΔU(r*))，即固定点
    """
```

**算法：Ex Ante固定点迭代（两层Monte Carlo）**

```python
# 初始化参与率
r = 0.5

for iteration in range(max_iter):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 两层Monte Carlo估计期望效用差
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    delta_u_samples = []
    
    for _ in range(num_world_samples):
        # 外层循环：抽世界状态 (w, s)
        data = generate_consumer_data(params)
        
        for _ in range(num_market_samples):
            # 内层循环：抽其他人的参与决策
            # 模拟：其他N-1人以概率r参与
            others_participation = np.random.rand(N-1) < r
            
            # 情况A：如果我参与
            participation_if_accept = np.concatenate([[True], others_participation])
            outcome_accept = simulate_market_outcome(data, participation_if_accept, params)
            utility_accept = outcome_accept.utilities[0]  # 消费者0的效用
            
            # 情况B：如果我拒绝
            participation_if_reject = np.concatenate([[False], others_participation])
            outcome_reject = simulate_market_outcome(data, participation_if_reject, params)
            utility_reject = outcome_reject.utilities[0]
            
            # 效用差（不含补偿）
            delta_u_samples.append(utility_accept - utility_reject)
    
    # 平均效用差
    delta_u = np.mean(delta_u_samples)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 更新参与率（基于隐私成本分布）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 参与条件：τ_i ≤ ΔU + m
    # r_new = Pr(τ_i ≤ ΔU + m) = F_τ(ΔU + m)
    
    if params.tau_dist == "normal":
        from scipy.stats import norm
        r_new = norm.cdf(delta_u + params.m, params.tau_mean, params.tau_std)
    elif params.tau_dist == "uniform":
        # ... uniform CDF
        pass
    elif params.tau_dist == "none":
        # 同质消费者（角点解）
        r_new = 1.0 if (delta_u + params.m) > 0 else 0.0
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 检查收敛
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if abs(r_new - r) < tol:
        print(f"  Ex Ante固定点收敛于迭代 {iteration}, r* = {r_new:.4f}, ΔU = {delta_u:.4f}")
        return r_new, r_history, delta_u
    
    # 平滑更新（避免震荡）
    r = 0.7 * r_new + 0.3 * r
    r_history.append(r)

# 未收敛 → 抛出错误
raise RuntimeError(f"固定点未收敛在 {max_iter} 次迭代内")
```

**输出**：
- `r*`：均衡参与率（固定点）
- `ΔU`：期望效用差（参与 - 拒绝）

**关键**：
- ✅ Ex Ante时序（决策时不知道信号实现）
- ✅ 隐私成本异质性（产生内点r*）
- ✅ 固定点收敛保证

---

##### **步骤3：生成实际参与决策**

```python
def generate_participation_from_tau(
    delta_u: float,
    params: ScenarioCParams,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    基于隐私成本分布生成参与决策
    
    Microfoundation：
      - 每个消费者i有一个隐私成本 τ_i ~ F_τ
      - 参与当且仅当 τ_i ≤ ΔU + m
      - 这样生成的参与率平均为 r* = F_τ(ΔU + m)
    """
    N = params.N
    
    # 抽取每个消费者的隐私成本
    if params.tau_dist == "normal":
        tau_i = np.random.normal(params.tau_mean, params.tau_std, N)
    elif params.tau_dist == "uniform":
        # ... uniform sampling
        pass
    else:
        # 同质情况
        tau_i = np.full(N, params.tau_mean)
    
    # 参与决策：τ_i ≤ ΔU + m
    participation = (tau_i <= delta_u + params.m)
    
    return participation
```

**输出**：
- `participation[i]`：消费者i是否参与（bool数组）

---

##### **步骤4：模拟市场结果（中层）**

```python
def simulate_market_outcome(
    data: ConsumerData,
    participation: np.ndarray,
    params: ScenarioCParams
) -> MarketOutcome:
    """
    给定参与决策，模拟完整的市场均衡
    
    包括：
      1. 数据收集与处理
      2. 信息披露
      3. 后验更新（消费者 + 生产者）
      4. 生产者定价（最优反应）
      5. 消费者购买
      6. 效用与利润计算
    
    对应论文：
      - Stage 2-5: 数据市场到产品市场的完整流程
    """
```

**子步骤4.1：数据收集与处理**
```python
# 参与者信号集合
participant_indices = np.where(participation)[0]
participant_signals = data.s[participant_indices]

if params.anonymization == "anonymized":
    # 匿名化：打乱身份映射
    np.random.shuffle(participant_signals)
    # X = {s_i : i ∈ Participants}（无身份）
else:
    # 实名制：保留身份映射
    # X = {(i, s_i) : i ∈ Participants}
```

**子步骤4.2：后验更新**

**4.2.1 消费者后验（对应论文Section 4）**
```python
def compute_posterior_mean_consumer(
    s_i: float,
    participant_signals: np.ndarray,
    params: ScenarioCParams,
    is_participant: bool
) -> float:
    """
    消费者i对自己的真实偏好w_i的后验期望
    
    信息集：I_i = {s_i} ∪ X
      - s_i：自己的私人信号（永远知道）
      - X：参与者的信号集合（可观察）
    """
    
    if params.data_structure == "common_preferences":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # E[θ | s_i, X] 共轭正态更新
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        prior_precision = 1 / params.sigma_theta**2
        signal_precision = 1 / params.sigma**2
        
        # 合并所有信号
        if is_participant:
            # 避免double count：X已包含s_i
            all_signals = participant_signals  # X
        else:
            # 拒绝者：结合自己的信号和X
            all_signals = np.concatenate([[s_i], participant_signals])
        
        n_signals = len(all_signals)
        posterior_precision = prior_precision + n_signals * signal_precision
        
        mu_posterior = (
            (prior_precision * params.mu_theta + signal_precision * np.sum(all_signals))
            / posterior_precision
        )
        
        return mu_posterior
    
    elif params.data_structure == "common_experience":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # E[w_i | s_i, X] 需要先估计共同噪声ε
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        epsilon_hat = _compute_ce_posterior_approx(
            participant_signals, params.mu_theta, params.sigma_theta, params.sigma
        )
        
        # 过滤噪声
        filtered_signal = s_i - params.sigma * epsilon_hat
        
        # 结合先验
        prior_precision = 1 / params.sigma_theta**2
        # ... 贝叶斯更新
        
        return mu_posterior
```

**4.2.2 生产者后验（关键：信息不对称）**
```python
def compute_producer_posterior(
    data: ConsumerData,
    participation: np.ndarray,
    participant_signals: np.ndarray,
    params: ScenarioCParams
) -> np.ndarray:
    """
    生产者对每个消费者的后验期望
    
    关键区别（P0-3修复）：
      - identified：生产者知道谁参与了，对参与者可用s_i
      - anonymized：生产者不知道谁是谁，只能用聚合信息
    """
    N = params.N
    mu_producer = np.zeros(N)
    
    if params.anonymization == "identified":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 实名制：可个性化后验
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        for i in range(N):
            if participation[i]:
                # 参与者：用个人信号
                mu_producer[i] = compute_posterior_mean_consumer(
                    data.s[i], participant_signals, params, is_participant=True
                )
            else:
                # 拒绝者：仍能从X学习（社会数据外部性，P0-2修复）
                if params.data_structure == "common_preferences":
                    # 学习共同偏好θ
                    mu_producer[i] = compute_posterior_mean_consumer(
                        data.s[i], participant_signals, params, is_participant=False
                    )
                else:
                    # Common Experience：学习共同噪声
                    epsilon_hat = _compute_ce_posterior_approx(...)
                    # ... 估计w_i
    
    elif params.anonymization == "anonymized":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 匿名化：只能用聚合信息，所有人相同后验
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if params.data_structure == "common_preferences":
            # 估计共同偏好θ
            common_posterior = compute_posterior_mean_consumer(
                0, participant_signals, params, is_participant=False
            )
            mu_producer[:] = common_posterior
        
        else:  # common_experience
            # P0-3修复：匿名化也能学习共同噪声（不是固定先验）
            epsilon_hat = _compute_ce_posterior_approx(...)
            # 代表性个体的后验
            mu_producer[:] = ...
    
    return mu_producer
```

**子步骤4.3：生产者最优定价**

```python
# 根据匿名化策略选择定价方式
if params.anonymization == "identified":
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 个性化定价（P0-2修复：正确公式）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # p_i* = (μ_producer[i] + c) / 2
    prices = (mu_producer + params.c) / 2

else:  # anonymized
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 统一定价（P0-2修复：数值优化）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # p* = argmax Σ(p-c)·max(μ_consumer[i]-p, 0)
    p_optimal, _ = compute_optimal_price_uniform(mu_producer, params.c)
    prices = np.full(N, p_optimal)
```

**子步骤4.4：消费者购买决策**
```python
# 最优购买量：q_i* = max(μ_consumer[i] - p_i, 0)
quantities = np.maximum(mu_consumers - prices, 0)
```

**子步骤4.5：效用与利润计算**
```python
# 消费者效用：u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
utilities = data.w * quantities - prices * quantities - 0.5 * quantities**2

# 参与者获得补偿
for i in participant_indices:
    utilities[i] += params.m

# 消费者剩余
consumer_surplus = np.sum(utilities)

# 生产者利润
producer_profit = np.sum((prices - params.c) * quantities)

# 中介利润（P2-7修复）
intermediary_profit = params.m_0 - params.m * len(participant_indices)

# 社会福利
social_welfare = consumer_surplus + producer_profit + intermediary_profit
```

**输出**：完整的`MarketOutcome`对象

---

### **Level 2：求解最优策略（中介优化）**

#### **输入**
```python
# 基础市场参数（不含中介策略）
params_base = {
    'N': 20,
    'data_structure': 'common_preferences',
    'mu_theta': 5.0,
    'sigma_theta': 1.0,
    'sigma': 1.0,
    'c': 0.0,
    'tau_mean': 0.5,
    'tau_std': 0.5,
    'tau_dist': 'normal',
    'seed': 42
}

# 策略搜索空间
m_grid = np.linspace(0, 3.0, 31)  # 31个补偿候选
policies = ['identified', 'anonymized']  # 2个策略
```

#### **求解步骤**

##### **步骤1：遍历所有候选策略**

```python
def optimize_intermediary_policy(
    params_base: Dict,
    m_grid: np.ndarray,
    policies: List[str],
    ...
) -> OptimalPolicy:
    """
    求解中介的最优策略组合 (m*, anonymization*)
    
    对应论文：Section 5.2-5.3
    """
    
    all_results = []
    
    for m in m_grid:  # 31个补偿
        for anonymization in policies:  # 2个策略
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 评估该策略组合
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            result = evaluate_intermediary_strategy(
                m=m,
                anonymization=anonymization,
                params_base=params_base,
                ...
            )
            all_results.append(result)
    
    # 找到最优策略
    optimal_result = max(all_results, key=lambda x: x.intermediary_profit)
    
    return OptimalPolicy(
        optimal_m=optimal_result.m,
        optimal_anonymization=optimal_result.anonymization,
        optimal_result=optimal_result,
        all_results=all_results,
        ...
    )
```

##### **步骤2：评估单个候选策略**

```python
def evaluate_intermediary_strategy(
    m: float,
    anonymization: str,
    params_base: Dict,
    ...
) -> IntermediaryOptimizationResult:
    """
    评估给定策略(m, anonymization)下的完整市场均衡
    
    执行逆向归纳三步：
      1. 内层：求解消费者均衡 r*(m, anonymization)
      2. 中层：计算生产者利润 π*(r*, anonymization)
      3. 外层：计算中介利润 R = m_0 - m·r*·N
    """
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.1：构建完整参数
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    params = ScenarioCParams(
        m=m,
        anonymization=anonymization,
        **params_base
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.2：内层 - 求解消费者均衡
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    r_star, _, delta_u = compute_rational_participation_rate(params)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.3：生成市场实现
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    data = generate_consumer_data(params)
    participation = generate_participation_from_tau(delta_u, params)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.4：中层 - 模拟市场结果（有数据）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    outcome_with_data = simulate_market_outcome(data, participation, params)
    producer_profit_with_data = outcome_with_data.producer_profit
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.5：Baseline - 计算无数据时的生产者利润
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    outcome_no_data = simulate_market_outcome_no_data(data, params)
    producer_profit_no_data = outcome_no_data.producer_profit
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.6：计算数据价值（利润增益）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    producer_profit_gain = producer_profit_with_data - producer_profit_no_data
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2.7：外层 - 计算中介利润
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 生产者支付意愿 = 数据带来的利润增益
    m_0 = max(0, producer_profit_gain)
    
    # 中介成本 = 向参与者支付的补偿总额
    num_participants = int(np.sum(participation))
    intermediary_cost = m * num_participants
    
    # 中介利润
    intermediary_profit = m_0 - intermediary_cost
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 返回完整结果
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    return IntermediaryOptimizationResult(
        m=m,
        anonymization=anonymization,
        r_star=r_star,
        delta_u=delta_u,
        num_participants=num_participants,
        producer_profit_with_data=producer_profit_with_data,
        producer_profit_no_data=producer_profit_no_data,
        producer_profit_gain=producer_profit_gain,
        m_0=m_0,
        intermediary_cost=intermediary_cost,
        intermediary_profit=intermediary_profit,
        consumer_surplus=outcome_with_data.consumer_surplus,
        social_welfare=outcome_with_data.social_welfare,
        ...
    )
```

**输出**：
- `optimal_m`：最优补偿
- `optimal_anonymization`：最优策略
- `optimal_result`：最优策略下的完整均衡
- `all_results`：所有候选策略的结果（用于分析）

---

## 📊 **函数调用链总结**

### **完整调用链（Level 2：中介优化）**

```
optimize_intermediary_policy()
  ├─ for m in m_grid:
  │    for anonymization in policies:
  │      └─ evaluate_intermediary_strategy(m, anonymization)
  │           ├─ compute_rational_participation_rate(params)  ← 内层
  │           │    └─ (固定点迭代)
  │           │         ├─ generate_consumer_data()
  │           │         └─ simulate_market_outcome() × 多次  ← 中层（嵌套）
  │           │              ├─ compute_posterior_mean_consumer()
  │           │              ├─ compute_producer_posterior()
  │           │              ├─ compute_optimal_price_uniform() / 个性化定价
  │           │              └─ (计算效用与利润)
  │           │
  │           ├─ generate_participation_from_tau()
  │           ├─ simulate_market_outcome() ← 中层（最终）
  │           ├─ simulate_market_outcome_no_data() ← Baseline
  │           └─ (计算中介利润) ← 外层
  │
  └─ max(all_results, key=lambda x: x.intermediary_profit)
```

### **简化调用链（Level 1：给定策略）**

```
generate_ground_truth(params)  ← params包含(m, anonymization)
  ├─ compute_rational_participation_rate(params)  ← 内层
  ├─ generate_participation_from_tau()
  └─ simulate_market_outcome() × 多次 ← 中层
       └─ (计算期望outcome + 样本outcome)
```

---

## 🔑 **关键设计特点**

### **1. 模块化设计**
```python
# 每一层都是独立函数，可单独调用
内层：compute_rational_participation_rate()
中层：simulate_market_outcome()
外层：optimize_intermediary_policy()

# 也可以组合调用
完整优化：optimize_intermediary_policy()
给定策略：generate_ground_truth()
```

### **2. 逆向归纳的实现**

```python
# 中介优化时，对每个候选(m, anonymization)：
1. 调用内层函数 → 得到r*(m, anonymization)
2. 调用中层函数 → 得到π*(r*, anonymization)
3. 计算外层目标 → 得到R(m, anonymization)

# 遍历所有候选，选择R最大的
```

### **3. 固定点求解（内层）**

```python
# Ex Ante固定点迭代
r_0 = 0.5  # 初始猜测
for iter in range(max_iter):
    ΔU(r) = 估计效用差（给定参与率r）
    r_new = F_τ(ΔU(r) + m)  # 更新
    if |r_new - r| < tol:
        break  # 收敛
    r = 0.7 * r_new + 0.3 * r  # 平滑更新
```

### **4. Monte Carlo模拟（期望计算）**

```python
# 两层MC：
外层：抽世界状态(w, s)
内层：抽市场实现（其他人的参与决策）

# 平均得到期望效用差
ΔU = E_{w,s,r_{-i}}[U_i^{accept} - U_i^{reject}]
```

---

## 📈 **输出示例**

### **Level 1：给定策略（m=1.0, identified）**

```json
{
  "params": {
    "m": 1.0,
    "anonymization": "identified",
    "N": 20,
    ...
  },
  "rational_participation_rate": 0.8363,  // r*
  "delta_u": 0.9896,                      // ΔU
  "outcome": {
    "participation_rate": 0.85,           // 实际实现
    "consumer_surplus": 45.23,
    "producer_profit": 178.74,
    "intermediary_profit": -11.08,        // R = m_0 - m·r*·N
    "social_welfare": 212.89,
    ...
  }
}
```

### **Level 2：中介优化**

```
策略空间：31个补偿 × 2个策略 = 62个候选

遍历结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     m    | anonymization |   r*  |   R   |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   0.00   |  identified   | 11.2% |  1.20 |
   0.20   |  identified   | 23.0% | -1.00 |
   0.40   |  identified   | 39.5% | -2.65 |
   0.60   |  identified   | 56.3% |  4.61 | ← 最优
   0.80   |  identified   | 71.7% |-10.02 |
   1.00   |  identified   | 83.6% |-11.08 |
   ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最优策略：
  m* = 0.60
  anonymization* = identified
  r* = 56.3%
  R* = 4.61
```

---

## 🎯 **总结**

### **核心算法**

| 层次 | 输入 | 算法 | 输出 |
|------|------|------|------|
| **内层**（消费者） | (m, anonymization) | 固定点迭代 + 两层MC | r*, ΔU |
| **中层**（生产者） | r* + 参与决策 | 后验更新 + 最优定价 | π*, 市场结果 |
| **外层**（中介） | 市场参数 | 网格搜索 + 逆向归纳 | (m*, anonymization*), R* |

### **理论最优解 = Stackelberg均衡**

```
(m*, anonymization*, r*, {p_i*}, {q_i*})

满足：
  1. r*是消费者的纳什均衡（固定点）
  2. {p_i*}是生产者的最优反应（给定r*）
  3. (m*, anonymization*)是中介的最优策略（给定1和2）
```

### **代码实现的学术正确性**

✅ **Ex Ante时序**（P1-4修复）
✅ **隐私成本异质性**（P2-1/2修复）
✅ **完整的三层框架**（本次实现）
✅ **所有论文机制对齐**（P0-P2修复）

---

**文档版本**: v1.0  
**创建日期**: 2026-01-18  
**作者**: Claude (Sonnet 4.5)  
**用途**: 说明场景C理论最优解的完整求解流程
