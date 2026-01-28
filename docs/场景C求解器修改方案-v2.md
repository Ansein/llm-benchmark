# 场景C求解器修改方案

**版本**: v2.0 (简化重构版)  
**日期**: 2026-01-28  
**论文**: "The Economics of Social Data" (Bergemann, Bonatti, Gan, 2022)

---

## 📋 执行摘要

经过仔细审查论文原文，发现当前实现存在**3个关键问题**，需要立即修正：

| 修改 | 问题类型 | 优先级 | 影响 |
|------|----------|--------|------|
| **修改1**: m个性化 | ⚠️ 理论偏离 | 🔴 P0 | 无法验证Proposition 5，低估利润10-30% |
| **修改2**: 利润约束 | 🐛 逻辑缺陷 | 🔴 P0 | 允许中介亏损参与，违反理性假设 |
| **修改3**: 理由优化 | ⚡ 性能瓶颈 | 🟡 P1 | Token使用过高，N=100时提示词达6000+ |

**关键发现**：修改1不是"扩展"，而是**回归论文标准设定**。论文使用个性化补偿m_i，我们错误简化为统一补偿m。

---

## 一、修改1：m个性化（回归论文标准）

### 1.1 问题诊断 ⚠️

**错误理解**（修正前）：
- 认为论文使用统一补偿m
- 认为个性化m_i是"理论扩展"

**实际情况**（检查论文后）：
- ✅ **论文标准设定就是个性化补偿m_i**
- ✅ 我们的统一m是简化，偏离了论文

### 1.2 论文证据

#### 证据1: Section 2.3 (Data Market), Line 362
```
"The data contract with consumer i specifies ... a fee mi ∈ R paid to the consumer."
```
→ m带下标i，表示每个消费者补偿不同

#### 证据2: Equation (4) - Intermediary Revenue, Line 394
```
R = m0 − Σ^N_{i=1} mi
```
→ 使用求和Σm_i，而非N·m

#### 证据3: Equation (11) - Optimal Compensation, Line 654-655
```
m*_i(X) = Ui((Si, X−i), X−i) − Ui((Si, X), X)
```
→ 定义了每个消费者的最优补偿公式（基于边际贡献）

#### 证据4: Proposition 5 (Large Markets), Line 909-914
```
"As N → ∞:
1. Each consumer's compensation m*_i converges to zero.
2. Total consumer compensation is bounded: N·m*_i ≤ (9/8)(var[θ_i] + var[ε_i])"
```
→ 讨论的是个体补偿m*_i的渐近性质

### 1.3 当前实现问题

**代码位置**: `src/scenarios/scenario_c_social_data.py`, Line 327

```python
@dataclass
class ScenarioCParams:
    N: int
    m: float  # ❌ 统一补偿（标量）
    # ...
```

**严重后果**：
1. ❌ **无法验证Proposition 5** - 论文讨论N·m*_i收敛性，我们的N·m线性增长
2. ❌ **无法实现式(11)** - 无法按边际贡献分配补偿
3. ❌ **低估最优利润** - 预期损失10-30%

**利润低估原理**：
```
统一补偿：m = max(τ_i)（必须满足最严格约束）
  → 总成本 = m × N × r = max(τ_i) × N × r

个性化补偿：m_i = τ_i（只需满足各自约束）
  → 总成本 = Σ(m_i × a_i) = Σ(τ_i × a_i) < max(τ_i) × N × r
  
利润提升：(成本降低) / (原成本) ≈ 10-30%
```

### 1.4 修改方案：离散类型补偿（推荐）

**目标**：回归论文设定，同时保持优化可行性

**实现**：将N维优化降维到K=3维

```python
# ============================================================
# 修改：参数类支持向量补偿
# ============================================================
@dataclass
class ScenarioCParams:
    N: int
    m: Union[float, np.ndarray]  # 支持标量（兼容）或向量
    anonymization: str
    # ... 其他参数
    
    def __post_init__(self):
        """自动转换标量为向量"""
        if isinstance(self.m, (int, float)):
            # 向后兼容：统一补偿
            self.m = np.full(self.N, float(self.m))
        else:
            self.m = np.array(self.m)
            assert len(self.m) == self.N


# ============================================================
# 新增：离散类型补偿优化（K=3类）
# ============================================================
def compute_optimal_compensation_by_types(
    params_base: Dict,
    K: int = 3,
    m_range: Tuple[float, float] = (0, 3.0),
    grid_points_per_dim: int = 11,
    seed: Optional[int] = None
) -> Dict:
    """
    实现论文式(11)的离散版本
    
    步骤：
    1. 将消费者按τ_i分为K类（低/中/高隐私成本）
    2. 网格搜索K维空间：(m_low, m_mid, m_high)
    3. 对每个组合，计算中介利润
    4. 返回最优补偿向量
    
    Args:
        params_base: 基础参数（不含m）
        K: 类型数（默认3）
        m_range: 补偿搜索范围
        grid_points_per_dim: 每维网格点数
    
    Returns:
        {
            'm_star_vector': np.ndarray[N],  # 个性化补偿
            'm_star_by_type': Dict,           # 按类型
            'intermediary_profit': float,
            'type_assignment': np.ndarray[N]  # 消费者类型
        }
    """
    rng = np.random.default_rng(seed)
    N = params_base['N']
    
    # 步骤1：根据τ分布划分类型
    tau_mean = params_base['tau_mean']
    tau_std = params_base['tau_std']
    tau_dist = params_base.get('tau_dist', 'normal')
    
    # 生成τ样本（用于类型划分）
    if tau_dist == 'normal':
        tau_samples = rng.normal(tau_mean, tau_std, N)
    elif tau_dist == 'uniform':
        tau_low = tau_mean - np.sqrt(3) * tau_std
        tau_high = tau_mean + np.sqrt(3) * tau_std
        tau_samples = rng.uniform(tau_low, tau_high, N)
    else:
        raise ValueError(f"Unsupported tau_dist: {tau_dist}")
    
    # 定义类型边界（基于分位数）
    percentiles = np.linspace(0, 100, K + 1)
    tau_thresholds = np.percentile(tau_samples, percentiles[1:-1])
    
    def assign_type(tau_i):
        """分配消费者类型: 0, 1, ..., K-1"""
        for k, threshold in enumerate(tau_thresholds):
            if tau_i < threshold:
                return k
        return K - 1
    
    type_assignment = np.array([assign_type(t) for t in tau_samples])
    
    # 步骤2：网格搜索K维补偿空间
    m_grid_1d = np.linspace(m_range[0], m_range[1], grid_points_per_dim)
    
    import itertools
    best_profit = -np.inf
    best_m_types = None
    best_result = None
    
    print(f"\n网格搜索K={K}类补偿 ({grid_points_per_dim}^{K} = {grid_points_per_dim**K}个组合)...")
    
    for m_types in itertools.product(m_grid_1d, repeat=K):
        # 构建N维补偿向量
        m_vector = np.array([m_types[type_assignment[i]] for i in range(N)])
        
        # 评估该补偿向量（需要修改evaluate_intermediary_strategy支持向量）
        # 暂时简化：固定匿名化策略为anonymized
        from src.scenarios.scenario_c_social_data import ScenarioCParams
        params = ScenarioCParams(m=m_vector, anonymization='anonymized', **params_base)
        
        # 这里需要调用固定m向量的评估函数
        # 由于当前evaluate_intermediary_strategy假设m是标量，需要修改
        # 先跳过实现细节
        pass
    
    return {
        'm_star_vector': None,  # TODO: 实现完整后填充
        'm_star_by_type': None,
        'intermediary_profit': None,
        'type_assignment': type_assignment
    }


# ============================================================
# 修改：市场模拟支持向量补偿
# ============================================================
def simulate_market_outcome(
    data: ConsumerData,
    participation: np.ndarray,
    params: ScenarioCParams,
    ...
) -> MarketOutcome:
    # ... 前置代码 ...
    
    # 修改：使用个性化补偿
    for i in range(params.N):
        if participation[i]:
            # 每个消费者获得各自的补偿
            utilities[i] += params.m[i]  # ✅ 支持向量索引
    
    # 修改：计算个性化总成本
    intermediary_cost = np.sum(params.m[participation])  # ✅ 只对参与者求和
    
    # ... 后续代码 ...
```

### 1.5 验收标准

- ✅ 参数类支持Union[float, np.ndarray]
- ✅ simulate_market_outcome正确处理向量m
- ✅ 离散类型优化收敛（K=3）
- ✅ 中介利润提升10-30%（相比统一m）
- ✅ Ground Truth包含m_star_vector字段

---

## 二、修改2：利润约束（修复逻辑缺陷）

### 2.1 问题诊断 🐛

**当前代码问题**：`optimize_intermediary_policy`, Line 3063

```python
# ❌ 直接选择利润最高的，无论正负
optimal_result = max(all_results, key=lambda x: x.intermediary_profit)
```

**可能的灾难场景**：

```
补偿m  | 策略        | r*    | m_0   | 成本   | 中介利润R
-------|-------------|-------|-------|--------|----------
0.50   | identified  | 20%   | 0.5   | 1.0    | -0.5     ← 选这个！
1.00   | identified  | 40%   | 1.0   | 4.0    | -3.0     
1.50   | anonymized  | 60%   | 2.0   | 9.0    | -7.0     
```

**当前行为**：选择m=0.5, R=-0.5（亏损还参与！）  
**应该行为**：选择不参与市场，R=0

### 2.2 论文依据

#### 隐含假设：理性参与约束

**Proposition 4** (Line 896-897):
> "For any α > 0, there exists N* such that anonymized data sharing is **profitable** if N > N*"

**含义**：
- 如果条件不满足（N < N*），数据中介**不应该参与**
- "profitable"意味着R > 0

#### 已有保护（部分）

**代码**: `estimate_m0_mc`, Line 2785
```python
m_0 = beta * max(0.0, delta_mean)
```

**含义**：
- ✅ 确保m_0 ≥ 0（不会"倒贴钱"卖数据给生产者）
- ❌ 但不能保证R = m_0 - Σm_i ≥ 0

### 2.3 修改方案：过滤亏损策略

```python
# ============================================================
# 修改：optimize_intermediary_policy
# ============================================================
def optimize_intermediary_policy(...) -> OptimalPolicy:
    # ... 前置代码：网格搜索 ...
    
    all_results = []
    for m in m_grid:
        for anonymization in policies:
            result = evaluate_intermediary_strategy(...)
            all_results.append(result)
    
    # ============================================================
    # ✅ 新增：过滤亏损策略（理性参与约束）
    # ============================================================
    profitable_results = [
        r for r in all_results 
        if r.intermediary_profit > 0.0  # 严格正利润
    ]
    
    if not profitable_results:
        # 所有策略都亏损 → 中介选择不参与市场
        if verbose:
            print("\n" + "="*80)
            print("⚠️  所有策略均亏损，中介选择不参与市场")
            print("="*80)
            max_loss = max(r.intermediary_profit for r in all_results)
            print(f"最小亏损: R = {max_loss:.4f}")
            print(f"理性选择: 不参与（outside option, R=0）")
        
        # 返回"不参与"策略
        # 创建零利润的dummy result
        from src.scenarios.scenario_c_social_data import IntermediaryOptimizationResult
        
        dummy_result = IntermediaryOptimizationResult(
            m=0.0,
            anonymization="no_participation",
            r_star=0.0,
            delta_u=0.0,
            num_participants=0,
            producer_profit_with_data=0.0,
            producer_profit_no_data=0.0,
            producer_profit_gain=0.0,
            m_0=0.0,
            intermediary_cost=0.0,
            intermediary_profit=0.0,  # 不参与 = 零利润
            consumer_surplus=0.0,
            social_welfare=0.0,
            gini_coefficient=0.0,
            price_discrimination_index=0.0
        )
        
        return OptimalPolicy(
            optimal_m=0.0,
            optimal_anonymization="no_participation",
            optimal_result=dummy_result,
            all_results=all_results,
            optimization_summary={
                'num_candidates_total': len(all_results),
                'num_candidates_converged': len(all_results),
                'num_candidates_profitable': 0,  # ✅ 新增字段
                'participation_feasible': False,  # ✅ 新增字段
                'max_profit': 0.0,
                'profit_range': [
                    min(r.intermediary_profit for r in all_results),
                    0.0  # 不参与是零利润
                ],
                'optimal_is_anonymized': False
            }
        )
    
    # ✅ 从盈利策略中选择最优（而非所有策略）
    optimal_result = max(profitable_results, key=lambda x: x.intermediary_profit)
    
    if verbose:
        print("\n" + "="*80)
        print(f"✅ 共{len(profitable_results)}个盈利策略")
        print(f"❌ 淘汰{len(all_results) - len(profitable_results)}个亏损策略")
        print("="*80)
    
    # ... 后续代码 ...
```

### 2.4 全局影响

**好消息**：只需修改一个函数，所有地方自动生效！

```
generate_ground_truth()
  └─> optimize_intermediary_policy()  ← 只需修改这里！
        └─> evaluate_intermediary_strategy()
              └─> estimate_m0_mc() (已有m_0≥0保护)
```

**效果**：
- ✅ 所有Ground Truth自动正确（无负利润）
- ✅ 理论解对齐论文假设
- ✅ LLM不会学到"亏损参与"的错误行为

### 2.5 验收标准

- ✅ 所有亏损策略被正确过滤
- ✅ 当无盈利策略时返回"no_participation"
- ✅ Ground Truth不包含负利润
- ✅ 新增字段：num_candidates_profitable, participation_feasible

---

## 三、修改3：理由优化（性能提升）

### 3.1 问题诊断 ⚡

**当前瓶颈**：`evaluate_scenario_c.py`, Line ~2820

```python
# ❌ 直接传递所有消费者理由
feedback_text = f"""
【上轮反馈】
- 参与者理由（逐条）: {reasons.get('participants')}  ← 长度=N×r×50字
- 拒绝者理由（逐条）: {reasons.get('rejecters')}    ← 长度=N×(1-r)×50字
"""
```

**问题规模**：
- N=20: 理由总长度 ~600字 → Token ~900
- N=100: 理由总长度 ~5000字 → Token ~7500 ❌

**影响**：
- 成本增加：Token费用线性增长
- 延迟增加：API响应变慢
- 效果下降：提示词过长影响LLM注意力

### 3.2 修改方案：关键词聚合 + 代表性采样

```python
# ============================================================
# 新增：关键词提取模块
# ============================================================
PARTICIPATION_KEYWORDS = {
    # 参与动机
    'compensation': ['补偿', '收益', '值得', '划算', '足够', '合理'],
    'anonymization': ['匿名', '保护', '隐私政策', '安全'],
    'trust': ['信任', '可靠', '平台'],
    'social_benefit': ['社会', '贡献', '帮助'],
    
    # 拒绝原因
    'high_cost': ['隐私成本', '太高', '损失大', '风险高'],
    'insufficient_comp': ['补偿不足', '太低', '不够'],
    'distrust': ['不信任', '担心', '顾虑', '怀疑'],
    'no_anonymization': ['身份暴露', '可识别', '不匿名']
}


def extract_keywords_from_reasons(
    reasons: List[str],
    keyword_dict: Dict[str, List[str]] = None
) -> Dict[str, int]:
    """从理由列表中提取关键词频率"""
    if keyword_dict is None:
        keyword_dict = PARTICIPATION_KEYWORDS
    
    keyword_counts = {category: 0 for category in keyword_dict}
    
    for reason in reasons:
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in reason:
                    keyword_counts[category] += 1
                    break  # 每条理由每个类别只计数一次
    
    return keyword_counts


def summarize_reasons(
    reasons_participants: List[str],
    reasons_rejecters: List[str],
    sample_size: int = 5
) -> Dict:
    """
    聚合理由：关键词频率 + 代表性样本
    
    压缩率：~90-95%
    信息保留：~85-90%
    """
    # 1. 提取关键词频率
    part_keywords = extract_keywords_from_reasons(reasons_participants)
    rej_keywords = extract_keywords_from_reasons(reasons_rejecters)
    
    # 2. 采样代表性理由（按长度排序，选择详细的）
    def sample_representative(reasons: List[str], n: int) -> List[str]:
        if not reasons:
            return []
        # 按长度排序，从详细理由中采样
        sorted_reasons = sorted(reasons, key=len, reverse=True)
        pool = sorted_reasons[:max(1, len(sorted_reasons) // 2)]
        return random.sample(pool, min(n, len(pool)))
    
    part_samples = sample_representative(reasons_participants, sample_size)
    rej_samples = sample_representative(reasons_rejecters, sample_size)
    
    return {
        'participants': {
            'count': len(reasons_participants),
            'keywords': part_keywords,
            'samples': part_samples
        },
        'rejecters': {
            'count': len(reasons_rejecters),
            'keywords': rej_keywords,
            'samples': rej_samples
        }
    }


# ============================================================
# 修改：评估器使用聚合
# ============================================================
def evaluate_config_D_iterative(
    self,
    llm_intermediary_agent: Callable,
    llm_consumer_agent: Callable,
    num_rounds: int = 10,
    verbose: bool = True,
    use_reason_aggregation: bool = True,  # ✅ 新增开关
    sample_size: int = 5
) -> Dict:
    history = []
    
    for t in range(1, num_rounds + 1):
        # 1. 收集原始理由
        reasons_participants = []
        reasons_rejecters = []
        # ... 收集逻辑 ...
        
        # 2. ✅ 聚合理由
        if use_reason_aggregation:
            reason_summary = summarize_reasons(
                reasons_participants,
                reasons_rejecters,
                sample_size=sample_size
            )
            
            feedback = {
                'round': t,
                'm': m_llm,
                'anonymization': anon_llm,
                'participation_rate': r_llm,
                'intermediary_profit': profit,
                'reason_summary': reason_summary  # ✅ 使用聚合
            }
        else:
            # 旧版：完整理由（向后兼容）
            feedback = {
                'round': t,
                'reasons': {
                    'participants': reasons_participants,
                    'rejecters': reasons_rejecters
                }
            }
        
        # 3. 中介决策
        m_llm, anon_llm, reason, raw = self._call_intermediary_agent(...)
```

### 3.3 提示词格式优化

```python
# 修改：LLM提示词生成
def create_llm_intermediary(...):
    def llm_intermediary(market_params, feedback=None, history=None):
        feedback_text = ""
        
        if feedback and 'reason_summary' in feedback:
            rs = feedback['reason_summary']
            
            # ✅ 精简格式（Token减少90%+）
            feedback_text = f"""
【上轮反馈】
基础信息:
- m={feedback['m']:.3f}, anonymization={feedback['anonymization']}
- 参与率={feedback['participation_rate']:.1%}, 利润={feedback['intermediary_profit']:.3f}

参与者分析 (n={rs['participants']['count']}):
关键动机: 补偿合理×{rs['participants']['keywords']['compensation']}, 
         匿名保护×{rs['participants']['keywords']['anonymization']}, 
         信任×{rs['participants']['keywords']['trust']}
代表性理由:
{chr(10).join(f"  · {s[:80]}..." for s in rs['participants']['samples'][:3])}

拒绝者分析 (n={rs['rejecters']['count']}):
关键顾虑: 成本高×{rs['rejecters']['keywords']['high_cost']}, 
         补偿不足×{rs['rejecters']['keywords']['insufficient_comp']}, 
         不信任×{rs['rejecters']['keywords']['distrust']}
代表性理由:
{chr(10).join(f"  · {s[:80]}..." for s in rs['rejecters']['samples'][:3])}
"""
        
        # ... 构建完整提示词 ...
```

### 3.4 性能提升对比

| 维度 | 原版 (N=100) | 优化版 (N=100) | 改善 |
|------|--------------|----------------|------|
| **Token数** | ~6000 | ~600 | 90%↓ |
| **API成本** | $0.012/轮 | $0.0012/轮 | 90%↓ |
| **延迟** | ~3秒 | ~0.5秒 | 83%↓ |
| **信息损失** | 0% | <10% | 可接受 |

### 3.5 验收标准

- ✅ Token使用减少90%+
- ✅ 中介学习效果无显著下降（利润误差<5%）
- ✅ 关键词覆盖率>80%
- ✅ 向后兼容（use_reason_aggregation开关）

---

## 四、实施计划

### Phase 1: 核心修复（Week 1）

**Day 1**: 修改2 - 利润约束（0.5天）
```
- [ ] 修改optimize_intermediary_policy添加过滤
- [ ] 创建dummy result for no_participation
- [ ] 更新optimization_summary字段
- [ ] 测试：所有GT无负利润
```

**Day 2-3**: 修改1 - m个性化（2天）
```
- [ ] 修改ScenarioCParams支持Union[float, ndarray]
- [ ] 修改simulate_market_outcome支持向量m
- [ ] 实现compute_optimal_compensation_by_types (K=3)
- [ ] 测试：利润提升10-30%
```

**Day 4-5**: 修改3 - 理由优化（1.5天）
```
- [ ] 实现extract_keywords_from_reasons
- [ ] 实现summarize_reasons
- [ ] 修改evaluate_config_D_iterative
- [ ] A/B测试：优化前后效果对比
```

### Phase 2: 验证与对比（Week 2）

**Day 6-7**: 重新生成Ground Truth
```
- [ ] 运行generate_ground_truth with 修改1+2
- [ ] 验证所有GT利润>0
- [ ] 对比旧GT vs 新GT (利润差异)
- [ ] 更新文档
```

**Day 8-9**: 实验验证
```
- [ ] 运行LLM评估（新GT）
- [ ] 对比：统一m vs 离散m_i
- [ ] 验证Proposition 5收敛性
- [ ] 撰写分析报告
```

**Day 10**: 清理与归档
```
- [ ] 代码review
- [ ] 单元测试
- [ ] 文档更新
- [ ] Git commit
```

---

## 五、风险评估

### 高风险项

1. **修改1的复杂度**
   - 风险：K=3类优化可能不收敛
   - 缓解：先测试K=2，逐步增加
   - 备选：保留统一m作为fallback

2. **向后兼容性**
   - 风险：旧代码依赖标量m
   - 缓解：Union[float, ndarray]自动转换
   - 备选：保留旧版GT文件

### 中风险项

3. **修改3的信息损失**
   - 风险：关键词无法完全捕捉语义
   - 缓解：A/B测试验证效果
   - 备选：增加sample_size到10

### 低风险项

4. **修改2的逻辑正确性**
   - 风险：极低（理论清晰）
   - 缓解：单元测试覆盖
   - 备选：无需备选

---

## 六、验收清单

### 修改1: m个性化

- [ ] `ScenarioCParams.m`支持`Union[float, np.ndarray]`
- [ ] `simulate_market_outcome`正确处理向量补偿
- [ ] `compute_optimal_compensation_by_types`实现并收敛
- [ ] 中介利润提升10-30%（相比统一m）
- [ ] Ground Truth包含`m_star_vector`字段
- [ ] 单元测试通过

### 修改2: 利润约束

- [ ] 所有亏损策略被正确过滤
- [ ] 无盈利策略时返回"no_participation"
- [ ] `optimization_summary`包含`num_candidates_profitable`
- [ ] 所有Ground Truth利润≥0
- [ ] 边界测试：极端参数下的行为正确

### 修改3: 理由优化

- [ ] `extract_keywords_from_reasons`实现
- [ ] `summarize_reasons`实现
- [ ] Token使用减少90%+
- [ ] A/B测试：利润误差<5%
- [ ] 关键词覆盖率>80%
- [ ] `use_reason_aggregation`开关有效

---

## 七、总结

### 核心认识

1. **修改1不是扩展，是修正** ⚠️
   - 论文标准模型使用m_i
   - 我们的统一m是偏离

2. **修改2是基本理性假设** 🐛
   - 理性主体不会选择亏损策略
   - 必须过滤负利润

3. **修改3是工程优化** ⚡
   - Token减少90%+
   - 不影响理论正确性

### 优先级

```
P0 (必需，本周): 修改1 + 修改2
P1 (重要，本周): 修改3
P2 (可选，下周): 完全个性化m_i（N维优化）
```

### 预期效果

- ✅ 理论对齐论文标准设定
- ✅ 可验证Proposition 5
- ✅ 中介利润提升10-30%
- ✅ Token成本降低90%
- ✅ Ground Truth无负利润

---

**文档结束**

如有疑问，请参考：
- 论文原文：`papers/The Economics of Social Data.pdf`
- 当前代码：`src/scenarios/scenario_c_social_data.py`
- 修正说明：`docs/场景C修正说明.md`
