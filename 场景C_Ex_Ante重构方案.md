# 场景C Ex Ante参与决策重构方案

## 📋 问题诊断

### **当前实现（Ex Post）**：
```python
# 1. 先生成realized状态
data = generate_consumer_data(params)  # 固定的(w,s)

# 2. 再计算参与决策（基于realized s_i）
for i in range(N):
    utility_accept = E[u_i | participate, r, realized data]  # ❌
    utility_reject = E[u_i | reject, r, realized data]       # ❌
```

**问题**: 消费者在决策时已经观察到了s_i，这是**interim/ex post决策**

---

### **论文要求（Ex Ante）**：
```python
# 1. 消费者在不知道(w,s)实现的情况下决策
# 2. 期望效用对所有随机性取平均：
#    - 信号realization
#    - 偏好realization  
#    - 他人参与集合
#    - 价格/需求
```

---

## 🔧 **重构方案**

### **方案A: 两层Monte Carlo（推荐）**

#### **算法结构**：
```python
def compute_expected_utility_ex_ante(
    consumer_id: int,
    participates: bool,
    others_participation_rate: float,
    params: ScenarioCParams,
    num_world_samples: int = 50,    # 外层：世界状态
    num_market_samples: int = 20     # 内层：参与者集合
) -> float:
    """
    Ex Ante期望效用：对所有随机性取平均
    
    外层循环：抽取世界状态(w, s)
    内层循环：抽取参与者集合
    """
    total_utility = 0.0
    
    # 外层：遍历可能的世界状态
    for world_sample in range(num_world_samples):
        # 1. 生成一个可能的世界状态
        data = generate_consumer_data_sample(params, seed=...)
        
        # 内层：在这个世界状态下，遍历可能的参与者集合
        world_utility = 0.0
        for market_sample in range(num_market_samples):
            # 2. 抽取他人参与决策
            participation = sample_participation(
                N, consumer_id, participates, 
                others_participation_rate
            )
            
            # 3. 模拟市场结果
            outcome = simulate_market_outcome(data, participation, params)
            
            # 4. 累加效用
            world_utility += outcome.utilities[consumer_id]
        
        total_utility += world_utility / num_market_samples
    
    return total_utility / num_world_samples
```

---

### **方案B: 引入异质性（学术标准）**

#### **为什么需要异质性**：
- 同质消费者下，ex ante均衡通常是r*∈{0,1}（全参与或全不参与）
- 无法产生内点参与率，难以做LLM偏差分析
- **学术标准做法**：引入隐私成本异质性τ_i

#### **数学模型**：
```
参与条件：
E[u_i | participate, r] + m - τ_i ≥ E[u_i | reject, r]

阈值：
τ_i ≤ ΔU(r) = E[u_i | 1, r] - E[u_i | 0, r] + m

参与率（固定点）：
r* = F_τ(ΔU(r*))

其中 F_τ 是τ_i的累积分布函数
```

#### **实现**：
```python
@dataclass
class ScenarioCParams:
    # ... 现有参数
    
    # 新增：隐私成本分布
    tau_mean: float = 0.5      # 隐私成本均值
    tau_std: float = 0.3       # 隐私成本标准差
    tau_dist: str = "normal"   # 分布类型（normal, uniform, lognormal）
    
    # 决策模式
    participation_timing: Literal["ex_ante", "interim", "ex_post"] = "ex_ante"

def compute_rational_participation_rate_ex_ante(
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_world_samples: int = 50,
    num_market_samples: int = 20
) -> Tuple[float, List[float]]:
    """
    Ex Ante固定点：使用异质性
    
    固定点方程：
    r = F_τ(ΔU(r))
    
    其中 ΔU(r) = E[u|1,r] - E[u|0,r] + m（代表性消费者）
    """
    r = 0.5
    r_history = [r]
    
    for iteration in range(max_iter):
        # 1. 计算代表性消费者的期望效用差
        # 注意：不需要为每个i计算，只要代表性agent即可
        u_accept = compute_expected_utility_ex_ante(
            consumer_id=0,  # 代表性消费者
            participates=True,
            others_participation_rate=r,
            params=params,
            num_world_samples=num_world_samples,
            num_market_samples=num_market_samples
        )
        
        u_reject = compute_expected_utility_ex_ante(
            consumer_id=0,
            participates=False,
            others_participation_rate=r,
            params=params,
            num_world_samples=num_world_samples,
            num_market_samples=num_market_samples
        )
        
        delta_u = u_accept - u_reject  # m已经在效用中计入
        
        # 2. 计算参与率：r = P(τ_i ≤ ΔU)
        if params.tau_dist == "normal":
            from scipy.stats import norm
            r_new = norm.cdf(delta_u, loc=params.tau_mean, scale=params.tau_std)
        elif params.tau_dist == "uniform":
            # τ_i ~ Uniform[tau_mean - sqrt(3)*tau_std, tau_mean + sqrt(3)*tau_std]
            a = params.tau_mean - np.sqrt(3) * params.tau_std
            b = params.tau_mean + np.sqrt(3) * params.tau_std
            r_new = np.clip((delta_u - a) / (b - a), 0, 1)
        else:
            raise ValueError(f"Unsupported tau_dist: {params.tau_dist}")
        
        r_history.append(r_new)
        
        # 3. 检查收敛
        if abs(r_new - r) < tol:
            print(f"  Ex Ante固定点收敛于迭代 {iteration + 1}, r* = {r_new:.4f}")
            return r_new, r_history
        
        # 4. 平滑更新
        r = 0.6 * r_new + 0.4 * r
    
    print(f"  警告: Ex Ante固定点未在{max_iter}次迭代内收敛, 当前 r = {r:.4f}")
    return r, r_history
```

---

## 📊 **实现路线图**

### **阶段1: 最小Ex Ante实现（保持同质性）** ⏱️ 2-3小时
```python
# 修改函数：
1. compute_expected_utility_given_participation() 
   → compute_expected_utility_ex_ante()
   - 外层循环：世界状态采样
   - 内层循环：参与者集合采样

2. compute_rational_participation_rate()
   → compute_rational_participation_rate_ex_ante()
   - 不传入fixed data
   - 调用新的期望效用函数

# 保留旧函数：
- 重命名为 *_ex_post()
- 标记为"扩展/鲁棒性"
```

### **阶段2: 引入异质性（学术标准）** ⏱️ 1-2小时
```python
# 新增参数：
- tau_mean, tau_std, tau_dist

# 修改固定点：
- 计算代表性ΔU
- 通过F_τ(ΔU)得到r*

# 好处：
- 内点参与率
- 学术上更标准
- 可解释性强
```

### **阶段3: 支持多种时序模式** ⏱️ 1小时
```python
params.participation_timing = "ex_ante"  # 主模型
params.participation_timing = "interim"   # 扩展：观察s_i后决策
params.participation_timing = "ex_post"   # 鲁棒性：当前实现

# 统一接口：
def compute_rational_participation_rate(
    params: ScenarioCParams, ...
):
    if params.participation_timing == "ex_ante":
        return compute_rational_participation_rate_ex_ante(...)
    elif params.participation_timing == "interim":
        return compute_rational_participation_rate_interim(...)
    else:  # ex_post
        return compute_rational_participation_rate_ex_post(...)
```

---

## 🎓 **学术叙事**

### **论文/报告中的写法**：

#### **主结果**：
> "我们的基准模型采用ex ante参与决策，与Acemoglu et al. (2022)的合约时序一致。消费者在观察到信号实现之前决定是否参与数据共享。为了产生内点参与率，我们引入隐私成本异质性τ_i ~ N(μ_τ, σ_τ)，这在隐私经济学文献中是标准做法（Acquisti et al., 2016）。"

#### **扩展/鲁棒性**：
> "作为鲁棒性检验，我们还考虑interim参与决策，即消费者在观察到私人信号s_i后再决定参与。这捕捉了现实中消费者可能在获得更多信息后才做决定的情况。结果见附录X。"

---

## 💻 **代码结构（重构后）**

```
src/scenarios/scenario_c_social_data.py
│
├── 核心参数类
│   └── ScenarioCParams
│       ├── 原有参数（N, data_structure, anonymization, ...）
│       ├── 异质性参数（tau_mean, tau_std, tau_dist）
│       └── 时序模式（participation_timing）
│
├── 数据生成（轻量化）
│   ├── generate_consumer_data_sample()  # 单次采样
│   └── generate_consumer_data()         # 批量生成（向后兼容）
│
├── Ex Ante期望效用（新）
│   └── compute_expected_utility_ex_ante()
│       ├── 外层循环：世界状态
│       └── 内层循环：参与者集合
│
├── Ex Ante固定点（新）
│   ├── compute_rational_participation_rate_ex_ante()  # 同质版
│   └── compute_rational_participation_rate_ex_ante_hetero()  # 异质版
│
├── Ex Post期望效用（旧，保留）
│   └── compute_expected_utility_ex_post()
│
├── 统一接口（新）
│   └── compute_rational_participation_rate()
│       └── 根据participation_timing分发
│
└── Ground Truth生成
    └── generate_ground_truth()
        └── 调用统一接口
```

---

## 🧪 **验证计划**

### **1. 理论一致性检验**：
```python
# 在无异质性、高补偿下，应该r*→1
params_high_m = ScenarioCParams(..., m=5.0, tau_mean=0.1, tau_std=0.01)
r_star, _ = compute_rational_participation_rate_ex_ante(params_high_m)
assert r_star > 0.95

# 在无异质性、低补偿下，应该r*→0
params_low_m = ScenarioCParams(..., m=0.0, tau_mean=2.0, tau_std=0.01)
r_star, _ = compute_rational_participation_rate_ex_ante(params_low_m)
assert r_star < 0.05
```

### **2. 对比Ex Ante vs Ex Post**：
```python
# 同一参数下，对比两种时序
r_ex_ante = compute_rate_ex_ante(params)
r_ex_post = compute_rate_ex_post(params, data)

print(f"Ex Ante: {r_ex_ante:.2%}")
print(f"Ex Post: {r_ex_post:.2%}")
# 分析差异及其经济学含义
```

### **3. 收敛性验证**：
```python
# 固定点应该稳定收敛
_, r_history = compute_rational_participation_rate_ex_ante(params)
plot_convergence(r_history)  # 应该是平滑的收敛曲线
```

---

## 📈 **预期影响**

### **学术可信度**：
- ✅ 与论文时序对齐
- ✅ 不会被审稿人质疑"求解了另一个模型"
- ✅ 异质性是标准做法

### **Benchmark质量**：
- ✅ 内点参与率（便于测LLM偏差）
- ✅ 可扩展（支持多种时序）
- ✅ 理论基础扎实

### **计算成本**：
- ⚠️ 两层MC会增加计算量（约50×20 = 1000倍）
- 💡 **优化**：使用Common Random Numbers减少方差
- 💡 **并行**：外层循环可并行化

---

## 🚀 **立即行动**

### **最小可行方案（MVP）**：
1. ✅ 实现`compute_expected_utility_ex_ante()`（两层MC）
2. ✅ 修改固定点函数（不传入fixed data）
3. ✅ 测试收敛性和合理性

### **学术标准方案**：
4. ✅ 引入τ_i异质性
5. ✅ 实现异质版固定点
6. ✅ 对比Ex Ante vs Ex Post结果

### **完整系统**：
7. ✅ 支持多种时序模式（统一接口）
8. ✅ 更新文档和报告
9. ✅ 重新生成Ground Truth

---

**预计总工时**：4-6小时（包含测试和文档）

**优先级**：🔴 **最高**（影响学术可信度）

**建议**：先做MVP验证可行性，再扩展到完整系统。
