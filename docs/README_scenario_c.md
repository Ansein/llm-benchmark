# 场景C：The Economics of Social Data 使用说明

## 📋 场景概述

场景C基于Bergemann, Bonatti, Gan (2022)的论文《The Economics of Social Data》，研究社会数据的外部性和匿名化政策的影响。

### 核心机制

1. **社会数据外部性**
   - 个人数据不仅预测自己的行为，也预测他人的行为
   - 数据的相关性结构决定外部性的强度

2. **搭便车问题**
   - 拒绝者仍能从参与者的数据中学习
   - 降低了个体参与的激励

3. **匿名化政策**
   - 实名制：允许个性化定价，增加隐私风险
   - 匿名化：只能统一定价，保护消费者

4. **参与决策**
   - 消费者权衡：补偿 + 学习收益 vs 价格歧视风险
   - 理性均衡：固定点参与率

---

## 🏗️ 两种数据结构

### Common Preferences（共同偏好）

**设定**：
```python
w_i = θ  for all i = 1, ..., N
e_i ~ i.i.d. N(0, 1)
s_i = θ + σ·e_i
```

**特点**：
- 所有消费者对产品的真实价值相同（θ）
- 但每个人的初步评估有独立噪声
- 多人数据可以通过平均滤掉噪声

**学习价值**：
```
E[θ | s_1, ..., s_N] ≈ (1/N) Σ s_i  (N大时)
```

### Common Experience（共同经历）

**设定**：
```python
w_i ~ i.i.d. N(μ, σ²)
e_i = ε  for all i
s_i = w_i + σ·ε
```

**特点**：
- 每个消费者的真实偏好不同
- 但所有人受到相同的噪声冲击（ε）
- 多人数据可以识别并过滤共同噪声

**学习价值**：
```
估计 ε → 过滤 → 更准确估计各自的 w_i
```

---

## 📐 数学模型

### 产品市场

**消费者效用**：
```
u_i = w_i·q_i - p_i·q_i - (1/2)·q_i²
```

**最优需求**：
```
q_i* = max{μ_i - p_i, 0}
其中 μ_i = E[w_i | s_i, Y_i]
```

**间接效用**：
```
v_i(μ_i, p_i) = (1/2)·max{μ_i - p_i, 0}²
```

### 定价机制

**个性化定价**（实名制下）：
```
p_i* = (μ_i + c) / 2  (闭式解)
```

**统一定价**（匿名化下）：
```
p* = argmax_p Σ_i (p - c)·max{μ_i - p, 0}
```

### 参与决策

**消费者i的参与条件**：
```
E[u_i | participate] + m ≥ E[u_i | not participate]
```

**理性均衡**（固定点）：
```
r* = Pr(consumer participates | others participate with rate r*)
```

---

## 🚀 快速开始

### 1. 测试场景C实现

```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"
python test_scenario_c.py
```

**测试内容**：
- ✅ 数据生成（两种结构）
- ✅ 后验估计算法
- ✅ 市场模拟
- ✅ 匿名化对比
- ✅ Ground Truth生成

### 2. 生成Ground Truth

```bash
python generate_scenario_c_gt.py
```

**生成内容**：
- MVP配置：Common Preferences + Identified
- 核心对比：2种数据结构 × 2种匿名化 = 4个配置
- 补偿扫描：5个补偿水平（绘制参与率曲线）

**输出文件**：
```
data/ground_truth/
├── scenario_c_result.json                        # MVP配置（默认）
├── scenario_c_common_preferences_identified.json
├── scenario_c_common_preferences_anonymized.json
├── scenario_c_common_experience_identified.json
├── scenario_c_common_experience_anonymized.json
└── scenario_c_payment_sweep.json                 # 补偿扫描
```

### 3. 运行LLM评估

```bash
# 确保已配置LLM API
# 编辑 configs/model_configs.json

# 单个模型评估
python -m src.evaluators.evaluate_scenario_c

# 或使用主评估脚本（待集成）
python run_evaluation.py --scenarios C --models gpt-4.1-mini
```

---

## 📊 Ground Truth示例

### MVP配置参数

```python
ScenarioCParams(
    N=20,                                # 20个消费者
    data_structure="common_preferences",  # 共同偏好
    anonymization="identified",           # 实名制
    mu_theta=5.0,                        # 先验均值
    sigma_theta=1.0,                     # 先验标准差
    sigma=1.0,                           # 噪声水平
    m=1.0,                               # 补偿金额
    c=0.0,                               # 边际成本
    seed=42
)
```

### 预期结果（示例）

```json
{
  "rational_participation_rate": 0.65,
  "outcome": {
    "consumer_surplus": 12.34,
    "producer_profit": 8.56,
    "social_welfare": 20.90,
    "gini_coefficient": 0.15,
    "price_discrimination_index": 1.23
  }
}
```

---

## 🎯 评估指标

### 主要指标

| 指标 | 含义 | 计算方法 |
|------|------|---------|
| **参与率偏差** | LLM vs 理论参与率 | \|PR_LLM - PR_theory\| |
| **消费者剩余偏差** | 福利差异 | \|CS_LLM - CS_theory\| |
| **生产者利润偏差** | 利润差异 | \|π_LLM - π_theory\| |
| **社会福利偏差** | 总福利差异 | \|SW_LLM - SW_theory\| |

### 次要指标

| 指标 | 含义 |
|------|------|
| **Gini系数** | 效用不平等程度（0-1） |
| **价格歧视指数** | max(p_i) - min(p_i) |
| **学习质量** | mean\|μ_i - w_i\|（参与者 vs 拒绝者） |
| **搭便车收益** | 拒绝者的学习收益 |

### 标签分类

**参与率分桶**：
- Low: < 33%
- Medium: 33-67%
- High: > 67%

**方向标签**：
- Match: \|rate_diff\| < 10%
- Over-participation: rate_diff > 10%
- Under-participation: rate_diff < -10%

---

## 🔬 实验设计

### Phase 1: MVP（最小可行版本）

**目标**：验证框架，得到初步结果

**配置**：
```
N = 20
data_structure = common_preferences
anonymization = [identified, anonymized]
sigma = 1.0
m = [0, 0.5, 1.0, 2.0, 3.0]
seeds = 10

总计: 2 × 5 × 10 = 100 runs
```

### Phase 2: 核心扩展

**目标**：覆盖两种数据结构，探索噪声影响

**配置**：
```
N = 20
data_structure = [common_preferences, common_experience]
anonymization = [identified, anonymized]
sigma = [0.5, 1.0, 2.0]
m = [0, 0.5, 1.0, 2.0, 3.0]
seeds = 20

总计: 2 × 2 × 3 × 5 × 20 = 1200 runs
```

### Phase 3: 完整benchmark

**目标**：市场规模效应，更细粒度

**配置**：
```
N = [10, 20, 50, 100]
data_structure = [common_preferences, common_experience]
anonymization = [identified, anonymized]
sigma = [0.5, 1.0, 2.0]
m = linspace(0, 5, 11)
seeds = 50
```

---

## 📈 预期发现

### H1: 参与率偏离

**假设**：LLM在实名制下参与率**低于**理论
- **原因**：高估价格歧视风险，展现风险厌恶

### H2: 匿名化效应

**假设**：匿名化会提高LLM参与率
- **原因**：隐私保护缓解顾虑

### H3: 搭便车行为

**假设**：LLM会过度搭便车
- **原因**：理解"拒绝仍能学习"的收益

### H4: 福利影响

**假设**：LLM的低参与降低社会福利
- **原因**：数据稀缺 → 估计不准 → 定价低效

---

## 🛠️ 技术实现要点

### 后验估计（贝叶斯更新）

**Common Preferences**：
```python
def posterior_theta(signals, mu_theta, sigma_theta, sigma):
    n = len(signals)
    prior_precision = 1 / sigma_theta**2
    likelihood_precision = n / sigma**2
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_mean = (prior_precision * mu_theta + 
                     likelihood_precision * mean(signals)) / posterior_precision
    
    return posterior_mean
```

**Common Experience**：
```python
def posterior_wi(s_i, all_signals, mu_w, sigma_w, sigma):
    # 1. 估计共同噪声 ε
    epsilon_hat = estimate_common_noise(all_signals, mu_w, sigma_w, sigma)
    
    # 2. 过滤噪声
    filtered_signal = s_i - sigma * epsilon_hat
    
    # 3. 贝叶斯更新
    posterior_mean = bayesian_update(filtered_signal, mu_w, sigma_w)
    
    return posterior_mean
```

### 固定点迭代

```python
def compute_rational_participation_rate(data, params):
    r = 0.5  # 初始参与率
    
    for iteration in range(max_iter):
        # 在参与率r下，计算每个消费者的效用差
        accept_decisions = []
        for i in range(N):
            utility_accept = expected_utility(i, True, r)
            utility_reject = expected_utility(i, False, r)
            delta_u = utility_accept - utility_reject
            should_accept = (delta_u + m) > 0
            accept_decisions.append(should_accept)
        
        # 更新参与率
        r_new = mean(accept_decisions)
        
        # 检查收敛
        if abs(r_new - r) < tol:
            return r_new
        
        r = 0.6 * r_new + 0.4 * r  # 平滑更新
    
    return r
```

---

## 📝 文件结构

```
场景C相关文件:
├── src/
│   ├── scenarios/
│   │   └── scenario_c_social_data.py      # 理论求解器
│   └── evaluators/
│       └── evaluate_scenario_c.py         # LLM评估器
├── docs/
│   ├── README_scenario_c.md               # 本文件
│   └── 论文解析_The_Economics_of_Social_Data.md
├── data/
│   └── ground_truth/
│       ├── scenario_c_result.json         # MVP配置
│       ├── scenario_c_*.json              # 其他配置
│       └── scenario_c_payment_sweep.json  # 补偿扫描
├── test_scenario_c.py                     # 单元测试
├── generate_scenario_c_gt.py              # GT生成器
└── 场景C新方案.md                         # 设计文档
```

---

## 🔗 相关论文

**主要论文**：
- Bergemann, D., Bonatti, A., & Gan, T. (2022). "The Economics of Social Data"
  - **核心贡献**：社会数据外部性、匿名化最优性、中介渐近利润

**相关工作**：
- Taylor (2004): "Consumer Privacy and the Market for Customer Information"
- Lizzeri (1999): "Information Revelation and Certification Intermediaries"
- Acquisti et al. (2016): "The Economics of Privacy" (综述)

---

## 💡 使用技巧

### 1. 调试Ground Truth生成

如果固定点不收敛，尝试：
- 减少蒙特卡洛样本数（从50降到20）
- 增加收敛容差（从1e-3增到1e-2）
- 减少消费者数量（从20降到10）

### 2. 加快评估速度

- 减少`num_trials`（从3降到1）
- 减少`max_iterations`（从10降到5）
- 使用更快的模型（如grok-3-mini）

### 3. 理解结果

**如果LLM过度拒绝**：
- 可能高估价格歧视风险
- 可能低估搭便车收益
- 可能不理解匿名化保护

**如果LLM过度参与**：
- 可能低估价格歧视风险
- 可能高估补偿价值
- 可能忽略搭便车机会

---

## 🐛 常见问题

### Q1: Ground Truth生成太慢？

**A**: 降低`num_mc_samples`参数：
```python
gt = generate_ground_truth(params, num_mc_samples=20)  # 默认50
```

### Q2: 固定点不收敛？

**A**: 检查参数设置，特别是`m`和`sigma`的组合。某些参数下可能存在多重均衡。

### Q3: LLM响应格式错误？

**A**: 检查prompt是否清晰，必要时增加格式示例。评估器会自动处理大多数格式问题。

### Q4: 如何添加新的数据结构？

**A**: 在`generate_consumer_data`和`compute_posterior_mean_consumer`中添加新的case。

---

## 📞 技术支持

如有问题，请查看：
1. 本文档（README_scenario_c.md）
2. 设计文档（场景C新方案.md）
3. 论文解析（论文解析_The_Economics_of_Social_Data.md）
4. 代码注释（scenario_c_social_data.py）

---

**祝实验顺利！** 🎉
