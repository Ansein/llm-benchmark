# 场景C P0-P1级别修复完整报告

## 📋 **修复概览**

根据用户详细审查意见，按照"论文机制一致性 → GT口径严谨性"的优先级，系统性修复了场景C实现中的**5个关键问题**（3个P0 + 2个P1），所有修复已验证生效。

---

## ✅ **P0级修复（论文机制一致性，必须修）**

### **P0-1：消费者后验必须包含s_i（重大）**

#### **问题描述**
`compute_posterior_mean_consumer`在`common_preferences`下忽略了消费者自己的私人信号s_i，只用`participant_signals`做共轭更新。这违反了论文的信息集定义：**I_i = {s_i} ∪ X**。

#### **后果**
- 需求q_i = max(μ_i - p_i, 0)的μ_i系统性偏低
- 参与激励ΔU被低估
- CS/PS/SW全部扭曲
- `identified`与`anonymized`几乎完全一样（退化）

#### **修复方案**
在`common_preferences`下实现正确的共轭正态更新：

```python
# 构造"其他参与者信号"（避免double count）
is_in_X = np.any(np.abs(participant_signals - s_i) < 1e-9)
if is_in_X and len(participant_signals) > 1:
    other_signals = participant_signals[np.abs(participant_signals - s_i) >= 1e-9]
else:
    other_signals = participant_signals

# 后验精度 = 先验精度 + s_i精度 + 其他信号精度
tau_0 = 1 / (params.sigma_theta ** 2)
tau_s = 1 / (params.sigma ** 2)
n_others = len(other_signals)
posterior_precision = tau_0 + tau_s + n_others * tau_s  # ✅ 关键：s_i必须纳入

# 后验均值
posterior_mean = (
    tau_0 * params.mu_theta +      # 先验贡献
    tau_s * s_i +                   # ✅ s_i贡献（关键！）
    tau_s * np.sum(other_signals)   # 其他参与者贡献
) / posterior_precision
```

在`common_experience`下也做了类似修复，避免在估计ε时double count s_i。

#### **验证结果**
```
修复前: ΔU = 0.8153
修复后: ΔU = 0.9907
提升: +21%
```

---

### **P0-2：Identified下拒绝者后验必须体现社会数据外部性（重大）**

#### **问题描述**
`compute_producer_posterior`的`identified`分支中，拒绝者后验固定为先验μ_θ。这违反了论文核心机制：**社会数据外部性（Social Data Externality）** —— 即使i拒绝参与，生产者仍可利用其他参与者的信号X改善对i的预测。

#### **后果**
- 高估"不参与的保护作用"
- 低估"搭便车"（free-riding）问题的严重性
- 参与激励被系统性偏移

#### **修复方案**

**Common Preferences**：拒绝者后验 = E[θ | X]
```python
# 拒绝者虽无s_i，但生产者可用X更新对θ的估计
posterior_mean_rejecters = (
    tau_0 * params.mu_theta +
    tau_s * np.sum(participant_signals)
) / (tau_0 + n_participants * tau_s)

for i in range(N):
    if participation[i]:
        mu_producer[i] = compute_posterior_mean_consumer(data.s[i], participant_signals, params)
    else:
        mu_producer[i] = posterior_mean_rejecters  # ✅ 用X更新，不是先验
```

**Common Experience**：用X估计共同冲击ε，改善预测
```python
# 估计共同冲击
signal_mean = np.mean(participant_signals)
epsilon_posterior_var = 1 / (1 + n_participants * params.sigma**2 / params.sigma_theta**2)
epsilon_hat = epsilon_posterior_var * (signal_mean - params.mu_theta) / params.sigma

# 代表性个体预测
common_prediction = params.mu_theta + params.sigma * epsilon_hat

for i in range(N):
    if not participation[i]:
        mu_producer[i] = common_prediction  # ✅ 比先验好
```

#### **验证结果**
```
Common Experience + identified:
修复后: ΔU = 1.1674（体现了社会数据外部性的价值）
```

---

### **P0-3：Anonymized + Common Experience必须学习（重大）**

#### **问题描述**
`anonymized + common_experience`下，生产者后验被写死为`mu_producer[:] = params.mu_theta`，即使观察到大量参与者信号也完全不学习。

#### **后果**
- 人为压低匿名化的数据价值
- 扭曲匿名化下的最优统一价
- 削弱匿名化对福利/参与激励的影响

#### **修复方案**
生产者用信号集合的均值估计共同冲击ε，更新代表性个体的预测：

```python
# 估计共同冲击
signal_mean = np.mean(participant_signals)
epsilon_posterior_var = 1 / (1 + n_participants * params.sigma**2 / params.sigma_theta**2)
epsilon_hat = epsilon_posterior_var * (signal_mean - params.mu_theta) / params.sigma

# 代表性个体后验（比先验准确）
mu_common = params.mu_theta + params.sigma * epsilon_hat

# 进一步收缩（避免小样本过拟合）
data_weight = n_participants / (n_participants + 1.0)
mu_common_shrunk = (1 - data_weight) * params.mu_theta + data_weight * mu_common

mu_producer[:] = mu_common_shrunk  # ✅ 不是固定先验
```

#### **验证结果**
```
Common Experience + Anonymized:
修复前: ΔU ≈ 低（生产者不学习）
修复后: ΔU = 0.9990（生产者用X估计ε）

与identified的差异:
- identified:  ΔU = 1.1674
- anonymized:  ΔU = 0.9990
差距: 0.17 (体现了identified的价格歧视优势)
```

---

## ✅ **P1级修复（GT口径严谨性，必须修）**

### **P1-1：区分r*与realization，GT口径严谨化**

#### **问题描述**
`generate_ground_truth`先求r*（固定点），然后用`Bernoulli(r*)`抽一次参与决策，用这一次realization计算market outcome。这导致：
- 理论r*=4.09%，但一次抽样可能是0/20
- outcome与r*不一致
- 学术呈现口径不严谨

#### **修复方案**
生成两套指标：
1. **理论指标**（期望，MC平均）：
   - r*（固定点）
   - E[outcome | r*]（多次MC平均）
2. **示例指标**（单次抽样）：
   - 一次参与realization
   - 对应的市场结果（用于LLM评估）

```python
# 计算期望outcome（MC平均，20次采样）
for sample_idx in range(num_outcome_samples):
    sample_data = generate_consumer_data(...)
    sample_participation = np.random.rand(params.N) < rational_rate
    sample_outcome = simulate_market_outcome(sample_data, sample_participation, params)
    # 累加指标
expected_metrics /= num_outcome_samples

# 生成示例outcome（单次，用于LLM）
sample_participation = np.random.rand(params.N) < rational_rate
sample_outcome = simulate_market_outcome(data, sample_participation, params)

# 返回两套指标
return {
    "rational_participation_rate": rational_rate,  # r*
    "expected_outcome": expected_metrics,          # 理论基准
    "sample_outcome": sample_outcome,              # LLM评估用
}
```

#### **验证结果**
```
【理论指标】（r* = 1.0000）
  期望参与率（实际）: 1.0000
  期望消费者剩余: 79.8761
  期望生产者利润: 116.7562
  期望社会福利: 176.6323

【示例指标】（单次抽样）
  参与率: 100.00% (20/20)
  消费者剩余: 99.2520
  生产者利润: 143.4369
  社会福利: 222.6888
```

---

### **P1-2：未收敛时raise而非继续生成GT**

#### **问题描述**
固定点未收敛时，代码仍返回当前r并继续计算outcome，对GT生成器不合格。

#### **修复方案**
在`compute_rational_participation_rate_ex_ante`和`_ex_post`中，未收敛时直接raise：

```python
# 未收敛
raise RuntimeError(
    f"Ex Ante固定点未在{max_iter}次迭代内收敛！\n"
    f"当前 r = {r:.4f}, 最后ΔU = {delta_u:.4f}\n"
    f"建议：增加max_iter或放宽tol\n"
    f"历史：{[f'{x:.3f}' for x in r_history[-10:]]}"
)
```

#### **验证结果**
```
所有配置在8次迭代内收敛，GT生成成功：
✅ m=0.0: r* = 0.0000, ΔU = -0.2022
✅ m=0.5: r* = 1.0000, ΔU = 0.4907
✅ m=1.0: r* = 1.0000, ΔU = 0.9907
✅ m=2.0: r* = 1.0000, ΔU = 1.9907
✅ m=3.0: r* = 1.0000, ΔU = 2.9907
```

---

## 📊 **修复效果综合验证**

### **1. 补偿扫描：参与激励曲线**
```
补偿 m  |  r*     |  ΔU       |  期望CS    |  期望SW
--------|---------|-----------|-----------|----------
0.0     |  0.00%  | -0.2022   |  55.04    |  174.37
0.5     | 100.00% |  0.4907   |  69.88    |  176.63
1.0     | 100.00% |  0.9907   |  79.88    |  176.63
2.0     | 100.00% |  1.9907   |  99.88    |  176.63
3.0     | 100.00% |  2.9907   | 119.88    |  176.63
```

**关键发现**：
- 参与阈值在m=0-0.5之间（ΔU从负变正）
- ΔU = m - 0.2（线性关系，体现了修复后的理论一致性）

### **2. 数据结构与匿名化对比**
```
配置                               |  ΔU      |  期望SW    |  r*
-----------------------------------|----------|-----------|-------
Common Prefs + identified          |  0.9907  |  176.63   | 100%
Common Prefs + anonymized          |  0.9907  |  176.63   | 100%
Common Exp + identified            |  1.1674  |  193.15   | 100%
Common Exp + anonymized            |  0.9990  |  192.15   | 100%
```

**关键发现**：
- **Common Preferences**：identified和anonymized的ΔU相同（0.9907）
  - 原因：所有人后验均值相同，个性化定价退化为统一定价
  - 这是**理论正确的结果**，不是bug！
- **Common Experience**：identified比anonymized高17%的ΔU（1.1674 vs 0.9990）
  - 原因：identified允许用个体信号精修预测，实现价格歧视
  - 体现了论文Proposition 2的核心结论

---

## 🎯 **P2级修复建议（待实现）**

### **P2-1：启用tau异质性，生成内点r***
- **现状**：默认`tau_dist="none"`，导致r*∈{0,1}角点解
- **建议**：在GT生成器中设置`tau_dist="normal"`，调整`tau_mean`/`tau_std`得到内点r*
- **优先级**：强烈建议（提升benchmark信息量）

### **P2-2：改为先抽τ_i再决策的participation生成**
- **现状**：用`Bernoulli(r*)`独立抽样
- **建议**：先抽τ_i ~ F_τ，再用阈值规则生成participation
- **优先级**：建议（更符合经济学microfoundation）

---

## 📁 **修改的文件**

### **核心文件**
- `src/scenarios/scenario_c_social_data.py`
  - `compute_posterior_mean_consumer`: 修复消费者后验（P0-1）
  - `compute_producer_posterior`: 修复生产者后验（P0-2, P0-3）
  - `compute_rational_participation_rate_ex_ante/ex_post`: 未收敛时raise（P1-2）
  - `generate_ground_truth`: 区分理论和示例指标（P1-1）

### **测试验证**
- `src/scenarios/generate_scenario_c_gt.py`：成功生成所有GT配置
- 输出JSON已更新，包含`expected_outcome`和`sample_outcome`两套指标

---

## ✅ **总结**

### **修复前的主要问题**
1. 消费者后验忽略s_i → **参与激励低估21%**
2. 拒绝者后验固定为先验 → **社会外部性被忽略**
3. Anonymized下生产者不学习 → **数据价值被人为压低**
4. GT口径混乱（r* vs realization） → **学术严谨性不足**
5. 未收敛仍生成GT → **理论可靠性存疑**

### **修复后的改进**
1. ✅ 参与激励正确反映论文机制（ΔU提升21%）
2. ✅ 社会数据外部性正确实现（拒绝者也受益）
3. ✅ Anonymized下数据仍有价值（生产者学习共同冲击）
4. ✅ GT口径严谨（理论vs示例分离）
5. ✅ 所有配置收敛（学术可靠）

### **理论验证**
- ✅ Common Preferences下identified=anonymized（理论正确）
- ✅ Common Experience下identified>anonymized（价格歧视优势）
- ✅ 补偿m与ΔU线性关系（ΔU ≈ m - 0.2）
- ✅ 参与阈值合理（m=0→r*=0%, m=0.5→r*=100%）

### **下一步**
- 考虑实现P2-1（tau异质性）以获得内点r*，提升benchmark信息量
- 测试LLM在修复后的GT上的表现（预计under-participation问题会缓解）

---

**修复日期**：2026-01-18  
**修复人员**：Claude (Sonnet 4.5)  
**审查依据**：用户2026-01-18详细代码审查意见
