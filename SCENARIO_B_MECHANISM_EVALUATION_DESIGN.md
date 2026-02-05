# 场景B虚拟博弈机制理解能力评估指标设计

## 🎯 设计目标

在场景B（推断外部性博弈）中，除了评估LLM决策结果与贝叶斯纳什均衡(BNE)的数值距离外，更重要的是评估**LLM对隐私外部性机制的理解能力**。

### 核心思想
- **数值准确度**：LLM的结果是否接近理论均衡（已有指标：Jaccard相似度、利润MAE等）
- **机制理解能力**（新增）：LLM是否理解变量间的因果关系和弹性特征

---

## 📊 场景B的核心变量

### 1. 结果变量（Outcomes）
- **share_rate**：分享率（|S|/n，S为分享集合）
- **profit**：平台利润
- **welfare**：社会福利（所有参与者效用之和）
- **total_leakage**：总信息泄露量

### 2. 参数变量（Parameters）
- **n**：用户数量
- **rho**：信息相关系数（控制推断外部性强度）
- **sigma_sq**：观测噪声
- **v_mean**：隐私偏好均值
- **price_level**：价格水平（如均值或中位数）

---

## 🔧 三个新指标设计

### 指标1：TC (Trend Consistency) - 趋势一致性

**定义**：衡量LLM结果在不同参数设置下，各核心指标与BNE的趋势一致性。

#### 计算方法

1. **准备多个实例**：运行多组实验，改变关键参数（如rho, v_mean, price）
2. **对每个核心指标**，计算LLM结果与BNE之间的Spearman秩相关系数
3. **取平均**：

```python
indicators = ['share_rate', 'profit', 'welfare', 'total_leakage']

TC_scores = []
for indicator in indicators:
    # 提取所有实例的该指标值
    llm_values = [result_i[indicator] for result_i in llm_results]
    bne_values = [result_i[indicator] for result_i in bne_results]
    
    # 计算Spearman秩相关系数
    rho_spearman, p_value = scipy.stats.spearmanr(llm_values, bne_values)
    TC_scores.append(rho_spearman)

TC = np.mean(TC_scores)  # 范围 [-1, 1]，越接近1越好
```

#### 解释
- **TC ≈ 1**：LLM的趋势与BNE完全一致，说明理解了参数如何影响结果
- **TC ≈ 0**：无相关性，LLM的变化与BNE无关
- **TC < 0**：负相关，LLM的理解可能是反向的

---

### 指标2：DCS (Direction Consistency Score) - 方向一致性

**定义**：衡量LLM对关键机制关系方向的理解是否正确。

#### 五对关键机制关系

基于场景B的理论机制，定义以下机制对：

| 机制对 | 自变量 | 因变量 | 理论预期方向 | 经济含义 |
|--------|--------|--------|-------------|----------|
| M1 | price_level | share_rate | + | 价格越高，分享意愿越强 |
| M2 | rho | share_rate | - | 相关性越高，次模性越强，边际成本越低，但总体分享可能减少（需验证） |
| M3 | share_rate | total_leakage | + | 分享的人越多，总泄露越大（推断外部性） |
| M4 | v_mean | share_rate | - | 隐私偏好越高，分享越少 |
| M5 | share_rate | welfare | 非单调 | 存在最优分享率（次模性权衡） |

#### 计算方法

对每对机制关系，使用线性回归估计斜率：

```python
from sklearn.linear_model import LinearRegression

mechanism_pairs = [
    ('price_level', 'share_rate', '+'),
    ('rho', 'share_rate', '-'),
    ('share_rate', 'total_leakage', '+'),
    ('v_mean', 'share_rate', '-'),
    # M5较复杂，可用二次回归或分段处理
]

DCS_scores = []
for x_var, y_var, expected_direction in mechanism_pairs:
    # 提取数据
    X_llm = np.array([r[x_var] for r in llm_results]).reshape(-1, 1)
    y_llm = np.array([r[y_var] for r in llm_results])
    
    X_bne = np.array([r[x_var] for r in bne_results]).reshape(-1, 1)
    y_bne = np.array([r[y_var] for r in bne_results])
    
    # 拟合线性回归
    model_llm = LinearRegression().fit(X_llm, y_llm)
    model_bne = LinearRegression().fit(X_bne, y_bne)
    
    slope_llm = model_llm.coef_[0]
    slope_bne = model_bne.coef_[0]
    
    # 判断方向是否一致
    if slope_llm * slope_bne > 0:  # 同号
        DCS_scores.append(1)
    else:
        DCS_scores.append(0)

DCS = np.mean(DCS_scores)  # 范围 [0, 1]
```

#### 解释
- **DCS = 1**：所有机制关系方向都正确
- **DCS = 0.6**：60%的机制关系方向正确
- **DCS = 0**：所有方向都错误（极端情况）

---

### 指标3：EAS (Elasticity Alignment Score) - 弹性对齐分数

**定义**：衡量LLM对机制关系弹性大小的理解，即变化幅度是否合理。

#### 弹性定义

对于机制对 `X → Y`，定义弹性为标准化斜率：

```
Elasticity = β * (σ_X / σ_Y)
```

其中：
- `β` 是回归斜率
- `σ_X`, `σ_Y` 是X和Y的标准差

#### 计算方法

```python
EAS_scores = []

for x_var, y_var, _ in mechanism_pairs[:-1]:  # 排除非单调的M5
    # 计算LLM的弹性
    X_llm = np.array([r[x_var] for r in llm_results]).reshape(-1, 1)
    y_llm = np.array([r[y_var] for r in llm_results])
    
    model_llm = LinearRegression().fit(X_llm, y_llm)
    slope_llm = model_llm.coef_[0]
    elasticity_llm = slope_llm * (np.std(X_llm) / np.std(y_llm))
    
    # 计算BNE的弹性
    X_bne = np.array([r[x_var] for r in bne_results]).reshape(-1, 1)
    y_bne = np.array([r[y_var] for r in bne_results])
    
    model_bne = LinearRegression().fit(X_bne, y_bne)
    slope_bne = model_bne.coef_[0]
    elasticity_bne = slope_bne * (np.std(X_bne) / np.std(y_bne))
    
    # 避免除零
    if abs(elasticity_bne) < 1e-6:
        continue
    
    # 计算弹性比的对数衰减
    ratio = elasticity_llm / elasticity_bne
    score = np.exp(-abs(np.log(abs(ratio))))  # 范围 [0, 1]
    
    EAS_scores.append(score)

EAS = np.mean(EAS_scores)  # 范围 [0, 1]
```

#### 解释
- **EAS = 1**：LLM的弹性与BNE完全一致
- **EAS = 0.5**：弹性比约为e或1/e（相差约2.7倍）
- **EAS → 0**：弹性差异极大

---

## 📐 综合评估框架

### 机制理解能力综合得分

```python
MUS = (TC + 1) / 2 * 0.4 + DCS * 0.3 + EAS * 0.3
```

**说明**：
- `(TC + 1) / 2`：将TC从[-1, 1]映射到[0, 1]
- 权重分配：
  - TC (40%)：趋势一致性最重要
  - DCS (30%)：方向正确是基础
  - EAS (30%)：弹性对齐是进阶

**MUS范围**：[0, 1]，越高越好

---

### 数值准确度综合得分

使用标准化欧氏距离：

```python
# 计算4个核心指标的标准化距离
indicators = ['share_rate', 'profit', 'welfare', 'total_leakage']

distances = []
for ind in indicators:
    llm_val = llm_result[ind]
    bne_val = bne_result[ind]
    
    # 标准化
    range_val = max_val[ind] - min_val[ind]  # 跨实例的值域
    normalized_dist = abs(llm_val - bne_val) / range_val
    distances.append(normalized_dist)

Euclidean_Distance = np.sqrt(np.mean([d**2 for d in distances]))
```

**距离范围**：[0, +∞)，越小越好

---

## 📊 四象限可视化

### 坐标轴定义

- **横轴（X）**：Euclidean Distance（数值准确度）
  - 越小越好（左侧）
  - 范围：[0, 1+]

- **纵轴（Y）**：MUS（机制理解能力）
  - 越高越好（上方）
  - 范围：[0, 1]

### 四象限划分

设定阈值：
- `distance_threshold = 0.3`（可根据数据调整）
- `mus_threshold = 0.6`

| 象限 | 条件 | 评价 | 含义 |
|------|------|------|------|
| **I（右上）** | 距离大 & MUS高 | **理解对但不准确** | LLM理解机制，但可能因其他因素（如随机性、次优策略）导致结果偏离均衡 |
| **II（左上）** | 距离小 & MUS高 | **理想区域** | LLM既理解机制又能达到均衡附近 |
| **III（左下）** | 距离小 & MUS低 | **巧合区域** | 结果碰巧接近均衡，但不理解机制（可能记忆或过拟合） |
| **IV（右下）** | 距离大 & MUS低 | **两者都不行** | 既不理解机制也达不到均衡 |

### 可视化代码框架

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制每个模型的点
for model_name, results in all_model_results.items():
    x = results['euclidean_distance']
    y = results['MUS']
    ax.scatter(x, y, s=100, label=model_name, alpha=0.7)
    ax.text(x+0.01, y+0.01, model_name, fontsize=9)

# 绘制象限分割线
ax.axvline(x=distance_threshold, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=mus_threshold, color='gray', linestyle='--', alpha=0.5)

# 标注象限
ax.text(0.6, 0.85, 'I: High MUS, High Dist\n(Understands but Inaccurate)', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.text(0.1, 0.85, 'II: High MUS, Low Dist\n(Ideal)', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.1, 0.3, 'III: Low MUS, Low Dist\n(Lucky)', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.6, 0.3, 'IV: Low MUS, High Dist\n(Poor)', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

ax.set_xlabel('Euclidean Distance to BNE', fontsize=12, fontfamily='Times New Roman')
ax.set_ylabel('Mechanism Understanding Score (MUS)', fontsize=12, fontfamily='Times New Roman')
ax.set_title('LLM Performance: Accuracy vs. Mechanism Understanding', 
            fontsize=14, fontweight='bold', fontfamily='Times New Roman')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('scenario_b_mechanism_evaluation.png', dpi=150, bbox_inches='tight')
```

---

## 🔬 实施步骤

### Step 1：数据准备

需要收集多组实验数据，变化关键参数：

```python
# 参数网格
param_grid = {
    'rho': [0.3, 0.5, 0.7, 0.9],
    'v_mean': [0.5, 0.75, 1.0],
    'price_level': [0.3, 0.5, 0.7]
}

# 对每组参数，运行：
# 1. BNE计算（理论解）
# 2. LLM虚拟博弈（多个模型）
```

**最少需要**：10-15组不同参数组合的实验

### Step 2：计算三个指标

```python
def calculate_mechanism_scores(llm_results_list, bne_results_list):
    """
    Args:
        llm_results_list: List[Dict]，每个Dict包含一组实验的结果
        bne_results_list: List[Dict]，对应的BNE结果
    
    Returns:
        Dict: {'TC': float, 'DCS': float, 'EAS': float, 'MUS': float}
    """
    # 计算TC
    TC = compute_trend_consistency(llm_results_list, bne_results_list)
    
    # 计算DCS
    DCS = compute_direction_consistency(llm_results_list, bne_results_list)
    
    # 计算EAS
    EAS = compute_elasticity_alignment(llm_results_list, bne_results_list)
    
    # 综合MUS
    MUS = (TC + 1) / 2 * 0.4 + DCS * 0.3 + EAS * 0.3
    
    return {'TC': TC, 'DCS': DCS, 'EAS': EAS, 'MUS': MUS}
```

### Step 3：计算距离

```python
def calculate_average_distance(llm_results_list, bne_results_list):
    """计算所有实例的平均标准化欧氏距离"""
    distances = []
    for llm_res, bne_res in zip(llm_results_list, bne_results_list):
        dist = compute_normalized_euclidean(llm_res, bne_res)
        distances.append(dist)
    return np.mean(distances)
```

### Step 4：生成可视化

```python
# 对多个LLM模型
models = ['gpt-4', 'claude-3', 'deepseek-v3', 'qwen-2.5']

results_summary = {}
for model in models:
    llm_results = load_llm_results(model)
    bne_results = load_bne_results()
    
    # 计算指标
    scores = calculate_mechanism_scores(llm_results, bne_results)
    distance = calculate_average_distance(llm_results, bne_results)
    
    results_summary[model] = {
        'MUS': scores['MUS'],
        'TC': scores['TC'],
        'DCS': scores['DCS'],
        'EAS': scores['EAS'],
        'distance': distance
    }

# 绘制四象限图
plot_quadrant_chart(results_summary)
```

---

## 📋 输出示例

### 指标报告表

| Model | TC | DCS | EAS | MUS | Distance | Quadrant |
|-------|-----|-----|-----|-----|----------|----------|
| GPT-4 | 0.85 | 0.80 | 0.72 | 0.79 | 0.25 | II (Ideal) |
| Claude-3 | 0.78 | 0.60 | 0.65 | 0.68 | 0.35 | I (Understands) |
| DeepSeek-v3 | 0.45 | 0.40 | 0.35 | 0.40 | 0.28 | III (Lucky) |
| Qwen-2.5 | 0.60 | 0.60 | 0.50 | 0.57 | 0.45 | IV (Poor) |

### 解释建议

针对不同象限的模型，给出改进建议：

- **象限II（理想）**：继续保持，可以尝试更复杂的场景
- **象限I（理解但不准）**：
  - 可能原因：学习率不够、收敛不充分、虚拟博弈轮数不足
  - 建议：增加轮数、调整信念窗口、改进提示词
- **象限III（巧合）**：
  - 可能原因：记忆特定模式、过拟合训练数据
  - 建议：测试泛化能力、引入新的参数组合
- **象限IV（差）**：
  - 可能原因：根本不理解机制
  - 建议：改进提示词、增加机制解释、使用更强的模型

---

## 🎯 研究价值

### 1. 区分"理解"与"巧合"
传统指标只看数值距离，可能将"巧合接近均衡"误判为"理解机制"。

### 2. 多维度评估
- **TC**：长期趋势
- **DCS**：基本方向
- **EAS**：精细程度

### 3. 指导改进方向
通过四象限定位，明确模型的短板，指导提示词优化或模型选择。

---

## 📝 注意事项

### 1. 数据量要求
- 至少需要**10-15组**不同参数组合
- 每组重复**3-5次**取平均（控制随机性）

### 2. 参数选择
- 变化的参数应该**覆盖合理范围**
- 避免极端值（如rho=0.99）导致数值不稳定

### 3. 理论预期验证
- 部分机制关系（如M2: rho → share_rate）理论方向可能复杂
- 需要先用理性agent验证理论预期

### 4. 归一化处理
- 不同指标量纲不同，需要标准化
- 建议使用z-score或min-max归一化

---

## 🚀 后续扩展

### 1. 细粒度分析
针对每个机制对单独分析，而不只是平均

### 2. 时间序列分析
分析虚拟博弈过程中机制理解的演化

### 3. 跨场景对比
将此框架应用到场景C，对比不同场景下的机制理解能力

### 4. 因果分析
使用因果推断方法（如IV）进一步验证机制关系
