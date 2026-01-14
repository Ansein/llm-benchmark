# 场景实验脚本说明

根据 `最终设计方案.md` 实现的两个场景实验脚本。

## 文件清单

- `scenario_a_personalization.py` - 场景A：个性化与隐私选择（价格传导外部性）
- `scenario_b_too_much_data.py` - 场景B：Too Much Data（推断外部性）

## 使用方法

### 场景A运行
```bash
python scenario_a_personalization.py
```

**输出文件**：`scenario_a_result.json`

### 场景B运行
```bash
python scenario_b_too_much_data.py
```

**输出文件**：`scenario_b_result.json`

## 场景A：Personalization & Privacy Choice

### 核心机制
- **外部性来源**：消费者披露数据 → 平台个性化定价 → 统一价格上升 → 未披露者受损
- **求解方法**：枚举所有披露集合（2^n），找Nash均衡和社会最优

### 参数设置
- `n`：消费者数（默认8）
- `theta`：每个消费者的愿付（从离散网格{0.2,0.4,0.6,0.8,1.0}抽样）
- `c_privacy`：隐私成本（从[0.05, 0.3]均匀抽样）
- `seller_cost`：卖家边际成本（默认0.1）

### 输出结构

#### gt_numeric（数值真值）
```json
{
  "eq_disclosure_set": [],           // Nash均衡的披露集合
  "eq_uniform_price": 0.6,           // 均衡统一价格
  "eq_profit": 3.5,                  // 均衡平台利润
  "eq_CS": 1.4,                      // 均衡消费者剩余
  "eq_W": 4.9,                       // 均衡社会福利
  "fb_disclosure_set": [5],          // 社会最优披露集合
  "fb_uniform_price": 0.6,           // 最优统一价格
  "fb_profit": 3.8,                  // 最优平台利润
  "fb_CS": 1.17,                     // 最优消费者剩余
  "fb_W": 4.97,                      // 最优社会福利
  "externality_gap_W": 0.07,         // 福利损失
  "externality_gap_CS": -0.23        // 消费者剩余损失
}
```

#### gt_labels（抽象标签）
```json
{
  "disclosure_rate_bucket": "low",   // 披露率桶（low/med/high）
  "disclosure_rate": 0.0,            // 实际披露率
  "over_disclosure": 0,              // 是否过度披露（1/0）
  "eq_size": 0,                      // 均衡披露人数
  "fb_size": 1                       // 最优披露人数
}
```

### 实验观察
该实例展示了**披露不足**（under-disclosure）的情况：
- 均衡：无人披露（统一价格0.6，高愿付者购买）
- 最优：用户5应该披露（隐私成本较低）
- 原因：用户5披露虽然能提升整体效率，但其个人收益不足以补偿隐私成本

---

## 场景B：Too Much Data（推断外部性）

### 核心机制
- **外部性来源**：类型相关 → 他人分享泄露你的信息 → 边际信息递减（次模性）→ 数据价格压低 → 过度分享
- **求解方法**：枚举所有分享集合（2^n），计算后验协方差与泄露信息量

### 参数设置
- `n`：用户数（默认8）
- `rho`：相关系数（默认0.7，控制外部性强度）
- `Sigma`：协方差矩阵（等相关结构）
- `sigma_noise_sq`：观测噪声方差（默认0.1）
- `alpha`：平台收益系数（默认1.0）
- `v`：用户隐私偏好（从[0.3, 1.2]均匀抽样）

### 输出结构

#### gt_numeric（数值真值）
```json
{
  "eq_share_set": [4, 5, 6],         // 均衡分享集合（平台利润最大）
  "eq_prices": [...],                // 最小支持价格向量
  "eq_value": 5.694,                 // 均衡平台价值
  "eq_profit": 4.562,                // 均衡平台利润
  "eq_W": 1.816,                     // 均衡社会福利
  "eq_total_leakage": 5.694,         // 均衡总泄露
  "fb_share_set": [0,2,3,4,5,6],     // 社会最优分享集合
  "fb_W": 1.998,                     // 最优社会福利
  "fb_total_leakage": 6.805,         // 最优总泄露
  "shutdown_W": 0.0,                 // 关停福利
  "shutdown_leakage": 0.0            // 关停泄露
}
```

#### gt_labels（抽象标签）
```json
{
  "over_sharing": 0,                 // 是否过度分享（1/0）
  "shutdown_better": 0,              // 关停是否更好（1/0）
  "leakage_bucket": "high",          // 泄露量级（low/med/high）
  "eq_size": 3,                      // 均衡分享人数
  "fb_size": 6,                      // 最优分享人数
  "share_rate": 0.375                // 分享率
}
```

### 实验观察
该实例展示了**分享不足**（under-sharing）的情况：
- 均衡：3人分享（隐私偏好较低的用户）
- 最优：6人分享更优（社会收益 > 隐私成本总和）
- 次模性验证：用户0的边际泄露随他人分享递减
  - 无人分享 → 用户1分享：泄露增加0.4455
  - 用户1分享 → 用户1,2分享：泄露仅增加0.0990

---

## 修改参数进行实验

### 修改场景A参数
```python
# 在 main() 函数中修改
params = generate_instance(
    n=10,      # 消费者数
    seed=123   # 随机种子
)
```

### 修改场景B参数
```python
# 在 main() 函数中修改
params = generate_instance(
    n=10,       # 用户数
    rho=0.9,    # 更高相关性 → 更强外部性
    seed=123    # 随机种子
)
```

### 批量生成实例
两个脚本都可以很容易扩展成批量生成：
```python
# 示例：生成100个不同种子的实例
for seed in range(100):
    params = generate_instance(n=8, seed=seed)
    result = solve_scenario_a(params)  # 或 solve_scenario_b
    # 保存到数据库或多个JSON文件
```

---

## 关键发现总结

### 场景A（价格传导外部性）
- ✅ 成功复现"披露决策外部性"：统一价格会因披露而变化
- ✅ 找到Nash均衡（单边偏离检验通过）
- ✅ 社会最优 ≠ 均衡（存在效率损失）
- ⚠️ 该实例出现**披露不足**，与"过度披露"假说不同（取决于参数）

### 场景B（推断外部性）
- ✅ 成功计算后验协方差与泄露信息量
- ✅ 验证次模性：边际泄露递减
- ✅ 平台利润最大 ≠ 社会最优（存在效率损失）
- ✅ 该实例显示关停不如有限分享（shutdown_W < eq_W）
- ⚠️ 该实例出现**分享不足**，与"过度分享"假说不同（取决于参数配置）

---

## 依赖库
```bash
pip install numpy
```

## 下一步扩展

1. **批量生成benchmark数据集**：按外部性强度（rho, n等）分层抽样
2. **添加LLM评测接口**：让LLM基于规则描述做决策，与理论基准对比
3. **可视化**：绘制福利、泄露量随参数变化的曲线
4. **政策对比**：实现税收、关停、去相关等政策的反事实分析
5. **扩展到场景C/D/E**：社会数据、动态学习、两边市场

## 许可
按照项目原始设计方案实现。
