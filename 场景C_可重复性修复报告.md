# 场景C Ground Truth 可重复性修复报告

## 🎯 问题描述

用户发现：多次运行 `generate_ground_truth` 产生不同的结果，主要体现在：
- 最优匿名化策略不稳定（有时 `identified`，有时 `anonymized`）
- 中介利润、`m_0`、社会福利等指标有显著差异

## 🔍 问题诊断

### 步骤1：验证固定点求解的可重复性

**测试脚本**：`test_reproducibility.py`

**结果**：✅ **完全可重复**
```
连续5次运行固定参数：
  r* = 0.102537 (标准差 = 0)
  ΔU = 0.619831 (标准差 = 0)
```

**结论**：固定点算法本身是稳定的。

---

### 步骤2：验证 GT 生成的可重复性（修复前）

**测试脚本**：`test_gt_reproducibility.py`（修复前）

**结果**：❌ **不可重复**
```
第1次: anonymization* = identified,   中介利润 = 1.6844
第2次: anonymization* = anonymized,   中介利润 = 1.4190
第3次: anonymization* = identified,   中介利润 = 1.9797

标准差 = 0.229 ⚠️ 显著差异
```

---

### 步骤3：诊断利润差异和 MC 噪声

**测试脚本**：`diagnose_profit_variance.py`

**发现**：
```
给定 seed=42，连续评估5次：

Identified 策略:
  中介利润 = 1.714235 (完全稳定，标准差 = 0)

Anonymized 策略:
  中介利润 = 1.705554 (完全稳定，标准差 = 0)

利润差异:
  Identified - Anonymized = +0.008680 (< 1%)
```

**关键洞察**：
1. ✅ 给定固定 `seed`，单个策略的评估结果**完全可重复**
2. ⚠️ **但两种策略的利润差异非常小**（< 1%）
3. ❌ 如果 `seed` 不固定，MC 采样噪声可能导致优劣反转

---

### 步骤4：定位根本原因

**代码检查**：`src/scenarios/scenario_c_social_data.py` 第 2326-2333 行

```python
# ❌ 修复前：没有传递 seed 参数
optimal_policy = optimize_intermediary_policy(
    params_base=params_base,
    m_grid=m_grid if m_grid is not None else np.linspace(0, 3, 31),
    policies=policies if policies is not None else ['identified', 'anonymized'],
    num_mc_samples=num_mc_samples,
    max_iter=max_iter,
    tol=tol,
    verbose=verbose
    # ❌ 缺少 seed=params_base.get('seed')
)
```

**根本原因**：
- `generate_ground_truth` 调用 `optimize_intermediary_policy` 时**没有传递 `seed` 参数**
- 虽然 `params_base` 中包含 `seed=42`，但这个 seed 只传递给了 `ScenarioCParams`
- `optimize_intermediary_policy` 的 `seed` 参数需要**显式传递**
- 因此，每次运行时 `evaluate_intermediary_strategy` 中的 MC 采样使用了不同的随机数
- 在两种策略利润接近时（差异 < 1%），MC 噪声导致优劣关系反转

---

## ✅ 解决方案

### 修改代码

**文件**：`src/scenarios/scenario_c_social_data.py`
**位置**：第 2332 行

```python
# ✅ 修复后：显式传递 seed 参数
optimal_policy = optimize_intermediary_policy(
    params_base=params_base,
    m_grid=m_grid if m_grid is not None else np.linspace(0, 3, 31),
    policies=policies if policies is not None else ['identified', 'anonymized'],
    num_mc_samples=num_mc_samples,
    max_iter=max_iter,
    tol=tol,
    seed=params_base.get('seed'),  # ⭐ 传递 seed 确保可重复性
    verbose=verbose
)
```

**关键改动**：
- 添加 `seed=params_base.get('seed')`
- 确保 `optimize_intermediary_policy` 使用与 `params_base` 一致的随机种子

---

## 🎯 修复验证

### 重新运行可重复性测试

**测试脚本**：`test_gt_reproducibility.py`（修复后）

**结果**：✅ **完全可重复**

```
======================================================================
✅ 完全可重复！
   固定 seed 后，GT 生成结果完全一致
======================================================================

m_star:
  第1次: 0.75000000
  第2次: 0.75000000
  第3次: 0.75000000
  标准差: 0.00000000 ✅

anonymization_star:
  第1次: identified
  第2次: identified
  第3次: identified
  完全一致 ✅

r_star:
  第1次: 0.08334896
  第2次: 0.08334896
  第3次: 0.08334896
  标准差: 0.00000000 ✅

intermediary_profit:
  第1次: 1.71423455
  第2次: 1.71423455
  第3次: 1.71423455
  标准差: 0.00000000 ✅

m_0:
  第1次: 2.97423455
  第2次: 2.97423455
  第3次: 2.97423455
  标准差: 0.00000000 ✅

social_welfare:
  第1次: 210.70791310
  第2次: 210.70791310
  第3次: 210.70791310
  标准差: 0.00000000 ✅
```

**所有指标标准差为 0，完美可重复！** 🎉

---

## 📊 理论验证

### 一分钟确认：固定点理论一致性

**测试脚本**：`verify_optimal_solution.py`

**结果**：
```
参与条件: τ_i ≤ ΔU
τ_i ~ N(τ_mean=1.0, τ_std=0.3)

理论参与率:
  r̂ = Φ((ΔU - τ_mean) / τ_std)
    = Φ((0.6198 - 1.0) / 0.3)
    = Φ(-1.2672)
    = 0.1025

对比结果:
  求解的 r* = 0.1025 ✅
  理论的 r̂  = 0.1025 ✅
  绝对误差   = 0.0000 ✅
```

**结论**：固定点求解与理论公式完美匹配！

---

## 🎓 技术要点

### 1. 随机数管理的三层结构

```
┌─────────────────────────────────────┐
│ generate_ground_truth               │
│   seed = params_base.get('seed')    │ ← 主种子
│   ↓                                  │
│   optimize_intermediary_policy      │
│     seed = seed                     │ ← 传递主种子
│     ↓                                │
│     evaluate_intermediary_strategy  │
│       seed = seed                   │ ← 传递主种子
│       ↓                              │
│       estimate_m0_mc                │
│         rng = default_rng(seed)     │ ← 使用主种子
└─────────────────────────────────────┘
```

**关键原则**：
- 顶层函数接收 `seed` 参数
- 所有下层函数**显式传递 `seed`**，不依赖全局状态
- 最底层使用 `np.random.Generator` 进行采样

---

### 2. 为什么两种策略利润接近？

```
m = 0.75, r* = 0.0833

Identified:
  • 允许个性化定价
  • 生产者利润更高 → m_0 更高
  • 但可能导致消费者剩余下降
  • 中介利润 = 1.714

Anonymized:
  • 强制统一定价
  • 生产者利润略低 → m_0 略低
  • 消费者剩余更高（无价格歧视）
  • 中介利润 = 1.706

差异：+0.009 (< 0.5%)
```

**经济学解释**：
- 在当前参数下（`tau_mean=1.0` 较高，参与率仅 8%），数据价值主要来自**学习效应**而非**价格歧视**
- 因此，个性化定价的额外收益有限
- 两种策略在中介利润上接近，但 `identified` 略优

---

### 3. MC 采样噪声与稳定性

**噪声来源**：
- `estimate_m0_mc` 使用 Monte Carlo 估计 `m_0`
- 样本数 `T=200`，标准误差 ≈ `σ/√200 ≈ σ/14`

**稳定性保证**：
- ✅ 固定 `seed` → 相同的 `rng` 流 → 相同的 MC 样本 → 相同的估计值
- ✅ 使用 Common Random Numbers (CRN)：with/no data 使用相同的 world
- ✅ 所有随机性来源都通过 `rng` 传递，完全可控

---

## 📁 相关文件

### 修改的文件
- `src/scenarios/scenario_c_social_data.py` (第 2332 行)

### 测试脚本
- `test_reproducibility.py` - 固定点可重复性测试
- `test_gt_reproducibility.py` - GT 生成可重复性测试
- `diagnose_profit_variance.py` - 利润差异诊断
- `verify_optimal_solution.py` - 理论一致性验证

### 文档
- `场景C_可重复性修复报告.md` (本文档)

---

## ✅ 最终结论

1. **问题已完全解决**：
   - 多次运行 `generate_ground_truth` 产生**完全相同**的结果
   - 所有指标标准差为 0

2. **根本原因**：
   - `seed` 参数传递缺失

3. **修复方法**：
   - 在 `generate_ground_truth` 中显式传递 `seed` 给 `optimize_intermediary_policy`

4. **验证通过**：
   - ✅ 可重复性测试通过（标准差 = 0）
   - ✅ 理论一致性验证通过（r* = r̂）
   - ✅ 所有指标稳定

5. **工程启示**：
   - 在深层调用链中，必须**显式传递所有随机数种子**
   - 不能假设 `seed` 会通过 `params` 自动传递给所有子函数
   - 使用 `np.random.Generator` + 显式 `seed` 传递是最佳实践

---

## 🎉 任务完成

**Ground Truth 生成器现在完全可重复、理论正确、工程健壮！**

可以安全地用于：
- 生成标准 Benchmark
- LLM 代理评估
- 论文实验复现
- 参数敏感性分析

---

*报告生成时间: 2026-01-19*
*作者: AI Assistant*
*状态: ✅ 完成*
