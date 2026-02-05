# 快速修复：添加1600次评估早停

## 🚀 最简单的方法（推荐）

在服务器上执行：

```bash
cd /mnt/llm-benchmark

# 方法1：直接修改maxfun参数（最简单）
sed -i "s/'maxiter': 20,  # ✅ 从100减少到20（5倍加速）/'maxiter': 100,  # 迭代次数上限/" src/scenarios/scenario_c_social_data_optimization.py

sed -i "s/'maxfun': 500   # ✅ 最大函数评估次数限制/'maxfun': 1600   # ✅ 早停：最多1600次函数评估/" src/scenarios/scenario_c_social_data_optimization.py
```

## 📝 手动修改

编辑文件：`src/scenarios/scenario_c_social_data_optimization.py`

找到第180-188行左右：

```python
# 运行优化
result = minimize(
    fun=objective,
    x0=m_init,
    method=method,
    bounds=bounds,
    options={
        'disp': verbose,
        'maxiter': 20,
        'maxfun': 500
    }
)
```

修改为：

```python
# 运行优化（早停：最多1600次评估）
print(f"开始优化（最多1600次函数评估）...")
result = minimize(
    fun=objective,
    x0=m_init,
    method=method,
    bounds=bounds,
    options={
        'disp': False,         # 关闭详细输出
        'maxiter': 100,        # 迭代上限
        'maxfun': 1600         # ✅ 函数评估上限（早停）
    }
)
```

## ⚡ 进一步加速（可选）

如果还是太慢，修改 `src/scenarios/generate_scenario_c_gt.py` 第101-112行：

```python
result = optimize_intermediary_policy_personalized(
    params_base=params_base,
    policies=['identified', 'anonymized'],
    optimization_method='hybrid',
    m_bounds=(0.0, 3.0),
    num_mc_samples=10,   # ✅ 从15减少到10（1.5倍加速）
    max_iter=3,          # 已经是3了
    grid_size=11,        # 已经是11了
    n_jobs=-1,
    verbose=True,
    seed=42
)
```

## 🎯 预期效果

修改后：
- **identified策略**: ~5分钟（已完成）
- **anonymized策略**: ~10-15分钟（而非无限长）
- **总时间**: ~15-20分钟

## 🔍 验证修改

```bash
# 检查是否修改成功
grep -n "maxfun" src/scenarios/scenario_c_social_data_optimization.py
# 应该看到: 'maxfun': 1600
```

## 🚀 重新运行

```bash
cd /mnt/llm-benchmark
python -m src.scenarios.generate_scenario_c_gt
```

## 📊 监控进度

由于有96核CPU，你可以在另一个终端监控：

```bash
# 查看CPU使用率
htop

# 查看进程
ps aux | grep python

# 查看输出（如果重定向了）
tail -f output.log
```

---

**关键点**：
- `maxfun=1600` 是硬性限制，达到后立即停止
- 1600次评估 ≈ 76次迭代（1600/21）
- 这足够找到一个不错的解，即使不是全局最优
