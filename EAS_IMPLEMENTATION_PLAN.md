# EAS (Elasticity Alignment Score) 实现方案

## 🎯 目标

先只实现弹性对齐分数（EAS），评估LLM对场景B机制关系弹性的理解。

---

## 📊 需要的数据

### 数据来源
需要多组**不同参数设置**的实验结果，包括：
- 多个LLM的FP结果（`fp_{model}/eval_*.json`）
- 对应的静态博弈结果（`static_{model}/eval_*.json`）作为对照
- 理论BNE结果（从`gt_numeric`中提取）

### 关键变量
每个结果文件应包含：
- **参数**：`n`, `rho`, `sigma_noise_sq`, `v_mean`, `price_level`（价格向量的均值/中位数）
- **结果**：`share_rate`, `profit`, `welfare`, `total_leakage`

---

## 🔧 机制对定义

基于场景B的理论，选择4个关键机制对（暂时排除复杂的非单调关系）：

| 机制对ID | 自变量X | 因变量Y | 理论预期 | 经济含义 |
|---------|---------|---------|---------|---------|
| M1 | price_level | share_rate | + | 价格↑ → 分享↑ |
| M2 | rho | share_rate | ? | 相关性↑ → 次模性↑ → 效果复杂 |
| M3 | share_rate | total_leakage | + | 分享↑ → 泄露↑（推断外部性） |
| M4 | v_mean | share_rate | - | 隐私偏好↑ → 分享↓ |

**注意**：M2的理论方向需要先验证（可能是非线性关系）

---

## 💻 实现步骤

### Step 1：数据收集函数

```python
def collect_multi_param_results(result_dir: str, mode: str = 'fp') -> pd.DataFrame:
    """
    从目录中读取多个结果JSON文件，提取关键信息
    
    Args:
        result_dir: 结果目录，如 "evaluation_results/fp_gpt-4"
        mode: 'fp' 或 'static'
    
    Returns:
        DataFrame with columns: 
            n, rho, sigma_noise_sq, v_mean, price_level,
            share_rate, profit, welfare, total_leakage,
            model_name, timestamp
    """
    import json
    import glob
    from pathlib import Path
    
    results = []
    json_files = glob.glob(f"{result_dir}/eval_*.json")
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 从results或final_share_set等提取结果
        if mode == 'fp':
            final_metrics = data['metrics']['final']
            share_set = data['final_share_set']
        else:  # static
            final_metrics = data['metrics']
            share_set = data['share_set']
        
        # 提取参数（从params对象或其他地方）
        # 注意：需要确保JSON中保存了这些参数
        params = data.get('params', {})
        
        results.append({
            'n': params.get('n'),
            'rho': params.get('rho'),
            'sigma_noise_sq': params.get('sigma_noise_sq'),
            'v_mean': params.get('v_mean'),  # 需要计算
            'price_level': np.mean(data['platform']['prices']),  # 价格均值
            'share_rate': final_metrics['share_rate'],
            'profit': final_metrics['profit'],
            'welfare': final_metrics['welfare'],
            'total_leakage': final_metrics['total_leakage'],
            'model_name': data['model_name'],
            'timestamp': Path(json_path).stem
        })
    
    return pd.DataFrame(results)
```

**问题**：当前JSON可能不包含所有参数（如v的均值）

**解决方案**：
1. 修改评估代码，确保保存完整参数到JSON
2. 或者从`gt_numeric`路径反推参数（如果文件名包含参数信息）

---

### Step 2：计算弹性

```python
def compute_elasticity(X: np.ndarray, Y: np.ndarray) -> float:
    """
    计算X对Y的标准化弹性
    
    Elasticity = β * (σ_X / σ_Y)
    
    Args:
        X: 自变量数组
        Y: 因变量数组
    
    Returns:
        弹性值（标准化斜率）
    """
    from sklearn.linear_model import LinearRegression
    
    # 检查数据有效性
    if len(X) < 3 or np.std(X) < 1e-6 or np.std(Y) < 1e-6:
        return np.nan
    
    # 线性回归
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    
    # 标准化弹性
    elasticity = slope * (np.std(X) / np.std(Y))
    
    return elasticity


def compute_mechanism_elasticities(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算所有机制对的弹性
    
    Args:
        df: 包含多组实验数据的DataFrame
    
    Returns:
        Dict: {mechanism_id: elasticity}
    """
    mechanisms = {
        'M1': ('price_level', 'share_rate'),
        'M2': ('rho', 'share_rate'),
        'M3': ('share_rate', 'total_leakage'),
        'M4': ('v_mean', 'share_rate')
    }
    
    elasticities = {}
    for mech_id, (x_var, y_var) in mechanisms.items():
        # 提取数据
        X = df[x_var].values
        Y = df[y_var].values
        
        # 计算弹性
        elasticity = compute_elasticity(X, Y)
        elasticities[mech_id] = elasticity
    
    return elasticities
```

---

### Step 3：计算EAS

```python
def compute_EAS(llm_elasticities: Dict[str, float], 
                bne_elasticities: Dict[str, float]) -> Dict[str, Any]:
    """
    计算弹性对齐分数
    
    Args:
        llm_elasticities: LLM的弹性字典
        bne_elasticities: BNE的弹性字典
    
    Returns:
        Dict: {
            'EAS': float,  # 总体EAS
            'mechanism_scores': Dict[str, float],  # 每个机制的得分
            'mechanism_ratios': Dict[str, float]   # 每个机制的弹性比
        }
    """
    scores = []
    mechanism_scores = {}
    mechanism_ratios = {}
    
    for mech_id in llm_elasticities.keys():
        e_llm = llm_elasticities[mech_id]
        e_bne = bne_elasticities[mech_id]
        
        # 检查有效性
        if np.isnan(e_llm) or np.isnan(e_bne) or abs(e_bne) < 1e-6:
            continue
        
        # 计算弹性比
        ratio = e_llm / e_bne
        mechanism_ratios[mech_id] = ratio
        
        # 计算对齐分数（对数衰减）
        score = np.exp(-abs(np.log(abs(ratio))))
        mechanism_scores[mech_id] = score
        scores.append(score)
    
    # 总体EAS
    EAS = np.mean(scores) if scores else 0.0
    
    return {
        'EAS': EAS,
        'mechanism_scores': mechanism_scores,
        'mechanism_ratios': mechanism_ratios
    }
```

---

### Step 4：主函数

```python
def analyze_mechanism_understanding(
    llm_result_dirs: List[str],
    bne_result_dir: str,
    mode: str = 'fp'
) -> pd.DataFrame:
    """
    分析多个LLM模型的机制理解能力
    
    Args:
        llm_result_dirs: LLM结果目录列表
        bne_result_dir: BNE结果目录（或静态博弈理性agent）
        mode: 'fp' 或 'static'
    
    Returns:
        DataFrame with EAS results for each model
    """
    # 收集BNE数据
    print("[1] 收集BNE/理性基准数据...")
    bne_df = collect_multi_param_results(bne_result_dir, mode=mode)
    bne_elasticities = compute_mechanism_elasticities(bne_df)
    
    print(f"BNE弹性: {bne_elasticities}")
    
    # 分析每个LLM
    results = []
    for llm_dir in llm_result_dirs:
        model_name = Path(llm_dir).name
        print(f"\n[2] 分析模型: {model_name}")
        
        # 收集LLM数据
        llm_df = collect_multi_param_results(llm_dir, mode=mode)
        
        if len(llm_df) < 3:
            print(f"  [WARN] 数据点太少({len(llm_df)})，跳过")
            continue
        
        # 计算LLM弹性
        llm_elasticities = compute_mechanism_elasticities(llm_df)
        print(f"  LLM弹性: {llm_elasticities}")
        
        # 计算EAS
        eas_result = compute_EAS(llm_elasticities, bne_elasticities)
        
        # 记录结果
        results.append({
            'model': model_name,
            'EAS': eas_result['EAS'],
            'n_samples': len(llm_df),
            **{f'EAS_{k}': v for k, v in eas_result['mechanism_scores'].items()},
            **{f'ratio_{k}': v for k, v in eas_result['mechanism_ratios'].items()},
            **{f'elasticity_{k}': v for k, v in llm_elasticities.items()}
        })
    
    return pd.DataFrame(results)
```

---

## 📊 使用示例

```python
# 在evaluate_scenario_b.py的main函数中添加

if args.analyze_mechanism:
    print("\n" + "="*60)
    print("[机制理解分析] 计算弹性对齐分数(EAS)")
    print("="*60)
    
    # 指定要分析的模型
    llm_dirs = [
        "evaluation_results/fp_gpt-4",
        "evaluation_results/fp_claude-3",
        "evaluation_results/fp_deepseek-v3"
    ]
    
    # 使用静态博弈的理性agent作为基准
    # 或者使用理论BNE（如果有多组参数的GT）
    bne_dir = "evaluation_results/static_rational"  # 假设有理性agent的结果
    
    # 分析
    eas_results = analyze_mechanism_understanding(
        llm_result_dirs=llm_dirs,
        bne_result_dir=bne_dir,
        mode='fp'
    )
    
    # 打印结果
    print("\n[EAS分析结果]")
    print(eas_results.to_string())
    
    # 保存
    output_path = "evaluation_results/eas_analysis.csv"
    eas_results.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
```

---

## 🚧 当前限制和解决方案

### 限制1：JSON中可能缺少参数信息

**问题**：当前JSON可能不保存`v_mean`, `rho`等参数

**解决方案A**：修改评估代码，在保存结果时添加参数
```python
# 在simulate_fictitious_play或evaluate_static中
results['params'] = {
    'n': self.params.n,
    'rho': self.params.rho,
    'sigma_noise_sq': self.params.sigma_noise_sq,
    'v_mean': np.mean(self.params.v),
    'v_std': np.std(self.params.v)
}
```

**解决方案B**：从GT路径推断参数
```python
# 从gt_numeric路径提取参数
# 例如："data/scenario_b_gt/n10_rho0.8_sigma1.0/gt.json"
```

### 限制2：需要多组不同参数的实验

**当前情况**：可能只有1-2组参数的结果

**解决方案**：
1. 运行参数扫描实验
2. 或者先用现有的少量数据进行概念验证

### 限制3：价格是向量，如何提取"price_level"

**方案**：
- 使用价格向量的均值：`price_level = np.mean(prices)`
- 或使用中位数、标准差等统计量

---

## 📝 输出示例

```
[机制理解分析] 计算弹性对齐分数(EAS)
============================================================

[1] 收集BNE/理性基准数据...
BNE弹性: {'M1': 0.85, 'M2': -0.32, 'M3': 0.92, 'M4': -0.68}

[2] 分析模型: fp_gpt-4
  LLM弹性: {'M1': 0.78, 'M2': -0.28, 'M3': 0.88, 'M4': -0.62}
  EAS: 0.87

[2] 分析模型: fp_deepseek-v3
  LLM弹性: {'M1': 0.45, 'M2': 0.15, 'M3': 0.75, 'M4': -0.35}
  EAS: 0.52

[EAS分析结果]
         model   EAS  n_samples  EAS_M1  EAS_M2  EAS_M3  EAS_M4  ratio_M1  ratio_M2  ratio_M3  ratio_M4
0      fp_gpt-4  0.87         15    0.95    0.92    0.98    0.95      0.92     0.88      0.96      0.91
1  fp_deepseek  0.52         15    0.65    0.20    0.82    0.68      0.53    -0.47      0.82      0.51

结果已保存到: evaluation_results/eas_analysis.csv
```

---

## 🔄 下一步

1. **验证现有JSON结构**，确认需要添加哪些字段
2. **修改保存逻辑**，确保参数信息被保存
3. **实现上述函数**，添加到`evaluate_scenario_b.py`
4. **添加命令行参数**：`--analyze-mechanism`
5. **测试**：用现有数据进行概念验证
6. **如果数据不够**：设计参数扫描实验

要我现在开始实现吗？还是先检查现有JSON的结构？
