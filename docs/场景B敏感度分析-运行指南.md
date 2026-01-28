# 场景B敏感度分析 - 运行指南

## 快速开始

### 配置说明
- **参数网格**: 3×3（ρ={0.3, 0.6, 0.9} × v={[0.3,0.6], [0.6,0.9], [0.9,1.2]}）
- **模型**: deepseek-v3.2, gpt-5.1, qwen-plus
- **提示词**: b.v4
- **重复次数**: 3次
- **总实验数**: 9组 × 3模型 × 3次 = **81个实验**

### 预计资源消耗
- **LLM调用次数**: 81 × 10用户 = **810次**
- **预计时间**: 1.5-2.5小时
- **预计成本**: $2-5（取决于模型定价）

---

## Step 1: 生成Ground Truth文件

**目的**: 为9个参数组合生成理论均衡解

**命令**:
```bash
cd d:\benchmark
python scripts\generate_sensitivity_b_gt.py
```

**预期输出**:
```
================================================================================
场景B敏感度分析 - Ground Truth生成
================================================================================
参数网格：
  ρ值: [0.3, 0.6, 0.9]
  v范围: [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
  固定参数: n=10, sigma_noise_sq=0.1, alpha=1.0
  输出目录: data\ground_truth\sensitivity_b
================================================================================

[1/9] 生成 GT: ρ=0.3, v=[0.3, 0.6]
  参数: n=10, rho=0.3, v=[0.3, 0.6]
  v值: min=0.300, max=0.600, mean=0.450
  求解成功!
    分享集合: [0, 1, 2, 3, 4, 5, 6, 7, 8] (分享率: 90.00%)
    平台利润: 0.xxxx
    社会福利: 0.xxxx
    总泄露量: 0.xxxx
  ✅ 已保存: scenario_b_rho0.3_v0.3-0.6.json

... (共9个)

================================================================================
✅ 所有GT文件生成完成！
   输出目录: data\ground_truth\sensitivity_b
   文件数量: 9
================================================================================
```

**验证**:
```bash
dir data\ground_truth\sensitivity_b
```

应该看到9个JSON文件：
- scenario_b_rho0.3_v0.3-0.6.json
- scenario_b_rho0.3_v0.6-0.9.json
- scenario_b_rho0.3_v0.9-1.2.json
- ... (共9个)

**时间**: 约1-2分钟

---

## Step 2: 运行敏感度实验

**目的**: 使用3个模型对9个参数组合进行评估，每个组合重复3次

**命令**:
```bash
python run_sensitivity_b.py ^
  --models deepseek-v3.2 gpt-5.1 qwen-plus ^
  --prompt-version b.v4 ^
  --num-trials 3
```

**注意**:
- Windows CMD使用 `^` 换行
- PowerShell使用 `` ` `` 换行
- Linux/Mac使用 `\` 换行

**预期输出**:
```
================================================================================
场景B敏感度分析 - 方案D（多模型×多试验）
================================================================================
参数网格:
  ρ值: [0.3, 0.6, 0.9]
  v范围: [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
提示词版本: b.v4
模型列表: ['deepseek-v3.2', 'gpt-5.1', 'qwen-plus']
重复次数: 3
总实验数: 9 × 3 × 3 = 81

实验配置已保存: sensitivity_results\scenario_b\sensitivity_3x3_b.v4_YYYYMMDD_HHMMSS\experiment_config.json

================================================================================
参数组合: ρ=0.3, v=[0.3, 0.6]
================================================================================

--- 模型: deepseek-v3.2 ---

  [试验 1/3] (1/81)
    分享率: LLM=85.00%, GT=90.00%
    Jaccard相似度: 0.889
    ✅ 已保存: result_rho0.3_v0.3-0.6_trial1.json

  [试验 2/3] (2/81)
    ...

... (继续运行所有81个实验)

================================================================================
✅ 实验完成！
================================================================================
总实验数: 81/81
结果目录: sensitivity_results\scenario_b\sensitivity_3x3_b.v4_YYYYMMDD_HHMMSS
  - 配置文件: experiment_config.json
  - 汇总结果: summary_all_results.json
  - 模型结果: deepseek-v3.2/gpt-5.1/qwen-plus/
  - LLM日志: llm_logs/
================================================================================

快速统计摘要
================================================================================

模型: deepseek-v3.2
  实验数: 27
  Jaccard相似度: 均值=0.XXX, 标准差=0.XXX
  分享率误差: 均值=X.XX%, 标准差=X.XX%

模型: gpt-5.1
  实验数: 27
  Jaccard相似度: 均值=0.XXX, 标准差=0.XXX
  分享率误差: 均值=X.XX%, 标准差=X.XX%

模型: qwen-plus
  实验数: 27
  Jaccard相似度: 均值=0.XXX, 标准差=0.XXX
  分享率误差: 均值=X.XX%, 标准差=X.XX%

================================================================================
```

**输出目录结构**:
```
sensitivity_results/scenario_b/sensitivity_3x3_b.v4_YYYYMMDD_HHMMSS/
├── experiment_config.json           # 实验配置
├── summary_all_results.json         # 所有结果汇总
├── deepseek-v3.2/                   # deepseek结果
│   ├── result_rho0.3_v0.3-0.6_trial1.json
│   ├── result_rho0.3_v0.3-0.6_trial2.json
│   ├── result_rho0.3_v0.3-0.6_trial3.json
│   └── ... (共27个文件)
├── gpt-5.1/                         # gpt-5.1结果
│   └── ... (共27个文件)
├── qwen-plus/                       # qwen-plus结果
│   └── ... (共27个文件)
└── llm_logs/                        # LLM对话日志
    ├── deepseek-v3.2/
    ├── gpt-5.1/
    └── qwen-plus/
```

**时间**: 约1.5-2.5小时

**中断恢复**:
如果实验中断，重新运行相同命令会跳过已完成的实验（基于文件存在性检测）

---

## Step 3: 生成可视化

**目的**: 为实验结果生成热力图和对比表格

**命令**:
```bash
python plot_sensitivity_b_heatmap.py sensitivity_results\scenario_b\sensitivity_3x3_b.v4_YYYYMMDD_HHMMSS
```

**替换实际目录名**:
```bash
# 查找最新的实验目录
dir sensitivity_results\scenario_b /O-D

# 使用实际目录名
python plot_sensitivity_b_heatmap.py sensitivity_results\scenario_b\sensitivity_3x3_b.v4_20260128_143022
```

**预期输出**:
```
✅ 加载模型 deepseek-v3.2: 27 个结果
✅ 加载模型 gpt-5.1: 27 个结果
✅ 加载模型 qwen-plus: 27 个结果

============================================================
生成热力图...
============================================================
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_share_rate_llm.png
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_share_rate_gt.png
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_jaccard.png

计算分享率误差...
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_share_rate_error.png
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_profit_mae.png
✅ 已保存: sensitivity_results\...\heatmaps\heatmap_welfare_mae.png

============================================================
✅ 所有热力图已保存至: sensitivity_results\...\heatmaps
============================================================

生成模型对比表格...
✅ 对比表格已保存: sensitivity_results\...\heatmaps\comparison_table.csv

============================================================
模型对比摘要
============================================================

模型: deepseek-v3.2
  平均Jaccard: 0.XXX ± 0.XXX
  平均分享率误差: X.XX% ± X.XX%

模型: gpt-5.1
  平均Jaccard: 0.XXX ± 0.XXX
  平均分享率误差: X.XX% ± X.XX%

模型: qwen-plus
  平均Jaccard: 0.XXX ± 0.XXX
  平均分享率误差: X.XX% ± X.XX%

============================================================
```

**生成的文件**:
```
heatmaps/
├── heatmap_share_rate_llm.png       # LLM分享率（3个子图）
├── heatmap_share_rate_gt.png        # 理论分享率（单图）
├── heatmap_jaccard.png              # Jaccard相似度（3个子图）
├── heatmap_share_rate_error.png     # 分享率误差（3个子图）
├── heatmap_profit_mae.png           # 利润偏差（3个子图）
├── heatmap_welfare_mae.png          # 福利偏差（3个子图）
└── comparison_table.csv             # 对比表格
```

**时间**: 约30秒

---

## 输出解读

### 热力图说明

#### 1. LLM分享率热力图
- **3个子图**: 分别对应3个模型
- **每个格子**: 显示均值±标准差（3次试验）
- **颜色**: 蓝色越深分享率越高
- **预期模式**: 右上角（高ρ×高v）分享率最低

#### 2. Jaccard相似度热力图
- **含义**: LLM决策与理论的相似度（0-1）
- **颜色**: 绿色越深相似度越高，红色越深越差
- **目标**: 接近1.0为理想
- **关注点**: 哪些参数区域LLM表现差？

#### 3. 分享率误差热力图
- **含义**: |LLM分享率 - 理论分享率|
- **颜色**: 红色越深误差越大
- **目标**: 接近0为理想
- **关注点**: 误差分布模式

### 对比表格说明

**CSV列含义**:
- **模型**: 模型名称
- **ρ**: 相关系数
- **v范围**: 隐私偏好范围
- **试验数**: 重复次数
- **Jaccard(均值)**: 相似度均值
- **Jaccard(标准差)**: 相似度稳定性
- **LLM分享率(均值)**: LLM预测的平均分享率
- **GT分享率**: 理论分享率（基准）
- **分享率误差(均值)**: 平均误差
- **分享率误差(标准差)**: 误差稳定性

**分析维度**:
1. **模型排名**: 按Jaccard均值排序
2. **参数敏感度**: 哪些参数下误差大？
3. **稳定性**: 标准差小的模型更稳定
4. **系统性偏差**: LLM是否系统性高估/低估？

---

## 进阶分析

### 分析1: ρ的边际效应

**问题**: 相关系数对LLM决策的影响？

**方法**: 固定v范围，画ρ的曲线

**手动操作**（Python）:
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取对比表格
df = pd.read_csv('sensitivity_results/.../heatmaps/comparison_table.csv')

# 对每个v范围画曲线
for v_range in df['v范围'].unique():
    subset = df[df['v范围'] == v_range]
    
    for model in df['模型'].unique():
        model_data = subset[subset['模型'] == model]
        plt.plot(model_data['ρ'], model_data['LLM分享率(均值)'], 
                 marker='o', label=f'{model}')
    
    plt.plot(subset[subset['模型'] == df['模型'].iloc[0]]['ρ'], 
             subset[subset['模型'] == df['模型'].iloc[0]]['GT分享率'],
             'k--', label='理论', linewidth=2)
    
    plt.xlabel('相关系数 ρ')
    plt.ylabel('分享率')
    plt.title(f'分享率 vs ρ (v={v_range})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 分析2: 模型优势区域

**问题**: 每个模型在哪些参数下表现最好？

**方法**: 对比每个格子的Jaccard相似度

**Excel操作**:
1. 打开 `comparison_table.csv`
2. 数据透视表：行=ρ, 列=v范围, 值=Jaccard(均值), 筛选=模型
3. 条件格式：色阶显示优劣

### 分析3: 交互效应

**问题**: 高ρ×高v的组合是否特别困难？

**方法**: 查看对角线误差是否最大

**Python**:
```python
# 提取(ρ=0.9, v=[0.9,1.2])的结果
extreme = df[(df['ρ'] == 0.9) & (df['v范围'] == '[0.9, 1.2]')]

print("极端参数组合的表现:")
print(extreme[['模型', 'Jaccard(均值)', '分享率误差(均值)']])
```

---

## 常见问题

### Q1: GT生成失败
**症状**: `scenario_b_rho0.X_vX.X-X.X.json` 文件缺失

**原因**: 求解器失败（可能是数值问题）

**解决**:
1. 检查 `src/scenarios/scenario_b_too_much_data.py` 中的求解器
2. 尝试调整参数（如增大sigma_noise_sq）
3. 查看错误日志

### Q2: 实验中断
**症状**: 运行到一半停止

**解决**:
1. 检查API配额/余额
2. 重新运行相同命令（会自动跳过已完成的）
3. 或手动删除未完成的结果文件

### Q3: 可视化失败
**症状**: 热力图空白或报错

**原因**: 结果文件格式不对或缺失

**解决**:
1. 检查结果JSON文件是否完整
2. 验证是否所有81个文件都存在
3. 查看错误信息具体是哪个指标路径

### Q4: 内存不足
**症状**: 运行时内存溢出

**解决**:
1. 减少num_trials（从3改为1）
2. 分批运行模型（先运行1个，再运行另外2个）
3. 关闭其他程序

### Q5: 模型不可用
**症状**: `Model xxx not found in config`

**解决**:
1. 检查 `configs/model_configs.json` 是否有该模型配置
2. 添加模型配置（参考现有格式）
3. 或使用已有模型替代

---

## 估算检查表

### 运行前检查
- [ ] GT文件已生成（9个JSON文件）
- [ ] API密钥已配置
- [ ] API余额充足（≥$5）
- [ ] 磁盘空间充足（≥1GB）
- [ ] 模型配置正确
- [ ] 网络连接稳定

### 运行后验证
- [ ] 81个结果JSON文件完整
- [ ] 日志目录存在
- [ ] summary_all_results.json 包含81条记录
- [ ] 6张热力图生成成功
- [ ] comparison_table.csv 有81行数据

### 结果合理性
- [ ] Jaccard相似度大部分 > 0.5
- [ ] 分享率误差大部分 < 20%
- [ ] GT分享率呈现预期趋势（ρ↑→分享率↓）
- [ ] 标准差 < 均值的30%（稳定性合格）

---

## 后续工作

### 立即可做
1. **撰写分析报告**: 基于热力图和表格总结发现
2. **统计检验**: 模型间差异是否显著？
3. **可视化改进**: 添加误差棒、置信区间等

### 扩展实验
1. **增加参数点**: ρ从3个扩展到5个
2. **测试其他提示词**: 对比b.v4 vs b.v6
3. **虚拟博弈版本**: 运行FP模式观察学习轨迹
4. **跨场景对比**: 与场景A、C的敏感度对比

---

**文档版本**: v1.0  
**创建日期**: 2026-01-28  
**预计执行时间**: 2-3小时  
**状态**: 就绪
