"""
场景C高级可视化分析
提供更深入的科学洞察
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from matplotlib.patches import Rectangle

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
df = pd.read_csv('evaluation_results/summary_report_20260205_205348.csv')
df_c = df[df['场景'] == 'C'].copy()

# 创建输出目录
output_dir = Path('evaluation_results/scenario_c/visualizations/advanced')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("场景C高级可视化分析")
print("=" * 80)
print(f"输出目录: {output_dir}")
print()

# 准备数据
df_theory = df_c[df_c['Config'] == 'A (Theory)'].iloc[0]
df_b = df_c[df_c['Config'] == 'B'].copy()
df_c_config = df_c[df_c['Config'] == 'C'].copy()
df_d = df_c[df_c['Config'] == 'D'].copy()

# 提取数值
theory_profit = df_theory['Intermediary Profit']
theory_cs = df_theory['Consumer Surplus']

# ============================================================================
# 1. 帕累托前沿分析：利润 vs 消费者福利权衡
# ============================================================================
print("[1/6] 生成帕累托前沿分析图...")

fig, ax = plt.subplots(figsize=(14, 10))

# 准备数据
df_d['m_value'] = df_d['m'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x))
df_d['cs_change'] = df_d['Consumer Surplus Gap']

# 颜色映射：按模型系列
def get_model_family(model):
    if 'gpt' in model.lower():
        return 'GPT'
    elif 'deepseek' in model.lower():
        return 'DeepSeek'
    elif 'qwen' in model.lower():
        return 'Qwen'
    else:
        return 'Other'

df_d['family'] = df_d['模型'].apply(get_model_family)
family_colors = {'GPT': '#E74C3C', 'DeepSeek': '#3498DB', 'Qwen': '#2ECC71'}

# 绘制散点
for family, color in family_colors.items():
    df_family = df_d[df_d['family'] == family]
    ax.scatter(df_family['Intermediary Profit'], 
              df_family['cs_change'],
              s=300, alpha=0.7, color=color, 
              edgecolors='black', linewidth=2,
              label=f'{family} Series', zorder=3)
    
    # 添加模型标签
    for idx, row in df_family.iterrows():
        ax.annotate(row['模型'], 
                   (row['Intermediary Profit'], row['cs_change']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, alpha=0.85,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))

# 添加理论最优点
ax.scatter(theory_profit, 0, s=500, marker='*', color='gold', 
          edgecolors='black', linewidth=2, label='Theoretical Optimum', zorder=5)

# 添加帕累托前沿
# 找出帕累托最优点（没有其他点同时在profit和cs_change上都更好）
pareto_points = []
for idx, row in df_d.iterrows():
    is_pareto = True
    for idx2, row2 in df_d.iterrows():
        if idx != idx2:
            if (row2['Intermediary Profit'] > row['Intermediary Profit'] and 
                row2['cs_change'] >= row['cs_change']) or \
               (row2['Intermediary Profit'] >= row['Intermediary Profit'] and 
                row2['cs_change'] > row['cs_change']):
                is_pareto = False
                break
    if is_pareto:
        pareto_points.append((row['Intermediary Profit'], row['cs_change'], row['模型']))

if pareto_points:
    pareto_points.sort()
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.5, label='Pareto Frontier', zorder=2)

# 添加象限划分
ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

# 添加象限标签
ax.text(2, 1, 'Win-Win\n(High Profit + Consumer Benefit)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(-0.2, 1, 'Consumer Benefit\nBut Low Profit', 
        ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(2, -40, 'Profit But\nConsumer Harmed', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(-0.2, -40, 'Lose-Lose', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

ax.set_xlabel('Intermediary Profit', fontsize=14, fontweight='bold')
ax.set_ylabel('Consumer Surplus Change', fontsize=14, fontweight='bold')
ax.set_title('Pareto Frontier Analysis: Profit vs Consumer Welfare Trade-off\n(Config D)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_pareto_frontier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. 策略空间地图：补偿 vs 参与率（带利润气泡）
# ============================================================================
print("[2/6] 生成策略空间地图...")

fig, ax = plt.subplots(figsize=(14, 10))

# 准备数据
df_d['profit_normalized'] = (df_d['Intermediary Profit'] - df_d['Intermediary Profit'].min()) / \
                             (df_d['Intermediary Profit'].max() - df_d['Intermediary Profit'].min() + 1e-6)

# 气泡大小表示利润（正利润大，负利润小）
df_d['bubble_size'] = 200 + df_d['profit_normalized'] * 800

# 颜色表示匿名化策略
anon_colors = {'anonymized': '#3498DB', 'identified': '#E67E22'}
df_d['color'] = df_d['Anonymization'].map(anon_colors)

# 绘制气泡
scatter = ax.scatter(df_d['m_value'], 
                    df_d['Participation Rate'] * 100,
                    s=df_d['bubble_size'],
                    c=df_d['color'],
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=2)

# 添加模型标签和利润值
for idx, row in df_d.iterrows():
    ax.annotate(f"{row['模型']}\n(π={row['Intermediary Profit']:.2f})", 
               (row['m_value'], row['Participation Rate'] * 100),
               ha='center', va='center',
               fontsize=8, fontweight='bold')

# 添加理论最优点
ax.scatter(0.6, 3.42, s=600, marker='*', color='gold', 
          edgecolors='black', linewidth=2, label='Theoretical Optimum', zorder=10)
ax.annotate('Theory\n(m=0.6, r=3.4%)', (0.6, 3.42),
           xytext=(20, 20), textcoords='offset points',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5),
           arrowprops=dict(arrowstyle='->', lw=2))

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498DB', edgecolor='black', label='Anonymized'),
    Patch(facecolor='#E67E22', edgecolor='black', label='Identified'),
    plt.scatter([], [], s=300, c='gray', alpha=0.6, edgecolors='black', label='Bubble Size = Profit')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.set_xlabel('Compensation m', fontsize=14, fontweight='bold')
ax.set_ylabel('Participation Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Strategy Space Map: Compensation vs Participation Rate\n(Bubble Size = Profit, Color = Anonymization)', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_strategy_space_map.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. 模型系列演进分析
# ============================================================================
print("[3/6] 生成模型系列演进分析图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 定义模型系列
gpt_models = ['gpt-5', 'gpt-5.1-2025-11-13', 'gpt-5.1', 'gpt-5.2']
deepseek_models = ['deepseek-v3-0324', 'deepseek-v3.1', 'deepseek-v3.2']
qwen_models = ['qwen-plus', 'qwen-plus-2025-12-01', 'qwen3-max', 'qwen3-max-2026-01-23']

model_series = [
    ('GPT Series', gpt_models, '#E74C3C'),
    ('DeepSeek Series', deepseek_models, '#3498DB'),
    ('Qwen Series', qwen_models, '#2ECC71')
]

metrics = [
    ('Intermediary Profit', 'Profit'),
    ('Consumer Surplus Gap', 'CS Gap'),
    ('Participation Rate', 'Part. Rate (%)'),
    ('Individual Accuracy', 'Accuracy')
]

for idx, (metric_name, metric_label) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    for series_name, models, color in model_series:
        # 获取该系列在配置D的数据
        series_data = []
        series_labels = []
        for model in models:
            model_data = df_d[df_d['模型'] == model]
            if not model_data.empty:
                if metric_name == 'Participation Rate':
                    value = model_data.iloc[0][metric_name] * 100
                else:
                    value = model_data.iloc[0][metric_name]
                series_data.append(value)
                series_labels.append(model.split('-')[-1] if len(model.split('-')) > 1 else model)
        
        if series_data:
            x_pos = np.arange(len(series_data))
            ax.plot(x_pos, series_data, 'o-', linewidth=2.5, markersize=10, 
                   label=series_name, color=color, alpha=0.8)
            
            # 添加数值标签
            for x, y in zip(x_pos, series_data):
                ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 添加理论最优线（如果适用）
    if metric_name == 'Intermediary Profit':
        ax.axhline(theory_profit, color='red', linestyle='--', linewidth=2, 
                  label=f'Theory ({theory_profit:.2f})', alpha=0.7)
    elif metric_name == 'Consumer Surplus Gap':
        ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    elif metric_name == 'Participation Rate':
        ax.axhline(3.42, color='red', linestyle='--', linewidth=2, 
                  label='Theory (3.42%)', alpha=0.7)
    
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Model Series Evolution Analysis (Config D)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '3_model_series_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. 指标相关性热力图
# ============================================================================
print("[4/6] 生成指标相关性分析图...")

# 准备相关性数据（配置D）
corr_data = df_d[['m_value', 'Participation Rate', 'Intermediary Profit', 
                   'Consumer Surplus Gap', 'Individual Accuracy', 'Gini Consumer Surplus']].copy()
corr_data.columns = ['Compensation', 'Part.Rate', 'Profit', 'CS Gap', 'Accuracy', 'Gini']
corr_data['Part.Rate'] = corr_data['Part.Rate'] * 100

# 计算相关系数矩阵
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))

# 绘制热力图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=2,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            vmin=-1, vmax=1,
            ax=ax)

ax.set_title('Metric Correlation Analysis (Config D)\n(Lower Triangle Shows Pearson Correlation)', 
             fontsize=15, fontweight='bold', pad=20)

# 添加显著性标注说明
textstr = 'Interpretation:\n|r| > 0.7: Strong\n|r| > 0.4: Moderate\n|r| > 0.2: Weak'
ax.text(1.15, 0.5, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '4_metric_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. 配置间性能对比（平行坐标图）
# ============================================================================
print("[5/6] 生成配置间性能对比图...")

from pandas.plotting import parallel_coordinates

# 准备数据：每个模型在B、C、D三个配置的利润
parallel_data = []
models = df_d['模型'].unique()

for model in models:
    b_profit = df_c[(df_c['模型'] == model) & (df_c['Config'] == 'B')]['Intermediary Profit'].values[0]
    c_profit = df_c[(df_c['模型'] == model) & (df_c['Config'] == 'C')]['Intermediary Profit'].values[0]
    d_profit = df_c[(df_c['模型'] == model) & (df_c['Config'] == 'D')]['Intermediary Profit'].values[0]
    
    family = 'GPT' if 'gpt' in model.lower() else ('DeepSeek' if 'deepseek' in model.lower() else 'Qwen')
    
    parallel_data.append({
        'Model': model,
        'Family': family,
        'Config B': b_profit,
        'Config C': c_profit,
        'Config D': d_profit
    })

df_parallel = pd.DataFrame(parallel_data)

fig, ax = plt.subplots(figsize=(14, 10))

# 绘制平行坐标图
parallel_coordinates(df_parallel, 'Family', 
                    cols=['Config B', 'Config C', 'Config D'],
                    color=['#E74C3C', '#3498DB', '#2ECC71'],
                    linewidth=2.5, alpha=0.8, ax=ax)

# 添加理论最优线
ax.axhline(theory_profit, color='gold', linestyle='--', linewidth=3, 
          label=f'Theoretical Optimum ({theory_profit:.2f})', alpha=0.8, zorder=0)

ax.set_ylabel('Intermediary Profit', fontsize=14, fontweight='bold')
ax.set_title('Cross-Configuration Profit Comparison\n(Parallel Coordinates)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '5_parallel_coordinates_profit.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. 效率前沿分析：利润效率 vs 消费者福利效率
# ============================================================================
print("[6/6] 生成效率前沿分析图...")

fig, ax = plt.subplots(figsize=(14, 10))

# 计算效率指标
df_d['profit_efficiency'] = df_d['Intermediary Profit'] / theory_profit
df_d['welfare_efficiency'] = (df_d['Consumer Surplus Gap'] + theory_cs) / theory_cs

# 绘制散点
for family, color in family_colors.items():
    df_family = df_d[df_d['family'] == family]
    ax.scatter(df_family['profit_efficiency'], 
              df_family['welfare_efficiency'],
              s=300, alpha=0.7, color=color, 
              edgecolors='black', linewidth=2,
              label=f'{family} Series', zorder=3)
    
    # 添加模型标签
    for idx, row in df_family.iterrows():
        ax.annotate(row['模型'], 
                   (row['profit_efficiency'], row['welfare_efficiency']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, alpha=0.85)

# 添加理论最优点
ax.scatter(1.0, 1.0, s=500, marker='*', color='gold', 
          edgecolors='black', linewidth=2, label='Theoretical Optimum (1.0, 1.0)', zorder=5)

# 添加45度线（帕累托改进线）
ax.plot([0, 2], [0, 2], 'k--', linewidth=2, alpha=0.3, label='Equal Efficiency Line')

# 添加效率区域
ax.axhline(1.0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax.axvline(1.0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

# 添加区域标签
ax.text(1.6, 1.05, 'Super-Efficient\n(Both > Theory)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.3, 1.05, 'Consumer Surplus\nOver-Achievement', 
        ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(1.6, 0.85, 'Profit-Focused\nStrategy', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

ax.set_xlabel('Profit Efficiency (Profit / Theoretical Profit)', fontsize=14, fontweight='bold')
ax.set_ylabel('Consumer Welfare Efficiency (CS / Theoretical CS)', fontsize=14, fontweight='bold')
ax.set_title('Efficiency Frontier: Profit Efficiency vs Welfare Efficiency\n(Values > 1.0 = Exceeds Theory)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.3, 2.0)
ax.set_ylim(0.85, 1.08)

plt.tight_layout()
plt.savefig(output_dir / '6_efficiency_frontier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 生成分析摘要
# ============================================================================
print()
print("=" * 80)
print("[OK] 高级可视化完成！")
print("=" * 80)
print(f"\n生成了6张高级分析图表：")
print(f"1. 帕累托前沿分析 - 利润vs消费者福利权衡")
print(f"2. 策略空间地图 - 补偿vs参与率（气泡=利润）")
print(f"3. 模型系列演进分析 - GPT/DeepSeek/Qwen内部对比")
print(f"4. 指标相关性热力图 - 6个关键指标的相关性")
print(f"5. 配置间性能对比 - 平行坐标图")
print(f"6. 效率前沿分析 - 利润效率vs福利效率")
print(f"\n所有图表已保存到: {output_dir}/")

# 关键发现
print("\n" + "=" * 80)
print("[分析] 关键发现")
print("=" * 80)

# 帕累托最优模型
print("\n[帕累托前沿]")
if pareto_points:
    print(f"帕累托最优模型 ({len(pareto_points)}个):")
    for profit, cs, model in pareto_points:
        print(f"  - {model}: Profit={profit:.3f}, CS Gap={cs:.2f}")

# Win-Win模型
win_win = df_d[(df_d['Intermediary Profit'] > 0) & (df_d['Consumer Surplus Gap'] > 0)]
print(f"\n[Win-Win策略] 既盈利又让消费者受益 ({len(win_win)}个):")
for _, row in win_win.iterrows():
    print(f"  - {row['模型']}: Profit={row['Intermediary Profit']:.3f}, CS Gap={row['Consumer Surplus Gap']:.2f}")

# 相关性发现
print("\n[相关性分析]")
print(f"  Compensation vs Profit: r={corr_matrix.loc['Compensation', 'Profit']:.3f}")
print(f"  Profit vs CS Gap: r={corr_matrix.loc['Profit', 'CS Gap']:.3f}")
print(f"  Participation Rate vs Profit: r={corr_matrix.loc['Part.Rate', 'Profit']:.3f}")

print("\n" + "=" * 80)
