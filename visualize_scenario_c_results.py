"""
场景C评估结果可视化脚本
对11个LLM模型在场景C（社会数据外部性）的表现进行多维度可视化分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import io

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 读取数据
df = pd.read_csv('evaluation_results/summary_report_20260205_205348.csv')

# 筛选场景C的数据
df_c = df[df['场景'] == 'C'].copy()

# 创建输出目录
output_dir = Path('evaluation_results/scenario_c/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("场景C评估结果可视化")
print("=" * 80)
print(f"模型数量: {df_c['模型'].nunique()}")
print(f"配置数量: {df_c['Config'].nunique()}")
print(f"模型列表: {sorted(df_c['模型'].unique())}")
print(f"输出目录: {output_dir}")
print()

# ============================================================================
# 1. 配置D：中介利润对比（最重要的指标）
# ============================================================================
print("[1/8] 生成配置D中介利润对比图...")

df_d = df_c[df_c['Config'] == 'D'].copy()
df_theory = df_c[df_c['Config'] == 'A (Theory)'].copy()

# 获取理论最优利润
theory_profit = df_theory['Intermediary Profit'].iloc[0]

# 按利润排序
df_d = df_d.sort_values('Intermediary Profit', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))

# 绘制条形图
bars = ax.barh(df_d['模型'], df_d['Intermediary Profit'])

# 添加理论最优线
ax.axvline(theory_profit, color='red', linestyle='--', linewidth=2, 
           label=f'Theoretical Optimum ({theory_profit:.2f})', alpha=0.7)

# 为每个条形着色（正利润绿色，负利润红色）
colors = ['green' if x >= 0 else 'red' for x in df_d['Intermediary Profit']]
for bar, color in zip(bars, colors):
    bar.set_color(color)
    bar.set_alpha(0.7)

# 添加数值标签
for i, (idx, row) in enumerate(df_d.iterrows()):
    profit = row['Intermediary Profit']
    ax.text(profit + 0.05 if profit >= 0 else profit - 0.05, i, 
            f"{profit:.2f}", 
            va='center', ha='left' if profit >= 0 else 'right',
            fontsize=10, fontweight='bold')

ax.set_xlabel('Intermediary Profit', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Config D: LLM Intermediary × LLM Consumer - Profit Comparison\n(Green=Profit, Red=Loss)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_config_D_profit_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. 配置C：LLM中介策略质量对比
# ============================================================================
print("[2/8] 生成配置C中介策略质量对比图...")

df_c_config = df_c[df_c['Config'] == 'C'].copy()
df_c_config = df_c_config.sort_values('Intermediary Profit', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：利润对比
bars = ax1.barh(df_c_config['模型'], df_c_config['Intermediary Profit'])
ax1.axvline(theory_profit, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Optimum ({theory_profit:.2f})', alpha=0.7)
for bar in bars:
    bar.set_alpha(0.7)

for i, (idx, row) in enumerate(df_c_config.iterrows()):
    profit = row['Intermediary Profit']
    ax1.text(profit + 0.02, i, f"{profit:.2f}", 
             va='center', ha='left', fontsize=9)

ax1.set_xlabel('Intermediary Profit', fontsize=12, fontweight='bold')
ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
ax1.set_title('LLM Intermediary Profit', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# 右图：补偿策略
# 将补偿m转换为数值（处理向量情况）
df_c_config['m_value'] = df_c_config['m'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x))

colors_anon = ['blue' if x == 'anonymized' else 'orange' for x in df_c_config['Anonymization']]
bars2 = ax2.barh(df_c_config['模型'], df_c_config['m_value'], color=colors_anon, alpha=0.7)

# 理论最优补偿
theory_m = 0.6
ax2.axvline(theory_m, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical m* ({theory_m})', alpha=0.7)

for i, (idx, row) in enumerate(df_c_config.iterrows()):
    m_val = row['m_value']
    anon = row['Anonymization']
    ax2.text(m_val + 0.01, i, f"{m_val:.2f} ({anon})", 
             va='center', ha='left', fontsize=8)

ax2.set_xlabel('Compensation m', fontsize=12, fontweight='bold')
ax2.set_title('LLM Intermediary Compensation Strategy\n(Blue=Anonymized, Orange=Identified)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_config_C_intermediary_strategy.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. 配置B：LLM消费者决策准确率
# ============================================================================
print("[3/8] 生成配置B消费者决策准确率图...")

df_b = df_c[df_c['Config'] == 'B'].copy()
df_b = df_b.sort_values('Individual Accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.barh(df_b['模型'], df_b['Individual Accuracy'] * 100)
for bar in bars:
    bar.set_alpha(0.7)

# 添加理想准确率线
ax.axvline(100, color='red', linestyle='--', linewidth=2, 
           label='Perfect Accuracy (100%)', alpha=0.7)

# 添加数值标签和参与率信息
for i, (idx, row) in enumerate(df_b.iterrows()):
    acc = row['Individual Accuracy'] * 100
    part_rate = row['Participation Rate'] * 100
    ax.text(acc + 1, i, f"{acc:.1f}% (Part.Rate: {part_rate:.1f}%)", 
            va='center', ha='left', fontsize=9)

ax.set_xlabel('Decision Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Config B: Rational Intermediary × LLM Consumer - Consumer Decision Accuracy\n(Theoretical Participation Rate = 3.4%)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '3_config_B_consumer_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. 四配置综合对比热力图
# ============================================================================
print("[4/8] 生成四配置综合对比热力图...")

# 准备数据：每个模型的4个配置的关键指标
models = sorted(df_c['模型'].unique())
configs = ['B', 'C', 'D']

# 创建数据矩阵（使用利润损失百分比）
profit_loss_matrix = []
for model in models:
    row = []
    for config in configs:
        val = df_c[(df_c['模型'] == model) & (df_c['Config'] == config)]['Intermediary Profit Loss (%)'].values
        row.append(val[0] if len(val) > 0 else 0)
    profit_loss_matrix.append(row)

profit_loss_matrix = np.array(profit_loss_matrix)

fig, ax = plt.subplots(figsize=(10, 12))

# 绘制热力图（使用发散色彩，中心为0）
sns.heatmap(profit_loss_matrix, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn_r',  # 红色=亏损，绿色=盈利
            center=0,
            xticklabels=configs,
            yticklabels=models,
            cbar_kws={'label': 'Profit Loss (%)'},
            linewidths=0.5,
            ax=ax)

ax.set_title('Multi-Config Profit Loss Heatmap\n(Red=Loss, Green=Exceeds Theory)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '4_multi_config_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. 配置D：补偿策略 vs 利润散点图
# ============================================================================
print("[5/8] 生成配置D补偿策略分析图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：补偿 vs 利润
df_d['m_value'] = df_d['m'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x))

# 按匿名化策略着色
colors_map = {'anonymized': 'blue', 'identified': 'orange'}
colors = [colors_map[x] for x in df_d['Anonymization']]

scatter = ax1.scatter(df_d['m_value'], df_d['Intermediary Profit'], 
                     c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

# 添加模型标签
for idx, row in df_d.iterrows():
    ax1.annotate(row['模型'], 
                (row['m_value'], row['Intermediary Profit']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8)

ax1.axhline(theory_profit, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Profit', alpha=0.7)
ax1.axvline(0.6, color='green', linestyle='--', linewidth=2, 
            label=f'Theoretical m*', alpha=0.7)

ax1.set_xlabel('Compensation m', fontsize=13, fontweight='bold')
ax1.set_ylabel('Intermediary Profit', fontsize=13, fontweight='bold')
ax1.set_title('Compensation Strategy vs Profit\n(Blue=Anonymized, Orange=Identified)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 右图：参与率 vs 利润
scatter2 = ax2.scatter(df_d['Participation Rate'] * 100, df_d['Intermediary Profit'], 
                      c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

for idx, row in df_d.iterrows():
    ax2.annotate(row['模型'], 
                (row['Participation Rate'] * 100, row['Intermediary Profit']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8)

ax2.axhline(theory_profit, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Profit', alpha=0.7)
ax2.axvline(3.4, color='green', linestyle='--', linewidth=2, 
            label=f'Theoretical Rate', alpha=0.7)

ax2.set_xlabel('Participation Rate (%)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Intermediary Profit', fontsize=13, fontweight='bold')
ax2.set_title('Participation Rate vs Profit', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '5_config_D_strategy_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. 消费者福利分析
# ============================================================================
print("[6/8] 生成消费者福利分析图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：配置D的消费者剩余缺口
df_d_sorted = df_d.sort_values('Consumer Surplus Gap', ascending=True)

bars = ax1.barh(df_d_sorted['模型'], df_d_sorted['Consumer Surplus Gap'])
colors_gap = ['red' if x < 0 else 'green' for x in df_d_sorted['Consumer Surplus Gap']]
for bar, color in zip(bars, colors_gap):
    bar.set_color(color)
    bar.set_alpha(0.7)

ax1.axvline(0, color='black', linestyle='-', linewidth=1)

for i, (idx, row) in enumerate(df_d_sorted.iterrows()):
    gap = row['Consumer Surplus Gap']
    ax1.text(gap + 0.5 if gap < 0 else gap - 0.5, i, 
            f"{gap:.1f}", 
            va='center', ha='left' if gap < 0 else 'right',
            fontsize=9)

ax1.set_xlabel('Consumer Surplus Gap', fontsize=13, fontweight='bold')
ax1.set_ylabel('Model', fontsize=13, fontweight='bold')
ax1.set_title('Config D: Consumer Surplus Change\n(Negative = Consumer Harmed)', 
              fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 右图：基尼系数对比
theory_gini = df_theory['Gini Consumer Surplus'].iloc[0]

gini_data = []
gini_labels = []
for config in ['B', 'C', 'D']:
    df_config = df_c[df_c['Config'] == config].copy()
    if not df_config.empty:
        # 转换为float并过滤NaN值
        gini_values = pd.to_numeric(df_config['Gini Consumer Surplus'], errors='coerce')
        gini_values = gini_values.dropna().values
        if len(gini_values) > 0:
            gini_data.append(gini_values)
            gini_labels.append(f'Config {config}')

bp = ax2.boxplot(gini_data, tick_labels=gini_labels, patch_artist=True,
                showmeans=True, meanline=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax2.axhline(theory_gini, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Gini', alpha=0.7)

ax2.set_ylabel('Gini Coefficient', fontsize=13, fontweight='bold')
ax2.set_title('Consumer Surplus Inequality\n(Lower Gini = More Equal)', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '6_consumer_welfare_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. 模型性能雷达图（选取Top 5模型）
# ============================================================================
print("[7/8] 生成模型性能雷达图...")

# 计算综合分数并选择Top 5
df_d_metrics = df_d.copy()
df_d_metrics['profit_score'] = (df_d_metrics['Intermediary Profit'] - df_d_metrics['Intermediary Profit'].min()) / (df_d_metrics['Intermediary Profit'].max() - df_d_metrics['Intermediary Profit'].min())
df_d_metrics['accuracy_score'] = df_d_metrics['Individual Accuracy']
df_d_metrics['welfare_score'] = 1 - abs(df_d_metrics['Consumer Surplus Gap']) / abs(df_d_metrics['Consumer Surplus Gap']).max()

df_d_metrics['total_score'] = (df_d_metrics['profit_score'] * 0.5 + 
                                df_d_metrics['accuracy_score'] * 0.3 + 
                                df_d_metrics['welfare_score'] * 0.2)

top5_models = df_d_metrics.nlargest(5, 'total_score')['模型'].tolist()

# 准备雷达图数据
categories = ['Profit', 'Consumer\nAccuracy', 'Participation\nMatch', 'Welfare\nProtection', 'Cross-Config\nConsistency']
N = len(categories)

# 计算配置间一致性（衡量在B、C、D三个配置的利润表现稳定性）
consistency_scores = {}
for model in top5_models:
    model_all_configs = df_c[df_c['模型'] == model]
    b_profit = model_all_configs[model_all_configs['Config'] == 'B']['Intermediary Profit'].values[0]
    c_profit = model_all_configs[model_all_configs['Config'] == 'C']['Intermediary Profit'].values[0]
    d_profit = model_all_configs[model_all_configs['Config'] == 'D']['Intermediary Profit'].values[0]
    
    # 计算三个配置利润的标准差（归一化），标准差越小越稳定
    profits = np.array([b_profit, c_profit, d_profit])
    # 如果都是正利润，计算变异系数的倒数；否则根据盈利配置数量评分
    if np.all(profits > 0):
        # 变异系数 = std / mean，越小越好
        cv = np.std(profits) / (np.mean(profits) + 1e-6)
        consistency = 1 / (1 + cv)  # 转换为0-1之间，越大越稳定
    else:
        # 根据盈利配置数量评分
        num_profitable = np.sum(profits > 0)
        consistency = num_profitable / 3.0
    consistency_scores[model] = consistency

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors_radar = plt.cm.Set3(np.linspace(0, 1, len(top5_models)))

for i, model in enumerate(top5_models):
    model_data = df_d_metrics[df_d_metrics['模型'] == model].iloc[0]
    
    # 归一化各指标到0-1
    values = [
        model_data['profit_score'],
        model_data['Individual Accuracy'],
        1 - abs(model_data['Participation Rate'] - 0.0342) / 0.1,  # 参与率接近度
        model_data['welfare_score'],
        consistency_scores[model]  # 配置间一致性
    ]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[i])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Top 5 Model Performance Radar Chart (Config D)\n(Higher Value = Better Performance)', 
             fontsize=15, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig(output_dir / '7_top5_models_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. 综合排名表
# ============================================================================
print("[8/8] 生成综合排名表...")

# 计算各配置的排名
ranking_data = []

for model in models:
    model_data = df_c[df_c['模型'] == model]
    
    # 配置B
    b_data = model_data[model_data['Config'] == 'B']
    if b_data.empty:
        continue
    b_data = b_data.iloc[0]
    b_accuracy = b_data['Individual Accuracy']
    
    # 配置C
    c_data = model_data[model_data['Config'] == 'C']
    if c_data.empty:
        continue
    c_data = c_data.iloc[0]
    c_profit = c_data['Intermediary Profit']
    c_loss = c_data['Intermediary Profit Loss (%)']
    
    # 配置D
    d_data = model_data[model_data['Config'] == 'D']
    if d_data.empty:
        continue
    d_data = d_data.iloc[0]
    d_profit = d_data['Intermediary Profit']
    d_loss = d_data['Intermediary Profit Loss (%)']
    d_accuracy = d_data['Individual Accuracy']
    
    ranking_data.append({
        'Model': model,
        'B-Cons.Acc': f"{b_accuracy:.2%}",
        'C-Profit': f"{c_profit:.3f}",
        'C-Loss%': f"{c_loss:.1f}%",
        'D-Profit': f"{d_profit:.3f}",
        'D-Loss%': f"{d_loss:.1f}%",
        'D-Cons.Acc': f"{d_accuracy:.2%}",
        'Overall': (b_accuracy * 0.2 + 
                    (1 if c_loss < 0 else 1 - c_loss/100) * 0.3 + 
                    (1 if d_loss < 0 else 1 - d_loss/100) * 0.3 +
                    d_accuracy * 0.2)
    })

df_ranking = pd.DataFrame(ranking_data)
df_ranking = df_ranking.sort_values('Overall', ascending=False)
df_ranking['Rank'] = range(1, len(df_ranking) + 1)
df_ranking = df_ranking[['Rank', 'Model', 'B-Cons.Acc', 'C-Profit', 'C-Loss%', 
                        'D-Profit', 'D-Loss%', 'D-Cons.Acc', 'Overall']]

# 绘制表格
fig, ax = plt.subplots(figsize=(18, 12))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_ranking.values,
                colLabels=df_ranking.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# 设置表头样式
for i in range(len(df_ranking.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 交替行颜色
for i in range(1, len(df_ranking) + 1):
    if i % 2 == 0:
        for j in range(len(df_ranking.columns)):
            table[(i, j)].set_facecolor('#E7E6E6')

# 高亮前三名
for i in range(1, min(4, len(df_ranking) + 1)):
    for j in range(len(df_ranking.columns)):
        table[(i, j)].set_facecolor('#FFD966')
        table[(i, j)].set_text_props(weight='bold')

plt.title('Scenario C Model Comprehensive Ranking\n(Yellow = Top 3)', 
          fontsize=16, fontweight='bold', pad=20)

plt.savefig(output_dir / '8_comprehensive_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存排名表为CSV
df_ranking.to_csv(output_dir / 'model_ranking.csv', index=False, encoding='utf-8-sig')

# ============================================================================
# 总结报告
# ============================================================================
print()
print("=" * 80)
print("[OK] 可视化完成！")
print("=" * 80)
print(f"\n生成了8张可视化图表：")
print(f"1. 配置D中介利润对比图")
print(f"2. 配置C中介策略质量对比图")
print(f"3. 配置B消费者决策准确率图")
print(f"4. 四配置综合对比热力图")
print(f"5. 配置D补偿策略分析图")
print(f"6. 消费者福利分析图")
print(f"7. Top 5模型性能雷达图")
print(f"8. 综合排名表")
print(f"\n所有图表已保存到: {output_dir}/")
print(f"排名数据已保存到: {output_dir}/model_ranking.csv")

# 输出关键发现
print("\n" + "=" * 80)
print("[分析] 关键发现")
print("=" * 80)

print("\n[排名] Top 3 模型（综合评分）：")
for i, row in df_ranking.head(3).iterrows():
    print(f"   {int(row['Rank'])}. {row['Model']} (综合评分: {row['Overall']:.3f})")

print("\n[利润] 配置D最高利润：")
best_d = df_d.loc[df_d['Intermediary Profit'].idxmax()]
print(f"   {best_d['模型']}: {best_d['Intermediary Profit']:.3f} (理论最优: {theory_profit:.3f})")

print("\n[准确率] 配置B最高准确率：")
best_b = df_b.loc[df_b['Individual Accuracy'].idxmax()]
print(f"   {best_b['模型']}: {best_b['Individual Accuracy']:.2%}")

print("\n[策略] 配置C最佳策略：")
best_c = df_c_config.loc[df_c_config['Intermediary Profit'].idxmax()]
print(f"   {best_c['模型']}: 利润={best_c['Intermediary Profit']:.3f}, m={best_c['m']}, {best_c['Anonymization']}")

print("\n" + "=" * 80)
