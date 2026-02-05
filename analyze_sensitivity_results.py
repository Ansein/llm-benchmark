"""
Sensitivity Analysis Results with Visualization
"""
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 读取结果
with open('sensitivity_results/scenario_b/sensitivity_3x3_b.v4_20260129_143102/summary_all_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f'Total Experiments: {len(results)}')
print()

# Statistics by parameter combinations
stats_by_params = defaultdict(list)
for r in results:
    key = (r['sensitivity_params']['rho'], 
           r['sensitivity_params']['v_min'], 
           r['sensitivity_params']['v_max'])
    stats_by_params[key].append({
        'jaccard': r['equilibrium_quality']['share_set_similarity'],
        'profit_mae': r['metrics']['deviations']['profit_mae'],
        'welfare_mae': r['metrics']['deviations']['welfare_mae'],
        'share_rate': r['metrics']['llm']['share_rate'],
        'gt_share_rate': r['metrics']['ground_truth']['share_rate'],
        'correct_eq': r['equilibrium_quality']['correct_equilibrium'],
    })

print('='*80)
print('Average Performance by Parameter Combinations (Mean of 3 trials):')
print('='*80)
header = f"{'Parameters':<25} {'Jaccard':<10} {'Profit MAE':<12} {'Welfare MAE':<12} {'Share Rate':<12} {'Accuracy':<10}"
print(header)
print('-'*80)

for key in sorted(stats_by_params.keys()):
    rho, v_min, v_max = key
    trials = stats_by_params[key]
    
    avg_jaccard = np.mean([t['jaccard'] for t in trials])
    avg_profit_mae = np.mean([t['profit_mae'] for t in trials])
    avg_welfare_mae = np.mean([t['welfare_mae'] for t in trials])
    avg_share = np.mean([t['share_rate'] for t in trials])
    gt_share = trials[0]['gt_share_rate']  # GT is fixed for same parameters
    correct_rate = np.mean([t['correct_eq'] for t in trials])
    
    param_str = f'ρ={rho} v=[{v_min},{v_max}]'
    row = f'{param_str:<25} {avg_jaccard:<10.3f} {avg_profit_mae:<12.4f} {avg_welfare_mae:<12.4f} {avg_share:<12.2%} {correct_rate:<10.1%}'
    print(row)
    print(f'  {"":>25} (GT Share Rate: {gt_share:.2%})')

print()
print('='*80)
print('Overall Statistics:')
print('='*80)
all_jaccard = [t['jaccard'] for trials in stats_by_params.values() for t in trials]
all_profit_mae = [t['profit_mae'] for trials in stats_by_params.values() for t in trials]
all_welfare_mae = [t['welfare_mae'] for trials in stats_by_params.values() for t in trials]
all_correct = [t['correct_eq'] for trials in stats_by_params.values() for t in trials]

print(f"Average Jaccard Similarity: {np.mean(all_jaccard):.3f} ± {np.std(all_jaccard):.3f}")
print(f"Average Profit MAE: {np.mean(all_profit_mae):.4f} ± {np.std(all_profit_mae):.4f}")
print(f"Average Welfare MAE: {np.mean(all_welfare_mae):.4f} ± {np.std(all_welfare_mae):.4f}")
print(f"Correct Equilibrium Rate: {np.mean(all_correct):.1%}")
print()

# Impact of ρ
print('='*80)
print('Impact of ρ:')
print('='*80)
rho_stats = defaultdict(list)
for key, trials in stats_by_params.items():
    rho = key[0]
    rho_stats[rho].extend([t['jaccard'] for t in trials])

for rho in sorted(rho_stats.keys()):
    jaccard_vals = rho_stats[rho]
    print(f"ρ={rho}: Average Jaccard={np.mean(jaccard_vals):.3f} ± {np.std(jaccard_vals):.3f}")

print()

# Impact of v range
print('='*80)
print('Impact of v Range:')
print('='*80)
v_stats = defaultdict(list)
for key, trials in stats_by_params.items():
    v_range = (key[1], key[2])
    v_stats[v_range].extend([t['jaccard'] for t in trials])

for v_range in sorted(v_stats.keys()):
    jaccard_vals = v_stats[v_range]
    print(f"v=[{v_range[0]},{v_range[1]}]: Average Jaccard={np.mean(jaccard_vals):.3f} ± {np.std(jaccard_vals):.3f}")

print()
print('='*80)
print('Generating Visualizations...')
print('='*80)

# 创建输出目录
output_dir = Path('sensitivity_results/scenario_b/sensitivity_3x3_b.v4_20260129_143102/analysis_plots')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 1: 3x3 Heatmap Matrix - Multiple Metrics
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Scenario B Sensitivity Analysis - Parameter Impact Heatmap (gpt-5.2, b.v4)', 
             fontsize=20, fontweight='bold', y=0.995)

# Prepare data matrices
rho_vals = [0.3, 0.6, 0.9]
v_ranges_vals = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
v_labels = ['v=[0.3,0.6]', 'v=[0.6,0.9]', 'v=[0.9,1.2]']
rho_labels = ['ρ=0.3', 'ρ=0.6', 'ρ=0.9']

# Create matrices
jaccard_matrix = np.zeros((3, 3))
profit_mae_matrix = np.zeros((3, 3))
welfare_mae_matrix = np.zeros((3, 3))
share_rate_matrix = np.zeros((3, 3))

for i, rho in enumerate(rho_vals):
    for j, (v_min, v_max) in enumerate(v_ranges_vals):
        key = (rho, v_min, v_max)
        trials = stats_by_params[key]
        jaccard_matrix[j, i] = np.mean([t['jaccard'] for t in trials])
        profit_mae_matrix[j, i] = np.mean([t['profit_mae'] for t in trials])
        welfare_mae_matrix[j, i] = np.mean([t['welfare_mae'] for t in trials])
        share_rate_matrix[j, i] = np.mean([t['share_rate'] for t in trials])

# Subplot 1: Jaccard Similarity
ax1 = axes[0, 0]
im1 = ax1.imshow(jaccard_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax1.set_title('Jaccard Similarity (Higher is Better)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(rho_labels, fontsize=11)
ax1.set_yticklabels(v_labels, fontsize=11)
ax1.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
ax1.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        text = ax1.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=10)

# Subplot 2: Profit MAE
ax2 = axes[0, 1]
im2 = ax2.imshow(profit_mae_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0)
ax2.set_title('Profit MAE (Lower is Better)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(rho_labels, fontsize=11)
ax2.set_yticklabels(v_labels, fontsize=11)
ax2.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, f'{profit_mae_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=10)

# Subplot 3: Welfare MAE
ax3 = axes[1, 0]
im3 = ax3.imshow(welfare_mae_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0)
ax3.set_title('Welfare MAE (Lower is Better)', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(range(3))
ax3.set_yticks(range(3))
ax3.set_xticklabels(rho_labels, fontsize=11)
ax3.set_yticklabels(v_labels, fontsize=11)
ax3.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{welfare_mae_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.ax.tick_params(labelsize=10)

# Subplot 4: LLM Share Rate
ax4 = axes[1, 1]
im4 = ax4.imshow(share_rate_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax4.set_title('LLM Share Rate', fontsize=14, fontweight='bold', pad=10)
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels(rho_labels, fontsize=11)
ax4.set_yticklabels(v_labels, fontsize=11)
ax4.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        text = ax4.text(j, i, f'{share_rate_matrix[i, j]:.1%}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.ax.tick_params(labelsize=10)

plt.tight_layout()
heatmap_path = output_dir / 'sensitivity_heatmap_matrix.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {heatmap_path}')
plt.close()

# ============================================================================
# Figure 2: Parameter Impact Comparison - Grouped Bar Charts
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Impact of Parameters on Model Performance', fontsize=18, fontweight='bold')

# Impact of ρ
ax1 = axes[0]
rho_means = []
rho_stds = []
for rho in sorted(rho_stats.keys()):
    jaccard_vals = rho_stats[rho]
    rho_means.append(np.mean(jaccard_vals))
    rho_stds.append(np.std(jaccard_vals))

x_pos = np.arange(len(rho_means))
bars1 = ax1.bar(x_pos, rho_means, yerr=rho_stds, capsize=8, 
                color=['#2ecc71', '#f39c12', '#e74c3c'], 
                edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_xlabel('Correlation Coefficient ρ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Average Jaccard Similarity', fontsize=14, fontweight='bold')
ax1.set_title('Impact of ρ', fontsize=15, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'ρ={rho}' for rho in sorted(rho_stats.keys())], fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Medium Level')
ax1.legend(fontsize=11)

# Annotate values on bars
for i, (mean, std) in enumerate(zip(rho_means, rho_stds)):
    ax1.text(i, mean + std + 0.05, f'{mean:.3f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Impact of v range
ax2 = axes[1]
v_means = []
v_stds = []
v_range_labels = []
for v_range in sorted(v_stats.keys()):
    jaccard_vals = v_stats[v_range]
    v_means.append(np.mean(jaccard_vals))
    v_stds.append(np.std(jaccard_vals))
    v_range_labels.append(f'[{v_range[0]},{v_range[1]}]')

x_pos = np.arange(len(v_means))
bars2 = ax2.bar(x_pos, v_means, yerr=v_stds, capsize=8,
                color=['#2ecc71', '#f39c12', '#e74c3c'],
                edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_xlabel('Privacy Preference Range v', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Jaccard Similarity', fontsize=14, fontweight='bold')
ax2.set_title('Impact of v Range', fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(v_range_labels, fontsize=12)
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Medium Level')
ax2.legend(fontsize=11)

# Annotate values on bars
for i, (mean, std) in enumerate(zip(v_means, v_stds)):
    ax2.text(i, mean + std + 0.05, f'{mean:.3f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
param_impact_path = output_dir / 'parameter_impact.png'
plt.savefig(param_impact_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {param_impact_path}')
plt.close()

# ============================================================================
# Figure 3: Comprehensive Performance Comparison - Radar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Select 3 representative scenarios
scenarios = [
    ('ρ=0.3, v=[0.3,0.6]', (0.3, 0.3, 0.6), '#2ecc71'),  # Best
    ('ρ=0.6, v=[0.6,0.9]', (0.6, 0.6, 0.9), '#f39c12'),  # Medium
    ('ρ=0.9, v=[0.9,1.2]', (0.9, 0.9, 1.2), '#e74c3c'),  # Worst
]

categories = ['Jaccard\nSimilarity', 'Profit\nAccuracy\n(1-MAE)', 'Welfare\nAccuracy\n(1-MAE)', 
              'Share Rate\nAccuracy', 'Correctness\nRate']
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for label, key, color in scenarios:
    trials = stats_by_params[key]
    
    # Calculate normalized metrics (higher is better)
    jaccard = np.mean([t['jaccard'] for t in trials])
    profit_acc = 1 - min(np.mean([t['profit_mae'] for t in trials]) / 20, 1)  # Normalized
    welfare_acc = 1 - min(np.mean([t['welfare_mae'] for t in trials]) / 2, 1)  # Normalized
    
    # Share rate accuracy
    llm_sr = np.mean([t['share_rate'] for t in trials])
    gt_sr = trials[0]['gt_share_rate']
    share_acc = 1 - abs(llm_sr - gt_sr)
    
    correct = np.mean([t['correct_eq'] for t in trials])
    
    values = [jaccard, profit_acc, welfare_acc, share_acc, correct]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3, label=label, color=color, markersize=8)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Comprehensive Performance Comparison of Representative Scenarios', fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)

radar_path = output_dir / 'performance_radar.png'
plt.savefig(radar_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {radar_path}')
plt.close()

# ============================================================================
# Figure 4: Detailed Comparison for Each Parameter Combination
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 8))

param_labels = []
jaccard_vals_all = []
colors_all = []

for i, key in enumerate(sorted(stats_by_params.keys())):
    rho, v_min, v_max = key
    trials = stats_by_params[key]
    
    param_labels.append(f'ρ={rho}\nv=[{v_min},{v_max}]')
    jaccard_list = [t['jaccard'] for t in trials]
    jaccard_vals_all.append(jaccard_list)
    
    # Color based on performance
    mean_jaccard = np.mean(jaccard_list)
    if mean_jaccard > 0.8:
        colors_all.append('#2ecc71')
    elif mean_jaccard > 0.5:
        colors_all.append('#f39c12')
    else:
        colors_all.append('#e74c3c')

bp = ax.boxplot(jaccard_vals_all, tick_labels=param_labels, patch_artist=True,
                widths=0.6, showmeans=True,
                boxprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
                medianprops=dict(linewidth=3, color='darkblue'),
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

for patch, color in zip(bp['boxes'], colors_all):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Parameter Combination', fontsize=14, fontweight='bold')
ax.set_ylabel('Jaccard Similarity', fontsize=14, fontweight='bold')
ax.set_title('Performance Distribution of All Parameter Combinations (Box Plot)', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Medium Level')
ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent Level')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=12, loc='upper right')

plt.xticks(rotation=0, fontsize=11)
plt.tight_layout()
boxplot_path = output_dir / 'performance_boxplot.png'
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {boxplot_path}')
plt.close()

print()
print('='*80)
print(f'[SUCCESS] All visualization charts have been saved to: {output_dir}')
print('='*80)
print('Generated Charts:')
print(f'  1. sensitivity_heatmap_matrix.png - 4-Metric Heatmap Matrix')
print(f'  2. parameter_impact.png - Parameter Impact Comparison')
print(f'  3. performance_radar.png - Comprehensive Performance Radar Chart')
print(f'  4. performance_boxplot.png - Performance Distribution Box Plot')
print('='*80)
