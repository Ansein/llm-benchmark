"""
GPT Model Family Sensitivity Analysis Comparison
Compare gpt-5, gpt-5.1, gpt-5.1-2025-11-13, and gpt-5.2 across parameter combinations
"""
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Load results
print('Loading data...')
# Load the combined file with gpt-5, gpt-5.1, gpt-5.1-2025-11-13
with open('sensitivity_results/scenario_b/summary_all_results_gpt-5_gpt-5.1_gpt-5.1-2025-11-13.json', 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

# Load gpt-5.2 separately
with open('sensitivity_results/scenario_b/summary_all_results_gpt-5.2.json', 'r', encoding='utf-8') as f:
    gpt52_data = json.load(f)

# Combine all data
all_data = combined_data + gpt52_data

# Organize by model
all_model_data = {}
for result in all_data:
    model_name = result['model_name']
    if model_name not in all_model_data:
        all_model_data[model_name] = []
    all_model_data[model_name].append(result)

# Print summary
print(f'\nLoaded {len(all_data)} total experiments')
for model_name, data in sorted(all_model_data.items()):
    print(f'  {model_name}: {len(data)} experiments')

print()

# Organize data by model and parameter combinations
model_stats = {}
for model_name, results in all_model_data.items():
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
    model_stats[model_name] = stats_by_params

# Create output directory
output_dir = Path('sensitivity_results/scenario_b/gpt_family_comparison_plots')
output_dir.mkdir(parents=True, exist_ok=True)

print('='*80)
print('Generating GPT Model Family Comparison Visualizations...')
print('='*80)

# Define parameter grid
rho_vals = [0.3, 0.6, 0.9]
v_ranges_vals = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
v_labels = ['v=[0.3,0.6]', 'v=[0.6,0.9]', 'v=[0.9,1.2]']
rho_labels = ['ρ=0.3', 'ρ=0.6', 'ρ=0.9']

# Model colors - use high-contrast colors for better distinction
model_colors = {
    'gpt-5': '#3498DB',           # Blue
    'gpt-5.1': '#2ECC71',         # Green
    'gpt-5.1-2025-11-13': '#F39C12',  # Orange
    'gpt-5.2': '#9B59B6'          # Purple
}

# ============================================================================
# Figure 1: Model Evolution Heatmaps - Jaccard Similarity
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle('GPT Model Evolution: Jaccard Similarity Across Versions (Higher is Better)', 
             fontsize=18, fontweight='bold', y=0.995)

for idx, (model_name, color) in enumerate(model_colors.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Create matrix
    jaccard_matrix = np.zeros((3, 3))
    for i, rho in enumerate(rho_vals):
        for j, (v_min, v_max) in enumerate(v_ranges_vals):
            key = (rho, v_min, v_max)
            trials = model_stats[model_name][key]
            jaccard_matrix[j, i] = np.mean([t['jaccard'] for t in trials])
    
    im = ax.imshow(jaccard_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(rho_labels, fontsize=11)
    ax.set_yticklabels(v_labels, fontsize=11)
    ax.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
    
    # Annotate values
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", 
                         fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
heatmap_path = output_dir / 'gpt_evolution_heatmaps.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {heatmap_path}')
plt.close()

# ============================================================================
# Figure 2: Overall Performance Comparison - Grouped Bar Chart
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('GPT Model Family: Overall Performance Metrics', 
             fontsize=20, fontweight='bold', y=0.995)

metrics_to_compare = [
    ('jaccard', 'Average Jaccard Similarity', 'higher', [0, 1]),
    ('profit_mae', 'Average Profit MAE', 'lower', None),
    ('welfare_mae', 'Average Welfare MAE', 'lower', None),
    ('correct_eq', 'Correct Equilibrium Rate', 'higher', [0, 1])
]

for idx, (metric_key, title, direction, ylim_range) in enumerate(metrics_to_compare):
    ax = axes[idx // 2, idx % 2]
    
    model_names_list = ['gpt-5', 'gpt-5.1', 'gpt-5.1-2025-11-13', 'gpt-5.2']
    means = []
    stds = []
    
    for model_name in model_names_list:
        all_vals = []
        for trials in model_stats[model_name].values():
            all_vals.extend([t[metric_key] for t in trials])
        means.append(np.mean(all_vals))
        stds.append(np.std(all_vals))
    
    x_pos = np.arange(len(model_names_list))
    colors = [model_colors[m] for m in model_names_list]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                  color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Model Version', fontsize=13, fontweight='bold')
    ax.set_ylabel(title, fontsize=13, fontweight='bold')
    ax.set_title(f'{title} ({direction.capitalize()} is Better)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names_list, fontsize=10, rotation=20, ha='right')
    if ylim_range:
        ax.set_ylim(ylim_range)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotate values
    for i, (mean, std) in enumerate(zip(means, stds)):
        if metric_key in ['correct_eq']:
            label = f'{mean:.1%}'
        elif metric_key == 'jaccard':
            label = f'{mean:.3f}'
        else:
            label = f'{mean:.2f}'
        ax.text(i, mean + std + (max(means) * 0.05), label,
               ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
overall_path = output_dir / 'gpt_overall_performance.png'
plt.savefig(overall_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {overall_path}')
plt.close()

# ============================================================================
# Figure 3: Performance Evolution Line Chart
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('GPT Model Evolution: Performance Trends by Parameter Combinations', 
             fontsize=20, fontweight='bold', y=0.995)

model_names_list = ['gpt-5', 'gpt-5.1', 'gpt-5.1-2025-11-13', 'gpt-5.2']
x_positions = [0, 1, 2, 3]

for row_idx, (v_min, v_max) in enumerate(v_ranges_vals):
    for col_idx, rho in enumerate(rho_vals):
        ax = axes[row_idx, col_idx]
        
        key = (rho, v_min, v_max)
        
        means = []
        stds = []
        for model_name in model_names_list:
            trials = model_stats[model_name][key]
            jaccard_vals = [t['jaccard'] for t in trials]
            means.append(np.mean(jaccard_vals))
            stds.append(np.std(jaccard_vals))
        
        # Plot line with error bars
        ax.errorbar(x_positions, means, yerr=stds, 
                   marker='o', markersize=10, linewidth=2.5,
                   capsize=8, capthick=2, color='#2E86C1',
                   markerfacecolor='#3498DB', markeredgecolor='black',
                   markeredgewidth=1.5, label='Mean ± Std')
        
        # Plot individual trials as scatter
        for i, model_name in enumerate(model_names_list):
            trials = model_stats[model_name][key]
            jaccard_vals = [t['jaccard'] for t in trials]
            ax.scatter([i] * len(jaccard_vals), jaccard_vals, 
                      color=model_colors[model_name], s=80, alpha=0.4, zorder=2)
        
        ax.set_title(f'ρ={rho}, v=[{v_min},{v_max}]', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['gpt-5', 'gpt-5.1', 'gpt-5.1\n(2025-11)', 'gpt-5.2'], 
                          fontsize=9, rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.4)
        
        if row_idx == 2:
            ax.set_xlabel('Model Version', fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Jaccard Similarity', fontsize=11, fontweight='bold')

plt.tight_layout()
evolution_path = output_dir / 'gpt_evolution_trends.png'
plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {evolution_path}')
plt.close()

# ============================================================================
# Figure 4: Model Improvement Matrix
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('GPT Model Evolution: Version-to-Version Improvement Analysis', 
             fontsize=18, fontweight='bold')

# Compare: gpt-5 vs gpt-5.1, gpt-5.1 vs gpt-5.1-2025-11-13, gpt-5.1-2025-11-13 vs gpt-5.2
comparisons = [
    ('gpt-5', 'gpt-5.1', 'v5.0 → v5.1'),
    ('gpt-5.1', 'gpt-5.1-2025-11-13', 'v5.1 → v5.1 (2025-11)'),
    ('gpt-5.1-2025-11-13', 'gpt-5.2', 'v5.1 (2025-11) → v5.2')
]

for idx, (base_model, new_model, title) in enumerate(comparisons):
    ax = axes[idx]
    
    # Calculate improvement matrix
    improvement_matrix = np.zeros((3, 3))
    for i, rho in enumerate(rho_vals):
        for j, (v_min, v_max) in enumerate(v_ranges_vals):
            key = (rho, v_min, v_max)
            base_jaccard = np.mean([t['jaccard'] for t in model_stats[base_model][key]])
            new_jaccard = np.mean([t['jaccard'] for t in model_stats[new_model][key]])
            improvement = new_jaccard - base_jaccard
            improvement_matrix[j, i] = improvement
    
    # Use diverging colormap
    vmax = max(abs(improvement_matrix.min()), abs(improvement_matrix.max()))
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(rho_labels, fontsize=11)
    ax.set_yticklabels(v_labels, fontsize=11)
    ax.set_xlabel('Correlation Coefficient ρ', fontsize=12, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Privacy Preference Range v', fontsize=12, fontweight='bold')
    
    # Annotate values
    for i in range(3):
        for j in range(3):
            value = improvement_matrix[i, j]
            if value >= 0:
                text = f'+{value:.3f}'
                color = 'darkgreen' if value > 0.05 else 'black'
            else:
                text = f'{value:.3f}'
                color = 'darkred' if value < -0.05 else 'black'
            ax.text(j, i, text, ha="center", va="center", 
                   color=color, fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Improvement (ΔJaccard)', fontsize=11, fontweight='bold')

plt.tight_layout()
improvement_path = output_dir / 'gpt_improvement_matrix.png'
plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {improvement_path}')
plt.close()

# ============================================================================
# Figure 5: Radar Chart Comparison
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle('GPT Model Family: Comprehensive Performance Profiles', 
             fontsize=18, fontweight='bold', y=1.02)

categories = ['Jaccard\nSimilarity', 'Profit\nAccuracy\n(1-MAE/20)', 
              'Welfare\nAccuracy\n(1-MAE/2)', 'Share Rate\nAccuracy', 'Correctness\nRate']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for idx, (model_name, color) in enumerate(model_colors.items()):
    ax = plt.subplot(1, 4, idx + 1, projection='polar')
    
    # Calculate overall metrics
    all_trials = [t for trials in model_stats[model_name].values() for t in trials]
    
    jaccard = np.mean([t['jaccard'] for t in all_trials])
    profit_acc = 1 - min(np.mean([t['profit_mae'] for t in all_trials]) / 20, 1)
    welfare_acc = 1 - min(np.mean([t['welfare_mae'] for t in all_trials]) / 2, 1)
    
    # Share rate accuracy
    share_diffs = [abs(t['share_rate'] - t['gt_share_rate']) for t in all_trials]
    share_acc = 1 - np.mean(share_diffs)
    
    correct = np.mean([t['correct_eq'] for t in all_trials])
    
    values = [jaccard, profit_acc, welfare_acc, share_acc, correct]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3, color=color, markersize=10)
    ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'{model_name}', fontsize=13, fontweight='bold', 
                pad=20, color=color)

plt.tight_layout()
radar_path = output_dir / 'gpt_radar_profiles.png'
plt.savefig(radar_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {radar_path}')
plt.close()

# ============================================================================
# Figure 6: Parameter Impact Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Parameter Impact on GPT Model Family Performance', 
             fontsize=18, fontweight='bold')

# Impact of ρ
ax1 = axes[0]
width = 0.2
x_pos = np.arange(len(rho_vals))

for i, (model_name, color) in enumerate(model_colors.items()):
    rho_means = []
    rho_stds = []
    
    for rho in rho_vals:
        jaccard_vals = []
        for key, trials in model_stats[model_name].items():
            if key[0] == rho:
                jaccard_vals.extend([t['jaccard'] for t in trials])
        rho_means.append(np.mean(jaccard_vals))
        rho_stds.append(np.std(jaccard_vals))
    
    offset = (i - 1.5) * width
    bars = ax1.bar(x_pos + offset, rho_means, width, yerr=rho_stds, 
                   capsize=5, label=model_name, color=color,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

ax1.set_xlabel('Correlation Coefficient ρ', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Jaccard Similarity', fontsize=13, fontweight='bold')
ax1.set_title('Impact of ρ on Model Performance', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'ρ={rho}' for rho in rho_vals], fontsize=12)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)

# Impact of v range
ax2 = axes[1]
x_pos = np.arange(len(v_ranges_vals))

for i, (model_name, color) in enumerate(model_colors.items()):
    v_means = []
    v_stds = []
    
    for v_range in v_ranges_vals:
        jaccard_vals = []
        for key, trials in model_stats[model_name].items():
            if (key[1], key[2]) == v_range:
                jaccard_vals.extend([t['jaccard'] for t in trials])
        v_means.append(np.mean(jaccard_vals))
        v_stds.append(np.std(jaccard_vals))
    
    offset = (i - 1.5) * width
    bars = ax2.bar(x_pos + offset, v_means, width, yerr=v_stds,
                   capsize=5, label=model_name, color=color,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

ax2.set_xlabel('Privacy Preference Range v', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Jaccard Similarity', fontsize=13, fontweight='bold')
ax2.set_title('Impact of v Range on Model Performance', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'[{v[0]},{v[1]}]' for v in v_ranges_vals], fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)

plt.tight_layout()
param_impact_path = output_dir / 'gpt_parameter_impact.png'
plt.savefig(param_impact_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {param_impact_path}')
plt.close()

# ============================================================================
# Figure 7: Stability Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 8))

param_labels = []
model_cv_data = {model: [] for model in model_colors.keys()}

for key in sorted(model_stats['gpt-5'].keys()):
    rho, v_min, v_max = key
    param_labels.append(f'ρ={rho}\nv=[{v_min},{v_max}]')
    
    for model_name in model_colors.keys():
        trials = model_stats[model_name][key]
        jaccard_vals = [t['jaccard'] for t in trials]
        mean_val = np.mean(jaccard_vals)
        std_val = np.std(jaccard_vals)
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        model_cv_data[model_name].append(cv)

x_pos = np.arange(len(param_labels))
width = 0.2

for i, (model_name, color) in enumerate(model_colors.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x_pos + offset, model_cv_data[model_name], width,
                  label=model_name, color=color,
                  edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_xlabel('Parameter Combination', fontsize=13, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=13, fontweight='bold')
ax.set_title('GPT Model Stability Analysis (Lower CV = More Stable)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(param_labels, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
stability_path = output_dir / 'gpt_stability_comparison.png'
plt.savefig(stability_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {stability_path}')
plt.close()

# ============================================================================
# Figure 8: Best Model by Parameter Region
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        key = sorted(model_stats['gpt-5'].keys())[idx]
        rho, v_min, v_max = key
        
        # Get best model
        model_means = {}
        for model_name in model_colors.keys():
            trials = model_stats[model_name][key]
            model_means[model_name] = np.mean([t['jaccard'] for t in trials])
        best_model = max(model_means, key=model_means.get)
        best_score = model_means[best_model]
        
        # Draw colored rectangle
        color = model_colors[best_model]
        alpha = 0.3 + 0.6 * best_score
        rect = plt.Rectangle((j, 2-i), 1, 1, facecolor=color, 
                            edgecolor='black', linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        
        # Add text
        ax.text(j + 0.5, 2-i + 0.65, best_model, 
               ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j + 0.5, 2-i + 0.35, f'{best_score:.3f}',
               ha='center', va='center', fontsize=9)
        ax.text(j + 0.5, 2-i + 0.08, f'ρ={rho}, v=[{v_min},{v_max}]',
               ha='center', va='center', fontsize=7, style='italic')

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels(['Low v\n[0.3,0.6]', 'Medium v\n[0.6,0.9]', 'High v\n[0.9,1.2]'], 
                   fontsize=12, fontweight='bold')
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_yticklabels(['High ρ=0.9', 'Medium ρ=0.6', 'Low ρ=0.3'], 
                   fontsize=12, fontweight='bold')
ax.set_title('Best Performing GPT Model by Parameter Region\n(Color = Model, Intensity = Performance)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_aspect('equal')
ax.invert_yaxis()

# Add legend
import matplotlib.patches as mpatches
legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', 
                                 label=model_name, alpha=0.7)
                  for model_name, color in model_colors.items()]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left', 
         bbox_to_anchor=(1.02, 1), framealpha=0.9)

plt.tight_layout()
win_matrix_path = output_dir / 'gpt_best_by_region.png'
plt.savefig(win_matrix_path, dpi=300, bbox_inches='tight')
print(f'[OK] Saved: {win_matrix_path}')
plt.close()

# ============================================================================
# Statistical Summary Table
# ============================================================================
print()
print('='*90)
print('GPT Model Family Performance Summary:')
print('='*90)
print(f"{'Model':<25} {'Avg Jaccard':<15} {'Avg Profit MAE':<18} {'Avg Welfare MAE':<18} {'Correct %':<12}")
print('-'*90)

for model_name in ['gpt-5', 'gpt-5.1', 'gpt-5.1-2025-11-13', 'gpt-5.2']:
    all_trials = [t for trials in model_stats[model_name].values() for t in trials]
    
    avg_jaccard = np.mean([t['jaccard'] for t in all_trials])
    avg_profit_mae = np.mean([t['profit_mae'] for t in all_trials])
    avg_welfare_mae = np.mean([t['welfare_mae'] for t in all_trials])
    correct_rate = np.mean([t['correct_eq'] for t in all_trials])
    
    print(f"{model_name:<25} {avg_jaccard:<15.3f} {avg_profit_mae:<18.4f} {avg_welfare_mae:<18.4f} {correct_rate:<12.1%}")

print()
print('='*90)
print('[SUCCESS] All GPT model family comparison charts have been saved to:')
print(f'  {output_dir}')
print('='*90)
print('Generated Charts:')
print('  1. gpt_evolution_heatmaps.png - 2x2 heatmaps showing each model')
print('  2. gpt_overall_performance.png - Overall metrics comparison')
print('  3. gpt_evolution_trends.png - Performance trends across versions')
print('  4. gpt_improvement_matrix.png - Version-to-version improvements')
print('  5. gpt_radar_profiles.png - Comprehensive performance profiles')
print('  6. gpt_parameter_impact.png - Parameter impact comparison')
print('  7. gpt_stability_comparison.png - Stability analysis')
print('  8. gpt_best_by_region.png - Best model by parameter region')
print('='*90)
