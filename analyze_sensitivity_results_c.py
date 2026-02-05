"""
场景C敏感度分析结果可视化
生成直观美观的对比图表
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
sns.set_palette("husl")


def load_results(results_path):
    """加载敏感度分析结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def organize_results(results):
    """按参数组合和模型组织结果"""
    stats_by_params = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        key = (r['sensitivity_params']['tau_mean'], 
               r['sensitivity_params']['sigma'])
        model = r['model_name']
        
        # 提取关键指标（以配置D为主要评估对象）
        config_d = r['results'].get('config_D', {})
        metrics = r['metrics']
        
        stats_by_params[key][model].append({
            'config_d_profit': config_d.get('final_profit', 0),
            'config_d_convergence': config_d.get('converged', False),
            'anonymization_accuracy': metrics.get('anonymization_accuracy', 0),
            'profit_deviation': abs(config_d.get('final_profit', 0) - r['gt_summary']['intermediary_profit_star']),
            'welfare_deviation': abs(metrics.get('social_welfare', 0) - r['gt_summary']['social_welfare']),
        })
    
    return stats_by_params


def create_visualizations(results, output_dir):
    """生成所有可视化图表"""
    
    # 组织数据
    stats_by_params = organize_results(results)
    
    # 提取参数值
    tau_values = sorted(set([k[0] for k in stats_by_params.keys()]))
    sigma_values = sorted(set([k[1] for k in stats_by_params.keys()]))
    
    tau_labels = [f'τ={t}' for t in tau_values]
    sigma_labels = [f'σ={s}' for s in sigma_values]
    
    # 获取所有模型
    all_models = set()
    for param_dict in stats_by_params.values():
        all_models.update(param_dict.keys())
    all_models = sorted(all_models)
    
    print(f"Found {len(all_models)} models: {all_models}")
    print(f"Parameter combinations: {len(stats_by_params)}")
    
    # =========================================================================
    # 图1: 3x3热力图 - 平均收敛利润
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Scenario C Sensitivity Analysis - Average Intermediary Profit (Config D)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 为前4个模型各生成一个热力图
    for idx, model in enumerate(all_models[:4]):
        if idx >= 4:
            break
        
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 创建利润矩阵
        profit_matrix = np.zeros((len(sigma_values), len(tau_values)))
        
        for i, sigma in enumerate(sigma_values):
            for j, tau in enumerate(tau_values):
                key = (tau, sigma)
                if key in stats_by_params and model in stats_by_params[key]:
                    trials = stats_by_params[key][model]
                    avg_profit = np.mean([t['config_d_profit'] for t in trials])
                    profit_matrix[i, j] = avg_profit
                else:
                    profit_matrix[i, j] = np.nan
        
        # 绘制热力图
        im = ax.imshow(profit_matrix, cmap='RdYlGn', aspect='auto', vmin=0)
        ax.set_title(f'{model}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(range(len(tau_values)))
        ax.set_yticks(range(len(sigma_values)))
        ax.set_xticklabels(tau_labels, fontsize=11)
        ax.set_yticklabels(sigma_labels, fontsize=11)
        ax.set_xlabel('Privacy Sensitivity τ_mean', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal Noise σ', fontsize=12, fontweight='bold')
        
        # 标注数值
        for i in range(len(sigma_values)):
            for j in range(len(tau_values)):
                if not np.isnan(profit_matrix[i, j]):
                    text = ax.text(j, i, f'{profit_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", 
                                 fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    heatmap_path = output_dir / 'sensitivity_profit_heatmaps.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {heatmap_path}')
    plt.close()
    
    # =========================================================================
    # 图2: 参数影响 - 分组柱状图
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Parameter Impact on Intermediary Profit (Config D)', 
                 fontsize=18, fontweight='bold')
    
    # τ的影响
    ax1 = axes[0]
    width = 0.8 / len(all_models[:4])
    x_pos = np.arange(len(tau_values))
    
    for i, model in enumerate(all_models[:4]):
        tau_means = []
        tau_stds = []
        
        for tau in tau_values:
            profits = []
            for sigma in sigma_values:
                key = (tau, sigma)
                if key in stats_by_params and model in stats_by_params[key]:
                    trials = stats_by_params[key][model]
                    profits.extend([t['config_d_profit'] for t in trials])
            
            if profits:
                tau_means.append(np.mean(profits))
                tau_stds.append(np.std(profits))
            else:
                tau_means.append(0)
                tau_stds.append(0)
        
        offset = (i - len(all_models[:4])/2 + 0.5) * width
        ax1.bar(x_pos + offset, tau_means, width, yerr=tau_stds,
               capsize=5, label=model, alpha=0.8)
    
    ax1.set_xlabel('Privacy Sensitivity τ_mean', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Intermediary Profit', fontsize=13, fontweight='bold')
    ax1.set_title('Impact of τ_mean', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tau_labels, fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # σ的影响
    ax2 = axes[1]
    x_pos = np.arange(len(sigma_values))
    
    for i, model in enumerate(all_models[:4]):
        sigma_means = []
        sigma_stds = []
        
        for sigma in sigma_values:
            profits = []
            for tau in tau_values:
                key = (tau, sigma)
                if key in stats_by_params and model in stats_by_params[key]:
                    trials = stats_by_params[key][model]
                    profits.extend([t['config_d_profit'] for t in trials])
            
            if profits:
                sigma_means.append(np.mean(profits))
                sigma_stds.append(np.std(profits))
            else:
                sigma_means.append(0)
                sigma_stds.append(0)
        
        offset = (i - len(all_models[:4])/2 + 0.5) * width
        ax2.bar(x_pos + offset, sigma_means, width, yerr=sigma_stds,
               capsize=5, label=model, alpha=0.8)
    
    ax2.set_xlabel('Signal Noise σ', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Intermediary Profit', fontsize=13, fontweight='bold')
    ax2.set_title('Impact of σ', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sigma_labels, fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    param_impact_path = output_dir / 'parameter_impact.png'
    plt.savefig(param_impact_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {param_impact_path}')
    plt.close()
    
    # =========================================================================
    # 图3: 模型性能对比 - 箱线图
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 收集所有模型的利润数据
    model_profits = {model: [] for model in all_models}
    
    for key in stats_by_params:
        for model in all_models:
            if model in stats_by_params[key]:
                trials = stats_by_params[key][model]
                model_profits[model].extend([t['config_d_profit'] for t in trials])
    
    # 准备箱线图数据
    data_to_plot = [model_profits[model] for model in all_models if model_profits[model]]
    labels_to_plot = [model for model in all_models if model_profits[model]]
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                    widths=0.6, showmeans=True)
    
    # 着色
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intermediary Profit (Config D)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Distribution Across All Parameter Combinations', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    boxplot_path = output_dir / 'model_performance_boxplot.png'
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f'[OK] Saved: {boxplot_path}')
    plt.close()
    
    # =========================================================================
    # 统计摘要
    # =========================================================================
    print()
    print('='*80)
    print('Statistical Summary:')
    print('='*80)
    print(f"{'Model':<20} {'Avg Profit':<15} {'Std Profit':<15} {'Min Profit':<15} {'Max Profit':<15}")
    print('-'*80)
    
    for model in all_models:
        if model_profits[model]:
            profits = model_profits[model]
            print(f"{model:<20} {np.mean(profits):<15.4f} {np.std(profits):<15.4f} "
                  f"{np.min(profits):<15.4f} {np.max(profits):<15.4f}")
    
    print('='*80)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_sensitivity_results_c.py <results_path>")
        print("Example: python analyze_sensitivity_results_c.py sensitivity_results/scenario_c/sensitivity_3x3_20240129_123456/summary_all_results.json")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} experiments")
    
    # 创建输出目录
    output_dir = results_path.parent / "analysis_plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    create_visualizations(results, output_dir)
    
    print()
    print('='*80)
    print(f'[SUCCESS] All charts saved to: {output_dir}')
    print('='*80)
    print('Generated Charts:')
    print('  1. sensitivity_profit_heatmaps.png - Profit heatmaps by model')
    print('  2. parameter_impact.png - Parameter impact comparison')
    print('  3. model_performance_boxplot.png - Model performance distribution')
    print('='*80)


if __name__ == "__main__":
    main()
