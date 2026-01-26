"""
汇总场景A的实验结果
从evaluation_results/scenario_a/目录读取所有JSON文件，生成直观的CSV表格和可视化图表
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_model_name(filename: str) -> str:
    """从文件名提取模型名称"""
    if 'rational' in filename:
        return 'rational'
    
    # 匹配模型名称模式: eval_A_full_{model}_{timestamp}.json
    pattern = r'eval_A_full_(.+?)_\d{8}_\d{6}\.json'
    match = re.search(pattern, filename)
    if match:
        model = match.group(1)
        # 移除 rational_share_price_search 后缀
        model = re.sub(r'_rational_share_price_search$', '', model)
        return model
    
    return 'unknown'

def parse_all_results(result_dir: Path) -> List[Dict]:
    """解析所有结果文件"""
    all_data = []
    
    # 按模型分组
    model_files = {}
    for json_file in result_dir.glob("eval_A_full_*.json"):
        model = extract_model_name(json_file.name)
        if model not in model_files:
            model_files[model] = []
        model_files[model].append(json_file)
    
    print(f"找到 {len(model_files)} 个模型:")
    for model in sorted(model_files.keys()):
        print(f"  - {model}: {len(model_files[model])} 个文件")
    
    # 读取每个模型的结果
    for model, files in sorted(model_files.items()):
        # 按firm_num排序（从文件内容中提取）
        file_data = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    params = data.get('params', {})
                    n_firms = params.get('n_firms', 0)
                    file_data.append((n_firms, f, data))
            except Exception as e:
                print(f"  [WARNING] Failed to read {f.name}: {e}")
        
        # 按firm_num排序
        file_data.sort(key=lambda x: x[0])
        
        # 提取数据
        for n_firms, filepath, data in file_data:
            try:
                # 获取第一轮的结果（假设是单轮实验）
                if 'all_rounds' in data and len(data['all_rounds']) > 0:
                    round_data = data['all_rounds'][0]
                    
                    all_data.append({
                        'model': model,
                        'firm_num': n_firms,
                        'share_rate': round_data.get('share_rate', 0),
                        'avg_price': round_data.get('avg_price', 0),
                        'consumer_surplus': round_data.get('consumer_surplus', 0),
                        'firm_profit': round_data.get('firm_profit', 0),
                        'social_welfare': round_data.get('social_welfare', 0),
                        'avg_search_cost': round_data.get('avg_search_cost', 0),
                        'purchase_rate': round_data.get('purchase_rate', 0)
                    })
                    print(f"  [OK] {model} | firm={n_firms} | share={round_data.get('share_rate', 0):.2%}")
            except Exception as e:
                print(f"  [ERROR] Failed to process {filepath.name}: {e}")
    
    return all_data

def create_summary_tables(df: pd.DataFrame, output_dir: Path):
    """生成多种格式的汇总表"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 长格式 - 原始数据
    long_path = output_dir / f"summary_long_{timestamp}.csv"
    df.to_csv(long_path, index=False)
    print(f"\n[SAVED] Long format: {long_path}")
    
    # 2. 宽格式 - 每个模型一行，按指标展开
    wide_data = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model].sort_values('firm_num')
        row = {'model': model}
        
        for firm_num in sorted(df['firm_num'].unique()):
            firm_data = model_df[model_df['firm_num'] == firm_num]
            if len(firm_data) > 0:
                for metric in ['share_rate', 'avg_price', 'consumer_surplus', 
                              'firm_profit', 'social_welfare', 'avg_search_cost']:
                    col_name = f"{metric}_{firm_num}"
                    row[col_name] = firm_data.iloc[0][metric]
        
        wide_data.append(row)
    
    df_wide = pd.DataFrame(wide_data)
    wide_path = output_dir / f"summary_wide_{timestamp}.csv"
    df_wide.to_csv(wide_path, index=False)
    print(f"[SAVED] Wide format: {wide_path}")
    
    # 3. 按指标分组的透视表
    metrics = ['share_rate', 'avg_price', 'consumer_surplus', 'firm_profit', 
               'social_welfare', 'avg_search_cost']
    
    pivot_path = output_dir / f"summary_by_metric_{timestamp}.csv"
    with open(pivot_path, 'w', encoding='utf-8') as f:
        for i, metric in enumerate(metrics):
            if i > 0:
                f.write('\n')  # 指标间空行
            
            # 创建该指标的透视表
            pivot = df.pivot_table(
                index='model',
                columns='firm_num',
                values=metric,
                aggfunc='first'
            )
            
            # 写入指标名
            f.write(f"{metric}\n")
            pivot.to_csv(f, lineterminator='\n')
    
    print(f"[SAVED] By metric: {pivot_path}")
    
    # 4. 模型对比表（平均值）
    comparison = df.groupby('model').agg({
        'share_rate': 'mean',
        'avg_price': 'mean',
        'consumer_surplus': 'mean',
        'firm_profit': 'mean',
        'social_welfare': 'mean',
        'avg_search_cost': 'mean',
        'purchase_rate': 'mean'
    }).round(4)
    
    comparison_path = output_dir / f"summary_model_comparison_{timestamp}.csv"
    comparison.to_csv(comparison_path)
    print(f"[SAVED] Model comparison: {comparison_path}")
    
    print(f"\n[PREVIEW] Model average metrics:")
    print(comparison)

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """生成可视化图表"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = output_dir / f"visualizations_{timestamp}"
    vis_dir.mkdir(exist_ok=True)
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 定义模型颜色（一致性）
    model_colors = {
        'rational': '#2E86AB',      # 蓝色 - 理性基准
        'deepseek-v3.2': '#A23B72', # 紫红 - DeepSeek
        'gpt-5-mini-2025-08-07': '#F18F01', # 橙色 - GPT-5
        'qwen-plus': '#C73E1D',     # 红色 - Qwen
        'gemini-3-flash-preview': '#6A994E' # 绿色 - Gemini
    }
    
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")
    
    # 1. 数据共享率随企业数变化（折线图）
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('firm_num')
        ax.plot(model_data['firm_num'], model_data['share_rate'], 
                marker='o', linewidth=2.5, markersize=8, 
                label=model, color=model_colors.get(model, None), alpha=0.8)
    
    ax.set_xlabel('Number of Firms', fontsize=13, fontweight='bold')
    ax.set_ylabel('Data Sharing Rate', fontsize=13, fontweight='bold')
    ax.set_title('Data Sharing Strategies: LLM vs Rational', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(vis_dir / '1_share_rate_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 1_share_rate_trend.png")
    
    # 2. 定价策略对比（折线图，突出理性vs LLM差异）
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('firm_num')
        linewidth = 3.5 if model == 'rational' else 2.5
        linestyle = '-' if model == 'rational' else '--'
        ax.plot(model_data['firm_num'], model_data['avg_price'], 
                marker='o', linewidth=linewidth, markersize=8, 
                linestyle=linestyle, label=model, 
                color=model_colors.get(model, None), alpha=0.8)
    
    ax.set_xlabel('Number of Firms', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Price', fontsize=13, fontweight='bold')
    ax.set_title('Pricing Strategies: Rational (Solid) vs LLM (Dashed)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(vis_dir / '2_pricing_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 2_pricing_strategy.png")
    
    # 3. 经济福利分配（堆叠柱状图，按模型）
    avg_by_model = df.groupby('model').agg({
        'consumer_surplus': 'mean',
        'firm_profit': 'mean'
    }).round(2)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(avg_by_model))
    width = 0.6
    
    colors_cs = [model_colors.get(m, '#cccccc') for m in avg_by_model.index]
    colors_fp = [plt.cm.colors.to_rgba(model_colors.get(m, '#cccccc'), alpha=0.6) 
                 for m in avg_by_model.index]
    
    p1 = ax.bar(x, avg_by_model['consumer_surplus'], width, 
                label='Consumer Surplus', color=colors_cs)
    p2 = ax.bar(x, avg_by_model['firm_profit'], width, 
                bottom=avg_by_model['consumer_surplus'],
                label='Firm Profit', color=colors_fp)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Economic Welfare', fontsize=13, fontweight='bold')
    ax.set_title('Economic Welfare Distribution: Consumer vs Firm', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(avg_by_model.index, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加总福利数值标签
    for i, (idx, row) in enumerate(avg_by_model.iterrows()):
        total = row['consumer_surplus'] + row['firm_profit']
        ax.text(i, total + 0.2, f'{total:.2f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / '3_welfare_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 3_welfare_distribution.png")
    
    # 4. 社会福利随企业数变化
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('firm_num')
        ax.plot(model_data['firm_num'], model_data['social_welfare'], 
                marker='o', linewidth=2.5, markersize=8, 
                label=model, color=model_colors.get(model, None), alpha=0.8)
    
    ax.set_xlabel('Number of Firms', fontsize=13, fontweight='bold')
    ax.set_ylabel('Social Welfare', fontsize=13, fontweight='bold')
    ax.set_title('Social Welfare Trends Across Market Structures', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(vis_dir / '4_social_welfare_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 4_social_welfare_trend.png")
    
    # 5. 模型综合对比（雷达图）
    metrics_radar = ['share_rate', 'purchase_rate', 'social_welfare']
    
    # 归一化数据到0-1
    df_normalized = df.copy()
    for metric in metrics_radar:
        max_val = df_normalized[metric].max()
        min_val = df_normalized[metric].min()
        if max_val > min_val:
            df_normalized[f'{metric}_norm'] = (df_normalized[metric] - min_val) / (max_val - min_val)
        else:
            df_normalized[f'{metric}_norm'] = 0.5
    
    # 添加归一化的企业利润（越高越好）和搜索成本（越低越好，需要反转）
    max_profit = df['firm_profit'].max()
    min_profit = df['firm_profit'].min()
    df_normalized['firm_profit_norm'] = (df['firm_profit'] - min_profit) / (max_profit - min_profit) if max_profit > min_profit else 0.5
    
    max_cost = df['avg_search_cost'].max()
    df_normalized['search_efficiency_norm'] = 1 - (df['avg_search_cost'] / max_cost) if max_cost > 0 else 1.0
    
    avg_normalized = df_normalized.groupby('model').agg({
        'share_rate_norm': 'mean',
        'purchase_rate_norm': 'mean',
        'social_welfare_norm': 'mean',
        'firm_profit_norm': 'mean',
        'search_efficiency_norm': 'mean'
    })
    
    categories = ['Share Rate', 'Purchase Rate', 'Social Welfare', 'Firm Profit', 'Search Efficiency']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model in avg_normalized.index:
        values = avg_normalized.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, 
                label=model, color=model_colors.get(model, None), markersize=6)
        ax.fill(angles, values, alpha=0.15, color=model_colors.get(model, None))
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Multi-dimensional Model Comparison (Normalized)', 
                 fontsize=15, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(vis_dir / '5_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 5_radar_comparison.png")
    
    # 6. 搜索成本对比（柱状图）
    avg_search = df.groupby('model')['avg_search_cost'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = [model_colors.get(m, '#cccccc') for m in avg_search.index]
    bars = ax.bar(range(len(avg_search)), avg_search.values, color=colors_bar, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Search Cost', fontsize=13, fontweight='bold')
    ax.set_title('Recommendation Efficiency: Lower is Better', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(avg_search)))
    ax.set_xticklabels(avg_search.index, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(avg_search.values):
        ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / '6_search_cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 6_search_cost_comparison.png")
    
    # 7. 热力图：不同企业数下的关键指标
    # 先去重，每个model-firm_num组合只保留第一条记录
    df_unique = df.drop_duplicates(subset=['model', 'firm_num'], keep='first')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_heatmap = [
        ('share_rate', 'Data Sharing Rate', axes[0, 0]),
        ('avg_price', 'Average Price', axes[0, 1]),
        ('consumer_surplus', 'Consumer Surplus', axes[1, 0]),
        ('firm_profit', 'Firm Profit', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_heatmap:
        pivot = df_unique.pivot(index='model', columns='firm_num', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                    cbar_kws={'label': title}, ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Number of Firms', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.suptitle('Heatmap: Key Metrics Across Market Structures', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(vis_dir / '7_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 7_metrics_heatmap.png")
    
    # 8. 理性vs LLM平均差异（误差条形图）
    llm_models = [m for m in df_unique['model'].unique() if m != 'rational']
    rational_avg = df_unique[df_unique['model'] == 'rational'].groupby('firm_num').mean(numeric_only=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics_diff = [
        ('share_rate', 'Share Rate Gap', axes[0, 0]),
        ('avg_price', 'Pricing Gap', axes[0, 1]),
        ('firm_profit', 'Firm Profit Gap', axes[1, 0]),
        ('consumer_surplus', 'Consumer Surplus Gap', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_diff:
        for model in llm_models:
            model_data = df_unique[df_unique['model'] == model].sort_values('firm_num')
            rational_values = rational_avg.loc[model_data['firm_num'], metric].values
            diff = model_data[metric].values - rational_values
            ax.plot(model_data['firm_num'], diff, marker='o', linewidth=2.5, 
                    markersize=7, label=model, color=model_colors.get(model, None), alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Rational Baseline')
        ax.set_xlabel('Number of Firms', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{title} (LLM - Rational)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))
    
    plt.suptitle('LLM Deviation from Rational Baseline', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(vis_dir / '8_rational_llm_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] 8_rational_llm_gap.png")
    
    print(f"\n[SUCCESS] All visualizations saved to: {vis_dir}")
    return vis_dir

def main():
    project_root = Path(__file__).parent.parent
    result_dir = project_root / "evaluation_results" / "scenario_a"
    
    if not result_dir.exists():
        print(f"[ERROR] Result directory not found: {result_dir}")
        return
    
    print(f"{'='*60}")
    print("Scenario A Results Summary")
    print(f"{'='*60}")
    print(f"Result directory: {result_dir}\n")
    
    # 解析所有结果
    all_data = parse_all_results(result_dir)
    
    if not all_data:
        print("\n[ERROR] No valid result data found")
        return
    
    print(f"\nTotal: {len(all_data)} records")
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 生成汇总表
    create_summary_tables(df, result_dir)
    
    # 生成可视化图表
    vis_dir = create_visualizations(df, result_dir)
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Summary complete!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  - CSV tables in: {result_dir}")
    print(f"  - Visualizations in: {vis_dir}")

if __name__ == "__main__":
    main()
