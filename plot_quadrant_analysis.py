"""
绘制四象限分析图：Jaccard相似度 vs EAS
展示模型在水平准确度（Level Accuracy）和机制理解（Mechanism Understanding）两个维度的表现
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

def extract_model_metrics(sensitivity_file: str) -> Dict:
    """
    从敏感性分析结果中提取欧氏距离和其他指标
    """
    with open(sensitivity_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算欧氏距离：基于多个维度（share_rate, welfare, profit）
    euclidean_distances = []
    for d in data:
        if 'metrics' in d and 'deviations' in d['metrics']:
            deviations = d['metrics']['deviations']
            
            # 使用多个指标计算欧氏距离
            share_rate_error = deviations.get('share_rate_mae', 0)
            welfare_error = deviations.get('welfare_mae', 0)
            profit_error = deviations.get('profit_mae', 0)
            
            # 计算欧氏距离（归一化）
            # 由于welfare和profit的量纲可能很大，我们使用相对误差
            llm = d['metrics'].get('llm', {})
            gt = d['metrics'].get('ground_truth', {})
            
            # 计算相对误差
            welfare_rel_error = abs(llm.get('welfare', 0) - gt.get('welfare', 0)) / max(abs(gt.get('welfare', 1)), 1e-6)
            profit_rel_error = abs(llm.get('profit', 0) - gt.get('profit', 0)) / max(abs(gt.get('profit', 1)), 1e-6)
            
            # 欧氏距离
            euclidean_dist = np.sqrt(share_rate_error**2 + welfare_rel_error**2 + profit_rel_error**2)
            euclidean_distances.append(euclidean_dist)
    
    avg_euclidean = np.mean(euclidean_distances) if euclidean_distances else None
    
    # 提取模型名称
    model_name = None
    if len(data) > 0:
        model_name = data[0].get('model_name', 'Unknown')
    
    return {
        'model_name': model_name,
        'avg_euclidean': avg_euclidean,
        'euclidean_distances': euclidean_distances
    }

def load_eas_results(eas_files: List[str]) -> Dict[str, float]:
    """
    加载EAS分析结果
    """
    model_eas = {}
    
    for eas_file in eas_files:
        try:
            with open(eas_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 优先从JSON中读取model_name
            model_name = data.get('model_name', None)
            
            # 如果没有，从data_file路径中提取
            if not model_name:
                data_file = data.get('data_file', '')
                if 'gpt-5.2' in data_file:
                    model_name = 'gpt-5.2'
                elif 'deepseek-v3.2' in data_file:
                    model_name = 'deepseek-v3.2'
                elif 'qwen3-max' in data_file:
                    model_name = 'qwen3-max-2026-01-23'
                else:
                    # 从文件名中提取
                    filename = Path(eas_file).stem  # 'eas_analysis_modelname_timestamp'
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # 提取模型名称部分（去掉'eas_analysis_'前缀和时间戳后缀）
                        model_name = '_'.join(parts[2:-1])
                    else:
                        continue
            
            overall_eas = data.get('overall_eas', None)
            if overall_eas is not None and model_name:
                model_eas[model_name] = overall_eas
        except Exception as e:
            print(f"Warning: Failed to load {eas_file}: {e}")
    
    return model_eas

def plot_quadrant_analysis(
    model_data: List[Dict],
    output_path: str = 'evaluation_results/eas_analysis/quadrant_analysis.png'
):
    """
    绘制四象限分析图
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 提取数据
    models = []
    euclidean_values = []
    eas_values = []
    
    for data in model_data:
        if data['avg_euclidean'] is not None and data['eas'] is not None:
            models.append(data['model_name'])
            euclidean_values.append(data['avg_euclidean'])
            eas_values.append(data['eas'])
    
    if not models:
        print("Error: No valid model data found!")
        return
    
    # 计算中位数（分界线）
    median_euclidean = np.median(euclidean_values)
    median_eas = np.median(eas_values)
    
    print(f"\n{'='*60}")
    print(f"四象限分析")
    print(f"{'='*60}")
    print(f"中位数分界线:")
    print(f"  欧氏距离中位数: {median_euclidean:.3f}")
    print(f"  EAS中位数: {median_eas:.3f}")
    print(f"\n模型数据:")
    for i, model in enumerate(models):
        print(f"  {model}: Euclidean Distance={euclidean_values[i]:.3f}, EAS={eas_values[i]:.3f}")
    
    # 定义模型家族和颜色
    model_families = {
        'GPT Family': {
            'models': ['gpt-5', 'gpt-5.1', 'gpt-5.2', 'gpt-5.1-2025-11-13'],
            'color': '#FF8C42',
            'marker': 'o'
        },
        'DeepSeek Family': {
            'models': ['deepseek-v3', 'deepseek-v3.1', 'deepseek-v3.2', 'deepseek-v3-0324', 'deepseek-r1'],
            'color': '#2E86AB',
            'marker': 'o'
        },
        'Qwen Family': {
            'models': ['qwen-plus', 'qwen-3-max', 'qwen3-max', 'qwen-3-max-2026-01-23', 'qwen-plus-2025-12-01'],
            'color': '#A23B72',
            'marker': 'o'
        }
    }
    
    # 为每个模型分配家族
    def get_model_family(model_name):
        for family, info in model_families.items():
            for model_pattern in info['models']:
                if model_pattern in model_name.lower():
                    return family
        return 'Other'
    
    # 绘制散点 - 每个模型单独标注以在图例中显示
    for i, model in enumerate(models):
        family = get_model_family(model)
        if family != 'Other':
            family_info = model_families[family]
            color = family_info['color']
            marker = family_info['marker']
        else:
            color = '#95A3A4'
            marker = 'o'
        
        # 绘制点，每个模型作为单独的label显示在图例中
        ax.scatter(euclidean_values[i], eas_values[i], 
                  s=250, color=color, marker=marker, 
                  alpha=0.8, edgecolors='black', linewidth=1.5,
                  label=model)  # 每个模型单独显示在图例
    
    # 设置坐标轴范围（留出余量确保所有点都可见）
    x_range = max(euclidean_values) - min(euclidean_values)
    y_range = max(eas_values) - min(eas_values)
    ax.set_xlim(min(euclidean_values) - 0.1 * x_range, max(euclidean_values) + 0.1 * x_range)
    ax.set_ylim(min(eas_values) - 0.1 * y_range, max(eas_values) + 0.15 * y_range)
    
    # 绘制分界线
    ax.axvline(median_euclidean, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(median_eas, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 添加象限标签
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 计算象限标签位置
    x_left = xlim[0] + (median_euclidean - xlim[0]) * 0.1
    x_right = median_euclidean + (xlim[1] - median_euclidean) * 0.9
    y_top = ylim[1] - (ylim[1] - median_eas) * 0.05
    y_bottom = ylim[0] + (median_eas - ylim[0]) * 0.05
    
    # 象限标签（注意：欧氏距离是越小越好，所以左右象限含义相反）
    ax.text(x_right, y_top, 'Potential Models\n(Low Accuracy,\nHigh Mechanism)', 
           ha='right', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', alpha=0.7, edgecolor='blue'))
    
    ax.text(x_left, y_top, 'Strong Models\n(High Accuracy,\nHigh Mechanism)', 
           ha='left', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F4E6', alpha=0.7, edgecolor='green'))
    
    ax.text(x_right, y_bottom, 'Poor Models\n(Low Accuracy,\nLow Mechanism)', 
           ha='right', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8E8E8', alpha=0.7, edgecolor='brown'))
    
    ax.text(x_left, y_bottom, 'Surface Fitting\n(High Accuracy,\nLow Mechanism)', 
           ha='left', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E0', alpha=0.7, edgecolor='orange'))
    
    # 设置标题和标签
    ax.set_xlabel('Average Euclidean Distance from Rational (Level Accuracy)\n← Lower is Better', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Elasticity Alignment Score (Mechanism Understanding)\n→ Higher is Better', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Strategic Positioning: Level Accuracy vs Mechanism Understanding', 
                fontsize=15, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # 添加图例（放在图外面右侧，显示每个模型）
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             fontsize=10, title='Models', title_fontsize=11,
             frameon=True, fancybox=True, shadow=True)
    
    # 添加中位数标注
    ax.text(median_euclidean, ylim[0] + (median_eas - ylim[0]) * 0.02, 
           f'Median={median_euclidean:.3f}', 
           ha='center', va='bottom', fontsize=9, 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.text(xlim[0] + (median_euclidean - xlim[0]) * 0.02, median_eas, 
           f'Median={median_eas:.3f}', 
           ha='left', va='center', fontsize=9, rotation=90,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 调整布局，为右侧图例留出空间
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n四象限分析图已保存: {output_path}")
    plt.close()

def main():
    print("开始收集模型数据...")
    
    # 敏感性分析结果文件
    sensitivity_files = [
        'sensitivity_results/scenario_b/summary_all_results_gpt-5.2.json',
        'sensitivity_results/scenario_b/summary_all_results_deepseek-v3.2.json',
        'sensitivity_results/scenario_b/summary_all_results_qwen3-max-2026-01-23.json',
    ]
    
    # 加载所有EAS结果
    eas_dir = Path('evaluation_results/eas_analysis')
    eas_files = list(eas_dir.glob('eas_analysis_*.json'))
    model_eas = load_eas_results([str(f) for f in eas_files])
    
    print(f"\n找到 {len(model_eas)} 个模型的EAS数据:")
    for model, eas in model_eas.items():
        print(f"  {model}: EAS={eas:.4f}")
    
    # 收集所有模型数据
    model_data = []
    
    for sensitivity_file in sensitivity_files:
        if not Path(sensitivity_file).exists():
            print(f"Warning: {sensitivity_file} not found, skipping...")
            continue
        
        # 提取Jaccard相似度
        metrics = extract_model_metrics(sensitivity_file)
        model_name = metrics['model_name']
        
        # 匹配EAS数据
        eas = model_eas.get(model_name, None)
        
        if eas is not None and metrics['avg_euclidean'] is not None:
            model_data.append({
                'model_name': model_name,
                'avg_euclidean': metrics['avg_euclidean'],
                'eas': eas
            })
            print(f"\n[OK] {model_name}")
            print(f"    Euclidean Distance: {metrics['avg_euclidean']:.4f}")
            print(f"    EAS: {eas:.4f}")
        else:
            print(f"\n[SKIP] {model_name} - Data incomplete")
    
    if not model_data:
        print("\nError: No valid model data collected!")
        return
    
    # 绘制四象限图
    plot_quadrant_analysis(model_data)

if __name__ == '__main__':
    main()
