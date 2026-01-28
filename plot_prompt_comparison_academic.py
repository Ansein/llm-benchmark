"""
学术风格提示词对比可视化

只展示三个系列的趋势图，纵向排列，学术配色
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# ============================================================================
# 数据加载
# ============================================================================

def extract_model_name(filepath):
    """从文件名中提取模型名称"""
    filename = Path(filepath).stem
    match = re.match(r'summary_(.+?)_\d{8}_\d{6}', filename)
    if match:
        return match.group(1)
    return filename.replace('summary_', '')

def classify_model(model_name):
    """根据模型名称分类"""
    lower_name = model_name.lower()
    if 'gpt' in lower_name:
        return 'GPT'
    elif 'deepseek' in lower_name:
        return 'DeepSeek'
    elif 'qwen' in lower_name:
        return 'Qwen'
    else:
        return 'Other'

# 扫描所有summary文件
results_dir = Path("evaluation_results/prompt_experiments_b")
summary_files = list(results_dir.glob("summary_*.json"))

print(f"📂 Loading {len(summary_files)} model result files\n")

# 提取数据（跳过v5）
models_data = {}
model_names = []
raw_versions = ["b.v0", "b.v1", "b.v2", "b.v3", "b.v4", "b.v6"]
display_versions = ["v0", "v1", "v2", "v3", "v4", "v5"]

# 英文标签（带变动说明）
version_labels_en = [
    "v0\nBaseline",
    "v1\n+Market\nParams",
    "v2\n+Param\nExplanation",
    "v3\n+Inference\nExternality",
    "v4\n+Submodularity\n& Compensation",
    "v5\n+Rational\nExpectation"
]

for filepath in summary_files:
    model_name = extract_model_name(filepath)
    model_names.append(model_name)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        share_rates = []
        decision_distances = []
        
        for version in raw_versions:
            version_data = data["versions"].get(version, {})
            share_rates.append(version_data.get("share_rate_mean", 0))
            decision_distances.append(version_data.get("decision_distance_mean", 1))
        
        models_data[model_name] = {
            "share_rates": share_rates,
            "decision_distances": decision_distances,
            "series": classify_model(model_name)
        }
        
        print(f"✓ {model_name}")
        
    except Exception as e:
        print(f"✗ {model_name}: {str(e)}")

print(f"\n✅ Successfully loaded {len(models_data)} models\n")

# 按系列分组
series_models = {}
for model_name, data in models_data.items():
    series = data['series']
    if series not in series_models:
        series_models[series] = []
    series_models[series].append(model_name)

# ============================================================================
# 绘制学术风格的三个系列折线图（纵向排列）
# ============================================================================

# 学术配色方案（更柔和、专业）
academic_colors = {
    'GPT': {
        'colors': ['#6094ce', '#4dbe93', '#f5cc2f', '#4c2e90', '#FF5252'],
        'main': '#6094ce'
    },
    'DeepSeek': {
        'colors': ['#6094ce', '#4dbe93', '#f5cc2f', '#4c2e90', '#FF5252'],
        'main': '#6094ce'
    },
    'Qwen': {
        'colors': ['#6094ce', '#4dbe93', '#f5cc2f', '#4c2e90', '#FF5252'],
        'main': '#6094ce'
    }
}

# 创建图表（纵向排列，3行1列）
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.subplots_adjust(hspace=0.35)

x_positions = np.arange(len(display_versions))

# 为每个系列绘制子图
for idx, (series, models) in enumerate(sorted(series_models.items())):
    ax = axes[idx]
    colors = academic_colors.get(series, {}).get('colors', ['#757575'] * 5)
    
    # 按模型名称排序
    sorted_models = sorted(models)
    
    for i, model in enumerate(sorted_models):
        if model not in models_data:
            continue
        
        data = models_data[model]
        distances = data["decision_distances"]
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 绘制折线（统一使用小圆点标记）
        ax.plot(x_positions, distances,
               marker='o',  # 统一使用圆形标记
               color=color,
               linewidth=2.5,
               markersize=6,  # 更小的标记
               label=model,
               markeredgewidth=1.0,
               markeredgecolor='white',
               alpha=0.85)
    
    # 设置标题和标签
    ax.set_title(f'{series} Series', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Prompt Version', fontsize=12, fontweight='bold')
    ax.set_ylabel('Decision Distance', fontsize=12, fontweight='bold')
    
    # 设置x轴刻度和标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(version_labels_en, fontsize=9, ha='center')
    
    # 设置y轴范围
    ax.set_ylim(-0.05, 1.05)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 添加理论最优线
    ax.axhline(y=0, color='#2E7D32', linestyle='--', linewidth=2, 
              alpha=0.6, label='Perfect Alignment', zorder=0)
    
    # 设置图例
    ax.legend(loc='best', fontsize=9, framealpha=0.95, 
             edgecolor='gray', fancybox=False, shadow=False)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

# 添加总标题
fig.suptitle('Prompt Engineering Performance Across Model Series', 
            fontsize=16, fontweight='bold', y=0.995)

# 保存图表
plt.tight_layout(rect=[0, 0.01, 1, 0.99])
output_path = "evaluation_results/prompt_experiments_b/academic_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Academic style figure saved: {output_path}\n")

# 输出统计信息
print("="*80)
print("Statistical Summary")
print("="*80)

for series in sorted(series_models.keys()):
    print(f"\n{series} Series:")
    models = sorted(series_models[series])
    
    for model in models:
        if model not in models_data:
            continue
        
        data = models_data[model]
        distances = data["decision_distances"]
        
        best_idx = np.argmin(distances)
        best_version = display_versions[best_idx]
        best_distance = distances[best_idx]
        
        improvement = distances[0] - distances[-1]
        
        print(f"  {model:25s} | Best: {best_version} ({best_distance:.3f}) | Δ(v0→v5): {improvement:+.3f}")

print("\n" + "="*80 + "\n")

plt.show()
