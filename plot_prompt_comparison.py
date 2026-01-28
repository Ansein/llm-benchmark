"""
提示词版本对比可视化

自动扫描 evaluation_results/prompt_experiments_b/ 文件夹中的所有模型结果，
按系列设置色系进行可视化对比。

色系设计（增强区分度）：
- GPT系列（gpt-*）：鲜红色系 (#FF4444 - #FFCCCC)
- DeepSeek系列（deepseek-*）：深紫色系 (#6633FF - #AA77FF)
- Qwen系列（qwen-*）：青绿色系 (#00CED1 - #80F0F0)

特别说明：
- ⚠️ 跳过 v5 版本（结构化版本表现不佳）
- 将原 v6（理性预期）作为 v5 显示

使用说明：
1. 运行实验后，脚本会自动扫描所有 summary_*.json 文件
2. 直接运行此脚本即可生成对比图表

版本说明（显示版本）：
- v0 (最简)：仅包含报价和隐私偏好，基本决策框架
- v1 (+参数)：添加市场环境参数（n, ρ, σ², 分布），无详细解释
- v2 (+解释)：添加参数详细解释（ρ和σ²的含义和作用）
- v3 (+外部性)：引入推断外部性概念（基础泄露、边际泄露）
- v4 (+次模性)：添加次模性和补偿逻辑，完整机制说明
- v5 (+理性预期)：理性预期决策框架（原v6）
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 自动扫描所有summary文件
results_dir = Path("evaluation_results/prompt_experiments_b")
summary_files = list(results_dir.glob("summary_*.json"))

print(f"📂 扫描到 {len(summary_files)} 个模型结果文件\n")

# 提取模型名称（从文件名中提取）
def extract_model_name(filepath):
    """从文件名中提取模型名称"""
    filename = Path(filepath).stem
    # summary_<model-name>_<timestamp>
    match = re.match(r'summary_(.+?)_\d{8}_\d{6}', filename)
    if match:
        return match.group(1)
    return filename.replace('summary_', '')

# 按系列分组模型
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

# 生成系列色系（增强区分度）
def generate_color_schemes():
    """为每个系列生成色系"""
    color_schemes = {
        'GPT': {
            'base': [1.0, 0.3, 0.3],  # 鲜红色系
            'colors': ['#FF4444', '#FF6666', '#FF8888', '#FFAAAA', '#FFCCCC']  # 红色渐变
        },
        'DeepSeek': {
            'base': [0.5, 0.3, 1.0],  # 深紫色系
            'colors': ['#6633FF', '#7744FF', '#8855FF', '#9966FF', '#AA77FF']  # 紫色渐变
        },
        'Qwen': {
            'base': [0.2, 0.8, 0.8],  # 青绿色系
            'colors': ['#00CED1', '#20D8D8', '#40E0E0', '#60E8E8', '#80F0F0']  # 青绿渐变
        },
        'Other': {
            'base': [0.5, 0.5, 0.5],  # 灰色系
            'colors': ['#666666', '#777777', '#888888', '#999999', '#AAAAAA']  # 灰色渐变
        }
    }
    return color_schemes

# 为每个系列的模型生成渐变色
def assign_colors_to_models(model_names):
    """为所有模型分配颜色"""
    # 按系列分组
    series_models = {}
    for model_name in model_names:
        series = classify_model(model_name)
        if series not in series_models:
            series_models[series] = []
        series_models[series].append(model_name)
    
    # 生成色系
    color_schemes = generate_color_schemes()
    
    # 为每个模型分配颜色
    colors = {}
    markers = {}
    marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
    
    for series, models in series_models.items():
        n_models = len(models)
        color_list = color_schemes[series]['colors']
        
        # 根据模型数量分配颜色
        if n_models == 1:
            color_indices = [0]
        else:
            # 均匀分布在颜色列表中
            color_indices = [int(i * (len(color_list) - 1) / (n_models - 1)) for i in range(n_models)]
        
        for i, model_name in enumerate(sorted(models)):
            colors[model_name] = color_list[color_indices[i]]
            markers[model_name] = marker_list[i % len(marker_list)]
    
    return colors, markers, series_models

# 提取数据（跳过v5，将v6当作v5显示）
models_data = {}
model_names = []

# 原始版本列表（用于从JSON读取）
raw_versions = ["b.v0", "b.v1", "b.v2", "b.v3", "b.v4", "b.v6"]  # 跳过v5，直接用v6

# 显示用的版本列表（将v6显示为v5）
display_versions = ["b.v0", "b.v1", "b.v2", "b.v3", "b.v4", "b.v5"]
version_labels = [
    "v0\n(最简)",      # 仅报价+隐私偏好
    "v1\n(+参数)",      # 添加市场环境参数
    "v2\n(+解释)",      # 添加参数解释
    "v3\n(+外部性)",    # 引入推断外部性
    "v4\n(+次模性)",    # 添加次模性和补偿逻辑
    "v5\n(+理性预期)"   # 原v6，理性预期决策框架
]

print("⚠️  注意: 跳过 v5 版本，将 v6 (理性预期) 作为 v5 显示\n")

for filepath in summary_files:
    model_name = extract_model_name(filepath)
    model_names.append(model_name)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        share_rates = []
        decision_distances = []
        
        # 读取原始版本数据（跳过v5）
        for version in raw_versions:
            version_data = data["versions"].get(version, {})
            share_rates.append(version_data.get("share_rate_mean", 0))
            decision_distances.append(version_data.get("decision_distance_mean", 1))
        
        models_data[model_name] = {
            "share_rates": share_rates,
            "decision_distances": decision_distances
        }
        
        print(f"✓ 已加载: {model_name}")
        
    except Exception as e:
        print(f"✗ 加载失败 {model_name}: {str(e)}")

print(f"\n✅ 成功加载 {len(models_data)} 个模型的数据\n")

# 分配颜色和标记
colors, markers, series_models = assign_colors_to_models(model_names)

# 打印模型分组信息
print("📊 模型分组：")
for series, models in series_models.items():
    print(f"  {series}: {', '.join(sorted(models))}")
print()

# 创建图表（根据模型数量调整图表大小）
n_models = len(models_data)
n_versions = len(display_versions)
fig_width = max(16, 10 + n_models * 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 8))

x_positions = np.arange(n_versions)

# 图1: 分享率均值（按系列分组绘制，增强视觉区分）
for series, models in sorted(series_models.items()):
    for model_name in sorted(models):
        if model_name in models_data:
            data = models_data[model_name]
            # 添加系列标识到标签
            label_with_series = f"{model_name} ({series})"
            ax1.plot(x_positions, data["share_rates"], 
                     marker=markers[model_name], 
                     color=colors[model_name], 
                     linewidth=2.5,
                     markersize=9,
                     label=label_with_series,
                     alpha=0.9)

# 添加理论最优基准线
optimal_share_rate = 0.8  # 16/20
ax1.axhline(y=optimal_share_rate, color='red', linestyle='--', linewidth=2, 
            label='理论最优 (80%)', alpha=0.6)

ax1.set_xlabel('提示词版本', fontsize=14, fontweight='bold')
ax1.set_ylabel('分享率均值', fontsize=14, fontweight='bold')
ax1.set_title('不同模型在各提示词版本下的分享率', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(version_labels, fontsize=11)

# 根据模型数量调整图例位置和列数
if n_models <= 4:
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
elif n_models <= 8:
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
else:
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fontsize=8, framealpha=0.9, ncol=3)

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(-0.05, 1.05)

# 添加数据标签（仅当模型数量较少时）
if n_models <= 4:
    for model_name, data in models_data.items():
        for i, rate in enumerate(data["share_rates"]):
            if rate > 0:  # 只标注非零值
                ax1.annotate(f'{rate:.0%}', 
                            xy=(i, rate), 
                            xytext=(0, 8), 
                            textcoords='offset points',
                            ha='center',
                            fontsize=8,
                            alpha=0.6,
                            color=colors[model_name])

# 图2: 决策距离（按系列分组绘制，增强视觉区分）
for series, models in sorted(series_models.items()):
    for model_name in sorted(models):
        if model_name in models_data:
            data = models_data[model_name]
            # 添加系列标识到标签
            label_with_series = f"{model_name} ({series})"
            ax2.plot(x_positions, data["decision_distances"], 
                     marker=markers[model_name], 
                     color=colors[model_name], 
                     linewidth=2.5,
                     markersize=9,
                     label=label_with_series,
                     alpha=0.9)

ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, 
            label='完美对齐 (距离=0)', alpha=0.6)

ax2.set_xlabel('提示词版本', fontsize=14, fontweight='bold')
ax2.set_ylabel('决策距离 (1 - Jaccard相似度)', fontsize=14, fontweight='bold')
ax2.set_title('不同模型与理论最优决策的距离', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(version_labels, fontsize=11)

# 根据模型数量调整图例位置和列数
if n_models <= 4:
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
elif n_models <= 8:
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
else:
    ax2.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fontsize=8, framealpha=0.9, ncol=3)

ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(-0.05, 1.05)
ax2.invert_yaxis()  # 距离越小越好，倒置y轴

# 调整布局（为图例留出空间）
if n_models > 8:
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # 底部留出空间给图例
else:
    plt.tight_layout()

# 保存图表
output_path = "evaluation_results/prompt_experiments_b/prompt_versions_comparison_all_models.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*120}")
print(f"[OK] 图表已保存: {output_path}")
print(f"{'='*120}")

# 生成数据表格（按系列分组）
print("\n" + "="*120)
print("[数据] 分享率汇总表")
print("="*120)

for series, models in series_models.items():
    sorted_models = sorted(models)
    if not sorted_models:
        continue
    
    print(f"\n{series} 系列:")
    print("-"*120)
    
    # 动态生成表头
    header = f"{'版本':<12}"
    for model in sorted_models:
        header += f"{model:<20}"
    print(header)
    print("-"*120)
    
    # 输出每个版本的数据
    for i, version in enumerate(display_versions):
        row = f"{version:<12}"
        for model in sorted_models:
            if model in models_data:
                rate = models_data[model]["share_rates"][i]
                row += f"{rate:<20.1%}"
            else:
                row += f"{'N/A':<20}"
        print(row)

print("\n" + "="*120)
print("[分析] 关键洞察（按系列分组）")
print("="*120)

# 按系列分析每个模型
for series, models in series_models.items():
    print(f"\n{'='*60}")
    print(f"{series} 系列")
    print(f"{'='*60}")
    
    for model_name in sorted(models):
        if model_name not in models_data:
            continue
        
        data = models_data[model_name]
        distances = data["decision_distances"]
        best_idx = np.argmin(distances)
        best_version = display_versions[best_idx]
        best_distance = distances[best_idx]
        best_share_rate = data["share_rates"][best_idx]
        
        print(f"\n{model_name}:")
        print(f"  最佳提示词版本: {best_version}")
        print(f"  该版本分享率: {best_share_rate:.1%}")
        print(f"  与理论最优距离: {best_distance:.3f}")
        
        # 版本演进分析
        if distances[-1] < distances[0]:
            improvement = distances[0] - distances[-1]
            print(f"  版本演进效果: 改进 {improvement:.3f} (v0→v5)")
        else:
            degradation = distances[-1] - distances[0]
            print(f"  版本演进效果: 退化 {degradation:.3f} (v0→v5)")

# 找出全局最佳模型
print(f"\n{'='*120}")
print("[全局最佳]")
print(f"{'='*120}")

best_overall_model = None
best_overall_distance = float('inf')
best_overall_version = None

for model_name, data in models_data.items():
    distances = data["decision_distances"]
    best_idx = np.argmin(distances)
    if distances[best_idx] < best_overall_distance:
        best_overall_distance = distances[best_idx]
        best_overall_model = model_name
        best_overall_version = display_versions[best_idx]

if best_overall_model:
    best_idx = display_versions.index(best_overall_version)
    best_rate = models_data[best_overall_model]["share_rates"][best_idx]
    print(f"\n最佳模型: {best_overall_model}")
    print(f"最佳版本: {best_overall_version}")
    print(f"分享率: {best_rate:.1%}")
    print(f"与理论最优距离: {best_overall_distance:.3f}")

print("\n" + "="*120)
print("[参考] 理论最优参考")
print("="*120)
print(f"理论最优分享率: 80.0% (16/20 用户)")
print(f"理论最优分享集合规模: 16")

print("\n" + "="*120)
print("[版本] 提示词复杂度递进（跳过原v5）")
print("="*120)
print("v0 (最简)      → 仅报价+隐私偏好")
print("v1 (+参数)     → 添加市场参数（n,ρ,σ²,分布）")
print("v2 (+解释)     → 添加参数详细解释")
print("v3 (+外部性)   → 引入推断外部性、基础泄露、边际泄露")
print("v4 (+次模性)   → 添加次模性和补偿逻辑")
print("v5 (+理性预期) → 理性预期框架（效用函数、贝叶斯更新）[原v6]")

print("\n" + "="*120)
print(f"🎉 分析完成！共对比 {n_models} 个模型，{n_versions} 个提示词版本")
print("="*120 + "\n")

plt.show()
