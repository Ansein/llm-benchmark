"""
提示词版本对比可视化

对比不同模型在不同提示词版本下的表现
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取四个模型的汇总结果
summary_files = {
    "GPT-5-mini": "evaluation_results/prompt_experiments_b/summary_gpt-5-mini-2025-08-07_20260127_192541.json",
    "Qwen-Plus": "evaluation_results/prompt_experiments_b/summary_qwen-plus_20260127_193707.json",
    "Gemini-3-Flash": "evaluation_results/prompt_experiments_b/summary_gemini-3-flash-preview_20260127_201259.json",
    "DeepSeek-v3.2": "evaluation_results/prompt_experiments_b/summary_deepseek-v3.2_20260127_192052.json"
}

# 提取数据
models_data = {}
versions = ["b.v0", "b.v1", "b.v2", "b.v3", "b.v4", "b.v5", "b.v6"]
version_labels = ["v0\n(基础)", "v1\n(+参数)", "v2\n(+解释)", "v3\n(+外部性)", "v4\n(+完整)", "v5\n(+格式)", "v6\n(+机制)"]

for model_name, filepath in summary_files.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    share_rates = []
    decision_distances = []
    
    for version in versions:
        version_data = data["versions"].get(version, {})
        share_rates.append(version_data.get("share_rate_mean", 0))
        decision_distances.append(version_data.get("decision_distance_mean", 1))
    
    models_data[model_name] = {
        "share_rates": share_rates,
        "decision_distances": decision_distances
    }

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 颜色方案
colors = {
    "GPT-5-mini": "#FF6B6B",
    "Qwen-Plus": "#4ECDC4",
    "Gemini-3-Flash": "#FFD93D",
    "DeepSeek-v3.2": "#6C5CE7"
}

markers = {
    "GPT-5-mini": "o",
    "Qwen-Plus": "s",
    "Gemini-3-Flash": "^",
    "DeepSeek-v3.2": "D"
}

x_positions = np.arange(len(versions))

# 图1: 分享率均值
for model_name, data in models_data.items():
    ax1.plot(x_positions, data["share_rates"], 
             marker=markers[model_name], 
             color=colors[model_name], 
             linewidth=2.5,
             markersize=10,
             label=model_name,
             alpha=0.8)

# 添加理论最优基准线
optimal_share_rate = 0.8  # 16/20
ax1.axhline(y=optimal_share_rate, color='red', linestyle='--', linewidth=2, 
            label='理论最优 (80%)', alpha=0.6)

ax1.set_xlabel('提示词版本', fontsize=14, fontweight='bold')
ax1.set_ylabel('分享率均值', fontsize=14, fontweight='bold')
ax1.set_title('不同模型在各提示词版本下的分享率', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(version_labels, fontsize=11)
ax1.legend(loc='best', fontsize=12, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(-0.05, 1.05)

# 添加数据标签
for model_name, data in models_data.items():
    for i, rate in enumerate(data["share_rates"]):
        if rate > 0:  # 只标注非零值
            ax1.annotate(f'{rate:.0%}', 
                        xy=(i, rate), 
                        xytext=(0, 8), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        alpha=0.7)

# 图2: 决策距离（与理论最优的差异）
for model_name, data in models_data.items():
    ax2.plot(x_positions, data["decision_distances"], 
             marker=markers[model_name], 
             color=colors[model_name], 
             linewidth=2.5,
             markersize=10,
             label=model_name,
             alpha=0.8)

ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, 
            label='完美对齐 (距离=0)', alpha=0.6)

ax2.set_xlabel('提示词版本', fontsize=14, fontweight='bold')
ax2.set_ylabel('决策距离 (1 - Jaccard相似度)', fontsize=14, fontweight='bold')
ax2.set_title('不同模型与理论最优决策的距离', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(version_labels, fontsize=11)
ax2.legend(loc='best', fontsize=12, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(-0.05, 1.05)
ax2.invert_yaxis()  # 距离越小越好，倒置y轴

plt.tight_layout()

# 保存图表
output_path = "evaluation_results/prompt_experiments_b/prompt_versions_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] 图表已保存: {output_path}")

# 生成数据表格
print("\n" + "="*80)
print("[数据] 分享率汇总表")
print("="*80)
print(f"{'版本':<10} {'GPT-5-mini':<15} {'Qwen-Plus':<15} {'Gemini-3-Flash':<18} {'DeepSeek-v3.2':<15}")
print("-"*80)
for i, version in enumerate(versions):
    gpt_rate = models_data["GPT-5-mini"]["share_rates"][i]
    qwen_rate = models_data["Qwen-Plus"]["share_rates"][i]
    gemini_rate = models_data["Gemini-3-Flash"]["share_rates"][i]
    deepseek_rate = models_data["DeepSeek-v3.2"]["share_rates"][i]
    
    print(f"{version:<10} {gpt_rate:<15.1%} {qwen_rate:<15.1%} {gemini_rate:<18.1%} {deepseek_rate:<15.1%}")

print("\n" + "="*80)
print("[分析] 关键洞察")
print("="*80)

# 计算每个模型的最佳版本
for model_name, data in models_data.items():
    distances = data["decision_distances"]
    best_idx = np.argmin(distances)
    best_version = versions[best_idx]
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

print("\n" + "="*80)
print("[参考] 理论最优参考")
print("="*80)
print(f"理论最优分享率: 80.0% (16/20 用户)")
print(f"理论最优分享集合规模: 16")

plt.show()
