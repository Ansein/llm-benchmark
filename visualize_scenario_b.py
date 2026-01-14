"""
场景B可视化脚本
对比LLM决策与理论真值
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
gt_path = Path("data/ground_truth/scenario_b_result.json")
test_path = Path("data/test_results/test_eval_scenario_b.json")

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 提取参数
params = gt_data['params']
n = params['n']
v = params['v']
gt_share_set = set(gt_data['gt_numeric']['eq_share_set'])
llm_share_set = set(test_data['llm_share_set'])

# 创建图表
fig = plt.figure(figsize=(16, 10))

# ===== 图1: 用户隐私偏好与分享决策 =====
ax1 = plt.subplot(2, 3, 1)
users = list(range(n))
colors = ['green' if i in gt_share_set else 'lightblue' for i in users]
bars1 = ax1.bar(users, v, color=colors, alpha=0.7, label='理论均衡')

# 标记LLM决策（用X表示不分享）
for i, user in enumerate(users):
    if user in gt_share_set and user not in llm_share_set:
        ax1.text(i, v[i] + 0.05, '❌', ha='center', fontsize=20, color='red')
    elif user in gt_share_set:
        ax1.text(i, v[i] + 0.05, '✓', ha='center', fontsize=16, color='green')

ax1.axhline(y=np.median(v), color='red', linestyle='--', alpha=0.3, label=f'中位数={np.median(v):.3f}')
ax1.set_xlabel('用户ID', fontsize=12)
ax1.set_ylabel('隐私偏好 v_i', fontsize=12)
ax1.set_title('图1: 隐私偏好分布与分享决策\n(绿色=应该分享, ❌=LLM错误)', fontsize=13)
ax1.set_xticks(users)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# ===== 图2: 关键指标对比 =====
ax2 = plt.subplot(2, 3, 2)
metrics = ['平台利润', '社会福利', '总泄露量']
gt_values = [
    gt_data['gt_numeric']['eq_profit'],
    gt_data['gt_numeric']['eq_W'],
    gt_data['gt_numeric']['eq_total_leakage']
]
llm_values = [
    test_data['metrics']['llm']['profit'],
    test_data['metrics']['llm']['welfare'],
    test_data['metrics']['llm']['total_leakage']
]

x = np.arange(len(metrics))
width = 0.35

bars_gt = ax2.bar(x - width/2, gt_values, width, label='理论真值', color='steelblue', alpha=0.8)
bars_llm = ax2.bar(x + width/2, llm_values, width, label='LLM结果', color='coral', alpha=0.8)

# 添加数值标签
for bars in [bars_gt, bars_llm]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax2.set_ylabel('数值', fontsize=12)
ax2.set_title('图2: 关键指标对比', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# ===== 图3: MAE偏差可视化 =====
ax3 = plt.subplot(2, 3, 3)
mae_metrics = ['利润', '福利', '泄露量', '分享率']
mae_values = [
    test_data['metrics']['deviations']['profit_mae'],
    test_data['metrics']['deviations']['welfare_mae'],
    test_data['metrics']['deviations']['total_leakage_mae'],
    test_data['metrics']['deviations']['share_rate_mae'] * 10  # 缩放以便可视化
]

bars3 = ax3.barh(mae_metrics, mae_values, color=['red', 'orange', 'yellow', 'lightcoral'])
for i, (bar, val) in enumerate(zip(bars3, mae_values)):
    if mae_metrics[i] == '分享率':
        label = f'{val/10:.2%}'
    else:
        label = f'{val:.3f}'
    ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, label, 
            va='center', fontsize=11, fontweight='bold')

ax3.set_xlabel('绝对偏差 (MAE)', fontsize=12)
ax3.set_title('图3: 偏差指标 (MAE vs Ground Truth)', fontsize=13)
ax3.grid(axis='x', alpha=0.3)

# ===== 图4: 信息泄露对比（理论均衡） =====
ax4 = plt.subplot(2, 3, 4)

# 从all_outcomes中提取空集和均衡集合的泄露
empty_leakage = gt_data['all_outcomes']['[]']['leakage']
eq_key = str(sorted(gt_data['gt_numeric']['eq_share_set']))
eq_leakage = gt_data['all_outcomes'][eq_key]['leakage']

x_pos = np.arange(n)
width = 0.35

bars_empty = ax4.bar(x_pos - width/2, empty_leakage, width, 
                     label='空集S={}', color='lightgray', alpha=0.7)
bars_eq = ax4.bar(x_pos + width/2, eq_leakage, width, 
                  label='均衡S={4,5,6}', color='indianred', alpha=0.7)

# 标记分享者
for i in gt_share_set:
    ax4.text(i, eq_leakage[i] + 0.05, '📤', ha='center', fontsize=16)

ax4.set_xlabel('用户ID', fontsize=12)
ax4.set_ylabel('信息泄露量', fontsize=12)
ax4.set_title('图4: 信息泄露对比\n(📤=分享者)', fontsize=13)
ax4.set_xticks(x_pos)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# ===== 图5: 不同分享集合大小的平台利润 =====
ax5 = plt.subplot(2, 3, 5)

# 统计不同大小的分享集合的平台利润
size_profits = {}
for key, outcome in gt_data['all_outcomes'].items():
    size = len(outcome['S'])
    profit = outcome['platform_profit']
    if size not in size_profits:
        size_profits[size] = []
    size_profits[size].append(profit)

sizes = sorted(size_profits.keys())
avg_profits = [np.mean(size_profits[s]) for s in sizes]
max_profits = [np.max(size_profits[s]) for s in sizes]

ax5.plot(sizes, avg_profits, 'o-', label='平均利润', linewidth=2, markersize=8)
ax5.plot(sizes, max_profits, 's--', label='最大利润', linewidth=2, markersize=6, alpha=0.7)

# 标记均衡点
eq_size = len(gt_share_set)
eq_profit = gt_data['gt_numeric']['eq_profit']
ax5.scatter([eq_size], [eq_profit], color='red', s=200, marker='*', 
           zorder=5, label=f'均衡 (size={eq_size})')

ax5.set_xlabel('分享集合大小', fontsize=12)
ax5.set_ylabel('平台利润', fontsize=12)
ax5.set_title('图5: 分享集合大小 vs 平台利润', fontsize=13)
ax5.set_xticks(sizes)
ax5.legend()
ax5.grid(alpha=0.3)

# ===== 图6: 社会福利分解 =====
ax6 = plt.subplot(2, 3, 6)

scenarios = ['LLM均衡\n(S={})', '理论均衡\n(S={4,5,6})', '社会最优\n(S={0,2,3,4,5,6})']

# 提取数据
llm_welfare = test_data['metrics']['llm']['welfare']
eq_welfare = gt_data['gt_numeric']['eq_W']
fb_welfare = gt_data['gt_numeric']['fb_W']

eq_value = gt_data['gt_numeric']['eq_value']
eq_cost = eq_value - eq_welfare

fb_value = gt_data['gt_numeric']['fb_total_leakage']  # alpha=1.0
fb_cost = fb_value - fb_welfare

welfare_data = [
    [0, 0],  # LLM: 价值, 成本
    [eq_value, -eq_cost],  # 理论均衡
    [fb_value, -fb_cost]   # 社会最优
]

x_pos = np.arange(len(scenarios))
width = 0.35

bars_value = ax6.bar(x_pos - width/2, [w[0] for w in welfare_data], width, 
                     label='平台价值', color='steelblue', alpha=0.8)
bars_cost = ax6.bar(x_pos + width/2, [w[1] for w in welfare_data], width, 
                    label='用户成本', color='coral', alpha=0.8)

# 净福利线
net_welfare = [llm_welfare, eq_welfare, fb_welfare]
ax6_twin = ax6.twinx()
ax6_twin.plot(x_pos, net_welfare, 'go-', linewidth=3, markersize=10, 
              label='社会福利', zorder=10)

for i, w in enumerate(net_welfare):
    ax6_twin.text(i, w + 0.1, f'{w:.2f}', ha='center', fontsize=11, 
                 fontweight='bold', color='green')

ax6.set_ylabel('价值/成本', fontsize=12)
ax6_twin.set_ylabel('社会福利', fontsize=12, color='green')
ax6.set_title('图6: 社会福利分解', fontsize=13)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(scenarios, fontsize=10)
ax6.legend(loc='upper left')
ax6_twin.legend(loc='upper right')
ax6.grid(axis='y', alpha=0.3)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('scenario_b_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存到: scenario_b_analysis.png")
plt.show()
