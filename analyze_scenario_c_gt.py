"""
场景C Ground Truth 理论解分析脚本

功能:
1. 单个GT的详细分析
2. 多个GT的对比分析（2×2矩阵）
3. 补偿扫描曲线分析
4. 理论验证检查
5. 可视化图表生成

用法:
    python analyze_scenario_c_gt.py --mode single --file scenario_c_result.json
    python analyze_scenario_c_gt.py --mode compare --pattern "scenario_c_common_*.json"
    python analyze_scenario_c_gt.py --mode sweep --file scenario_c_payment_sweep.json
    python analyze_scenario_c_gt.py --mode all
"""

import sys
import io
# 修复Windows控制台编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns

# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 数据路径
DATA_DIR = Path("data/ground_truth")
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)


class GTAnalyzer:
    """Ground Truth分析器"""
    
    def __init__(self, gt_path: str):
        """
        Args:
            gt_path: GT文件路径
        """
        self.path = Path(gt_path)
        with open(self.path, 'r', encoding='utf-8') as f:
            self.gt = json.load(f)
        
        self.params = self.gt.get("params", {})
        self.r_star = self.gt.get("rational_participation_rate", 0)
        self.expected = self.gt.get("expected_outcome", {})
        self.sample = self.gt.get("sample_outcome", {})
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n" + "="*80)
        print(f"📊 Ground Truth 分析: {self.path.name}")
        print("="*80)
        
        # 参数配置
        print("\n【参数配置】")
        print(f"  消费者数量 N: {self.params['N']}")
        print(f"  数据结构: {self.params['data_structure']}")
        print(f"  匿名化策略: {self.params['anonymization']}")
        print(f"  补偿 m: {self.params['m']:.2f}")
        print(f"  噪声水平 σ: {self.params['sigma']:.2f}")
        print(f"  时序模式: {self.params.get('participation_timing', 'N/A')}")
        print(f"  异质性分布: {self.params.get('tau_dist', 'N/A')}")
        
        # 理论指标
        print("\n【理论指标】（固定点均衡）")
        print(f"  理性参与率 r*: {self.r_star:.4f} ({self.r_star*100:.2f}%)")
        
        if "r_history" in self.gt:
            r_hist = self.gt["r_history"]
            print(f"  收敛迭代次数: {len(r_hist)-1}")
            if len(r_hist) > 1:
                convergence = abs(r_hist[-1] - r_hist[-2])
                print(f"  最后一步变化: {convergence:.6f}")
                status = "✅ 已收敛" if convergence < 1e-3 else "⚠️ 未完全收敛"
                print(f"  收敛状态: {status}")
        
        # 期望福利指标
        print("\n【期望福利指标】（MC平均，理论基准）")
        print(f"  期望参与率（实现）: {self.expected.get('participation_rate_realized', 0):.4f}")
        print(f"  消费者剩余 CS: {self.expected.get('consumer_surplus', 0):.2f}")
        print(f"  生产者利润 PS: {self.expected.get('producer_profit', 0):.2f}")
        print(f"  中介利润 IS: {self.expected.get('intermediary_profit', 0):.2f}")
        print(f"  社会福利 SW: {self.expected.get('social_welfare', 0):.2f}")
        
        # 验证福利加总
        cs = self.expected.get('consumer_surplus', 0)
        ps = self.expected.get('producer_profit', 0)
        is_profit = self.expected.get('intermediary_profit', 0)
        sw = self.expected.get('social_welfare', 0)
        sw_check = cs + ps + is_profit
        diff = abs(sw - sw_check)
        status = "✅ 正确" if diff < 0.01 else f"❌ 误差={diff:.4f}"
        print(f"  福利加总验证 (CS+PS+IS=SW): {status}")
        
        # 不平等指标
        print("\n【不平等指标】")
        gini = self.expected.get('gini_coefficient', 0)
        print(f"  基尼系数 Gini: {gini:.4f}")
        if gini < 0.05:
            print(f"    → 低不平等（良好）")
        elif gini < 0.15:
            print(f"    → 中等不平等")
        else:
            print(f"    → 高不平等")
        
        pdi = self.expected.get('price_discrimination_index', 0)
        print(f"  价格歧视指数 PDI: {pdi:.6f}")
        
        # 验证匿名化机制
        if self.params['anonymization'] == 'anonymized':
            status = "✅ 正确" if pdi < 0.01 else f"❌ 应为0"
            print(f"    → 匿名化验证: {status}")
        
        # 示例结果对比
        if self.sample:
            print("\n【示例结果】（单次抽样）")
            sample_rate = self.sample.get('participation_rate', 0)
            print(f"  实际参与率: {sample_rate:.2%} ({self.sample.get('num_participants', 0)}/{self.params['N']})")
            
            deviation = abs(sample_rate - self.r_star)
            rel_dev = deviation / self.r_star if self.r_star > 0 else 0
            print(f"  与r*偏差: {deviation:.2%} (相对 {rel_dev*100:.1f}%)")
            
            if rel_dev < 0.1:
                print(f"    → ✅ 接近理论值")
            elif rel_dev < 0.3:
                print(f"    → ⚠️ 有一定偏差（N小时正常）")
            else:
                print(f"    → ❌ 偏差较大")
            
            # 参与者vs拒绝者
            acceptor_util = self.sample.get('acceptor_avg_utility', 0)
            rejecter_util = self.sample.get('rejecter_avg_utility', 0)
            if acceptor_util > 0 and rejecter_util > 0:
                print(f"\n  参与者平均效用: {acceptor_util:.3f}")
                print(f"  拒绝者平均效用: {rejecter_util:.3f}")
                diff = acceptor_util - rejecter_util
                print(f"  差值: {diff:+.3f}")
                if diff > 0:
                    print(f"    → 参与有利（补偿足够）")
                else:
                    print(f"    → 拒绝者搭便车成功（补偿不足）")
        
        print("\n" + "="*80)
    
    def check_quality(self) -> Dict[str, bool]:
        """检查GT质量"""
        checks = {}
        
        # 1. 收敛性
        if "r_history" in self.gt:
            r_hist = self.gt["r_history"]
            convergence = abs(r_hist[-1] - r_hist[-2]) if len(r_hist) > 1 else 0
            checks["converged"] = convergence < 1e-3
            checks["iterations_ok"] = len(r_hist) < 50
        else:
            checks["converged"] = False
            checks["iterations_ok"] = False
        
        # 2. r*合理性
        checks["r_star_interior"] = 0.05 < self.r_star < 0.95
        
        # 3. 福利加总
        cs = self.expected.get('consumer_surplus', 0)
        ps = self.expected.get('producer_profit', 0)
        is_profit = self.expected.get('intermediary_profit', 0)
        sw = self.expected.get('social_welfare', 0)
        checks["welfare_sum"] = abs(sw - (cs + ps + is_profit)) < 0.01
        
        # 4. 价格歧视验证
        pdi = self.expected.get('price_discrimination_index', 0)
        if self.params['anonymization'] == 'anonymized':
            checks["anonymization_correct"] = pdi < 0.01
        else:
            checks["anonymization_correct"] = True
        
        # 5. Gini系数合理性
        gini = self.expected.get('gini_coefficient', 0)
        checks["gini_reasonable"] = 0 <= gini <= 0.3
        
        # 6. sample vs expected一致性
        if self.sample:
            sample_rate = self.sample.get('participation_rate', 0)
            deviation = abs(sample_rate - self.r_star) / self.r_star if self.r_star > 0 else 1
            checks["sample_consistent"] = deviation < 0.5
        else:
            checks["sample_consistent"] = False
        
        return checks
    
    def plot_convergence(self, save_path: str = None):
        """绘制收敛曲线"""
        if "r_history" not in self.gt:
            print("无收敛历史数据")
            return
        
        r_hist = self.gt["r_history"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = list(range(len(r_hist)))
        ax.plot(iterations, r_hist, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=self.r_star, color='r', linestyle='--', 
                   label=f'Final r* = {self.r_star:.4f}')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Participation Rate', fontsize=12)
        ax.set_title(f'Fixed Point Convergence\n{self.path.name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 收敛曲线已保存: {save_path}")
        else:
            plt.savefig(OUTPUT_DIR / f"{self.path.stem}_convergence.png", dpi=300)
            print(f"✅ 收敛曲线已保存: {OUTPUT_DIR / f'{self.path.stem}_convergence.png'}")
        
        plt.close()


class GTComparator:
    """多个GT的对比分析"""
    
    def __init__(self, gt_files: List[str]):
        """
        Args:
            gt_files: GT文件路径列表
        """
        self.analyzers = [GTAnalyzer(f) for f in gt_files]
        self.configs = [a.path.stem for a in self.analyzers]
    
    def compare_summary(self):
        """对比摘要"""
        print("\n" + "="*80)
        print("📊 Ground Truth 对比分析")
        print("="*80)
        
        # 构建对比表
        print(f"\n{'配置':<40} | {'r*':>7} | {'CS':>8} | {'PS':>8} | {'SW':>8} | {'Gini':>6} | {'PDI':>8}")
        print("-"*100)
        
        for analyzer in self.analyzers:
            config_name = analyzer.path.stem.replace("scenario_c_", "")
            r = analyzer.r_star
            cs = analyzer.expected.get('consumer_surplus', 0)
            ps = analyzer.expected.get('producer_profit', 0)
            sw = analyzer.expected.get('social_welfare', 0)
            gini = analyzer.expected.get('gini_coefficient', 0)
            pdi = analyzer.expected.get('price_discrimination_index', 0)
            
            print(f"{config_name:<40} | {r:6.1%} | {cs:8.2f} | {ps:8.2f} | {sw:8.2f} | {gini:6.3f} | {pdi:8.4f}")
        
        print("-"*100)
        
        # 找出最优
        print("\n【关键发现】")
        
        sw_values = [a.expected.get('social_welfare', 0) for a in self.analyzers]
        max_sw_idx = sw_values.index(max(sw_values))
        print(f"  最高社会福利: {self.configs[max_sw_idx]}")
        
        r_values = [a.r_star for a in self.analyzers]
        max_r_idx = r_values.index(max(r_values))
        print(f"  最高参与率: {self.configs[max_r_idx]}")
        
        gini_values = [a.expected.get('gini_coefficient', 0) for a in self.analyzers]
        min_gini_idx = gini_values.index(min(gini_values))
        print(f"  最低不平等: {self.configs[min_gini_idx]}")
        
        cs_values = [a.expected.get('consumer_surplus', 0) for a in self.analyzers]
        max_cs_idx = cs_values.index(max(cs_values))
        print(f"  最高消费者剩余: {self.configs[max_cs_idx]}")
    
    def plot_comparison_matrix(self, save_path: str = None):
        """绘制2x2对比矩阵（如果有4个配置）"""
        if len(self.analyzers) != 4:
            print(f"需要4个配置才能绘制2x2矩阵（当前{len(self.analyzers)}个）")
            return
        
        # 提取指标
        metrics = {
            'r*': [a.r_star for a in self.analyzers],
            'CS': [a.expected.get('consumer_surplus', 0) for a in self.analyzers],
            'PS': [a.expected.get('producer_profit', 0) for a in self.analyzers],
            'SW': [a.expected.get('social_welfare', 0) for a in self.analyzers],
            'Gini': [a.expected.get('gini_coefficient', 0) for a in self.analyzers],
            'PDI': [a.expected.get('price_discrimination_index', 0) for a in self.analyzers]
        }
        
        # 假设顺序: CP-ID, CP-AN, CE-ID, CE-AN
        labels = [a.path.stem.replace("scenario_c_", "").replace("_", "\n") 
                  for a in self.analyzers]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx // 3, idx % 3]
            
            bars = ax.bar(range(4), values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Ground Truth Comparison Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 对比矩阵已保存: {save_path}")
        else:
            plt.savefig(OUTPUT_DIR / "comparison_matrix.png", dpi=300)
            print(f"✅ 对比矩阵已保存: {OUTPUT_DIR / 'comparison_matrix.png'}")
        
        plt.close()
    
    def plot_welfare_decomposition(self, save_path: str = None):
        """绘制福利分解堆叠图"""
        configs = [a.path.stem.replace("scenario_c_", "").replace("_", "\n") 
                   for a in self.analyzers]
        
        cs_values = [a.expected.get('consumer_surplus', 0) for a in self.analyzers]
        ps_values = [a.expected.get('producer_profit', 0) for a in self.analyzers]
        is_values = [a.expected.get('intermediary_profit', 0) for a in self.analyzers]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(configs))
        width = 0.6
        
        # 堆叠柱状图
        p1 = ax.bar(x, cs_values, width, label='Consumer Surplus (CS)', color='#2ca02c')
        p2 = ax.bar(x, ps_values, width, bottom=cs_values, 
                    label='Producer Profit (PS)', color='#1f77b4')
        
        # 中介利润（通常为负，从底部开始）
        bottoms = [cs + ps for cs, ps in zip(cs_values, ps_values)]
        p3 = ax.bar(x, is_values, width, bottom=bottoms,
                    label='Intermediary Profit (IS)', color='#d62728')
        
        # 总福利线
        sw_values = [a.expected.get('social_welfare', 0) for a in self.analyzers]
        ax.plot(x, sw_values, 'ko-', linewidth=2, markersize=8, label='Social Welfare (SW)')
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Welfare', fontsize=12)
        ax.set_title('Welfare Decomposition Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 福利分解图已保存: {save_path}")
        else:
            plt.savefig(OUTPUT_DIR / "welfare_decomposition.png", dpi=300)
            print(f"✅ 福利分解图已保存: {OUTPUT_DIR / 'welfare_decomposition.png'}")
        
        plt.close()


class SweepAnalyzer:
    """补偿扫描分析"""
    
    def __init__(self, sweep_file: str):
        """
        Args:
            sweep_file: 补偿扫描JSON文件路径
        """
        self.path = Path(sweep_file)
        with open(self.path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*80)
        print(f"📈 补偿扫描分析: {self.path.name}")
        print("="*80)
        
        print(f"\n补偿范围: m ∈ [{self.data[0]['m']:.2f}, {self.data[-1]['m']:.2f}]")
        print(f"扫描点数: {len(self.data)}")
        
        # 找关键点
        sw_values = [item['social_welfare'] for item in self.data]
        max_sw_idx = sw_values.index(max(sw_values))
        optimal_m = self.data[max_sw_idx]['m']
        max_sw = sw_values[max_sw_idx]
        
        print(f"\n【最优补偿】")
        print(f"  m* = {optimal_m:.2f}")
        print(f"  最大社会福利 SW* = {max_sw:.2f}")
        print(f"  对应参与率 r* = {self.data[max_sw_idx]['participation_rate']:.2%}")
        
        # 临界点分析
        print(f"\n【参与率趋势】")
        r_min = min(item['participation_rate'] for item in self.data)
        r_max = max(item['participation_rate'] for item in self.data)
        print(f"  最低参与率: {r_min:.2%} (m={self.data[0]['m']:.2f})")
        print(f"  最高参与率: {r_max:.2%} (m={self.data[-1]['m']:.2f})")
        
        # 找到r*达到50%的临界m
        for i, item in enumerate(self.data):
            if item['participation_rate'] > 0.5:
                print(f"  r*>50% 临界点: m ≈ {item['m']:.2f}")
                break
    
    def plot_curves(self, save_path: str = None):
        """绘制补偿扫描曲线"""
        m_values = [item['m'] for item in self.data]
        r_values = [item['participation_rate'] for item in self.data]
        cs_values = [item['consumer_surplus'] for item in self.data]
        ps_values = [item['producer_profit'] for item in self.data]
        sw_values = [item['social_welfare'] for item in self.data]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 参与率曲线
        ax1 = axes[0, 0]
        ax1.plot(m_values, r_values, 'o-', linewidth=2, markersize=6, color='#1f77b4')
        ax1.set_xlabel('Compensation (m)', fontsize=11)
        ax1.set_ylabel('Participation Rate (r*)', fontsize=11)
        ax1.set_title('r*(m) Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='r=50%')
        ax1.legend()
        
        # 2. 社会福利曲线
        ax2 = axes[0, 1]
        ax2.plot(m_values, sw_values, 's-', linewidth=2, markersize=6, color='#2ca02c')
        max_sw_idx = sw_values.index(max(sw_values))
        ax2.plot(m_values[max_sw_idx], sw_values[max_sw_idx], 'r*', 
                markersize=15, label=f'Optimal m*={m_values[max_sw_idx]:.2f}')
        ax2.set_xlabel('Compensation (m)', fontsize=11)
        ax2.set_ylabel('Social Welfare (SW)', fontsize=11)
        ax2.set_title('SW(m) Curve', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. CS vs PS
        ax3 = axes[1, 0]
        ax3.plot(m_values, cs_values, 'o-', linewidth=2, label='Consumer Surplus', color='#ff7f0e')
        ax3.plot(m_values, ps_values, 's-', linewidth=2, label='Producer Profit', color='#9467bd')
        ax3.set_xlabel('Compensation (m)', fontsize=11)
        ax3.set_ylabel('Welfare', fontsize=11)
        ax3.set_title('CS(m) vs PS(m)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 福利分解堆叠面积图
        ax4 = axes[1, 1]
        ax4.fill_between(m_values, 0, cs_values, alpha=0.5, label='CS', color='#ff7f0e')
        ax4.fill_between(m_values, cs_values, 
                        [cs + ps for cs, ps in zip(cs_values, ps_values)],
                        alpha=0.5, label='PS', color='#9467bd')
        ax4.plot(m_values, sw_values, 'k-', linewidth=2, label='SW (Total)')
        ax4.set_xlabel('Compensation (m)', fontsize=11)
        ax4.set_ylabel('Welfare', fontsize=11)
        ax4.set_title('Welfare Decomposition', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle('Compensation Sweep Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 补偿扫描曲线已保存: {save_path}")
        else:
            plt.savefig(OUTPUT_DIR / "compensation_sweep.png", dpi=300)
            print(f"✅ 补偿扫描曲线已保存: {OUTPUT_DIR / 'compensation_sweep.png'}")
        
        plt.close()


def analyze_single(file_path: str):
    """分析单个GT文件"""
    analyzer = GTAnalyzer(file_path)
    analyzer.print_summary()
    
    # 质量检查
    checks = analyzer.check_quality()
    print("\n【质量检查】")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    # 生成可视化
    analyzer.plot_convergence()
    
    return analyzer


def analyze_compare(pattern: str = None):
    """对比分析多个GT"""
    if pattern:
        gt_files = list(DATA_DIR.glob(pattern))
    else:
        # 默认对比核心4个配置
        gt_files = [
            DATA_DIR / "scenario_c_common_preferences_identified.json",
            DATA_DIR / "scenario_c_common_preferences_anonymized.json",
            DATA_DIR / "scenario_c_common_experience_identified.json",
            DATA_DIR / "scenario_c_common_experience_anonymized.json"
        ]
        gt_files = [f for f in gt_files if f.exists()]
    
    if not gt_files:
        print("❌ 未找到匹配的GT文件")
        return
    
    print(f"\n找到 {len(gt_files)} 个GT文件")
    for f in gt_files:
        print(f"  - {f.name}")
    
    comparator = GTComparator([str(f) for f in gt_files])
    comparator.compare_summary()
    
    # 生成可视化
    if len(gt_files) == 4:
        comparator.plot_comparison_matrix()
    comparator.plot_welfare_decomposition()
    
    return comparator


def analyze_sweep(file_path: str = None):
    """分析补偿扫描"""
    if file_path is None:
        file_path = DATA_DIR / "scenario_c_payment_sweep.json"
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    analyzer = SweepAnalyzer(file_path)
    analyzer.print_summary()
    analyzer.plot_curves()
    
    return analyzer


def analyze_all():
    """运行所有分析"""
    print("\n" + "🚀"*40)
    print("运行完整分析...")
    print("🚀"*40)
    
    # 1. 对比分析
    print("\n" + "="*80)
    print("第1步: 核心配置对比分析")
    print("="*80)
    analyze_compare()
    
    # 2. 补偿扫描
    print("\n" + "="*80)
    print("第2步: 补偿扫描分析")
    print("="*80)
    analyze_sweep()
    
    # 3. 单个分析（选第一个可用的）
    print("\n" + "="*80)
    print("第3步: 示例单文件详细分析")
    print("="*80)
    result_file = DATA_DIR / "scenario_c_result.json"
    if result_file.exists():
        analyze_single(str(result_file))
    else:
        # 找任意一个
        gt_files = list(DATA_DIR.glob("scenario_c_*.json"))
        if gt_files and not gt_files[0].name.endswith("sweep.json"):
            analyze_single(str(gt_files[0]))
    
    print("\n" + "🎉"*40)
    print(f"分析完成！所有图表已保存到: {OUTPUT_DIR}")
    print("🎉"*40)


def main():
    parser = argparse.ArgumentParser(
        description="场景C Ground Truth 理论解分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个文件
  python analyze_scenario_c_gt.py --mode single --file data/ground_truth/scenario_c_result.json
  
  # 对比分析（默认核心4配置）
  python analyze_scenario_c_gt.py --mode compare
  
  # 对比分析（自定义模式匹配）
  python analyze_scenario_c_gt.py --mode compare --pattern "scenario_c_common_*.json"
  
  # 补偿扫描分析
  python analyze_scenario_c_gt.py --mode sweep
  
  # 运行所有分析
  python analyze_scenario_c_gt.py --mode all
        """
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'compare', 'sweep', 'all'],
                       default='all',
                       help='分析模式 (默认: all)')
    
    parser.add_argument('--file', type=str,
                       help='GT文件路径（用于single和sweep模式）')
    
    parser.add_argument('--pattern', type=str,
                       help='文件匹配模式（用于compare模式，如 "scenario_c_*.json"）')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 场景C Ground Truth 理论解分析工具")
    print("="*80)
    
    if args.mode == 'single':
        if not args.file:
            print("❌ single模式需要指定--file参数")
            return
        analyze_single(args.file)
    
    elif args.mode == 'compare':
        analyze_compare(args.pattern)
    
    elif args.mode == 'sweep':
        analyze_sweep(args.file)
    
    elif args.mode == 'all':
        analyze_all()
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()
