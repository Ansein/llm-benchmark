"""
场景B敏感度分析可视化 - 3×3热力图（支持多模型）
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(exp_dir: str) -> Dict[str, List[Dict]]:
    """
    加载实验结果（按模型分组）
    
    Returns:
        {model_name: [result1, result2, ...]}
    """
    exp_path = Path(exp_dir)
    
    # 先尝试从模型子目录加载
    results_by_model = {}
    
    # 检查是否有模型子目录
    model_dirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name != "llm_logs" and d.name != "heatmaps"]
    
    if model_dirs:
        # 从模型子目录加载
        for model_dir in model_dirs:
            model_name = model_dir.name
            model_results = []
            
            for json_file in model_dir.glob("result_*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        model_results.append(json.load(f))
                except Exception as e:
                    print(f"[警告] 无法加载 {json_file}: {e}")
            
            if model_results:
                results_by_model[model_name] = model_results
                print(f"✅ 加载模型 {model_name}: {len(model_results)} 个结果")
    else:
        # 从根目录加载（旧格式）
        all_results = []
        for json_file in exp_path.glob("result_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    all_results.append(json.load(f))
            except Exception as e:
                print(f"[警告] 无法加载 {json_file}: {e}")
        
        if all_results:
            # 按model_name分组
            for result in all_results:
                model_name = result.get("model_name", "unknown")
                if model_name not in results_by_model:
                    results_by_model[model_name] = []
                results_by_model[model_name].append(result)
            
            print(f"✅ 加载 {len(all_results)} 个结果，分组为 {len(results_by_model)} 个模型")
    
    return results_by_model


def aggregate_results(results: List[Dict], rho_values: List[float], v_ranges: List[Tuple]) -> np.ndarray:
    """
    聚合多次试验的结果（计算均值）
    
    Returns:
        3×3矩阵（行=rho, 列=v_range）
    """
    # 创建字典存储每个组合的所有值
    values_dict = {}
    
    for result in results:
        params = result.get("sensitivity_params", {})
        rho = params.get("rho")
        v_min = params.get("v_min")
        v_max = params.get("v_max")
        
        key = (rho, v_min, v_max)
        if key not in values_dict:
            values_dict[key] = []
        
        values_dict[key].append(result)
    
    # 创建矩阵并填充均值
    matrix = np.zeros((len(rho_values), len(v_ranges)))
    
    for i, rho in enumerate(rho_values):
        for j, (v_min, v_max) in enumerate(v_ranges):
            key = (rho, v_min, v_max)
            if key in values_dict:
                matrix[i, j] = len(values_dict[key])
            else:
                matrix[i, j] = 0
    
    return matrix, values_dict


def extract_metric_value(result: Dict, metric_path: List[str]) -> float:
    """从结果中提取指标值"""
    value = result
    for key in metric_path:
        value = value.get(key, {})
        if not isinstance(value, dict) and key != metric_path[-1]:
            return 0.0
    
    # 如果最后一个键返回的是字典，可能出错了
    if isinstance(value, dict):
        return 0.0
    
    return float(value)


def create_heatmap(
    results_by_model: Dict[str, List[Dict]],
    metric_path: List[str],
    title: str,
    output_path: str,
    vmin=None,
    vmax=None,
    cmap='viridis',
    show_std=False
):
    """
    创建3×3热力图（支持多模型对比）
    
    Args:
        results_by_model: 按模型分组的结果
        metric_path: 指标路径，如 ["metrics", "llm", "share_rate"]
        title: 图表标题
        output_path: 输出路径
        vmin/vmax: 颜色范围
        cmap: 颜色映射
        show_std: 是否显示标准差
    """
    rho_values = [0.3, 0.6, 0.9]
    v_ranges = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
    
    num_models = len(results_by_model)
    
    if num_models == 1:
        # 单模型：1个子图
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    elif num_models == 2:
        # 双模型：1行2列
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    else:
        # 多模型：2行N列
        ncols = (num_models + 1) // 2
        fig, axes = plt.subplots(2, ncols, figsize=(8*ncols, 14))
        axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_by_model.items()):
        # 创建空矩阵
        matrix_mean = np.zeros((3, 3))
        matrix_std = np.zeros((3, 3))
        
        # 按(rho, v_min, v_max)分组
        grouped = {}
        for result in results:
            params = result.get("sensitivity_params", {})
            rho = params.get("rho")
            v_min = params.get("v_min")
            v_max = params.get("v_max")
            
            key = (rho, v_min, v_max)
            if key not in grouped:
                grouped[key] = []
            
            # 提取指标值
            value = extract_metric_value(result, metric_path)
            grouped[key].append(value)
        
        # 填充矩阵
        for i, rho in enumerate(rho_values):
            for j, (v_min, v_max) in enumerate(v_ranges):
                key = (rho, v_min, v_max)
                if key in grouped and len(grouped[key]) > 0:
                    values = grouped[key]
                    matrix_mean[i, j] = np.mean(values)
                    matrix_std[i, j] = np.std(values) if len(values) > 1 else 0
                else:
                    matrix_mean[i, j] = np.nan
                    matrix_std[i, j] = np.nan
        
        # 绘制热力图
        ax = axes[idx] if num_models > 1 else axes[0]
        
        # 准备标注（均值±标准差）
        if show_std:
            annot_matrix = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    if not np.isnan(matrix_mean[i, j]):
                        mean_val = matrix_mean[i, j]
                        std_val = matrix_std[i, j]
                        if std_val > 0:
                            annot_matrix[i, j] = f'{mean_val:.3f}\n±{std_val:.3f}'
                        else:
                            annot_matrix[i, j] = f'{mean_val:.3f}'
                    else:
                        annot_matrix[i, j] = 'N/A'
            
            sns.heatmap(
                matrix_mean,
                annot=annot_matrix,
                fmt='',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xticklabels=[f'[{v[0]}, {v[1]}]' for v in v_ranges],
                yticklabels=[f'{rho}' for rho in rho_values],
                cbar_kws={'label': title},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
        else:
            sns.heatmap(
                matrix_mean,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xticklabels=[f'[{v[0]}, {v[1]}]' for v in v_ranges],
                yticklabels=[f'{rho}' for rho in rho_values],
                cbar_kws={'label': title},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
        
        ax.set_xlabel('隐私偏好范围 v', fontsize=12)
        ax.set_ylabel('相关系数 ρ', fontsize=12)
        ax.set_title(f'{title}\n模型: {model_name}', fontsize=13, fontweight='bold')
    
    # 隐藏多余的子图
    if num_models > 1:
        for idx in range(len(results_by_model), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存: {output_path}")


def plot_all_heatmaps(exp_dir: str, show_std: bool = True):
    """生成所有热力图"""
    results_by_model = load_results(exp_dir)
    
    if not results_by_model:
        print("❌ 没有找到结果文件")
        return
    
    output_dir = Path(exp_dir) / "heatmaps"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"生成热力图...")
    print(f"{'='*60}")
    
    # 1. LLM分享率
    create_heatmap(
        results_by_model,
        ["metrics", "llm", "share_rate"],
        "LLM分享率",
        str(output_dir / "heatmap_share_rate_llm.png"),
        vmin=0, vmax=1,
        cmap='YlGnBu',
        show_std=show_std
    )
    
    # 2. 理论分享率
    create_heatmap(
        results_by_model,
        ["metrics", "ground_truth", "share_rate"],
        "理论分享率",
        str(output_dir / "heatmap_share_rate_gt.png"),
        vmin=0, vmax=1,
        cmap='YlGnBu',
        show_std=False  # GT没有标准差
    )
    
    # 3. Jaccard相似度
    create_heatmap(
        results_by_model,
        ["equilibrium_quality", "share_set_similarity"],
        "决策相似度 (Jaccard)",
        str(output_dir / "heatmap_jaccard.png"),
        vmin=0, vmax=1,
        cmap='RdYlGn',
        show_std=show_std
    )
    
    # 4. 分享率误差
    # 需要单独计算
    print("\n计算分享率误差...")
    error_results = {}
    for model_name, results in results_by_model.items():
        error_results[model_name] = []
        for result in results:
            # 复制结果并添加误差字段
            result_copy = result.copy()
            llm_rate = result.get("metrics", {}).get("llm", {}).get("share_rate", 0)
            gt_rate = result.get("metrics", {}).get("ground_truth", {}).get("share_rate", 0)
            
            # 添加嵌套字段
            if "custom_metrics" not in result_copy:
                result_copy["custom_metrics"] = {}
            result_copy["custom_metrics"]["share_rate_error"] = abs(llm_rate - gt_rate)
            
            error_results[model_name].append(result_copy)
    
    create_heatmap(
        error_results,
        ["custom_metrics", "share_rate_error"],
        "分享率误差 (|LLM - GT|)",
        str(output_dir / "heatmap_share_rate_error.png"),
        vmin=0,
        cmap='Reds',
        show_std=show_std
    )
    
    # 5. 利润偏差
    create_heatmap(
        results_by_model,
        ["metrics", "deviations", "profit_mae"],
        "利润偏差 (MAE)",
        str(output_dir / "heatmap_profit_mae.png"),
        vmin=0,
        cmap='Reds',
        show_std=show_std
    )
    
    # 6. 福利偏差
    create_heatmap(
        results_by_model,
        ["metrics", "deviations", "welfare_mae"],
        "福利偏差 (MAE)",
        str(output_dir / "heatmap_welfare_mae.png"),
        vmin=0,
        cmap='Reds',
        show_std=show_std
    )
    
    print(f"\n{'='*60}")
    print(f"✅ 所有热力图已保存至: {output_dir}")
    print(f"{'='*60}")
    
    # 生成对比表格
    generate_comparison_table(results_by_model, output_dir)


def generate_comparison_table(results_by_model: Dict[str, List[Dict]], output_dir: Path):
    """生成模型对比表格"""
    print(f"\n生成模型对比表格...")
    
    import pandas as pd
    
    rows = []
    
    rho_values = [0.3, 0.6, 0.9]
    v_ranges = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
    
    for model_name, results in results_by_model.items():
        # 按参数组合分组
        grouped = {}
        for result in results:
            params = result.get("sensitivity_params", {})
            rho = params.get("rho")
            v_min = params.get("v_min")
            v_max = params.get("v_max")
            
            key = (rho, v_min, v_max)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # 计算每个组合的统计量
        for rho in rho_values:
            for v_min, v_max in v_ranges:
                key = (rho, v_min, v_max)
                if key not in grouped or len(grouped[key]) == 0:
                    continue
                
                group_results = grouped[key]
                
                # 提取指标
                jaccards = [r.get("equilibrium_quality", {}).get("share_set_similarity", 0) for r in group_results]
                llm_rates = [r.get("metrics", {}).get("llm", {}).get("share_rate", 0) for r in group_results]
                gt_rates = [r.get("metrics", {}).get("ground_truth", {}).get("share_rate", 0) for r in group_results]
                rate_errors = [abs(l - g) for l, g in zip(llm_rates, gt_rates)]
                
                row = {
                    "模型": model_name,
                    "ρ": rho,
                    "v范围": f"[{v_min}, {v_max}]",
                    "试验数": len(group_results),
                    "Jaccard(均值)": np.mean(jaccards),
                    "Jaccard(标准差)": np.std(jaccards),
                    "LLM分享率(均值)": np.mean(llm_rates),
                    "GT分享率": gt_rates[0],  # GT应该相同
                    "分享率误差(均值)": np.mean(rate_errors),
                    "分享率误差(标准差)": np.std(rate_errors),
                }
                
                rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 保存CSV
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 对比表格已保存: {csv_path}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("模型对比摘要")
    print("="*60)
    
    for model_name in df["模型"].unique():
        model_df = df[df["模型"] == model_name]
        print(f"\n模型: {model_name}")
        print(f"  平均Jaccard: {model_df['Jaccard(均值)'].mean():.3f} ± {model_df['Jaccard(标准差)'].mean():.3f}")
        print(f"  平均分享率误差: {model_df['分享率误差(均值)'].mean():.3%} ± {model_df['分享率误差(标准差)'].mean():.3%}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='场景B敏感度分析可视化')
    parser.add_argument('exp_dir', type=str, help='实验结果目录')
    parser.add_argument('--no-std', action='store_true', help='不显示标准差')
    
    args = parser.parse_args()
    
    plot_all_heatmaps(args.exp_dir, show_std=not args.no_std)


if __name__ == "__main__":
    main()
