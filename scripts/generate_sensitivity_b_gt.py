"""
生成场景B敏感度分析的Ground Truth文件
3×3参数网格：ρ={0.3, 0.6, 0.9} × v范围={[0.3,0.6], [0.6,0.9], [0.9,1.2]}
"""
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scenarios.scenario_b_too_much_data import (
    ScenarioBParams,
    solve_stackelberg_personalized
)


def generate_sensitivity_gt():
    """生成3×3网格的9个GT文件"""
    
    # 参数网格
    rho_values = [0.3, 0.6, 0.9]
    v_ranges = [
        (0.3, 0.6),
        (0.6, 0.9),
        (0.9, 1.2),
    ]
    
    # 固定参数
    n = 20  # 用户数量
    sigma_noise_sq = 0.1
    alpha = 1.0
    seed = 42  # 固定随机种子确保可复现
    
    # 创建输出目录
    output_dir = Path("data/ground_truth/sensitivity_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("场景B敏感度分析 - Ground Truth生成")
    print("="*80)
    print(f"参数网格：")
    print(f"  ρ值: {rho_values}")
    print(f"  v范围: {v_ranges}")
    print(f"  固定参数: n={n}, sigma_noise_sq={sigma_noise_sq}, alpha={alpha}")
    print(f"  输出目录: {output_dir}")
    print("="*80)
    
    # 生成所有组合
    total = len(rho_values) * len(v_ranges)
    count = 0
    
    for rho in rho_values:
        for v_min, v_max in v_ranges:
            count += 1
            print(f"\n[{count}/{total}] 生成 GT: ρ={rho}, v=[{v_min}, {v_max}]")
            
            # 生成v值（均匀分布）
            v = np.linspace(v_min, v_max, n).tolist()
            
            # 生成相关矩阵（均匀相关）
            Sigma = np.full((n, n), rho)
            np.fill_diagonal(Sigma, 1.0)
            
            # 创建参数
            params = ScenarioBParams(
                n=n,
                rho=rho,
                Sigma=Sigma,  # 直接传numpy数组，不需要tolist()
                v=v,
                sigma_noise_sq=sigma_noise_sq,
                alpha=alpha,
                seed=seed
            )
            
            print(f"  参数: n={n}, rho={rho}, v=[{v_min:.1f}, {v_max:.1f}]")
            print(f"  v值: min={min(v):.3f}, max={max(v):.3f}, mean={np.mean(v):.3f}")
            
            # 求解均衡
            try:
                result = solve_stackelberg_personalized(params)
                
                # result是字典，包含: eq_share_set, eq_prices, eq_profit, eq_W, eq_total_leakage, diagnostics, solver_mode
                eq_share_set = result["eq_share_set"]
                eq_prices = result["eq_prices"]
                eq_profit = result["eq_profit"]
                eq_W = result["eq_W"]
                eq_total_leakage = result["eq_total_leakage"]
                solver_mode = result["solver_mode"]
                diagnostics = result.get("diagnostics", {})
                
                print(f"  求解成功!")
                print(f"    分享集合: {eq_share_set} (分享率: {len(eq_share_set)/n:.2%})")
                print(f"    平台利润: {eq_profit:.4f}")
                print(f"    社会福利: {eq_W:.4f}")
                print(f"    总泄露量: {eq_total_leakage:.4f}")
                print(f"    求解器模式: {solver_mode}")
                
                # 构造GT数据
                gt_data = {
                    "params": params.to_dict(),
                    "gt_numeric": {
                        "eq_share_set": eq_share_set,
                        "eq_prices": eq_prices,
                        "eq_profit": eq_profit,
                        "eq_W": eq_W,
                        "eq_total_leakage": eq_total_leakage,
                        "solver_mode": solver_mode,
                        "diagnostics": diagnostics,
                    },
                    "gt_labels": {
                        "leakage_bucket": _bucket_share_rate(len(eq_share_set) / n),
                        "over_sharing": 0,  # 这里简化，不计算first-best
                        "share_rate": len(eq_share_set) / n,
                    },
                    "metadata": {
                        "generated_by": "scripts/generate_sensitivity_b_gt.py",
                        "sensitivity_params": {
                            "rho": rho,
                            "v_min": v_min,
                            "v_max": v_max,
                        }
                    }
                }
                
                # 保存GT文件
                filename = f"scenario_b_rho{rho:.1f}_v{v_min:.1f}-{v_max:.1f}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(gt_data, f, indent=2, ensure_ascii=False)
                
                print(f"  ✅ 已保存: {filename}")
                
            except Exception as e:
                print(f"  ❌ 求解失败: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n"+"="*80)
    print(f"✅ 所有GT文件生成完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   文件数量: {len(list(output_dir.glob('*.json')))}")
    print("="*80)


def _bucket_share_rate(rate: float) -> str:
    """将分享率分桶"""
    if rate < 0.3:
        return "low"
    elif rate < 0.7:
        return "medium"
    else:
        return "high"


if __name__ == "__main__":
    generate_sensitivity_gt()
