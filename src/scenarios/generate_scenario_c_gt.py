"""
生成场景C的Ground Truth数据

包含多个配置的Ground Truth:
1. MVP配置（Common Preferences + Identified）
2. 核心对比配置（2种数据结构 × 2种匿名化策略）

位置: src/scenarios/generate_scenario_c_gt.py
运行: python -m src.scenarios.generate_scenario_c_gt
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from pathlib import Path
from .scenario_c_social_data import ScenarioCParams, generate_ground_truth


def generate_mvp_config():
    """生成MVP配置的Ground Truth"""
    print("\n" + "🎯 "*20)
    print("生成MVP配置 Ground Truth")
    print("🎯 "*20)
    
    params = ScenarioCParams(
        N=20,
        data_structure="common_preferences",
        anonymization="identified",
        mu_theta=5.0,
        sigma_theta=1.0,
        sigma=1.0,
        m=1.0,
        c=0.0,
        seed=42
    )
    
    gt = generate_ground_truth(
        params,
        max_iter=20,
        tol=1e-3,
        num_mc_samples=50
    )
    
    # 保存（从src/scenarios/向上两级到项目根目录）
    output_path = Path(__file__).parent.parent.parent / "data" / "ground_truth" / "scenario_c_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ MVP配置已保存到: {output_path}")
    return gt


def generate_core_configs():
    """生成核心对比配置的Ground Truth"""
    print("\n" + "🎯 "*20)
    print("生成核心对比配置 Ground Truth")
    print("🎯 "*20)
    
    configs = []
    
    # 获取输出目录路径
    output_dir = Path(__file__).parent.parent.parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2种数据结构 × 2种匿名化策略 = 4个配置
    for data_structure in ["common_preferences", "common_experience"]:
        for anonymization in ["identified", "anonymized"]:
            config_name = f"{data_structure}_{anonymization}"
            print(f"\n生成配置: {config_name}")
            
            params = ScenarioCParams(
                N=20,
                data_structure=data_structure,
                anonymization=anonymization,
                mu_theta=5.0,
                sigma_theta=1.0,
                sigma=1.0,
                m=1.0,
                c=0.0,
                seed=42
            )
            
            gt = generate_ground_truth(
                params,
                max_iter=20,
                tol=1e-3,
                num_mc_samples=50
            )
            
            # 保存
            output_path = output_dir / f"scenario_c_{config_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gt, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已保存到: {output_path}")
            
            configs.append({
                "name": config_name,
                "path": str(output_path),
                "participation_rate": gt["rational_participation_rate"],
                "social_welfare": gt["outcome"]["social_welfare"]
            })
    
    return configs


def generate_payment_sweep():
    """生成不同补偿水平的Ground Truth（用于绘制参与率曲线）"""
    print("\n" + "🎯 "*20)
    print("生成补偿扫描配置 Ground Truth")
    print("🎯 "*20)
    
    m_values = [0.0, 0.5, 1.0, 2.0, 3.0]
    results = []
    
    # 获取输出目录路径
    output_dir = Path(__file__).parent.parent.parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for m in m_values:
        print(f"\n生成配置: m={m:.1f}")
        
        params = ScenarioCParams(
            N=20,
            data_structure="common_preferences",
            anonymization="identified",
            mu_theta=5.0,
            sigma_theta=1.0,
            sigma=1.0,
            m=m,
            c=0.0,
            seed=42
        )
        
        gt = generate_ground_truth(
            params,
            max_iter=20,
            tol=1e-3,
            num_mc_samples=50
        )
        
        results.append({
            "m": m,
            "participation_rate": gt["rational_participation_rate"],
            "consumer_surplus": gt["outcome"]["consumer_surplus"],
            "producer_profit": gt["outcome"]["producer_profit"],
            "social_welfare": gt["outcome"]["social_welfare"]
        })
    
    # 保存汇总
    output_path = output_dir / "scenario_c_payment_sweep.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 补偿扫描结果已保存到: {output_path}")
    
    # 打印汇总表
    print(f"\n{'='*60}")
    print(f"补偿扫描结果汇总")
    print(f"{'='*60}")
    print(f"{'补偿':^10} | {'参与率':^10} | {'消费者剩余':^12} | {'社会福利':^12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['m']:^10.1f} | {r['participation_rate']:^10.2%} | {r['consumer_surplus']:^12.4f} | {r['social_welfare']:^12.4f}")
    
    return results


def main():
    """主函数"""
    print("\n" + "="*60)
    print("场景C Ground Truth 生成器")
    print("="*60)
    
    # 1. 生成MVP配置（默认）
    mvp_gt = generate_mvp_config()
    
    # 2. 生成核心对比配置
    core_configs = generate_core_configs()
    
    # 3. 生成补偿扫描配置
    payment_sweep = generate_payment_sweep()
    
    # 打印总结
    print("\n" + "="*60)
    print("✅ 所有Ground Truth生成完成!")
    print("="*60)
    
    print(f"\nMVP配置:")
    print(f"  参与率: {mvp_gt['rational_participation_rate']:.2%}")
    print(f"  社会福利: {mvp_gt['outcome']['social_welfare']:.4f}")
    
    print(f"\n核心对比配置 ({len(core_configs)}个):")
    for config in core_configs:
        print(f"  {config['name']:40s} | 参与率={config['participation_rate']:6.2%} | 福利={config['social_welfare']:8.4f}")
    
    print(f"\n补偿扫描: {len(payment_sweep)}个补偿水平")
    
    print(f"\n📁 所有文件已保存到 data/ground_truth/ 目录")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
