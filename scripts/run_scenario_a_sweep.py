"""
场景A参数扫描实验脚本
复现原始experiment_results.csv的完整实验

企业数从1递增到10，对比理性vs多个LLM模型

复现原始实验配置（对应原始：--consumer-num 20 --firm-num 1 --num-experiments 10 --num-rounds 10）：
python scripts/run_scenario_a_sweep.py --start-firm 1 --end-firm 10 --n-consumers 20 --rounds 10 --search-cost 0.02 --seed 42 --models deepseek-v3.2 gpt-5-mini-2025-08-07 qwen-plus gemini-3-flash-preview

快速测试（1轮）：
python scripts/run_scenario_a_sweep.py --start-firm 1 --end-firm 10 --n-consumers 10 --rounds 1 --search-cost 0.02 --seed 42 --models deepseek-v3.2
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List
import sys
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_scenario_a_sweep(
    firm_range: range = range(1, 11),
    n_consumers: int = 10,
    search_cost: float = 0.02,
    seed: int = 42,
    rounds: int = 1,
    models: List[str] = None
):
    """
    运行场景A的参数扫描实验
    
    Args:
        firm_range: 企业数量范围
        n_consumers: 消费者数量
        search_cost: 搜索成本
        seed: 随机种子
        models: 要测试的模型列表
    """
    if models is None:
        models = ["deepseek-v3.2", "gpt-5-mini-2025-08-07", "qwen-plus", "gemini-3-flash-preview"]
    
    print(f"\n{'='*60}")
    print(f"Scenario A Parameter Sweep")
    print(f"{'='*60}")
    print(f"Firm range: {list(firm_range)}")
    print(f"Consumers: {n_consumers}")
    print(f"Rounds per experiment: {rounds}")
    print(f"Search cost: {search_cost}")
    print(f"Models: {models}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    # 首先运行理性基准
    print("\n[RATIONAL BASELINE]")
    for firm_num in firm_range:
        print(f"\n[Rational] firm_num={firm_num}")
        
        cmd = [
            sys.executable,
            "src/evaluators/evaluate_scenario_a_full.py",
            "--rational-share",
            "--rational-price",
            "--rational-search",
            "--n-consumers", str(n_consumers),
            "--n-firms", str(firm_num),
            "--search-cost", str(search_cost),
            "--seed", str(seed),
            "--rounds", str(rounds)
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=project_root)
            
            # 读取最新结果
            result_dir = project_root / "evaluation_results" / "scenario_a"
            result_files = sorted(result_dir.glob("eval_A_full_*_rational_*.json"))
            
            if result_files:
                with open(result_files[-1], 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                round_data = result['all_rounds'][0]
                all_results.append({
                    'model': 'rational',
                    'firm_num': firm_num,
                    'share_rate': round_data['share_rate'],
                    'avg_price': round_data['avg_price'],
                    'consumer_surplus': round_data['consumer_surplus'],
                    'firm_profit': round_data['firm_profit'],
                    'social_welfare': round_data['social_welfare'],
                    'avg_search_cost': round_data['avg_search_cost'],
                    'purchase_rate': round_data['purchase_rate']
                })
                print(f"  [OK] share_rate={round_data['share_rate']:.2%}, "
                      f"price={round_data['avg_price']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # 然后运行LLM模型
    for model in models:
        print(f"\n[LLM MODEL: {model}]")
        
        for firm_num in firm_range:
            print(f"\n[{model}] firm_num={firm_num}")
            
            cmd = [
                sys.executable,
                "src/evaluators/evaluate_scenario_a_full.py",
                "--model", model,
                "--n-consumers", str(n_consumers),
                "--n-firms", str(firm_num),
                "--search-cost", str(search_cost),
                "--seed", str(seed),
                "--rounds", str(rounds)
            ]
            
            try:
                subprocess.run(cmd, check=True, cwd=project_root)
                
                # 读取最新结果
                result_dir = project_root / "evaluation_results" / "scenario_a"
                result_files = sorted(result_dir.glob(f"eval_A_full_{model.replace('-', '_')}*.json"))
                
                if result_files:
                    with open(result_files[-1], 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    round_data = result['all_rounds'][0]
                    all_results.append({
                        'model': model,
                        'firm_num': firm_num,
                        'share_rate': round_data['share_rate'],
                        'avg_price': round_data['avg_price'],
                        'consumer_surplus': round_data['consumer_surplus'],
                        'firm_profit': round_data['firm_profit'],
                        'social_welfare': round_data['social_welfare'],
                        'avg_search_cost': round_data['avg_search_cost'],
                        'purchase_rate': round_data['purchase_rate']
                    })
                    print(f"  [OK] share_rate={round_data['share_rate']:.2%}, "
                          f"price={round_data['avg_price']:.4f}")
                    
                    # API限流控制
                    time.sleep(1)
            except Exception as e:
                print(f"  [ERROR] {e}")
    
    # 保存结果
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    df = pd.DataFrame(all_results)
    
    output_dir = project_root / "evaluation_results" / "scenario_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存长格式（便于分析）
    long_csv_path = output_dir / f"sweep_results_long_{timestamp}.csv"
    df.to_csv(long_csv_path, index=False)
    print(f"[SAVED] Long format: {long_csv_path}")
    
    # 保存宽格式（类似原始CSV，每行一个模型）
    wide_csv_path = output_dir / f"sweep_results_wide_{timestamp}.csv"
    
    # 手动构建宽格式：每个模型一行，列名为 metric_firmnum
    models_list = df['model'].unique()
    firm_nums = sorted(df['firm_num'].unique())
    
    wide_data = []
    for model in models_list:
        model_data = df[df['model'] == model].sort_values('firm_num')
        row = {'model': model}
        
        for metric in ['share_rate', 'consumer_surplus', 'firm_profit', 
                       'avg_search_cost', 'avg_price']:
            for firm_num in firm_nums:
                value = model_data[model_data['firm_num'] == firm_num][metric]
                col_name = f"{metric}_{firm_num}"
                row[col_name] = value.values[0] if len(value) > 0 else None
        
        wide_data.append(row)
    
    df_wide = pd.DataFrame(wide_data)
    df_wide.to_csv(wide_csv_path, index=False)
    print(f"[SAVED] Wide format: {wide_csv_path}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Firm range: {df['firm_num'].min()}-{df['firm_num'].max()}")
    
    print(f"\nModel average metrics:")
    summary = df.groupby('model').agg({
        'share_rate': 'mean',
        'avg_price': 'mean',
        'avg_search_cost': 'mean',
        'social_welfare': 'mean'
    }).round(4)
    print(summary)
    
    return df, df_wide


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scenario A Parameter Sweep')
    parser.add_argument('--start-firm', type=int, default=1, help='Starting firm count')
    parser.add_argument('--end-firm', type=int, default=10, help='Ending firm count')
    parser.add_argument('--n-consumers', type=int, default=10, help='Number of consumers')
    parser.add_argument('--rounds', type=int, default=1, help='Rounds per experiment (to average results)')
    parser.add_argument('--search-cost', type=float, default=0.02, help='Search cost')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--models', nargs='+', default=["deepseek-v3.2", "gpt-5-mini-2025-08-07", "qwen-plus"],
                        help='Models to test')
    
    args = parser.parse_args()
    
    firm_range = range(args.start_firm, args.end_firm + 1)
    
    df, df_wide = run_scenario_a_sweep(
        firm_range=firm_range,
        n_consumers=args.n_consumers,
        search_cost=args.search_cost,
        seed=args.seed,
        rounds=args.rounds,
        models=args.models
    )
    
    print(f"\n{'='*60}")
    print("[SUCCESS] All experiments complete!")
    print(f"{'='*60}")
