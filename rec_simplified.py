# -*- coding: utf-8 -*-
"""简化版模拟 - 移除广播、CoT、记忆蒸馏等功能"""
import argparse
import random
import time
import numpy as np
import warnings
import logging
import os
import sys
import pickle
import json
import shutil

import matplotlib
matplotlib.use('Agg', force=True)  
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
import datetime 
import csv

# 保存原始的stdout和stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# 重设日志级别
logging.getLogger().setLevel(logging.INFO)
os.environ["AGENTSCOPE_LOG_LEVEL"] = "INFO"

from agents_complete import Consumer, Firm, Platform
import agentscope

for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith('agentscope'):
        logging.getLogger(logger_name).setLevel(logging.INFO)
        logging.getLogger(logger_name).propagate = True


# 添加详细数据记录类
class DetailedDataRecorder:
    """记录详细数值特征到CSV文件"""
    
    def __init__(self, base_dir: str, exp_idx: int, firm_num: int, rational_tag: str):
        self.base_dir = base_dir
        self.exp_idx = exp_idx
        self.firm_num = firm_num
        self.rational_tag = rational_tag
        
        self.detailed_data_dir = os.path.join(base_dir, 'detailed_data')
        if not os.path.exists(self.detailed_data_dir):
            os.makedirs(self.detailed_data_dir)
                    
        self.csv_path = os.path.join(
            self.detailed_data_dir, 
            f'detailed_exp_{exp_idx + 1}_firms_{firm_num}_{rational_tag}.csv'
        )
        
        self._init_csv()
        
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        fieldnames = [
            'Round', 'Agent_Type', 'Agent_Index', 
            # 消费者相关字段
            'Share_Decision', 'Share_Reason', 'Privacy_Cost', 'Valuations',
            'Searched_Firms', 'Purchase_Index', 'Total_Revenue', 'Search_Cost',
            'Decision_Sequence', 'Final_Reason',
            # 企业相关字段  
            'Price', 'Price_Reason', 'Share_Rate_Predicted', 'Revenue', 'Profit',
            'Personalized_Prices', 'Sales_Count',
            # 平台相关字段
            'Platform_Consumer_Surplus', 'Platform_Firm_Surplus', 'Platform_Share_Rate'
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    def record_round_data(self, round_num: int, consumers: list, firms: list, platform, share_ratio: float):
        """记录一轮的详细数据"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Round', 'Agent_Type', 'Agent_Index', 
                'Share_Decision', 'Share_Reason', 'Privacy_Cost', 'Valuations',
                'Searched_Firms', 'Purchase_Index', 'Total_Revenue', 'Search_Cost',
                'Decision_Sequence', 'Final_Reason',
                'Price', 'Price_Reason', 'Share_Rate_Predicted', 'Revenue', 'Profit',
                'Personalized_Prices', 'Sales_Count',
                'Platform_Consumer_Surplus', 'Platform_Firm_Surplus', 'Platform_Share_Rate'
            ])
            
            for consumer in consumers:
                row = {
                    'Round': round_num + 1,
                    'Agent_Type': 'Consumer',
                    'Agent_Index': consumer.index,
                    'Share_Decision': consumer.share,
                    'Share_Reason': str(consumer.temp_memory.get('share_reason', '')),
                    'Privacy_Cost': consumer.privacy_cost,
                    'Valuations': str(consumer.valuations.tolist()),
                    'Searched_Firms': str(consumer.searched_firms),
                    'Purchase_Index': consumer.purchase_index,
                    'Total_Revenue': consumer.total_revenue,
                    'Search_Cost': consumer.total_search_cost,
                    'Decision_Sequence': str(consumer.temp_memory.get('decision_sequence', [])),
                    'Final_Reason': str(consumer.temp_memory.get('final_reason', '')),
                    # 企业字段留空
                    'Price': '', 'Price_Reason': '', 'Share_Rate_Predicted': '', 
                    'Revenue': '', 'Profit': '', 'Personalized_Prices': '', 'Sales_Count': '',
                    # 平台字段
                    'Platform_Consumer_Surplus': platform.consumer_surplus,
                    'Platform_Firm_Surplus': platform.firm_surplus,
                    'Platform_Share_Rate': share_ratio
                }
                writer.writerow(row)
                
            for firm in firms:
                row = {
                    'Round': round_num + 1,
                    'Agent_Type': 'Firm',
                    'Agent_Index': firm.index,
                    'Share_Decision': '', 'Share_Reason': '', 'Privacy_Cost': '', 'Valuations': '',
                    'Searched_Firms': '', 'Purchase_Index': '', 'Total_Revenue': '', 'Search_Cost': '',
                    'Decision_Sequence': '', 'Final_Reason': '',
                    'Price': firm.price,
                    'Price_Reason': str(firm.temp_memory.get('price_reason', '')),
                    'Share_Rate_Predicted': str(firm.temp_memory.get('share_rate_predicted', '')),
                    'Revenue': firm.revenue,
                    'Profit': firm.profit,
                    'Personalized_Prices': str(firm.personalized_prices),
                    'Sales_Count': firm.temp_memory.get('sale_num', 0),
                    'Platform_Consumer_Surplus': platform.consumer_surplus,
                    'Platform_Firm_Surplus': platform.firm_surplus,
                    'Platform_Share_Rate': share_ratio
                }
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="简化版推荐模拟参数")
    parser.add_argument("--consumer-num", type=int, default=4, help="消费者数量")
    parser.add_argument("--firm-num", type=int, default=5, help="企业数量")
    parser.add_argument("--search-cost", type=float, default=0.02, help="消费者搜索成本")
    parser.add_argument("--agent-type", choices=["random", "llm"], default="llm", help="智能体决策类型")
    parser.add_argument("--waiting-time", type=float, default=0.5, help="步骤间等待时间")
    parser.add_argument("--use-dist", action="store_true", help="启用分布式模式")
    parser.add_argument("--visualize", action="store_true", help="启用可视化")
    parser.add_argument("--threads", type=int, default=5, help="并行处理线程数")
    parser.add_argument("--memory_truncate", type=int, default=3, help="记忆截断长度")
    parser.add_argument("--model_config_name", type=str, default="gpt-config", help="模型配置名称")
    parser.add_argument("--clean_start", action="store_true", help="删除之前实验结果，重新开始")
    parser.add_argument("--num-experiments", type=int, default=6, help="实验次数（递增firm_num）")
    parser.add_argument("--num-rounds", type=int, default=4, help="每个实验的模拟轮数")
    parser.add_argument("--force-fresh", action="store_true", help="强制从实验1开始，忽略检查点")
    parser.add_argument("--pricing-mode", choices=["fixed", "adaptive", "perfect"], default="adaptive", help="企业定价模式")
    parser.add_argument("--firm-cost", type=float, default=0.0, help="企业单位生产成本")
    
    # 理智决策参数 - 分别控制每个决策步骤的rational状态
    parser.add_argument("--rational-share", action="store_true", help="分享决策使用理性决策")
    parser.add_argument("--rational-search", action="store_true", help="搜索决策使用理性决策") 
    parser.add_argument("--rational-price", action="store_true", help="定价决策使用理性决策")
    
    # 添加详细数据记录参数
    parser.add_argument("--record-detailed-data", action="store_true", 
                        help="Record detailed agent data to CSV files and JSON")
    
    return parser.parse_args()


def main(
        consumer_num: int = 2,
        firm_num: int = 2,
        search_cost: float = 0,
        agent_type: str = "llm",
        waiting_time: float = 3.0,
        use_dist: bool = False,
        num_rounds: int = 6,
        visualize: bool = False,
        threads: int = 4,
        memory_truncate: int = 3,
        basic_price: float = 0.5,
        method: str = "adaptive",
        model_config_name: str = "gpt-config",
        model_names: dict = None,
        force_fresh: bool = False,
        firm_cost: float = 0.0,
        rational_share: bool = False,
        rational_search: bool = False,
        rational_price: bool = False,
        record_detailed_data: bool = True,
        data_recorder_params: dict = None,
) -> dict:

    try:
        if model_names is None:
            model_names = {
                'decide_share': model_config_name,
                'decide_search': model_config_name,
                'decide_purchase': model_config_name,
                'set_price': model_config_name,
                'set_personalized_price': model_config_name,
            }

        print(f"DEBUG: 初始化模拟 - consumer_num={consumer_num}, firm_num={firm_num}, search_cost={search_cost}")
        print(f"DEBUG: 使用模型配置: {model_names}")
        print(f"DEBUG: Rational设置 - Share: {rational_share}, Search: {rational_search}, Price: {rational_price}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configs/model_configs.json")
        print(f"DEBUG: 尝试从以下路径加载模型配置: {config_path}")

        agentscope.init(
            project="personalized_recommendation_simulation",
            name="main",
            save_code=False,
            save_api_invoke=True,
            model_configs=config_path,  
            use_monitor=False,
        )

        platform = Platform(
            search_cost=search_cost,
            memory_truncate=memory_truncate,
            model_config_name=model_config_name,
            model_names=model_names
        )

        # 设置rational决策所需的参数
        v_dist = {'type': 'uniform', 'low': 0, 'high': 1}
        r_value = 0.8

        # 创建消费者（所有消费者使用相同的模型配置）
        consumers = [
            Consumer(
                index=i,
                search_cost=platform.search_cost,
                privacy_cost=round(random.uniform(0.025, 0.055), 3),
                num_firms=firm_num,
                dist_type='uniform',
                memory_truncate=memory_truncate,
                model_config_name=model_config_name,
                model_names=model_names,
                r_value=r_value,
                v_dist=v_dist,
                rational_search_cost=search_cost,
                enable_cot=False  # 禁用CoT
            )
            for i in range(consumer_num)
        ]
        
        firms = [
            Firm(
                index=i,
                method=method,
                memory_truncate=memory_truncate,
                basic_price=basic_price,
                pricing_mode='adaptive',
                firm_cost=firm_cost,
                model_config_name=model_config_name,
                model_names=model_names,
                marginal_cost=firm_cost,
                v_dist=v_dist,
                r_value=r_value,
                enable_cot=False  # 禁用CoT
            )
            for i in range(firm_num)
        ]

        platform.get_num_consumers(consumers)
        platform.get_num_firms(firms)

        # 初始化列表
        share_ratio_list = []
        consumer_surplus_list = []
        firm_surplus_list = []
        total_search_cost_list = []
        avg_search_cost_list = []
        firm_prices_list = []

        # ===== 理性分享率均衡求解（仅在rational_share=True时执行）=====
        equilibrium_share_rate = None
        if rational_share:
            print("\n🎯 理性分享率均衡求解...")
            max_iter = 50
            tol = 1e-7
            σ = 0.4
            for iter_share in range(max_iter):
                share_decisions = []
                for consumer in consumers:
                    consumer.decide_share(rational=rational_share)
                    share_decisions.append(consumer.share)
                
                σ_new = np.mean(share_decisions)
                print(f"  迭代 {iter_share + 1}: σ = {σ:.4f} -> {σ_new:.4f}")
                
                if abs(σ_new - σ) < tol:
                    print(f"  分享率收敛于第 {iter_share + 1} 次迭代: σ = {σ_new:.4f}")
                    break
                σ = σ_new
            else:
                print(f"  注意：分享率未在 {max_iter} 次迭代内收敛，使用最终值: σ = {σ:.4f}")
            
            equilibrium_share_rate = σ
            print(f"🎯 理性均衡分享率: σ = {equilibrium_share_rate:.4f}")
            print("=" * 60)

        # 智能检查是否需要多线程：只有当所有三个关键步骤都是rational时才用单线程
        all_rational = rational_share and rational_search and rational_price
        effective_threads = 1 if all_rational else threads
        
        mode_desc = "fully rational (single thread)" if all_rational else f"mixed/LLM mode ({threads} threads)"
        print(f"DEBUG: Using {effective_threads} threads - {mode_desc}")
        print(f"DEBUG: Steps - Share: {'Rational' if rational_share else 'LLM'}, Search: {'Rational' if rational_search else 'LLM'}, Price: {'Rational' if rational_price else 'LLM'}")

        recorder = None
        if record_detailed_data and data_recorder_params:
            recorder = DetailedDataRecorder(
                base_dir=data_recorder_params['base_dir'],
                exp_idx=data_recorder_params['exp_idx'],
                firm_num=firm_num,
                rational_tag=data_recorder_params['rational_tag']
            )

        # 用于存储每轮的详细数据（包括决策理由）
        round_details_list = []

        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            # ===== 模拟循环 =====
            for round_num in range(num_rounds):
                print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

                try:
                    if rational_share:
                        # 理性分享模式：使用预计算的均衡分享率
                        print("📋 使用均衡分享决策...")
                        for consumer in consumers:
                            consumer.decide_share(rational=rational_share)
                    else:
                        # LLM分享模式：并行任务提交分享决策
                        consumer_tasks = []
                        for consumer in consumers:
                            consumer_tasks.append(executor.submit(
                                consumer.decide_share,
                                model_name=None,
                                broadcast_message="",  # 无广播
                                rational=rational_share
                            ))
                        [task.result() for task in consumer_tasks]

                    share_ratio = sum(consumer.share for consumer in consumers) / consumer_num
                    print(f"Platform: Share ratio (σ) is now {share_ratio:.2f}")
                    
                    if rational_share:
                        print(f"理论均衡分享率: {equilibrium_share_rate:.4f}, 实际模拟分享率: {share_ratio:.4f}")
                    
                    list(executor.map(lambda f: f.get_num_consumers_and_firms(platform), firms))

                    # 检查是否第一轮，如果是则初始化企业的预期
                    for firm in firms:
                        firm.temp_memory['share_rate'] = round(share_ratio, 4)

                    # 更新企业预期
                    if rational_share and equilibrium_share_rate is not None:
                        for firm in firms:
                            firm.share_rate_predicted = equilibrium_share_rate
                            firm.temp_memory['share_rate_predicted'] = [equilibrium_share_rate, 'equilibrium']
                        print(f"所有企业预期已更新为均衡分享率: {equilibrium_share_rate:.4f}")
                    else:
                        list(executor.map(lambda f: f.update_expectation(round_num), firms))

                    if rational_price:
                        # 理性价格模式：企业价格在均衡中确定
                        print("🎯 理性价格求解...")
                        max_iter = 50
                        tol = 1e-7
                        initial_prices = [max(0.1, r_value - 0.3) for _ in range(firm_num)]
                        current_prices = initial_prices.copy()
                        
                        for iter_price in range(max_iter):
                            market_price = np.mean(current_prices)
                            new_prices = []
                            
                            platform.firm_prices = current_prices.copy()
                            
                            for firm in firms:
                                if rational_share and equilibrium_share_rate is not None:
                                    firm.temp_memory['actual_share_rate'] = equilibrium_share_rate
                                    firm.share_rate_predicted = equilibrium_share_rate
                                else:
                                    firm.temp_memory['actual_share_rate'] = share_ratio
                                    firm.share_rate_predicted = share_ratio

                                firm.num_firms = firm_num
                                firm.set_price(rational=rational_price)
                                new_prices.append(firm.price)
                            
                            price_diff = np.max(np.abs(np.array(new_prices) - np.array(current_prices)))
                            print(f"  迭代 {iter_price + 1}: 价格差异 = {price_diff:.6f}, 均价 = {np.mean(new_prices):.4f}")
                            
                            if price_diff < tol:
                                print(f"  价格收敛于第 {iter_price + 1} 次迭代")
                                print(f"  最终价格: {[round(p, 4) for p in new_prices]}")
                                break
                                
                            current_prices = new_prices.copy()
                        else:
                            print(f"  价格未在 {max_iter} 次迭代内收敛")
                            print(f"  最终价格: {[round(p, 4) for p in current_prices]}")
                        
                        platform.firm_prices = current_prices.copy()
                        for i, firm in enumerate(firms):
                            firm.price = current_prices[i]
                            firm.temp_memory['equilibrium_price'] = current_prices[i]
                            if rational_share and equilibrium_share_rate is not None:
                                firm.temp_memory['equilibrium_share_rate'] = equilibrium_share_rate
                            else:
                                firm.temp_memory['equilibrium_share_rate'] = share_ratio
                    else:
                        # 非理性价格模式：并行处理企业定价
                        list(
                            executor.map(
                                lambda f: f.set_price(
                                    model_name=model_names.get('set_price'), 
                                    broadcast_message="",  # 无广播
                                    rational=rational_price
                                ), 
                                firms
                            )
                        )
                        print("set price has finished")

                    platform.get_consumer_valuations(consumers)
                    platform.generate_search_sequence(consumers)
                    platform.get_firm_prices(firms)

                    if rational_search:
                        # 理性搜索模式
                        print("使用理性搜索决策...")
                        for consumer in consumers:
                            consumer.decide_search(platform, rational=rational_search)
                    else:
                        # 非理性搜索模式：并行处理消费者搜索决策
                        list(
                            executor.map(
                                lambda c: c.decide_search(
                                    platform, 
                                    model_name=model_names.get('decide_search'), 
                                    broadcast_message="",  # 无广播
                                    rational=rational_search
                                ),
                                consumers
                            )
                        )

                    list(executor.map(lambda c: c.calculate_total_revenue_recommendation(), consumers))
                    list(executor.map(lambda c: c.update_memory(round_num, model_name=model_names.get('update_memory_distill')), consumers))

                    for consumer in consumers:
                        print(
                            f"\n--- Consumer {consumer.index + 1} ({'Shared Data' if consumer.share else 'Did Not Share Data'}) ---")
                        if consumer.share:
                            print(f"- Recommendation Order: {platform.search_sequence[consumer.index]}")

                        if consumer.purchase_index != -1:
                            goal_firms = [firm for firm in consumer.searched_firms if
                                          firm['index'] == consumer.purchase_index]
                            if goal_firms:
                                print(f"Outcome: Purchased Product {consumer.purchase_index} for {goal_firms[0]['price']}.")
                            else:
                                if consumer.purchase_index < len(platform.firm_prices):
                                    price = platform.firm_prices[consumer.purchase_index]
                                    print(f"Outcome: Purchased Product {consumer.purchase_index} for {price}.")
                                else:
                                    print(f"Outcome: Purchased Product {consumer.purchase_index} (price unknown).")
                        else:
                            print("Outcome: Did not purchase any product.")

                    platform.get_consumer_purchase_behavior_recommendation(consumers)
                    platform.calculate_sales(firms)

                    for firm in firms:
                        firm_sales_details = platform.firm_sales.get(firm.index, {}).get('sales_details', {})
                        actual_share_ratio = equilibrium_share_rate if rational_share and equilibrium_share_rate is not None else share_ratio
                        firm.get_revenue(platform, actual_share_ratio, firm_sales_details)
                        firm.update_memory(round_num, model_name=model_names.get('update_memory_distill'))

                    platform.calculate_surplus(consumers, firms)
                    
                    total_search_cost = sum(c.total_search_cost for c in consumers)
                    avg_search_cost = total_search_cost / consumer_num if consumer_num > 0 else 0
                    print(f"Total Search Cost: {total_search_cost}")
                    print(f"Average Search Cost: {avg_search_cost}")  
                    print(f"consumer_num: {consumer_num}")

                    # 更新列表
                    share_ratio_list.append(share_ratio)
                    consumer_surplus_list.append(platform.consumer_surplus)
                    firm_surplus_list.append(platform.firm_surplus)
                    total_search_cost_list.append(total_search_cost)
                    avg_search_cost_list.append(avg_search_cost)  
                    firm_prices_list.append(platform.firm_prices)  

                    print(f"\nDebug - Round {round_num + 1} values:")
                    print(f"Share Ratio: {share_ratio}")
                    print(f"Consumer Surplus: {platform.consumer_surplus}")
                    print(f"Firm Surplus: {platform.firm_surplus}")
                    print(f"Total Search Cost: {total_search_cost}")
                    print(f"Average Search Cost: {avg_search_cost}")  

                    platform.update_memory(round_num, model_name=model_names.get('update_memory_distill'))

                    # 在reset()之前收集本轮的详细数据（包括决策理由）
                    # 注意：必须在reset()之前收集，因为reset()会清空temp_memory
                    round_detail = {
                        'share_ratio': share_ratio,
                        'consumer_surplus': platform.consumer_surplus,
                        'firm_surplus': platform.firm_surplus,
                        'total_search_cost': total_search_cost,
                        'avg_search_cost': avg_search_cost,
                        'firm_prices': platform.firm_prices.copy(),
                        'avg_price': round(sum(platform.firm_prices) / len(platform.firm_prices), 4) if platform.firm_prices else 0.0,
                        'consumer_privacy_choice': {str(c.index): "Shared" if c.share else "Not Shared" for c in consumers},
                        'consumer_reason': {},
                        'firm_reason': {}
                    }
                    
                    # 收集消费者决策理由（优先从temp_memory获取，如果没有则从history_memory获取）
                    for c in consumers:
                        share_reason = c.temp_memory.get('share_reason', '')
                        # 如果temp_memory中没有，尝试从history_memory中获取（刚更新的那一轮）
                        if not share_reason and c.history_memory:
                            last_round_key = f"Round{round_num + 1}"
                            if last_round_key in c.history_memory:
                                share_reason = c.history_memory[last_round_key].get('share_reason', '')
                        # 如果还是没有，且是理性决策，设置默认理由
                        if not share_reason and rational_share:
                            share_reason = f"Rational decision: {'Shared' if c.share else 'Not shared'} based on privacy cost and expected benefit"
                        round_detail['consumer_reason'][str(c.index)] = str(share_reason) if share_reason else " "
                    
                    # 收集企业决策理由（优先从temp_memory获取，如果没有则从history_memory获取）
                    for f in firms:
                        price_reason = f.temp_memory.get('price_reason', '')
                        # 如果temp_memory中没有，尝试从history_memory中获取（刚更新的那一轮）
                        if not price_reason and f.history_memory:
                            last_round_key = f"Round{round_num + 1}"
                            if last_round_key in f.history_memory:
                                price_reason = f.history_memory[last_round_key].get('price_reason', '')
                        # 如果还是没有，且是理性决策，设置默认理由
                        if not price_reason and rational_price:
                            price_reason = f"Rational price: {f.price:.4f} based on equilibrium calculation"
                        round_detail['firm_reason'][str(f.index)] = str(price_reason) if price_reason else " "
                    
                    round_details_list.append(round_detail)

                    # 如果启用了详细数据记录，记录到CSV（在reset之前，确保能获取到决策理由）
                    if recorder:
                        recorder.record_round_data(round_num, consumers, firms, platform, share_ratio)

                    list(executor.map(lambda f: f.reset(), consumers))
                    list(executor.map(lambda f: f.reset(), firms))

                    print(f"\n--- Round {round_num + 1} Results ---")
                    print(f"Consumer Surplus: {platform.consumer_surplus}")
                    print(f"Platform: Share ratio (σ) is now {share_ratio:.2f}")
                    print(f"Prices: {platform.firm_prices}")
                    print(f"*Firm Surplus: {platform.firm_surplus}")

                    platform.reset()

                except Exception as e:
                    print(f"\n回合 {round_num + 1} 出现错误: {str(e)}")
                    import traceback
                    traceback.print_exc()

        if rational_share and equilibrium_share_rate is not None:
            print(f"\n🎉 理性分享模拟完成!")
            print(f"均衡分享率: {equilibrium_share_rate:.4f}")
            print(f"平均价格: {np.mean(platform.firm_prices):.4f}")

        print(f"\n--- Simulation Results (End of Experiment) ---")  
        print(f"Share Ratio: {share_ratio_list}")
        print(f"Firm Prices: {firm_prices_list}")
        print(f"Consumer Surplus: {consumer_surplus_list}")
        print(f"Firm Surplus: {firm_surplus_list}")
        print(f"Total Search Cost: {total_search_cost_list}")
        print(f"Average Search Cost: {avg_search_cost_list}")  

        return {
            'share_ratio': share_ratio_list,
            'consumer_surplus': consumer_surplus_list,
            'firm_surplus': firm_surplus_list,
            'total_search_cost': total_search_cost_list,
            'avg_search_cost': avg_search_cost_list,  
            'firm_prices': firm_prices_list,
            'round_details': round_details_list  # 添加每轮的详细数据
        }
    except Exception as e:
        print(f"\n模拟过程中出现严重错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'share_ratio': share_ratio_list if 'share_ratio_list' in locals() else [],
            'consumer_surplus': consumer_surplus_list if 'consumer_surplus_list' in locals() else [],
            'firm_surplus': firm_surplus_list if 'firm_surplus_list' in locals() else [],
            'total_search_cost': total_search_cost_list if 'total_search_cost_list' in locals() else [],
            'avg_search_cost': avg_search_cost_list if 'avg_search_cost_list' in locals() else [],
            'firm_prices': firm_prices_list if 'firm_prices_list' in locals() else [],
            'round_details': round_details_list if 'round_details_list' in locals() else []
        }


def run_multiple_experiments(num_experiments=6, clean_start=False, start_firm_num=1, force_fresh=False, **kwargs):
    """运行多次实验并保存可视化结果"""
    
    matplotlib.use('Agg', force=True)
    plt.ioff() 

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'DejaVu Sans',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 300
    })
    
    def generate_rational_title_suffix(**params):
        """生成基于rational参数的标题后缀"""
        rational_share = params.get('rational_share', False)
        rational_search = params.get('rational_search', False) 
        rational_price = params.get('rational_price', False)
        
        has_any_rational = rational_share or rational_search or rational_price
        
        if not has_any_rational:
            return " - LLM Simulation"
        
        rational_parts = []
        if rational_share:
            rational_parts.append("Share")
        if rational_search:
            rational_parts.append("Search")
        if rational_price:
            rational_parts.append("Price")
        
        if rational_parts:
            return f" - Rational: {', '.join(rational_parts)}"
        else:
            return " - LLM Simulation"
    
    original_stdout_f = sys.stdout  
    log_file_path = None
    log_file_handle = None

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pid = os.getpid()
        base_results_dir = 'recommendation_experiment_results'
        
        rational_tag = "rational" if any([kwargs.get('rational_share', False),
                                        kwargs.get('rational_search', False),
                                        kwargs.get('rational_price', False)]) else "llm"
        
        run_dir_name = f"run_{timestamp}_pid{pid}_{rational_tag}_startfirm{start_firm_num}_numexp{num_experiments}_simplified"
        run_results_dir = os.path.join(base_results_dir, run_dir_name)
        checkpoint_dir = os.path.join(run_results_dir, 'checkpoints')

        print(f"DEBUG: Base results directory: {base_results_dir}")
        print(f"DEBUG: Process ID (PID): {pid}")
        print(f"DEBUG: Rational mode: {rational_tag}")
        print(f"DEBUG: Results for this run will be saved in: {run_results_dir}")

        if clean_start and os.path.exists(base_results_dir):
            print(f"--clean_start specified. Deleting base results directory: {base_results_dir}")
            try:
                shutil.rmtree(base_results_dir)
                print(f"Successfully deleted directory: {base_results_dir}")
            except Exception as e:
                print(f"Error deleting base directory: {str(e)}")

        if not os.path.exists(run_results_dir):
            os.makedirs(run_results_dir)
            print(f"DEBUG: Created run results directory: {run_results_dir}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"DEBUG: Created checkpoints directory: {checkpoint_dir}")

        log_file_path = os.path.join(run_results_dir, f"run_{timestamp}_pid{pid}_{rational_tag}.log")
        print(f"DEBUG: Print output will be logged to: {log_file_path}",
              file=original_stdout_f)  

        log_file_handle = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = log_file_handle

        print(f"Starting run_multiple_experiments with following base config:")
        print(f"num_experiments={num_experiments}, start_firm_num={start_firm_num}")
        print(f"clean_start={clean_start}, force_fresh={force_fresh}")
        print(f"Base kwargs: {kwargs}")
        print(f"Results Dir: {run_results_dir}")
        print("---")

        all_results = []
        checkpoint_file = os.path.join(checkpoint_dir, 'experiment_checkpoint.pkl')
        experiment_offset = 0  

        if os.path.exists(checkpoint_file) and not clean_start and not force_fresh:
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    all_results = checkpoint_data['results']
                    experiment_offset = checkpoint_data.get('next_exp_offset', 0)
                    print(f"Resuming from checkpoint. Already completed {experiment_offset} experiments in this run.")
                    print(f"DEBUG: Loaded {len(all_results)} results from previous runs/checkpoints.")
            except Exception as e:
                print(f"Error reading checkpoint file '{checkpoint_file}': {str(e)}. Starting fresh for this run.")
                all_results = []
                experiment_offset = 0
        else:
            if force_fresh:
                print("--force-fresh specified. Ignoring any existing checkpoint and starting fresh for this run.")
            elif not clean_start:
                print(f"No checkpoint found at '{checkpoint_file}' or --force-fresh specified. Starting fresh for this run.")
            all_results = []
            experiment_offset = 0

        metrics = ['share_ratio', 'consumer_surplus', 'firm_surplus', 'avg_search_cost', 'avg_price']
        titles = ['Share Ratio', 'Consumer Surplus', 'Firm Surplus', 'Average Search Cost', 'Average Price']

        # 创建跨实验数据的CSV文件
        inter_exp_csv_path = os.path.join(run_results_dir, f'inter_experiment_avg_data_{rational_tag}.csv')
        with open(inter_exp_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            rational_info = generate_rational_title_suffix(**kwargs).strip(' -')
            writer.writerow(['# Experiment Configuration:'] + [rational_info])
            writer.writerow(['Experiment', 'Firm_Num'] + metrics)

        # 创建完整的实验结果JSON文件（包含所有轮次的详细数据）
        all_results_json_path = os.path.join(run_results_dir, f'all_results_{rational_tag}.json')

        def save_checkpoint(current_offset, results):
            try:
                checkpoint_data = {
                    'results': results,
                    'next_exp_offset': current_offset + 1  
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

                json_friendly_results = []
                for result in results:
                    json_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            json_result[k] = v.tolist()
                        elif isinstance(v, list) and all(isinstance(item, (int, float, np.number)) for item in v):
                            json_result[k] = [float(item) for item in v]
                        else:
                            json_result[k] = v
                    json_friendly_results.append(json_result)

                json_results_path = os.path.join(run_results_dir, 'experiment_results.json')
                with open(json_results_path, 'w', encoding='utf-8') as f:
                    json.dump(json_friendly_results, f, indent=2, ensure_ascii=False)
                
                # 保存完整的all_results JSON文件（包含所有轮次的详细数据）
                all_results_json_data = []
                for res in results:
                    # 为每个实验构建完整的JSON结构
                    exp_json = {
                        'scenario': 'A',
                        'model_name': kwargs.get('model_config_name', ''),
                        'result': {
                            'share_ratio': res.get('share_ratio', []),
                            'consumer_surplus': res.get('consumer_surplus', []),
                            'firm_surplus': res.get('firm_surplus', []),
                            'total_search_cost': res.get('total_search_cost', []),
                            'avg_search_cost': res.get('avg_search_cost', []),
                            'firm_prices': res.get('firm_prices', []),
                            'avg_price': res.get('avg_price', []),
                            'consumer_privacy_choice': {},
                            'consumer_reason': {},
                            'firm_reason': {}
                        },
                        '_exp_params': res.get('_exp_params', {}),
                        'round_details': res.get('round_details', [])  # 保存所有轮次的详细数据
                    }
                    
                    # 合并所有轮次的数据（取最后一轮作为代表性数据，用于兼容参考格式）
                    if 'round_details' in res and res['round_details']:
                        last_round = res['round_details'][-1]
                        exp_json['result']['consumer_privacy_choice'] = last_round.get('consumer_privacy_choice', {})
                        exp_json['result']['consumer_reason'] = last_round.get('consumer_reason', {})
                        exp_json['result']['firm_reason'] = last_round.get('firm_reason', {})
                    
                    all_results_json_data.append(exp_json)
                
                # 保存all_results JSON文件
                with open(all_results_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results_json_data, f, indent=2, ensure_ascii=False)

                print(f"Checkpoint saved for experiment offset {current_offset} in '{checkpoint_dir}'")
                print(f"JSON results updated at '{json_results_path}'")
                print(f"All results JSON updated at '{all_results_json_path}'")
            except Exception as e:
                print(f"Error saving checkpoint or JSON results: {str(e)}")
                import traceback
                traceback.print_exc()

        for exp_idx in range(experiment_offset, num_experiments):
            current_firm_num = start_firm_num + exp_idx
            current_run_params = kwargs.copy()
            current_run_params['firm_num'] = current_firm_num
            current_run_params['firm_cost'] = kwargs.get('firm_cost', 0.0)
            current_run_params['force_fresh'] = force_fresh
            current_run_params['rational_share'] = kwargs.get('rational_share', False)
            current_run_params['rational_search'] = kwargs.get('rational_search', False) 
            current_run_params['rational_price'] = kwargs.get('rational_price', False)
            current_run_params['record_detailed_data'] = kwargs.get('record_detailed_data', False)
            
            # 只有在启用详细数据记录时才添加数据记录器参数
            if kwargs.get('record_detailed_data', False):
                current_run_params['data_recorder_params'] = {
                    'base_dir': run_results_dir,
                    'exp_idx': exp_idx,
                    'rational_tag': rational_tag
                }

            print(f"\nRunning Experiment {exp_idx + 1}/{num_experiments} (firm_num={current_firm_num})")
            print(f"Passing params to main: {current_run_params}")

            try:
                result = main(**current_run_params)

                if not isinstance(result, dict):
                    print(f"Warning: Experiment {exp_idx + 1} returned invalid result type: {type(result)}")
                    continue

                avg_prices_this_exp = []
                if 'firm_prices' in result and result['firm_prices']:
                    for round_prices in result['firm_prices']:
                        if round_prices:  
                            avg_prices_this_exp.append(round(sum(round_prices) / len(round_prices), 4))
                        else:
                            avg_prices_this_exp.append(0.0)  
                result['avg_price'] = avg_prices_this_exp

                for metric in metrics:
                    if metric in result and isinstance(result[metric], list):
                        try:
                            if metric == 'avg_price':
                                result[metric] = [round(float(x), 4) if isinstance(x, (int, float, np.number)) else x for x in result[metric]]
                            else:
                                result[metric] = [round(float(x), 4) if isinstance(x, (int, float, np.number)) else x for x in result[metric]]
                        except (TypeError, ValueError) as round_err:
                            print(f"Warning: Could not round metric '{metric}' for exp {exp_idx + 1}. Data: {result[metric]}. Error: {round_err}")

                result['_exp_params'] = {'experiment_index': exp_idx + 1, 'firm_num': current_firm_num}
                all_results.append(result)

                save_checkpoint(exp_idx, all_results)

                print(f"Experiment {exp_idx + 1} Results Summary:")
                for metric in metrics:
                    if metric in result and result[metric]:
                        try:
                            if metric == 'avg_price':
                                avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                print(f"  Avg {metric}: {avg_value:.4f}")
                            else:
                                avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                            print(f"  Avg {metric}: {avg_value}")
                        except Exception as mean_err:
                            print(f"  Warning: Could not calculate mean for {metric}. Data: {result[metric]}. Error: {mean_err}")
                    else:
                        print(f"  Metric {metric}: No data or empty list.")

                intra_exp_csv_path = os.path.join(run_results_dir, f'experiment_{exp_idx + 1}_firms_{current_firm_num}_{rational_tag}_data.csv')
                with open(intra_exp_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    rational_info = generate_rational_title_suffix(**current_run_params).strip(' -')
                    writer.writerow(['# Experiment Configuration:'] + [rational_info])
                    writer.writerow(['# Firm Count:'] + [current_firm_num])
                    writer.writerow(['Round'] + metrics)  
                    
                    max_rounds = 0
                    for metric in metrics:
                        if metric in result and result[metric]:
                            max_rounds = max(max_rounds, len(result[metric]))
                    
                    for round_idx in range(max_rounds):
                        row = [round_idx + 1]  
                        for metric in metrics:
                            if metric in result and result[metric] and round_idx < len(result[metric]):
                                row.append(result[metric][round_idx])
                            else:
                                row.append('')  
                        writer.writerow(row)
                
                print(f"Experiment {exp_idx + 1} data saved to CSV: {intra_exp_csv_path}")

                with open(inter_exp_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [exp_idx + 1, current_firm_num]
                    
                    for metric in metrics:
                        if metric in result and result[metric]:
                            try:
                                if metric == 'avg_price':
                                    avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                    row.append(avg_value)
                                else:
                                    avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                    row.append(avg_value)
                            except Exception:
                                row.append('')  
                        else:
                            row.append('')  
                    
                    writer.writerow(row)

                if kwargs.get('visualize', False):
                    title_suffix = generate_rational_title_suffix(**current_run_params)
                    
                    fig_intra, axs_intra = plt.subplots(2, 3, figsize=(18, 12))
                    fig_intra.suptitle(f'Experiment {exp_idx + 1} Results (Firm Num: {current_firm_num}){title_suffix}', 
                                     fontsize=16, fontweight='bold')
                    axs_intra = axs_intra.flatten()

                    plot_metrics_intra = metrics  
                    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']  
                    
                    for j, metric in enumerate(plot_metrics_intra):
                        ax = axs_intra[j]
                        if metric in result and result[metric]:
                            numeric_data = [x for x in result[metric] if isinstance(x, (int, float, np.number))]
                            if numeric_data:
                                ax.plot(range(1, len(numeric_data) + 1), numeric_data, 
                                       color=colors[j], linewidth=2.5, marker='o', markersize=6, 
                                       markerfacecolor='white', markeredgewidth=2, alpha=0.8)
                            else:
                                ax.text(0.5, 0.5, 'No numeric data', horizontalalignment='center',
                                        verticalalignment='center', transform=ax.transAxes, 
                                        fontsize=12, color='red')
                        else:
                            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                                    transform=ax.transAxes, fontsize=12, color='red')
                        
                        ax.set_title(titles[j], fontsize=14, fontweight='bold', pad=15)
                        ax.set_xlabel('Round', fontsize=12)
                        ax.set_ylabel('Value', fontsize=12)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(1.5)
                        ax.spines['bottom'].set_linewidth(1.5)

                    if len(plot_metrics_intra) == 5:
                        axs_intra[-1].set_visible(False)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
                    
                    intra_plot_path = os.path.join(run_results_dir,
                                                   f'experiment_{exp_idx + 1}_firms_{current_firm_num}_{rational_tag}.png')
                    plt.savefig(intra_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_intra)
                    print(f"Intra-experiment plot saved to: {intra_plot_path}")

                if kwargs.get('visualize', False):
                    title_suffix = generate_rational_title_suffix(**current_run_params)
                    
                    fig_inter, axs_inter = plt.subplots(2, 3, figsize=(18, 12))
                    fig_inter.suptitle(f'Average Results Across Experiments (Up to Exp {exp_idx + 1}){title_suffix}', 
                                      fontsize=16, fontweight='bold')
                    axs_inter = axs_inter.flatten()

                    plot_metrics_inter = metrics
                    firm_nums_completed = [r['_exp_params']['firm_num'] for r in all_results if '_exp_params' in r]
                    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']  

                    for j, metric in enumerate(plot_metrics_inter):
                        ax = axs_inter[j]
                        avg_values = []
                        valid_firm_nums = []
                        for k, res in enumerate(all_results):
                            if metric in res and res[metric]:
                                try:
                                    numeric_metric_data = [x for x in res[metric] if
                                                           isinstance(x, (int, float, np.number))]
                                    if numeric_metric_data:
                                        avg_values.append(round(np.mean(numeric_metric_data), 4))
                                        if '_exp_params' in res:
                                            valid_firm_nums.append(res['_exp_params']['firm_num'])
                                except Exception as avg_err:
                                    print(f"Warning: Could not calculate average for {metric} in experiment {k + 1}. Error: {avg_err}")

                        if avg_values and valid_firm_nums:
                            ax.plot(valid_firm_nums, avg_values, color=colors[j], linewidth=3, 
                                   marker='o', markersize=8, markerfacecolor='white', 
                                   markeredgewidth=2.5, alpha=0.9, label=titles[j])
                        else:
                            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center',
                                    verticalalignment='center', transform=ax.transAxes,
                                    fontsize=12, color='red')
                        
                        ax.set_title(titles[j], fontsize=14, fontweight='bold', pad=15)
                        ax.set_xlabel('Number of Firms', fontsize=12)
                        ax.set_ylabel('Average Value', fontsize=12)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(1.5)
                        ax.spines['bottom'].set_linewidth(1.5)
                        
                        if valid_firm_nums:
                            ax.set_xticks(range(min(valid_firm_nums), max(valid_firm_nums) + 1))

                    if len(plot_metrics_inter) == 5:
                        axs_inter[-1].set_visible(False)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    inter_plot_path = os.path.join(run_results_dir, f'average_results_up_to_exp_{exp_idx + 1}_{rational_tag}.png')
                    plt.savefig(inter_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_inter)
                    print(f"Inter-experiment average plot saved to: {inter_plot_path}")

            except Exception as e:
                print(f"Experiment {exp_idx + 1} failed: {str(e)}")
                print("\nDetailed error trace:")
                import traceback
                traceback.print_exc()

                error_file = os.path.join(checkpoint_dir, f'error_experiment_{exp_idx + 1}_{rational_tag}.txt')
                with open(error_file, 'w') as f:
                    f.write(f"Experiment {exp_idx + 1} (firm_num={current_firm_num}) Error: {str(e)}\n")
                    f.write(f"Rational Mode: {generate_rational_title_suffix(**current_run_params).strip(' -')}\n\n")
                    traceback.print_exc(file=f)

                save_checkpoint(exp_idx, all_results)

        # 最终计算平均结果
        avg_results = {}
        if not all_results:
            print("No experiments completed successfully.")
            return avg_results

        for metric in metrics:
            try:
                metric_data_all_exp = []
                for r in all_results:
                    if metric in r and isinstance(r[metric], list):
                        numeric_data = [x for x in r[metric] if isinstance(x, (int, float, np.number))]
                        if numeric_data:
                            metric_data_all_exp.append(np.mean(numeric_data))
                        else:
                            metric_data_all_exp.append(np.nan)

                if not any(np.isnan(x) for x in metric_data_all_exp):
                    if metric == 'avg_price':
                        avg_results[metric] = [round(float(x), 4) for x in metric_data_all_exp]
                        overall_avg = round(np.nanmean(metric_data_all_exp), 4)
                        print(f"Overall Average for {metric}: {overall_avg:.4f}")
                        avg_results[f"{metric}_overall_avg"] = overall_avg
                    else:
                        avg_results[metric] = [round(float(x), 4) for x in metric_data_all_exp]
                        overall_avg = round(np.nanmean(metric_data_all_exp), 4)
                    print(f"Overall Average for {metric}: {overall_avg}")
                    avg_results[f"{metric}_overall_avg"] = overall_avg
                else:
                    print(f"Could not compute overall average for {metric} due to missing data.")
                    avg_results[metric] = []
            except Exception as e:
                print(f"Error calculating final average for {metric}: {str(e)}")
                avg_results[metric] = []

        print(f"\nAll {num_experiments} experiments finished. Results saved in: {run_results_dir}")
        
        # 创建实验总结文件
        summary_path = os.path.join(run_results_dir, f'experiment_summary_{rational_tag}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"- Decision Mode: {generate_rational_title_suffix(**kwargs).strip(' -')}\n")
            f.write(f"- Number of Experiments: {num_experiments}\n")
            f.write(f"- Firm Range: {start_firm_num} to {start_firm_num + num_experiments - 1}\n")
            f.write(f"- Consumer Count: {kwargs.get('consumer_num', 'N/A')}\n")
            f.write(f"- Rounds per Experiment: {kwargs.get('num_rounds', 'N/A')}\n")
            f.write(f"- Search Cost: {kwargs.get('search_cost', 'N/A')}\n\n")
            
            f.write("RESULTS OVERVIEW:\n")
            for metric in metrics:
                if metric in avg_results and f"{metric}_overall_avg" in avg_results:
                    if metric == 'avg_price':
                        f.write(f"- Average {metric.replace('_', ' ').title()}: {avg_results[f'{metric}_overall_avg']:.4f}\n")
                    else:
                        f.write(f"- Average {metric.replace('_', ' ').title()}: {avg_results[f'{metric}_overall_avg']}\n")
            
            f.write(f"\nFiles Generated:\n")
            f.write(f"- Raw Data: inter_experiment_avg_data_{rational_tag}.csv\n")
            f.write(f"- Visualizations: *_{rational_tag}.png\n")
            f.write(f"- Individual Data: experiment_*_{rational_tag}_data.csv\n")
            f.write(f"- Log File: run_*_{rational_tag}.log\n")
            
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Process ID: {pid}\n")
        
        print(f"Experiment summary saved to: {summary_path}")
        return avg_results
    except Exception as e:
        print(f"\nFATAL ERROR in run_multiple_experiments: {str(e)}", file=original_stdout_f)
        import traceback
        traceback.print_exc(file=original_stdout_f)
        if log_file_handle and not log_file_handle.closed:
            print(f"\nFATAL ERROR in run_multiple_experiments: {str(e)}", file=log_file_handle)
            traceback.print_exc(file=log_file_handle)

    finally:
        log_file_closed_properly = False
        if log_file_handle and not log_file_handle.closed:
            if sys.stdout == log_file_handle:
                sys.stdout = original_stdout_f
                print(f"Finished logging. Output redirected back to console.", file=original_stdout_f)
            else:
                print(f"Log file handle existed but stdout wasn't redirected to it upon finally. Closing file.",
                      file=original_stdout_f)
            log_file_handle.close()
            log_file_closed_properly = True
        elif log_file_path:
            print(f"Log file handle was not properly created or opened for {log_file_path}", file=original_stdout_f)
        else:
            print(f"Logging was not initiated.", file=original_stdout_f)

        try:
            plt.close('all')
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()
        except Exception as e:
            print(f"matplotlib清理时出现错误: {e}", file=original_stdout_f)

        # 清理日志文件
        if log_file_closed_properly and log_file_path and os.path.exists(log_file_path):
            print(f"清理日志文件: {log_file_path}", file=original_stdout_f)
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                filtered_lines = []
                for line in lines:
                    if 'INFO' not in line and 'WARNING' not in line:
                        filtered_lines.append(line)
                
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(filtered_lines)
                
                original_log_path = log_file_path + '.original'
                with open(original_log_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print(f"日志清理完成。原始日志备份至: {original_log_path}", file=original_stdout_f)
                print(f"过滤前行数: {len(lines)}, 过滤后行数: {len(filtered_lines)}", file=original_stdout_f)
            except Exception as e:
                print(f"日志清理失败: {str(e)}", file=original_stdout_f)


if __name__ == "__main__":
    args = parse_args()

    # 设置不同步骤使用的模型（所有主体使用相同配置）
    model_names = {
        'decide_share': 'gpt-config',
        'decide_search': 'gpt-config',
        'decide_purchase': 'gpt-config',
        'set_price': 'gpt-config',
        'set_personalized_price': 'gpt-config',
    }

    fallback_model = args.model_config_name

    print(f"\n### 使用模型配置: {model_names} (备用模型: {fallback_model}) ###\n")

    if args.clean_start:
        print("将删除之前的所有实验记录，从头开始实验...")

    print("开启详细调试模式，显示所有错误和警告...")

    try:
        experiment_params = {
            "consumer_num": args.consumer_num,
            "start_firm_num": args.firm_num,
            "search_cost": args.search_cost,
            "agent_type": args.agent_type,
            "use_dist": args.use_dist,
            "num_rounds": args.num_rounds,
            "visualize": args.visualize,
            "threads": args.threads,
            "memory_truncate": args.memory_truncate,
            "model_config_name": fallback_model,
            "model_names": model_names,
            "force_fresh": args.force_fresh,
            "firm_cost": args.firm_cost,
            "method": args.pricing_mode,
            "rational_share": args.rational_share,
            "rational_search": args.rational_search,
            "rational_price": args.rational_price,
            "record_detailed_data": args.record_detailed_data if hasattr(args, 'record_detailed_data') else True,
        }

        run_multiple_experiments(
            num_experiments=args.num_experiments,
            clean_start=args.clean_start,
            **experiment_params
        )
    except KeyboardInterrupt:
        print("\n程序被用户中断。已保存检查点，可以稍后恢复。")
    except Exception as e:
        print(f"\n程序遇到异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束。")
        plt.close('all')
