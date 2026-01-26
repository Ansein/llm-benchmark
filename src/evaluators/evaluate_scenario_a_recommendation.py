"""
场景A推荐系统的LLM评估器
评估LLM在个性化推荐与隐私选择场景下的决策能力

基于agents_complete.py和rec_simplified.py重构
支持LLM模式和理性模式对比
"""

import json
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import csv

# 添加项目根目录到路径
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.evaluators.llm_client import LLMClient, create_llm_client
from src.scenarios.scenario_a_recommendation import (
    ScenarioARecommendationParams,
    calculate_delta_sharing,
    rational_share_decision,
    optimize_firm_price
)


class ScenarioARecommendationEvaluator:
    """场景A推荐系统评估器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        ground_truth_path: str = "data/ground_truth/scenario_a_recommendation_result.json"
    ):
        """
        初始化评估器
        
        Args:
            llm_client: LLM客户端
            ground_truth_path: ground truth文件路径
        """
        self.llm_client = llm_client
        self.ground_truth_path = ground_truth_path
        
        # 加载ground truth
        if os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                self.gt_data = json.load(f)
            
            self.params = ScenarioARecommendationParams(**self.gt_data["params"])
            self.gt_numeric = self.gt_data["gt_numeric"]
            self.gt_labels = self.gt_data["gt_labels"]
        else:
            print(f"⚠️  Ground truth文件未找到: {ground_truth_path}")
            print("    将使用默认参数初始化")
            self.params = None
            self.gt_numeric = None
            self.gt_labels = None
    
    # ============================================================================
    # 提示词构建
    # ============================================================================
    
    def build_system_prompt_consumer(self) -> str:
        """构建消费者的系统提示"""
        return """你是一个理性的消费者，目标是最大化你的效用。
你必须输出严格JSON格式，不要包含任何额外的文本。"""
    
    def build_system_prompt_firm(self) -> str:
        """构建企业的系统提示"""
        return """你是一个理性的企业，目标是最大化你的利润。
你必须输出严格JSON格式，不要包含任何额外的文本。"""
    
    def build_share_decision_prompt(self, consumer_id: int, share_rate_estimate: float = 0.5) -> str:
        """
        构建数据分享决策提示
        
        Args:
            consumer_id: 消费者ID
            share_rate_estimate: 预估的数据分享率
        """
        τ_i = self.params.privacy_costs[consumer_id]
        s = self.params.search_cost
        n = self.params.n_firms
        r = self.params.r_value
        v_low = self.params.v_dist['low']
        v_high = self.params.v_dist['high']
        
        prompt = f"""
# 场景：个性化推荐与隐私选择

你是消费者 {consumer_id}，正在考虑是否向平台分享你的数据。

## 市场环境
- 有 {n} 家企业提供产品
- 你对每家企业产品的估值在 [{v_low}, {v_high}] 之间，服从均匀分布
- 每次搜索一家企业需要成本 {s}（首次搜索免费）
- 你的保留效用（不购买的底线）：{r}

## 你的私有信息
- 你的隐私成本（分享数据带来的心理成本）：{τ_i:.4f}

## 决策选项

### 选项1：分享数据
**好处**：
- 平台会根据你的偏好推荐产品（从最适合你的到最不适合的排序）
- 显著减少搜索成本（因为你可以按推荐顺序搜索，更快找到满意的产品）
- 预期搜索次数：1-2次

**成本**：
- 隐私成本：{τ_i:.4f}
- 仍需支付搜索成本（每次 {s}）

### 选项2：不分享数据
**好处**：
- 无隐私成本

**成本**：
- 需要随机搜索企业（效率低）
- 预期搜索次数：2-3次
- 更高的总搜索成本

## 市场信息
- 预估约有 {share_rate_estimate:.0%} 的消费者选择分享数据

## 决策框架
你需要权衡：
1. 分享数据带来的搜索成本节省
2. 推荐系统带来的效用提升
3. 隐私成本

**关键洞察**：
- 推荐系统能显著提高匹配效率（找到高估值产品的概率更高）
- 搜索成本节省 ≈ {s} × (随机搜索次数 - 推荐搜索次数)

请输出你的决策，格式为JSON：
{{
  "share": 0 或 1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过100字）"
}}
"""
        return prompt
    
    def build_price_decision_prompt(
        self,
        firm_id: int,
        share_rate: float,
        market_prices: List[float]
    ) -> str:
        """
        构建企业定价决策提示
        
        Args:
            firm_id: 企业ID
            share_rate: 实际数据分享率
            market_prices: 其他企业的价格
        """
        n = self.params.n_firms
        c = self.params.firm_cost
        r = self.params.r_value
        avg_market_price = np.mean([p for i, p in enumerate(market_prices) if i != firm_id]) if len(market_prices) > 1 else 0.5
        
        prompt = f"""
# 场景：企业定价决策

你是企业 {firm_id}，需要设定产品价格。

## 市场环境
- 总共有 {n} 家企业竞争
- 你的边际成本：{c}
- 消费者的保留效用：{r}
- 其他企业的平均价格：{avg_market_price:.4f}

## 消费者行为
当前市场中：
- **{share_rate:.0%}** 的消费者分享了数据，他们会按推荐顺序搜索
- **{1-share_rate:.0%}** 的消费者未分享数据，他们会随机搜索

**分享数据的消费者**：
- 按推荐顺序搜索（从最匹配到最不匹配）
- 如果你的价格合适，他们有 1/{n} 的机会搜索到你

**未分享数据的消费者**：
- 随机搜索企业
- 购买决策取决于价格和估值的比较

## 定价策略
你需要考虑：
1. **边际成本**：价格必须高于 {c} 才能盈利
2. **竞争压力**：其他企业平均定价 {avg_market_price:.4f}
3. **需求弹性**：价格越高，需求越少
4. **数据分享率**：{share_rate:.0%} 的消费者行为更可预测

**关键洞察**：
- 价格过高会失去需求
- 价格过低会损失利润
- 最优价格在 [{c}, {r}] 区间内

请输出你的定价决策，格式为JSON：
{{
  "price": float（你的定价，建议在 {c} 到 {r} 之间），
  "reason": "简要说明你的定价理由（不超过100字）"
}}
"""
        return prompt
    
    # ============================================================================
    # LLM决策查询
    # ============================================================================
    
    def query_llm_share_decision(
        self,
        consumer_id: int,
        share_rate_estimate: float = 0.5,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        查询LLM的数据分享决策
        
        Returns:
            {"share": 0/1, "reason": str}
        """
        prompt = self.build_share_decision_prompt(consumer_id, share_rate_estimate)
        
        decisions = []
        reasons = []
        
        for trial in range(num_trials):
            try:
                response = self.llm_client.generate_json([
                    {"role": "system", "content": self.build_system_prompt_consumer()},
                    {"role": "user", "content": prompt}
                ])
                
                share = int(response.get("share", 0))
                if share not in [0, 1]:
                    share = 0
                
                decisions.append(share)
                reasons.append(response.get("reason", ""))
                
            except Exception as e:
                print(f"  ⚠️  消费者{consumer_id} 试验{trial+1}失败: {e}")
                decisions.append(0)
                reasons.append("")
        
        final_share = 1 if sum(decisions) > len(decisions) / 2 else 0
        final_reason = reasons[0] if reasons else ""
        
        return {
            "share": final_share,
            "reason": final_reason
        }
    
    def query_llm_price_decision(
        self,
        firm_id: int,
        share_rate: float,
        market_prices: List[float],
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        查询LLM的定价决策
        
        Returns:
            {"price": float, "reason": str}
        """
        prompt = self.build_price_decision_prompt(firm_id, share_rate, market_prices)
        
        prices = []
        reasons = []
        
        for trial in range(num_trials):
            try:
                response = self.llm_client.generate_json([
                    {"role": "system", "content": self.build_system_prompt_firm()},
                    {"role": "user", "content": prompt}
                ])
                
                price = float(response.get("price", 0.5))
                # 限制价格在合理范围内
                price = max(self.params.firm_cost, min(self.params.r_value, price))
                
                prices.append(price)
                reasons.append(response.get("reason", ""))
                
            except Exception as e:
                print(f"  ⚠️  企业{firm_id} 试验{trial+1}失败: {e}")
                prices.append(0.5)
                reasons.append("")
        
        avg_price = np.mean(prices)
        final_reason = reasons[0] if reasons else ""
        
        return {
            "price": avg_price,
            "reason": final_reason
        }
    
    # ============================================================================
    # 模拟执行
    # ============================================================================
    
    def simulate_single_round(
        self,
        rational_share: bool = False,
        rational_price: bool = False,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        模拟单轮决策
        
        Args:
            rational_share: 是否使用理性分享决策
            rational_price: 是否使用理性定价决策
            num_trials: LLM查询重复次数
        
        Returns:
            单轮结果字典
        """
        print(f"\n{'='*60}")
        print(f"[模拟单轮] Rational Share: {rational_share}, Rational Price: {rational_price}")
        print(f"{'='*60}")
        
        # ===== 步骤1: 消费者分享决策 =====
        print("\n[步骤1] 消费者分享决策...")
        
        share_decisions = []
        share_reasons = []
        
        if rational_share:
            # 理性模式：基于贝叶斯纳什均衡
            delta = calculate_delta_sharing(
                self.params.v_dist,
                self.params.r_value,
                self.params.n_firms
            )
            
            for i in range(self.params.n_consumers):
                τ_i = self.params.privacy_costs[i]
                should_share = rational_share_decision(
                    τ_i, delta, self.params.search_cost
                )
                share_decisions.append(int(should_share))
                share_reasons.append(f"理性决策：Delta={delta:.4f}, τ={τ_i:.4f}, s={self.params.search_cost}")
        else:
            # LLM模式
            for i in range(self.params.n_consumers):
                result = self.query_llm_share_decision(i, 0.5, num_trials)
                share_decisions.append(result["share"])
                share_reasons.append(result["reason"])
        
        share_rate = np.mean(share_decisions)
        print(f"分享率：{share_rate:.2%} ({sum(share_decisions)}/{self.params.n_consumers})")
        
        # ===== 步骤2: 企业定价决策 =====
        print("\n[步骤2] 企业定价决策...")
        
        prices = []
        price_reasons = []
        
        if rational_price:
            # 理性模式：价格均衡迭代
            initial_price = max(0.1, self.params.r_value - 0.3)
            current_prices = [initial_price] * self.params.n_firms
            
            max_iter = 30
            tol = 1e-6
            
            for iter_p in range(max_iter):
                market_price = np.mean(current_prices)
                new_prices = []
                
                for firm_id in range(self.params.n_firms):
                    optimal_p = optimize_firm_price(
                        share_rate=share_rate,
                        n_firms=self.params.n_firms,
                        market_price=market_price,
                        v_dist=self.params.v_dist,
                        r_value=self.params.r_value,
                        firm_cost=self.params.firm_cost
                    )
                    new_prices.append(optimal_p)
                
                price_diff = np.max(np.abs(np.array(new_prices) - np.array(current_prices)))
                
                if price_diff < tol:
                    print(f"  价格收敛于第 {iter_p + 1} 次迭代")
                    break
                
                current_prices = new_prices
            
            prices = current_prices
            price_reasons = [f"理性均衡价格: {p:.4f}" for p in prices]
        else:
            # LLM模式
            for firm_id in range(self.params.n_firms):
                result = self.query_llm_price_decision(
                    firm_id, share_rate, prices, num_trials
                )
                prices.append(result["price"])
                price_reasons.append(result["reason"])
        
        avg_price = np.mean(prices)
        print(f"平均价格：{avg_price:.4f}")
        print(f"价格范围：[{min(prices):.4f}, {max(prices):.4f}]")
        
        # ===== 步骤3: 计算市场结果 =====
        print("\n[步骤3] 计算市场结果...")
        
        # 简化的消费者剩余计算
        consumer_surplus = 0.0
        for i in range(self.params.n_consumers):
            if share_decisions[i] == 1:
                # 分享数据：获得推荐，支付隐私成本
                delta = calculate_delta_sharing(
                    self.params.v_dist,
                    self.params.r_value,
                    self.params.n_firms
                )
                u_i = delta - self.params.privacy_costs[i] - self.params.search_cost
            else:
                # 未分享：随机搜索
                u_i = max(0, self.params.r_value - avg_price) - self.params.search_cost * 2
            
            consumer_surplus += u_i
        
        # 简化的企业利润计算
        avg_demand_per_firm = (share_rate * 0.8 + (1 - share_rate) * 0.5) / self.params.n_firms
        firm_profit = sum((p - self.params.firm_cost) * avg_demand_per_firm for p in prices)
        
        social_welfare = consumer_surplus + firm_profit
        
        print(f"消费者剩余：{consumer_surplus:.4f}")
        print(f"企业利润：{firm_profit:.4f}")
        print(f"社会福利：{social_welfare:.4f}")
        
        return {
            "share_rate": share_rate,
            "share_decisions": share_decisions,
            "share_reasons": share_reasons,
            "prices": prices,
            "avg_price": avg_price,
            "price_reasons": price_reasons,
            "consumer_surplus": consumer_surplus,
            "firm_profit": firm_profit,
            "social_welfare": social_welfare
        }
    
    def run_evaluation(
        self,
        num_rounds: int = 5,
        rational_share: bool = False,
        rational_price: bool = False,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        运行完整评估（多轮模拟）
        
        Args:
            num_rounds: 模拟轮数
            rational_share: 是否使用理性分享决策
            rational_price: 是否使用理性定价决策
            num_trials: LLM查询重复次数
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"[场景A推荐系统评估] 模型: {self.llm_client.config_name}")
        print(f"{'='*60}")
        print(f"参数:")
        print(f"  轮数: {num_rounds}")
        print(f"  理性分享: {rational_share}")
        print(f"  理性定价: {rational_price}")
        print(f"  LLM trials: {num_trials}")
        
        # 存储所有轮次的结果
        all_rounds = []
        
        share_rates = []
        avg_prices = []
        consumer_surpluses = []
        firm_profits = []
        social_welfares = []
        
        for round_num in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"轮次 {round_num + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            round_result = self.simulate_single_round(
                rational_share=rational_share,
                rational_price=rational_price,
                num_trials=num_trials
            )
            
            all_rounds.append(round_result)
            
            share_rates.append(round_result["share_rate"])
            avg_prices.append(round_result["avg_price"])
            consumer_surpluses.append(round_result["consumer_surplus"])
            firm_profits.append(round_result["firm_profit"])
            social_welfares.append(round_result["social_welfare"])
        
        # 计算平均结果
        avg_results = {
            "avg_share_rate": np.mean(share_rates),
            "avg_price": np.mean(avg_prices),
            "avg_consumer_surplus": np.mean(consumer_surpluses),
            "avg_firm_profit": np.mean(firm_profits),
            "avg_social_welfare": np.mean(social_welfares),
            "std_share_rate": np.std(share_rates),
            "std_price": np.std(avg_prices)
        }
        
        # 与ground truth比较（如果存在）
        comparison = {}
        if self.gt_numeric:
            comparison = {
                "share_rate_error": abs(avg_results["avg_share_rate"] - self.gt_numeric["eq_share_rate"]),
                "price_error": abs(avg_results["avg_price"] - self.gt_numeric["eq_avg_price"]),
                "welfare_error": abs(avg_results["avg_social_welfare"] - self.gt_numeric["eq_welfare"])
            }
        
        results = {
            "model_name": self.llm_client.config_name,
            "scenario": "A_recommendation",
            "num_rounds": num_rounds,
            "rational_share": rational_share,
            "rational_price": rational_price,
            "all_rounds": all_rounds,
            "share_rates": share_rates,
            "avg_prices": avg_prices,
            "consumer_surpluses": consumer_surpluses,
            "firm_profits": firm_profits,
            "social_welfares": social_welfares,
            "averages": avg_results,
            "ground_truth_comparison": comparison
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估结果摘要"""
        print(f"\n{'='*60}")
        print(f"[评估结果摘要]")
        print(f"{'='*60}")
        
        print(f"\n模型: {results['model_name']}")
        print(f"轮数: {results['num_rounds']}")
        print(f"决策模式: Share={'Rational' if results['rational_share'] else 'LLM'}, "
              f"Price={'Rational' if results['rational_price'] else 'LLM'}")
        
        print(f"\n【平均结果】")
        avg = results['averages']
        print(f"  分享率: {avg['avg_share_rate']:.2%} ± {avg['std_share_rate']:.3f}")
        print(f"  平均价格: {avg['avg_price']:.4f} ± {avg['std_price']:.4f}")
        print(f"  消费者剩余: {avg['avg_consumer_surplus']:.4f}")
        print(f"  企业利润: {avg['avg_firm_profit']:.4f}")
        print(f"  社会福利: {avg['avg_social_welfare']:.4f}")
        
        if results['ground_truth_comparison']:
            print(f"\n【与理论解偏差】")
            comp = results['ground_truth_comparison']
            print(f"  分享率偏差: {comp['share_rate_error']:.4f}")
            print(f"  价格偏差: {comp['price_error']:.4f}")
            print(f"  福利偏差: {comp['welfare_error']:.4f}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {output_path}")


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='场景A推荐系统评估器')
    parser.add_argument('--model', type=str, default='deepseek-v3.2', help='LLM模型名称')
    parser.add_argument('--rounds', type=int, default=5, help='模拟轮数')
    parser.add_argument('--rational-share', action='store_true', help='使用理性分享决策')
    parser.add_argument('--rational-price', action='store_true', help='使用理性定价决策')
    parser.add_argument('--num-trials', type=int, default=1, help='LLM查询重复次数')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/scenario_a',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建LLM客户端
    llm_client = create_llm_client(args.model)
    
    # 创建评估器
    evaluator = ScenarioARecommendationEvaluator(
        llm_client=llm_client,
        ground_truth_path="data/ground_truth/scenario_a_recommendation_result.json"
    )
    
    # 运行评估
    results = evaluator.run_evaluation(
        num_rounds=args.rounds,
        rational_share=args.rational_share,
        rational_price=args.rational_price,
        num_trials=args.num_trials
    )
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rational_tag = ""
    if args.rational_share:
        rational_tag += "_rational_share"
    if args.rational_price:
        rational_tag += "_rational_price"
    
    output_path = f"{args.output_dir}/eval_scenario_A_{llm_client.config_name}{rational_tag}_{timestamp}.json"
    evaluator.save_results(results, output_path)
    
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}")
