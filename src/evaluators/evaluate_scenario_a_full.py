"""
场景A推荐系统的完整LLM评估器
完整实现所有决策环节：分享决策、定价决策、搜索决策、购买决策

基于agents_complete.py和rec_simplified.py的完整重构
"""

import json
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re

# 添加项目根目录到路径
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.evaluators.llm_client import LLMClient, create_llm_client
from src.scenarios.scenario_a_recommendation import (
    ScenarioARecommendationParams,
    calculate_delta_sharing,
    rational_share_decision,
    optimize_firm_price,
    generate_recommendation_instance
)


class ConsumerState:
    """消费者状态（轻量级数据类）"""
    def __init__(self, index: int, privacy_cost: float, valuations: np.ndarray, search_cost: float, r_value: float):
        self.index = index
        self.privacy_cost = privacy_cost
        self.valuations = valuations  # 对各企业的估值向量
        self.search_cost = search_cost
        self.r_value = r_value
        
        # 决策状态
        self.share = False
        self.share_reason = ""
        self.searched_firms = []  # [{index, valuation, price}, ...]
        self.purchase_index = -1  # -1表示未购买
        self.total_search_cost = 0.0
        self.utility = 0.0
    
    def reset_for_round(self):
        """重置每轮状态"""
        self.searched_firms = []
        self.purchase_index = -1
        self.total_search_cost = 0.0
        self.utility = 0.0


class FirmState:
    """企业状态（轻量级数据类）"""
    def __init__(self, index: int, firm_cost: float):
        self.index = index
        self.firm_cost = firm_cost
        self.price = 0.0
        self.price_reason = ""
        self.sales_count = 0
        self.revenue = 0.0
        self.profit = 0.0
    
    def reset_for_round(self):
        """重置每轮状态"""
        self.sales_count = 0
        self.revenue = 0.0
        self.profit = 0.0


class ScenarioAFullEvaluator:
    """场景A推荐系统完整评估器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        params: ScenarioARecommendationParams
    ):
        """
        初始化评估器
        
        Args:
            llm_client: LLM客户端
            params: 场景参数
        """
        self.llm_client = llm_client
        self.params = params
        
        # 预计算Delta（推荐效用增益）
        self.delta = calculate_delta_sharing(
            params.v_dist,
            params.r_value,
            params.n_firms
        )
        print(f"预计算 Delta = {self.delta:.6f}")
    
    # ============================================================================
    # 提示词构建
    # ============================================================================
    
    def build_system_prompt_consumer(self) -> str:
        """消费者系统提示"""
        return """你是理性经济主体，目标是最大化你的效用。
你必须输出严格JSON格式，不要包含任何额外的文本。"""
    
    def build_system_prompt_firm(self) -> str:
        """企业系统提示"""
        return """你是理性企业，目标是最大化你的利润。
你必须输出严格JSON格式，不要包含任何额外的文本。"""
    
    def build_share_prompt(self, consumer_id: int, share_rate_estimate: float = 0.5) -> str:
        """构建数据分享决策提示"""
        τ = self.params.privacy_costs[consumer_id]
        s = self.params.search_cost
        n = self.params.n_firms
        r = self.params.r_value
        
        prompt = f"""
# 场景：个性化推荐与隐私选择

你是消费者 {consumer_id}，需要决定是否向平台分享你的偏好数据。

## 基本信息
- 市场中有 {n} 家企业
- 你对每家企业产品的估值在 [0, 1] 之间（你不知道具体值，但知道分布）
- 每次搜索企业需要成本 {s}（首次搜索免费，后续每次 {s}）
- 你的保留效用（不购买的底线）：{r}
- 你的隐私成本：{τ:.4f}

## 分享数据的影响

**如果分享数据**：
- **好处1**：平台会根据你的偏好推荐企业（从最适合到最不适合排序）
  这让你能更快找到高估值的产品，显著减少搜索次数
- **好处2**：推荐系统提高匹配效率，期望获得更高估值的产品
- **成本**：需要支付隐私成本 {τ:.4f}

**如果不分享数据**：
- **好处**：无隐私成本
- **成本**：需要随机搜索企业，搜索效率低，预期搜索次数更多

## 市场信息
预估约有 {share_rate_estimate:.0%} 的其他消费者选择分享数据。

## 决策框架
权衡：
- 推荐系统带来的效用提升（更高估值 + 搜索成本节省）
- vs 隐私成本 {τ:.4f}

请输出JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明理由（不超过100字）"
}}
"""
        return prompt
    
    def build_price_prompt(self, firm_id: int, share_rate: float, other_prices: List[float]) -> str:
        """构建企业定价提示"""
        c = self.params.firm_cost
        r = self.params.r_value
        avg_other_price = np.mean(other_prices) if other_prices else 0.5
        
        prompt = f"""
# 场景：企业定价决策

你是企业 {firm_id}，需要设定产品价格。

## 市场环境
- 总共 {self.params.n_firms} 家企业竞争
- 你的边际成本：{c}
- 消费者保留效用：{r}
- 其他企业平均价格：{avg_other_price:.4f}

## 消费者行为
- {share_rate:.0%} 的消费者分享了数据（按推荐顺序搜索）
- {1-share_rate:.0%} 的消费者未分享（随机搜索）

## 定价策略
考虑：
1. 价格需高于成本 {c} 才能盈利
2. 竞争压力：其他企业平均 {avg_other_price:.4f}
3. 价格过高失去需求，过低损失利润
4. 最优价格在 [{c}, {r}] 区间

请输出JSON：
{{
  "price": float（建议在 {c} 到 {r} 之间），
  "reason": "简要说明理由（不超过100字）"
}}
"""
        return prompt
    
    def build_search_prompt(
        self,
        consumer_id: int,
        share_data: bool,
        searched_firms: List[Dict],
        total_search_cost: float,
        can_search_more: bool
    ) -> str:
        """构建搜索/购买决策提示"""
        s = self.params.search_cost
        n = self.params.n_firms
        
        # 计算已搜索企业的潜在利润
        profits = [f['valuation'] - f['price'] for f in searched_firms]
        best_profit = max(profits) if profits else 0
        best_firm = searched_firms[profits.index(best_profit)]['index'] if profits else None
        
        # 构建已搜索企业列表
        firms_info = "\n".join([
            f"  企业{f['index']}: 估值={f['valuation']:.3f}, 价格={f['price']:.4f}, 净收益={f['valuation']-f['price']:.3f}"
            for f in searched_firms
        ])
        
        prompt = f"""
# 场景：搜索和购买决策

你是消费者 {consumer_id}，{'已分享数据（按推荐顺序搜索）' if share_data else '未分享数据（随机搜索）'}。

## 当前情况
- 已搜索 {len(searched_firms)} 家企业
- 已花费搜索成本：{total_search_cost:.4f}
- 还能搜索：{'是' if can_search_more else '否（已搜索完所有企业）'}

## 已搜索企业信息
{firms_info}

最佳选项：企业{best_firm}，净收益={best_profit:.3f}

## 决策选项

1. **购买**：选择已搜索过的某家企业购买
   - 你会获得 (估值 - 价格 - 搜索成本) 的效用
   - 例如：购买企业{best_firm}，效用={best_profit:.3f} - {total_search_cost:.4f} = {best_profit - total_search_cost:.3f}

2. **继续搜索**：{'搜索下一家企业（额外成本 ' + str(s) + '）' if can_search_more else '不可用（已搜索完所有企业）'}

3. **离开**：停止搜索，不购买任何产品
   - 效用 = -{total_search_cost:.4f}（已花费的搜索成本）

## 决策框架
- 如果已有正利润的选项，通常应该购买（除非预期搜索能找到更好的）
- 继续搜索的价值：期望找到更好的企业 vs 额外搜索成本 {s}
- 搜索成本已花费 {total_search_cost:.4f}，要考虑沉没成本

请输出JSON（选择一个选项）：
{{
  "action": "purchase" 或 "search" 或 "leave",
  "firm_index": int（如果action="purchase"，指定企业index；否则为null），
  "reason": "简要说明理由（不超过100字）"
}}
"""
        return prompt
    
    # ============================================================================
    # LLM决策查询
    # ============================================================================
    
    def query_llm_share(self, consumer_id: int, share_rate_estimate: float = 0.5) -> Tuple[bool, str]:
        """查询LLM分享决策"""
        prompt = self.build_share_prompt(consumer_id, share_rate_estimate)
        
        try:
            response = self.llm_client.generate_json([
                {"role": "system", "content": self.build_system_prompt_consumer()},
                {"role": "user", "content": prompt}
            ])
            
            share = int(response.get("share", 0))
            reason = response.get("reason", "")
            
            return bool(share), reason
        except Exception as e:
            print(f"  ⚠️  消费者{consumer_id} 分享决策失败: {e}")
            return False, ""
    
    def query_llm_price(self, firm_id: int, share_rate: float, other_prices: List[float]) -> Tuple[float, str]:
        """查询LLM定价决策"""
        prompt = self.build_price_prompt(firm_id, share_rate, other_prices)
        
        try:
            response = self.llm_client.generate_json([
                {"role": "system", "content": self.build_system_prompt_firm()},
                {"role": "user", "content": prompt}
            ])
            
            price = float(response.get("price", 0.5))
            reason = response.get("reason", "")
            
            # 限制价格在合理范围
            price = max(self.params.firm_cost, min(self.params.r_value, price))
            
            return price, reason
        except Exception as e:
            print(f"  ⚠️  企业{firm_id} 定价失败: {e}")
            return 0.5, ""
    
    def query_llm_search(
        self,
        consumer_id: int,
        share_data: bool,
        searched_firms: List[Dict],
        total_search_cost: float,
        can_search_more: bool
    ) -> Tuple[str, Optional[int], str]:
        """
        查询LLM搜索/购买决策
        
        Returns:
            (action, firm_index, reason)
            - action: "purchase", "search", "leave"
            - firm_index: 购买的企业index（仅当action="purchase"时有效）
            - reason: 决策理由
        """
        prompt = self.build_search_prompt(consumer_id, share_data, searched_firms, total_search_cost, can_search_more)
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = self.llm_client.generate_json([
                    {"role": "system", "content": self.build_system_prompt_consumer()},
                    {"role": "user", "content": prompt}
                ])
                
                action = response.get("action", "").lower()
                firm_index = response.get("firm_index", None)
                reason = response.get("reason", "")
                
                # 验证action
                if action not in ["purchase", "search", "leave"]:
                    print(f"  ⚠️  消费者{consumer_id} 无效action: {action}，重试...")
                    continue
                
                # 验证search可行性
                if action == "search" and not can_search_more:
                    print(f"  ⚠️  消费者{consumer_id} 选择search但不可行，改为leave")
                    action = "leave"
                    firm_index = None
                
                # 验证purchase的firm_index
                if action == "purchase":
                    if firm_index is None or firm_index not in [f['index'] for f in searched_firms]:
                        print(f"  ⚠️  消费者{consumer_id} 无效purchase index: {firm_index}，重试...")
                        continue
                
                return action, firm_index, reason
                
            except Exception as e:
                print(f"  ⚠️  消费者{consumer_id} 搜索决策失败（重试{retry+1}/{max_retries}）: {e}")
        
        # 失败后的默认策略：如果有正利润就买，否则离开
        if searched_firms:
            profits = [(f['index'], f['valuation'] - f['price']) for f in searched_firms]
            best = max(profits, key=lambda x: x[1])
            if best[1] > 0:
                return "purchase", best[0], "默认策略：购买最佳选项"
        
        return "leave", None, "默认策略：离开市场"
    
    # ============================================================================
    # 理性决策逻辑
    # ============================================================================
    
    def rational_share_decision_consumer(self, consumer: ConsumerState) -> bool:
        """理性分享决策"""
        expected_cost_saving = self.params.search_cost * 1.5
        benefit = self.delta + expected_cost_saving
        cost = consumer.privacy_cost
        return benefit >= cost
    
    def rational_search_decision_consumer(
        self,
        consumer: ConsumerState,
        search_order: List[int],
        firm_prices: List[float]
    ) -> Tuple[int, float]:
        """
        理性搜索决策（最优停止规则）
        
        Returns:
            (purchase_index, utility)
        """
        market_price = np.mean(firm_prices)
        
        if consumer.share:
            # 分享数据：直接选择最高估值的企业
            best_idx = np.argmax(consumer.valuations)
            v_best = consumer.valuations[best_idx]
            p_best = firm_prices[best_idx]
            
            if v_best >= p_best:
                utility = v_best - p_best - consumer.privacy_cost
                return best_idx, utility
            else:
                # 不购买
                return -1, -consumer.privacy_cost
        else:
            # 未分享：随机搜索，使用最优停止规则
            best_idx = -1
            best_net_utility = 0.0
            search_count = 0
            
            for firm_idx in search_order:
                search_count += 1
                v_i = consumer.valuations[firm_idx]
                p_i = firm_prices[firm_idx]
                net_utility = v_i - p_i
                
                # 记录搜索
                consumer.searched_firms.append({
                    'index': firm_idx,
                    'valuation': v_i,
                    'price': p_i
                })
                
                # 最优停止规则：v_i - p_i >= r - market_price
                if net_utility >= consumer.r_value - market_price:
                    best_idx = firm_idx
                    best_net_utility = net_utility
                    break
                
                if net_utility > best_net_utility:
                    best_idx = firm_idx
                    best_net_utility = net_utility
            
            # 计算搜索成本（首次免费）
            search_cost = max(search_count - 1, 0) * self.params.search_cost
            consumer.total_search_cost = search_cost
            
            # 如果找到正利润的选项，购买；否则不购买
            if best_net_utility > 0:
                utility = best_net_utility - search_cost
                return best_idx, utility
            else:
                return -1, -search_cost
    
    def rational_price_decision_firm(self, share_rate: float, market_price: float) -> float:
        """理性定价决策"""
        return optimize_firm_price(
            share_rate=share_rate,
            n_firms=self.params.n_firms,
            market_price=market_price,
            v_dist=self.params.v_dist,
            r_value=self.params.r_value,
            firm_cost=self.params.firm_cost
        )
    
    # ============================================================================
    # 完整市场模拟
    # ============================================================================
    
    def simulate_single_round(
        self,
        consumers: List[ConsumerState],
        firms: List[FirmState],
        rational_share: bool = False,
        rational_price: bool = False,
        rational_search: bool = False,
        round_num: int = 0
    ) -> Dict[str, Any]:
        """
        模拟单轮完整市场交互
        
        流程：
        1. 消费者分享决策
        2. 生成推荐序列
        3. 企业定价决策
        4. 消费者搜索和购买决策
        5. 计算市场结果
        """
        print(f"\n--- 轮次 {round_num + 1} ---")
        
        # 重置轮次状态
        for c in consumers:
            c.reset_for_round()
        for f in firms:
            f.reset_for_round()
        
        # ===== 阶段1：消费者分享决策 =====
        print(f"\n[阶段1] 消费者分享决策...")
        
        if rational_share:
            for consumer in consumers:
                consumer.share = self.rational_share_decision_consumer(consumer)
                consumer.share_reason = f"理性决策: Delta={self.delta:.4f}, τ={consumer.privacy_cost:.4f}"
        else:
            for consumer in consumers:
                share, reason = self.query_llm_share(consumer.index, 0.5)
                consumer.share = share
                consumer.share_reason = reason
        
        share_rate = np.mean([c.share for c in consumers])
        print(f"分享率: {share_rate:.2%} ({sum(c.share for c in consumers)}/{len(consumers)})")
        
        # ===== 阶段2：生成推荐序列 =====
        print(f"\n[阶段2] 生成推荐序列...")
        search_sequences = {}
        for consumer in consumers:
            if consumer.share:
                # 按估值从高到低排序
                search_sequences[consumer.index] = np.argsort(consumer.valuations)[::-1].tolist()
            else:
                # 随机排列
                search_sequences[consumer.index] = np.random.permutation(self.params.n_firms).tolist()
        
        # ===== 阶段3：企业定价决策 =====
        print(f"\n[阶段3] 企业定价决策...")
        
        if rational_price:
            # 价格均衡迭代
            prices = [0.5] * self.params.n_firms
            for iter_p in range(30):
                market_price = np.mean(prices)
                new_prices = []
                for firm in firms:
                    optimal_p = self.rational_price_decision_firm(share_rate, market_price)
                    new_prices.append(optimal_p)
                
                if np.max(np.abs(np.array(new_prices) - np.array(prices))) < 1e-6:
                    break
                prices = new_prices
            
            for i, firm in enumerate(firms):
                firm.price = prices[i]
                firm.price_reason = f"理性均衡价格: {prices[i]:.4f}"
        else:
            for firm in firms:
                other_prices = [f.price for f in firms if f.index != firm.index]
                price, reason = self.query_llm_price(firm.index, share_rate, other_prices)
                firm.price = price
                firm.price_reason = reason
        
        firm_prices = [f.price for f in firms]
        print(f"平均价格: {np.mean(firm_prices):.4f}, 范围: [{min(firm_prices):.4f}, {max(firm_prices):.4f}]")
        
        # ===== 阶段4：消费者搜索和购买决策 =====
        print(f"\n[阶段4] 消费者搜索和购买...")
        
        for consumer in consumers:
            search_order = search_sequences[consumer.index]
            
            if rational_search:
                # 理性搜索
                purchase_idx, utility = self.rational_search_decision_consumer(
                    consumer, search_order, firm_prices
                )
                consumer.purchase_index = purchase_idx
                consumer.utility = utility
            else:
                # LLM搜索（逐步决策）
                self._llm_search_process(consumer, search_order, firm_prices)
        
        # ===== 阶段5：计算市场结果 =====
        print(f"\n[阶段5] 计算市场结果...")
        
        # 统计企业销售
        for consumer in consumers:
            if consumer.purchase_index >= 0:
                firm = firms[consumer.purchase_index]
                firm.sales_count += 1
                firm.revenue += firm.price
                firm.profit = firm.revenue - firm.firm_cost * firm.sales_count
        
        # 计算剩余
        total_consumer_utility = sum(c.utility for c in consumers)
        total_firm_profit = sum(f.profit for f in firms)
        social_welfare = total_consumer_utility + total_firm_profit
        
        print(f"消费者剩余: {total_consumer_utility:.4f}")
        print(f"企业利润: {total_firm_profit:.4f}")
        print(f"社会福利: {social_welfare:.4f}")
        
        return {
            "share_rate": share_rate,
            "avg_price": np.mean(firm_prices),
            "consumer_surplus": total_consumer_utility,
            "firm_profit": total_firm_profit,
            "social_welfare": social_welfare,
            "total_search_cost": sum(c.total_search_cost for c in consumers),
            "avg_search_cost": np.mean([c.total_search_cost for c in consumers]),
            "purchase_rate": np.mean([c.purchase_index >= 0 for c in consumers]),
            "consumers_data": [{
                "index": c.index,
                "share": c.share,
                "share_reason": c.share_reason,
                "purchase_index": c.purchase_index,
                "search_cost": c.total_search_cost,
                "utility": c.utility,
                "privacy_cost": c.privacy_cost
            } for c in consumers],
            "firms_data": [{
                "index": f.index,
                "price": f.price,
                "price_reason": f.price_reason,
                "sales_count": f.sales_count,
                "profit": f.profit
            } for f in firms]
        }
    
    def _llm_search_process(
        self,
        consumer: ConsumerState,
        search_order: List[int],
        firm_prices: List[float]
    ):
        """
        LLM驱动的搜索过程（逐步决策）
        
        模拟消费者按推荐/随机顺序逐个搜索企业，每次决定购买/继续/离开
        """
        search_count = 0
        
        while search_count < len(search_order):
            # 搜索当前企业
            firm_idx = search_order[search_count]
            v_i = consumer.valuations[firm_idx]
            p_i = firm_prices[firm_idx]
            
            consumer.searched_firms.append({
                'index': firm_idx,
                'valuation': v_i,
                'price': p_i
            })
            
            # 更新搜索成本（首次免费）
            if search_count > 0:
                consumer.total_search_cost += self.params.search_cost
            
            search_count += 1
            
            # 查询LLM决策
            can_search_more = search_count < len(search_order)
            action, purchase_firm, reason = self.query_llm_search(
                consumer.index,
                consumer.share,
                consumer.searched_firms,
                consumer.total_search_cost,
                can_search_more
            )
            
            if action == "purchase" and purchase_firm is not None:
                # 购买
                consumer.purchase_index = purchase_firm
                purchase_data = next(f for f in consumer.searched_firms if f['index'] == purchase_firm)
                net_revenue = purchase_data['valuation'] - purchase_data['price']
                consumer.utility = net_revenue - consumer.total_search_cost
                
                if consumer.share:
                    consumer.utility -= consumer.privacy_cost
                
                break
            
            elif action == "leave":
                # 离开市场
                consumer.purchase_index = -1
                consumer.utility = -consumer.total_search_cost
                
                if consumer.share:
                    consumer.utility -= consumer.privacy_cost
                
                break
            
            elif action == "search":
                # 继续搜索下一家
                continue
    
    # ============================================================================
    # 主评估流程
    # ============================================================================
    
    def run_full_evaluation(
        self,
        num_rounds: int = 5,
        rational_share: bool = False,
        rational_price: bool = False,
        rational_search: bool = False
    ) -> Dict[str, Any]:
        """
        运行完整评估
        
        Args:
            num_rounds: 模拟轮数
            rational_share: 是否使用理性分享决策
            rational_price: 是否使用理性定价决策
            rational_search: 是否使用理性搜索决策
        
        Returns:
            完整评估结果
        """
        print(f"\n{'='*60}")
        print(f"[场景A完整评估] 模型: {self.llm_client.config_name}")
        print(f"{'='*60}")
        print(f"决策模式: Share={'理性' if rational_share else 'LLM'}, "
              f"Price={'理性' if rational_price else 'LLM'}, "
              f"Search={'理性' if rational_search else 'LLM'}")
        
        # 初始化消费者和企业
        consumers = []
        for i in range(self.params.n_consumers):
            # 为每个消费者生成估值向量
            rng = np.random.default_rng(self.params.seed + i)
            valuations = rng.uniform(
                self.params.v_dist['low'],
                self.params.v_dist['high'],
                self.params.n_firms
            )
            
            consumer = ConsumerState(
                index=i,
                privacy_cost=self.params.privacy_costs[i],
                valuations=valuations,
                search_cost=self.params.search_cost,
                r_value=self.params.r_value
            )
            consumers.append(consumer)
        
        firms = [FirmState(i, self.params.firm_cost) for i in range(self.params.n_firms)]
        
        # 多轮模拟
        all_rounds = []
        
        for round_num in range(num_rounds):
            round_result = self.simulate_single_round(
                consumers=consumers,
                firms=firms,
                rational_share=rational_share,
                rational_price=rational_price,
                rational_search=rational_search,
                round_num=round_num
            )
            
            all_rounds.append(round_result)
        
        # 聚合结果
        avg_results = {
            "avg_share_rate": np.mean([r["share_rate"] for r in all_rounds]),
            "avg_price": np.mean([r["avg_price"] for r in all_rounds]),
            "avg_consumer_surplus": np.mean([r["consumer_surplus"] for r in all_rounds]),
            "avg_firm_profit": np.mean([r["firm_profit"] for r in all_rounds]),
            "avg_social_welfare": np.mean([r["social_welfare"] for r in all_rounds]),
            "avg_search_cost": np.mean([r["avg_search_cost"] for r in all_rounds]),
            "avg_purchase_rate": np.mean([r["purchase_rate"] for r in all_rounds])
        }
        
        results = {
            "model_name": self.llm_client.config_name,
            "scenario": "A_recommendation",
            "num_rounds": num_rounds,
            "rational_share": rational_share,
            "rational_price": rational_price,
            "rational_search": rational_search,
            "params": self.params.to_dict(),
            "delta": self.delta,
            "all_rounds": all_rounds,
            "averages": avg_results
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"[评估结果摘要]")
        print(f"{'='*60}")
        
        print(f"\n模型: {results['model_name']}")
        print(f"轮数: {results['num_rounds']}")
        print(f"决策模式: Share={'理性' if results['rational_share'] else 'LLM'}, "
              f"Price={'理性' if results['rational_price'] else 'LLM'}, "
              f"Search={'理性' if results['rational_search'] else 'LLM'}")
        
        print(f"\n【平均结果】")
        avg = results['averages']
        print(f"  分享率: {avg['avg_share_rate']:.2%}")
        print(f"  平均价格: {avg['avg_price']:.4f}")
        print(f"  购买率: {avg['avg_purchase_rate']:.2%}")
        print(f"  平均搜索成本: {avg['avg_search_cost']:.4f}")
        print(f"  消费者剩余: {avg['avg_consumer_surplus']:.4f}")
        print(f"  企业利润: {avg['avg_firm_profit']:.4f}")
        print(f"  社会福利: {avg['avg_social_welfare']:.4f}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换numpy类型为Python原生类型
        def convert_to_python_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_to_python_types(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='场景A推荐系统完整评估器')
    parser.add_argument('--model', type=str, default='deepseek-v3.2', help='LLM模型')
    parser.add_argument('--rounds', type=int, default=5, help='模拟轮数')
    parser.add_argument('--rational-share', action='store_true', help='理性分享决策')
    parser.add_argument('--rational-price', action='store_true', help='理性定价决策')
    parser.add_argument('--rational-search', action='store_true', help='理性搜索决策')
    parser.add_argument('--n-consumers', type=int, default=10, help='消费者数量')
    parser.add_argument('--n-firms', type=int, default=5, help='企业数量')
    parser.add_argument('--search-cost', type=float, default=0.02, help='搜索成本')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/scenario_a', help='输出目录')
    
    args = parser.parse_args()
    
    # 生成参数
    params = generate_recommendation_instance(
        n_consumers=args.n_consumers,
        n_firms=args.n_firms,
        search_cost=args.search_cost,
        seed=args.seed
    )
    
    # 创建LLM客户端
    llm_client = create_llm_client(args.model)
    
    # 创建评估器
    evaluator = ScenarioAFullEvaluator(llm_client, params)
    
    # 运行评估
    results = evaluator.run_full_evaluation(
        num_rounds=args.rounds,
        rational_share=args.rational_share,
        rational_price=args.rational_price,
        rational_search=args.rational_search
    )
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rational_tags = []
    if args.rational_share:
        rational_tags.append("share")
    if args.rational_price:
        rational_tags.append("price")
    if args.rational_search:
        rational_tags.append("search")
    
    rational_suffix = "_rational_" + "_".join(rational_tags) if rational_tags else ""
    output_path = f"{args.output_dir}/eval_A_full_{llm_client.config_name}{rational_suffix}_{timestamp}.json"
    
    evaluator.save_results(results, output_path)
    
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}")
