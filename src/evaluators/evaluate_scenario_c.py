"""
场景C主评估器

支持4种配置的评估：
- 配置A：理性×理性（理论基准）
- 配置B：理性中介×LLM消费者
- 配置C：LLM中介×理性消费者
- 配置D：LLM中介×LLM消费者

所有指标都是完全量化的、客观的。
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass

# 处理直接运行和模块导入两种情况
if __name__ == "__main__":
    # 直接运行：添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    ConsumerData,
    simulate_market_outcome,
    compute_rational_participation_rate_ex_ante,
    evaluate_intermediary_strategy,
    generate_consumer_data
)
from src.evaluators.scenario_c_metrics import (
    compute_participation_metrics,
    compute_market_metrics,
    compute_inequality_metrics,
    compute_strategy_metrics,
    compute_profit_metrics,
    compute_ranking_metrics,
    compute_interaction_metrics
)


@dataclass
class LLMConsumerAgent:
    """LLM消费者代理的抽象接口"""
    
    def decide(
        self,
        consumer_params: Dict,
        m: float,
        anonymization: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        消费者决策
        
        Args:
            consumer_params: 消费者参数 {theta_i, tau_i, w_i (可选)}
            m: 补偿金额
            anonymization: 匿名化策略
            context: 额外上下文（可选）
        
        Returns:
            是否参与数据分享
        """
        raise NotImplementedError


@dataclass
class LLMIntermediaryAgent:
    """LLM中介代理的抽象接口"""
    
    def choose_strategy(
        self,
        market_params: Dict,
        context: Optional[Dict] = None
    ) -> Tuple[float, str]:
        """
        中介选择策略
        
        Args:
            market_params: 市场参数 {N, mu_theta, sigma_theta, tau_mean, tau_std, ...}
            context: 额外上下文（可选）
        
        Returns:
            (m, anonymization) 元组
        """
        raise NotImplementedError


class ScenarioCEvaluator:
    """场景C评估器"""
    
    def __init__(self, ground_truth_path: str):
        """
        初始化评估器
        
        Args:
            ground_truth_path: Ground Truth文件路径（配置A）
        """
        self.gt_A = self.load_ground_truth(ground_truth_path)
        self.params_base = self._extract_params_base()
        
    def load_ground_truth(self, path: str) -> Dict:
        """加载Ground Truth文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_params_base(self) -> Dict:
        """从GT中提取基础参数"""
        # GT文件中直接有params_base字段
        return dict(self.gt_A['params_base'])
    
    def _get_sample_consumers(self) -> List[Dict]:
        """
        获取样本消费者数据（包括生成tau值）
        
        Returns:
            消费者参数列表
        """
        sample_data = self.gt_A['sample_data']
        N = self.params_base['N']
        
        # 生成tau值（使用GT的seed确保可复现）
        rng = np.random.default_rng(self.params_base['seed'] + 1000)
        if self.params_base['tau_dist'] == 'normal':
            tau_values = rng.normal(
                self.params_base['tau_mean'],
                self.params_base['tau_std'],
                N
            )
        elif self.params_base['tau_dist'] == 'uniform':
            a = self.params_base['tau_mean'] - np.sqrt(3) * self.params_base['tau_std']
            b = self.params_base['tau_mean'] + np.sqrt(3) * self.params_base['tau_std']
            tau_values = rng.uniform(a, b, N)
        else:
            tau_values = np.zeros(N)
        
        consumers = []
        for i in range(N):
            consumer = {
                'tau_i': float(tau_values[i]),
            }
            
            # 根据data_structure添加theta和w
            if self.params_base['data_structure'] == 'common_preferences':
                consumer['theta_i'] = float(sample_data['theta'])
                consumer['w_i'] = float(sample_data['w'][i])
            elif self.params_base['data_structure'] == 'common_experience':
                consumer['theta_i'] = float(sample_data['w'][i])  # 在common_experience中，w实际上是theta
                consumer['w_i'] = float(sample_data['w'][i])
            
            consumers.append(consumer)
        
        return consumers
    
    def _get_theory_decisions(self, delta_u: float, consumers: List[Dict]) -> np.ndarray:
        """
        计算理论决策
        
        Args:
            delta_u: 参与vs拒绝的效用差
            consumers: 消费者参数列表
        
        Returns:
            理论决策数组（N个布尔值）
        """
        tau_values = np.array([c['tau_i'] for c in consumers])
        return tau_values <= delta_u

    # 修改：统一封装消费者LLM调用（可选返回理由）
    def _call_consumer_agent_with_reason(
        self,
        llm_consumer_agent: Callable,
        consumer_params: Dict,
        m: float,
        anonymization: str
    ) -> Tuple[bool, str]:
        """调用消费者代理并尽量返回理由；缺失时返回空理由"""
        if callable(llm_consumer_agent):
            if hasattr(llm_consumer_agent, "with_reason"):
                decision, reason = llm_consumer_agent.with_reason(
                    consumer_params=consumer_params,
                    m=m,
                    anonymization=anonymization
                )
                return bool(decision), str(reason)
            decision = llm_consumer_agent(
                consumer_params=consumer_params,
                m=m,
                anonymization=anonymization
            )
            return bool(decision), ""
        if hasattr(llm_consumer_agent, "decide_with_reason"):
            decision, reason = llm_consumer_agent.decide_with_reason(
                consumer_params=consumer_params,
                m=m,
                anonymization=anonymization
            )
            return bool(decision), str(reason)
        decision = llm_consumer_agent.decide(
            consumer_params=consumer_params,
            m=m,
            anonymization=anonymization
        )
        return bool(decision), ""

    # 修改：统一封装中介LLM调用（支持带反馈/历史的多轮学习）
    def _call_intermediary_agent(
        self,
        llm_intermediary_agent: Callable,
        market_params: Dict,
        feedback: Optional[Dict] = None,
        history: Optional[List[Dict]] = None
    ) -> Tuple[float, str]:
        """安全调用中介代理，兼容是否支持反馈/历史"""
        if callable(llm_intermediary_agent):
            try:
                return llm_intermediary_agent(
                    market_params=market_params,
                    feedback=feedback,
                    history=history
                )
            except TypeError:
                return llm_intermediary_agent(market_params=market_params)
        return llm_intermediary_agent.choose_strategy(
            market_params=market_params,
            feedback=feedback,
            history=history
        )
    
    def evaluate_config_B(
        self,
        llm_consumer_agent: Callable,
        verbose: bool = True
    ) -> Dict:
        """
        配置B：理性中介 × LLM消费者
        
        Args:
            llm_consumer_agent: LLM消费者代理（函数或对象）
            verbose: 是否打印详细信息
        
        Returns:
            完整的评估指标字典
        """
        if verbose:
            print("\n" + "="*70)
            print("配置B：理性中介 × LLM消费者")
            print("="*70)
        
        # 1. 获取理论最优策略
        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        r_star = self.gt_A['optimal_strategy']['r_star']
        delta_u = self.gt_A['optimal_strategy']['delta_u_star']
        
        if verbose:
            print(f"\n理论最优策略: m*={m_star:.4f}, {anon_star}")
            print(f"理论参与率: r*={r_star:.4f}")
        
        # 2. 获取消费者数据
        consumers = self._get_sample_consumers()
        N = self.params_base['N']
        
        # 3. LLM决策
        if verbose:
            print(f"\n正在收集{N}个LLM消费者的决策...")
        
        llm_decisions = []
        for consumer_params in consumers:
            # 调用LLM代理
            if callable(llm_consumer_agent):
                decision = llm_consumer_agent(
                    consumer_params=consumer_params,
                    m=m_star,
                    anonymization=anon_star
                )
            else:
                decision = llm_consumer_agent.decide(
                    consumer_params=consumer_params,
                    m=m_star,
                    anonymization=anon_star
                )
            
            llm_decisions.append(bool(decision))
        
        llm_decisions = np.array(llm_decisions)
        r_llm = float(np.mean(llm_decisions))
        
        if verbose:
            print(f"LLM参与率: r_llm={r_llm:.4f}")
            print(f"理论参与率: r*={r_star:.4f}")
            print(f"偏差: {abs(r_llm - r_star):.4f}")
        
        # 4. 计算理论决策
        theory_decisions = self._get_theory_decisions(delta_u, consumers)
        
        # 5. 计算市场结果（使用LLM的参与决策）
        params = ScenarioCParams(
            m=m_star,
            anonymization=anon_star,
            **self.params_base
        )
        
        # 生成消费者数据
        rng = np.random.default_rng(self.params_base['seed'])
        consumer_data = generate_consumer_data(params, rng=rng)
        
        # 模拟市场
        outcome_llm = simulate_market_outcome(
            consumer_data,
            llm_decisions,
            params,
            producer_info_mode="with_data",
            m0=self.gt_A['data_transaction']['m_0'],
            rng=rng
        )
        
        # 6. 提取理论市场结果
        outcome_theory = {
            'social_welfare': self.gt_A['equilibrium']['social_welfare'],
            'consumer_surplus': self.gt_A['equilibrium']['consumer_surplus'],
            'producer_profit': self.gt_A['equilibrium']['producer_profit'],
            'intermediary_profit': self.gt_A['equilibrium']['intermediary_profit'],
            'gini_coefficient': self.gt_A['equilibrium']['gini_coefficient'],
            'price_variance': self.gt_A['equilibrium'].get('price_variance', 0.0),
            'price_discrimination_index': self.gt_A['equilibrium'].get('price_discrimination_index', 0.0),
        }
        
        outcome_llm_dict = {
            'social_welfare': outcome_llm.social_welfare,
            'consumer_surplus': outcome_llm.consumer_surplus,
            'producer_profit': outcome_llm.producer_profit,
            'intermediary_profit': outcome_llm.intermediary_profit,
            'gini_coefficient': outcome_llm.gini_coefficient,
            'price_variance': outcome_llm.price_variance,
            'price_discrimination_index': outcome_llm.price_discrimination_index,
        }
        
        # 7. 计算所有指标
        metrics = {
            "config": "B_rational_intermediary_llm_consumer",
            "participation": compute_participation_metrics(
                llm_decisions,
                theory_decisions,
                r_star
            ),
            "market": compute_market_metrics(
                outcome_llm_dict,
                outcome_theory
            ),
            "inequality": compute_inequality_metrics(
                outcome_llm_dict,
                outcome_theory
            ),
        }
        
        if verbose:
            print(f"\n关键指标:")
            print(f"  参与率误差: {metrics['participation']['r_relative_error']:.2%}")
            print(f"  个体准确率: {metrics['participation']['individual_accuracy']:.2%}")
            print(f"  福利比率: {metrics['market']['social_welfare_ratio']:.4f}")
            print(f"  福利损失: {metrics['market']['welfare_loss_percent']:.2f}%")
        
        return metrics
    
    def evaluate_config_C(
        self,
        llm_intermediary_agent: Callable,
        verbose: bool = True
    ) -> Dict:
        """
        配置C：LLM中介 × 理性消费者
        
        Args:
            llm_intermediary_agent: LLM中介代理（函数或对象）
            verbose: 是否打印详细信息
        
        Returns:
            完整的评估指标字典
        """
        if verbose:
            print("\n" + "="*70)
            print("配置C：LLM中介 × 理性消费者")
            print("="*70)
        
        # 1. 获取理论最优策略
        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        profit_star = self.gt_A['optimal_strategy']['intermediary_profit_star']
        
        if verbose:
            print(f"\n理论最优策略: m*={m_star:.4f}, {anon_star}")
            print(f"理论最优利润: {profit_star:.4f}")
        
        # 2. LLM选择策略
        if verbose:
            print(f"\n请LLM中介选择策略...")
        
        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }
        
        if callable(llm_intermediary_agent):
            m_llm, anon_llm = llm_intermediary_agent(market_params=market_params)
        else:
            m_llm, anon_llm = llm_intermediary_agent.choose_strategy(market_params=market_params)
        
        if verbose:
            print(f"LLM选择: m={m_llm:.4f}, {anon_llm}")
        
        # 3. 计算理性消费者的反应（使用LLM的策略）
        if verbose:
            print(f"\n计算理性消费者对LLM策略的反应...")
        
        result_llm = evaluate_intermediary_strategy(
            m=m_llm,
            anonymization=anon_llm,
            params_base=self.params_base,
            num_mc_samples=50,
            max_iter=100,
            tol=1e-3,
            seed=self.params_base['seed']
        )
        
        profit_llm = result_llm.intermediary_profit
        r_given_llm = result_llm.r_star
        
        if verbose:
            print(f"理性参与率(给定LLM策略): r*={r_given_llm:.4f}")
            print(f"LLM策略利润: {profit_llm:.4f}")
            print(f"理论最优利润: {profit_star:.4f}")
            print(f"利润效率: {profit_llm / profit_star:.2%}")
        
        # 4. 计算市场结果
        outcome_llm = {
            'social_welfare': result_llm.social_welfare,
            'consumer_surplus': result_llm.consumer_surplus,
            'producer_profit': result_llm.producer_profit_with_data,
            'intermediary_profit': result_llm.intermediary_profit,
        }
        
        outcome_theory = {
            'social_welfare': self.gt_A['equilibrium']['social_welfare'],
            'consumer_surplus': self.gt_A['equilibrium']['consumer_surplus'],
            'producer_profit': self.gt_A['equilibrium']['producer_profit'],
            'intermediary_profit': self.gt_A['equilibrium']['intermediary_profit'],
        }
        
        # 5. 计算所有指标
        cost_llm = m_llm * result_llm.num_participants
        cost_theory = m_star * self.gt_A['optimal_strategy'].get('num_participants_expected', 0)
        
        metrics = {
            "config": "C_llm_intermediary_rational_consumer",
            "strategy": compute_strategy_metrics(
                m_llm, anon_llm,
                m_star, anon_star
            ),
            "profit": compute_profit_metrics(
                profit_llm, profit_star,
                cost_llm, cost_theory
            ),
            "market": compute_market_metrics(
                outcome_llm,
                outcome_theory
            ),
            "participation_given_llm_strategy": {
                "r_given_llm": r_given_llm,
                "r_optimal": self.gt_A['optimal_strategy']['r_star'],
                "r_ratio": r_given_llm / self.gt_A['optimal_strategy']['r_star'],
            }
        }
        
        if verbose:
            print(f"\n关键指标:")
            print(f"  策略m误差: {metrics['strategy']['m_relative_error']:.2%}")
            print(f"  匿名化匹配: {'✓' if metrics['strategy']['anon_match'] else '✗'}")
            print(f"  利润效率: {metrics['profit']['profit_ratio']:.2%}")
            print(f"  利润损失: {metrics['profit']['profit_loss_percent']:.2f}%")
        
        return metrics

    # 修改：新增多轮学习版（中介LLM通过反馈逐轮调整m）
    def evaluate_config_C_iterative(
        self,
        llm_intermediary_agent: Callable,
        rounds: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        配置C（多轮学习版）：LLM中介 × 理性消费者

        目标：通过多轮反馈让中介逐步提升利润
        """
        if verbose:
            print("\n" + "="*70)
            print("配置C（多轮学习）：LLM中介 × 理性消费者")
            print("="*70)

        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        profit_star = self.gt_A['optimal_strategy']['intermediary_profit_star']

        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }

        history: List[Dict] = []
        best_round: Optional[Dict] = None

        for t in range(1, rounds + 1):
            feedback = history[-1] if history else None
            if verbose:
                print(f"\n--- 轮次 {t}/{rounds} ---")
                if feedback:
                    print(f"上一轮利润: {feedback['intermediary_profit']:.4f}, "
                          f"参与率: {feedback['participation_rate']:.4f}")

            # 让中介根据反馈选择策略
            m_llm, anon_llm = self._call_intermediary_agent(
                llm_intermediary_agent,
                market_params=market_params,
                feedback=feedback,
                history=history
            )

            if verbose:
                print(f"LLM选择: m={m_llm:.4f}, {anon_llm}")

            # 计算理性消费者对该策略的反应
            result_llm = evaluate_intermediary_strategy(
                m=m_llm,
                anonymization=anon_llm,
                params_base=self.params_base,
                num_mc_samples=50,
                max_iter=100,
                tol=1e-3,
                seed=self.params_base['seed']
            )

            profit_llm = result_llm.intermediary_profit
            r_given_llm = result_llm.r_star

            # 修改：仅反馈与利润直接相关的指标 + 参与/拒绝理由逐条反馈（不做关键词汇总）
            consumers = self._get_sample_consumers()
            delta_u = float(result_llm.delta_u)
            reasons_participants: List[str] = []
            reasons_rejecters: List[str] = []
            for consumer in consumers:
                tau_i = float(consumer["tau_i"])
                if tau_i <= delta_u:
                    reasons_participants.append(
                        f"（理性）参与：τ={tau_i:.2f} <= ΔU={delta_u:.2f}"
                    )
                else:
                    reasons_rejecters.append(
                        f"（理性）拒绝：τ={tau_i:.2f} > ΔU={delta_u:.2f}"
                    )
            num_participants = int(len(reasons_participants))

            round_info = {
                "round": t,
                "m": float(m_llm),
                "anonymization": anon_llm,
                "participation_rate": float(r_given_llm),
                "num_participants": num_participants,
                "m0": float(result_llm.m_0),
                "intermediary_cost": float(result_llm.intermediary_cost),
                "intermediary_profit": float(profit_llm),
                "reasons": {
                    "participants": reasons_participants,
                    "rejecters": reasons_rejecters
                }
            }
            history.append(round_info)

            if (best_round is None) or (profit_llm > best_round["intermediary_profit"]):
                best_round = round_info

            if verbose:
                print(f"本轮利润: {profit_llm:.4f}, 参与率: {r_given_llm:.4f}, m0: {result_llm.m_0:.4f}")

        # 用最优轮次生成指标
        assert best_round is not None
        m_llm = best_round["m"]
        anon_llm = best_round["anonymization"]
        profit_llm = best_round["intermediary_profit"]

        # 重新评估最优轮对应的市场结果（用于与理论对比）
        result_best = evaluate_intermediary_strategy(
            m=m_llm,
            anonymization=anon_llm,
            params_base=self.params_base,
            num_mc_samples=50,
            max_iter=100,
            tol=1e-3,
            seed=self.params_base['seed']
        )

        outcome_llm = {
            'social_welfare': result_best.social_welfare,
            'consumer_surplus': result_best.consumer_surplus,
            'producer_profit': result_best.producer_profit_with_data,
            'intermediary_profit': result_best.intermediary_profit,
        }

        outcome_theory = {
            'social_welfare': self.gt_A['equilibrium']['social_welfare'],
            'consumer_surplus': self.gt_A['equilibrium']['consumer_surplus'],
            'producer_profit': self.gt_A['equilibrium']['producer_profit'],
            'intermediary_profit': self.gt_A['equilibrium']['intermediary_profit'],
        }

        cost_llm = m_llm * result_best.num_participants
        cost_theory = m_star * self.gt_A['optimal_strategy'].get('num_participants_expected', 0)

        metrics = {
            "config": "C_llm_intermediary_rational_consumer_iterative",
            "strategy": compute_strategy_metrics(
                m_llm, anon_llm,
                m_star, anon_star
            ),
            "profit": compute_profit_metrics(
                profit_llm, profit_star,
                cost_llm, cost_theory
            ),
            "market": compute_market_metrics(
                outcome_llm,
                outcome_theory
            ),
            "participation_given_llm_strategy": {
                "r_given_llm": result_best.r_star,
                "r_optimal": self.gt_A['optimal_strategy']['r_star'],
                "r_ratio": result_best.r_star / self.gt_A['optimal_strategy']['r_star'],
            },
            "learning_history": history
        }

        if verbose:
            print(f"\n最优轮策略: m={m_llm:.4f}, {anon_llm}")
            print(f"最优轮利润: {profit_llm:.4f} (理论最优 {profit_star:.4f})")

        return metrics

    # 修改：新增多轮学习版（LLM中介 × LLM消费者）
    def evaluate_config_D_iterative(
        self,
        llm_intermediary_agent: Callable,
        llm_consumer_agent: Callable,
        rounds: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        配置D（多轮学习版）：LLM中介 × LLM消费者
        """
        if verbose:
            print("\n" + "="*70)
            print("配置D（多轮学习）：LLM中介 × LLM消费者")
            print("="*70)

        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        profit_star = self.gt_A['optimal_strategy']['intermediary_profit_star']

        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }

        history: List[Dict] = []
        best_round: Optional[Dict] = None

        for t in range(1, rounds + 1):
            feedback = history[-1] if history else None
            if verbose:
                print(f"\n--- 轮次 {t}/{rounds} ---")
                if feedback:
                    print(f"上一轮利润: {feedback['intermediary_profit']:.4f}, "
                          f"参与率: {feedback['participation_rate']:.4f}")

            m_llm, anon_llm = self._call_intermediary_agent(
                llm_intermediary_agent,
                market_params=market_params,
                feedback=feedback,
                history=history
            )

            if verbose:
                print(f"LLM中介选择: m={m_llm:.4f}, {anon_llm}")

            # 让LLM消费者响应
            consumers = self._get_sample_consumers()
            llm_decisions: List[bool] = []
            reasons_participants: List[str] = []
            reasons_rejecters: List[str] = []
            for consumer_params in consumers:
                decision, reason = self._call_consumer_agent_with_reason(
                    llm_consumer_agent=llm_consumer_agent,
                    consumer_params=consumer_params,
                    m=m_llm,
                    anonymization=anon_llm
                )
                llm_decisions.append(bool(decision))
                if decision:
                    reasons_participants.append(
                        f"参与：{reason}".strip()
                    )
                else:
                    reasons_rejecters.append(
                        f"拒绝：{reason}".strip()
                    )

            llm_decisions_arr = np.array(llm_decisions)
            r_llm = float(np.mean(llm_decisions_arr))

            # 估算m0并计算市场结果
            params = ScenarioCParams(
                m=m_llm,
                anonymization=anon_llm,
                **self.params_base
            )
            rng = np.random.default_rng(self.params_base['seed'])
            consumer_data = generate_consumer_data(params, rng=rng)

            from src.scenarios.scenario_c_social_data import estimate_m0_mc

            def participation_rule(p, world, rng):
                return llm_decisions_arr

            m_0_D, _, _, _ = estimate_m0_mc(
                params=params,
                participation_rule=participation_rule,
                T=100,
                beta=1.0,
                seed=self.params_base['seed']
            )

            outcome_D = simulate_market_outcome(
                consumer_data,
                llm_decisions_arr,
                params,
                producer_info_mode="with_data",
                m0=m_0_D,
                rng=rng
            )

            round_info = {
                "round": t,
                "m": float(m_llm),
                "anonymization": anon_llm,
                "participation_rate": r_llm,
                "num_participants": int(np.sum(llm_decisions_arr)),
                "m0": float(m_0_D),
                "intermediary_cost": float(m_llm * np.sum(llm_decisions_arr)),
                "intermediary_profit": float(outcome_D.intermediary_profit),
                "reasons": {
                    "participants": reasons_participants,
                    "rejecters": reasons_rejecters
                }
            }
            history.append(round_info)

            if (best_round is None) or (round_info["intermediary_profit"] > best_round["intermediary_profit"]):
                best_round = round_info

            if verbose:
                print(f"本轮利润: {outcome_D.intermediary_profit:.4f}, 参与率: {r_llm:.4f}")

        # 用最优轮次生成指标
        assert best_round is not None
        m_llm = best_round["m"]
        anon_llm = best_round["anonymization"]

        # 重新计算一次最优轮的市场结果用于指标对比
        params = ScenarioCParams(
            m=m_llm,
            anonymization=anon_llm,
            **self.params_base
        )
        rng = np.random.default_rng(self.params_base['seed'])
        consumer_data = generate_consumer_data(params, rng=rng)
        llm_decisions_arr = np.array([
            self._call_consumer_agent_with_reason(
                llm_consumer_agent=llm_consumer_agent,
                consumer_params=c,
                m=m_llm,
                anonymization=anon_llm
            )[0]
            for c in consumers
        ])
        from src.scenarios.scenario_c_social_data import estimate_m0_mc

        def participation_rule(p, world, rng):
            return llm_decisions_arr

        m_0_best, _, _, _ = estimate_m0_mc(
            params=params,
            participation_rule=participation_rule,
            T=100,
            beta=1.0,
            seed=self.params_base['seed']
        )
        outcome_best = simulate_market_outcome(
            consumer_data,
            llm_decisions_arr,
            params,
            producer_info_mode="with_data",
            m0=m_0_best,
            rng=rng
        )

        outcome_A = {
            'social_welfare': self.gt_A['equilibrium']['social_welfare'],
            'consumer_surplus': self.gt_A['equilibrium']['consumer_surplus'],
            'producer_profit': self.gt_A['equilibrium']['producer_profit'],
            'intermediary_profit': self.gt_A['equilibrium']['intermediary_profit'],
        }
        outcome_best_dict = {
            'social_welfare': outcome_best.social_welfare,
            'consumer_surplus': outcome_best.consumer_surplus,
            'producer_profit': outcome_best.producer_profit,
            'intermediary_profit': outcome_best.intermediary_profit,
        }

        metrics = {
            "config": "D_llm_intermediary_llm_consumer_iterative",
            "strategy": {
                "m_llm": m_llm,
                "anon_llm": anon_llm,
                "r_llm": float(np.mean(llm_decisions_arr)),
            },
            "vs_theory": {
                "m_error": abs(m_llm - m_star),
                "anon_match": int(anon_llm == anon_star),
                "r_error": abs(float(np.mean(llm_decisions_arr)) - self.gt_A['optimal_strategy']['r_star']),
            },
            "market": compute_market_metrics(outcome_best_dict, outcome_A),
            "interaction": compute_interaction_metrics(
                outcome_best_dict,
                outcome_A
            ),
            "learning_history": history
        }

        if verbose:
            print(f"\n最优轮策略: m={m_llm:.4f}, {anon_llm}")

        return metrics
    
    def evaluate_config_D(
        self,
        llm_intermediary_agent: Callable,
        llm_consumer_agent: Callable,
        verbose: bool = True
    ) -> Dict:
        """
        配置D：LLM中介 × LLM消费者
        
        Args:
            llm_intermediary_agent: LLM中介代理
            llm_consumer_agent: LLM消费者代理
            verbose: 是否打印详细信息
        
        Returns:
            完整的评估指标字典
        """
        if verbose:
            print("\n" + "="*70)
            print("配置D：LLM中介 × LLM消费者")
            print("="*70)
        
        # 1. LLM中介选择策略
        if verbose:
            print(f"\n步骤1: LLM中介选择策略...")
        
        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }
        
        if callable(llm_intermediary_agent):
            m_llm, anon_llm = llm_intermediary_agent(market_params=market_params)
        else:
            m_llm, anon_llm = llm_intermediary_agent.choose_strategy(market_params=market_params)
        
        if verbose:
            print(f"LLM中介选择: m={m_llm:.4f}, {anon_llm}")
        
        # 2. LLM消费者反应
        if verbose:
            print(f"\n步骤2: 收集LLM消费者决策...")
        
        consumers = self._get_sample_consumers()
        N = self.params_base['N']
        
        llm_decisions = []
        for consumer_params in consumers:
            if callable(llm_consumer_agent):
                decision = llm_consumer_agent(
                    consumer_params=consumer_params,
                    m=m_llm,
                    anonymization=anon_llm
                )
            else:
                decision = llm_consumer_agent.decide(
                    consumer_params=consumer_params,
                    m=m_llm,
                    anonymization=anon_llm
                )
            
            llm_decisions.append(bool(decision))
        
        llm_decisions = np.array(llm_decisions)
        r_llm = float(np.mean(llm_decisions))
        
        if verbose:
            print(f"LLM消费者参与率: r={r_llm:.4f}")
        
        # 3. 计算市场结果
        if verbose:
            print(f"\n步骤3: 计算市场结果...")
        
        params = ScenarioCParams(
            m=m_llm,
            anonymization=anon_llm,
            **self.params_base
        )
        
        rng = np.random.default_rng(self.params_base['seed'])
        consumer_data = generate_consumer_data(params, rng=rng)
        
        # 估算m_0（简化：使用理论模型）
        from src.scenarios.scenario_c_social_data import estimate_m0_mc
        
        def participation_rule(p, world, rng):
            # 使用LLM决策作为参与规则的近似
            return llm_decisions
        
        m_0_D, _, _, _ = estimate_m0_mc(
            params=params,
            participation_rule=participation_rule,
            T=100,
            beta=1.0,
            seed=self.params_base['seed']
        )
        
        outcome_D = simulate_market_outcome(
            consumer_data,
            llm_decisions,
            params,
            producer_info_mode="with_data",
            m0=m_0_D,
            rng=rng
        )
        
        outcome_D_dict = {
            'social_welfare': outcome_D.social_welfare,
            'consumer_surplus': outcome_D.consumer_surplus,
            'producer_profit': outcome_D.producer_profit,
            'intermediary_profit': outcome_D.intermediary_profit,
        }
        
        # 4. 提取配置A的结果
        outcome_A = {
            'social_welfare': self.gt_A['equilibrium']['social_welfare'],
            'consumer_surplus': self.gt_A['equilibrium']['consumer_surplus'],
            'producer_profit': self.gt_A['equilibrium']['producer_profit'],
            'intermediary_profit': self.gt_A['equilibrium']['intermediary_profit'],
        }
        
        # 5. 计算所有指标
        metrics = {
            "config": "D_llm_intermediary_llm_consumer",
            "strategy": {
                "m_llm": m_llm,
                "anon_llm": anon_llm,
                "r_llm": r_llm,
            },
            "vs_theory": {
                "m_error": abs(m_llm - self.gt_A['optimal_strategy']['m_star']),
                "anon_match": int(anon_llm == self.gt_A['optimal_strategy']['anonymization_star']),
                "r_error": abs(r_llm - self.gt_A['optimal_strategy']['r_star']),
            },
            "market": compute_market_metrics(outcome_D_dict, outcome_A),
            "interaction": compute_interaction_metrics(
                outcome_D_dict,
                outcome_A
            )
        }
        
        if verbose:
            print(f"\n关键指标:")
            print(f"  vs理论最优:")
            print(f"    策略m误差: {metrics['vs_theory']['m_error']:.4f}")
            print(f"    参与率误差: {metrics['vs_theory']['r_error']:.4f}")
            print(f"    福利比率: {metrics['market']['social_welfare_ratio']:.4f}")
            print(f"    福利损失: {metrics['market']['welfare_loss_percent']:.2f}%")
            print(f"  交互指标:")
            print(f"    剥削指标: {metrics['interaction']['exploitation_indicator']:.4f}")
        
        return metrics
    
    def generate_report(
        self,
        results_B: Dict = None,
        results_C: Dict = None,
        results_D: Dict = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        生成评估报告
        
        Args:
            results_B: 配置B的评估结果
            results_C: 配置C的评估结果
            results_D: 配置D的评估结果
            output_path: 输出文件路径（可选）
        
        Returns:
            DataFrame格式的报告
        """
        rows = []
        
        # 配置A（理论基准）
        row_A = {
            "Config": "A (Theory)",
            "Intermediary": "Rational",
            "Consumers": "Rational",
            "m": self.gt_A['optimal_strategy']['m_star'],
            "Anonymization": self.gt_A['optimal_strategy']['anonymization_star'],
            "Participation Rate": self.gt_A['optimal_strategy']['r_star'],
            "Social Welfare": self.gt_A['equilibrium']['social_welfare'],
            "Consumer Surplus": self.gt_A['equilibrium']['consumer_surplus'],
            "Producer Profit": self.gt_A['equilibrium']['producer_profit'],
            "Intermediary Profit": self.gt_A['equilibrium']['intermediary_profit'],
            "Welfare Loss (%)": 0.0,
        }
        rows.append(row_A)
        
        # 配置B
        if results_B:
            row_B = {
                "Config": "B",
                "Intermediary": "Rational",
                "Consumers": "LLM",
                "m": self.gt_A['optimal_strategy']['m_star'],
                "Anonymization": self.gt_A['optimal_strategy']['anonymization_star'],
                "Participation Rate": results_B['participation']['r_llm'],
                "Social Welfare": results_B['market']['social_welfare_llm'],
                "Consumer Surplus": results_B['market']['consumer_surplus_llm'],
                "Producer Profit": results_B['market']['producer_profit_llm'],
                "Intermediary Profit": results_B['market']['intermediary_profit_llm'],
                "Welfare Loss (%)": results_B['market']['welfare_loss_percent'],
            }
            rows.append(row_B)
        
        # 配置C
        if results_C:
            row_C = {
                "Config": "C",
                "Intermediary": "LLM",
                "Consumers": "Rational",
                "m": results_C['strategy']['m_llm'],
                "Anonymization": results_C['strategy']['anon_llm'],
                "Participation Rate": results_C['participation_given_llm_strategy']['r_given_llm'],
                "Social Welfare": results_C['market']['social_welfare_llm'],
                "Consumer Surplus": results_C['market']['consumer_surplus_llm'],
                "Producer Profit": results_C['market']['producer_profit_llm'],
                "Intermediary Profit": results_C['profit']['profit_llm'],
                "Welfare Loss (%)": results_C['market']['welfare_loss_percent'],
            }
            rows.append(row_C)
        
        # 配置D
        if results_D:
            row_D = {
                "Config": "D",
                "Intermediary": "LLM",
                "Consumers": "LLM",
                "m": results_D['strategy']['m_llm'],
                "Anonymization": results_D['strategy']['anon_llm'],
                "Participation Rate": results_D['strategy']['r_llm'],
                "Social Welfare": results_D['market']['social_welfare_llm'],
                "Consumer Surplus": results_D['market']['consumer_surplus_llm'],
                "Producer Profit": results_D['market']['producer_profit_llm'],
                "Intermediary Profit": results_D['market']['intermediary_profit_llm'],
                "Welfare Loss (%)": results_D['market']['welfare_loss_percent'],
            }
            rows.append(row_D)
        
        df = pd.DataFrame(rows)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n报告已保存到: {output_path}")
        
        return df


# ============================================================================
# 直接运行示例
# ============================================================================

if __name__ == "__main__":
    """
    直接运行评估器
    
    使用configs/model_configs.json中配置的真实LLM模型
    """
    import sys
    import io
    from pathlib import Path
    from openai import OpenAI
    import re
    
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("场景C评估器 - 使用真实LLM模型")
    print("=" * 70)
    
    # ========================================================================
    # 加载模型配置
    # ========================================================================
    config_path = "configs/model_configs.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)
        print(f"\n✅ 成功加载模型配置: {config_path}")
        print(f"可用模型: {[cfg['config_name'] for cfg in model_configs]}")
    except FileNotFoundError:
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)
    
    # ========================================================================
    # 选择要评估的模型（直接指定模型名称）
    # ========================================================================
    # 修改这里来选择不同的模型
    TARGET_MODEL = "gpt-4.1-mini"  # 可选: grok-3-mini, gpt-4.1-mini, deepseek-v3, gemini-2.5-flash
    
    selected_model_config = None
    for config in model_configs:
        if config['config_name'] == TARGET_MODEL:
            selected_model_config = config
            break
    
    if selected_model_config is None:
        print(f"❌ 找不到模型配置: {TARGET_MODEL}")
        print(f"可用模型: {[cfg['config_name'] for cfg in model_configs]}")
        sys.exit(1)
    
    model_name = selected_model_config['config_name']
    print(f"\n🎯 选择模型: {model_name}")
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=selected_model_config['api_key'],
        **selected_model_config.get('client_args', {})
    )
    
    # ========================================================================
    # 真实LLM代理函数
    # ========================================================================
    
    def create_llm_consumer(client, model_config):
        """创建LLM消费者代理"""
        model_name = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        def _call_llm_consumer(consumer_params, m, anonymization):
            """调用LLM并返回(决策, 理由, 原始回复)"""
            # 构建提示词 v3：机制更清楚（但不提供“怎么算/该选什么”的步骤）
            prompt = f"""你是消费者，需要在“参与数据分享计划”与“拒绝参与”之间做选择。你的目标是最大化你的期望净效用（补偿 + 市场结果带来的收益 − 隐私成本）。

【数据分享计划】
- 若参与：你立刻获得补偿 m = {m:.2f}。
- 隐私机制：
  - identified：商家可能将你的数据与身份绑定，更容易对你做个性化定价（对高偏好者更可能不利）。
  - anonymized：商家只能利用匿名统计信息，通常更难针对个人定价，但仍可能从总体数据改进产品/定价。

【你的参数】
- 偏好强度 θ_i = {consumer_params['theta_i']:.2f} （越大表示你越喜欢该产品/更可能购买）
- 隐私成本 τ_i = {consumer_params['tau_i']:.2f} （参与会带来这项隐私损失成本）

【决策要求】
请你判断：参与是否“值得”。你不需要精确计算市场均衡，但要考虑以下要点：
1) 补偿 m 是参与的直接收益；
2) identified 可能导致对你更不利的个性化定价风险（尤其当 θ_i 较高时）；
3) anonymized 个性化定价风险较小，但补偿收益不变；
4) 参与会产生隐私成本 τ_i。

【输出格式（必须严格遵守）】
请按以下格式回答：
第1行：你的决策理由（50-100字）
第2行：决策：参与 或 决策：拒绝
"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **generate_args
            )
            answer = response.choices[0].message.content.strip()
            lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
            reason_line = lines[0] if lines else ""
            decision_line = ""
            for ln in reversed(lines):
                if ln.startswith("决策"):
                    decision_line = ln
                    break

            dl = decision_line.lower()
            if ("拒绝" in decision_line) or ("no" in dl):
                return False, reason_line, answer
            if ("参与" in decision_line) or ("yes" in dl):
                return True, reason_line, answer
            if "拒绝" in answer:
                return False, reason_line, answer
            if "参与" in answer:
                return True, reason_line, answer
            return bool(m > consumer_params['tau_i']), reason_line, answer

        def llm_consumer(consumer_params, m, anonymization):
            """LLM消费者决策"""
            try:
                decision, reason, answer = _call_llm_consumer(
                    consumer_params=consumer_params,
                    m=m,
                    anonymization=anonymization
                )
                # 打印LLM的完整回答（用于调试和理解）
                print(f"    [消费者 θ={consumer_params['theta_i']:.2f}, τ={consumer_params['tau_i']:.2f}] {answer[:80]}...")
                return bool(decision)
                    
            except Exception as e:
                print(f"⚠️ LLM调用失败: {e}")
                # 失败时使用简单的启发式
                return m > consumer_params['tau_i']
        
        # 修改：暴露带理由的调用接口（给配置D多轮学习使用）
        def llm_consumer_with_reason(consumer_params, m, anonymization):
            decision, reason, answer = _call_llm_consumer(
                consumer_params=consumer_params,
                m=m,
                anonymization=anonymization
            )
            return bool(decision), reason

        llm_consumer.with_reason = llm_consumer_with_reason

        return llm_consumer
    
    def create_llm_intermediary(client, model_config):
        """创建LLM中介代理"""
        model_name = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        def llm_intermediary(market_params, feedback=None, history=None):
            """LLM中介策略选择"""
            # 修改：在提示中加入“上一轮反馈/历史摘要”，引导多轮学习（但不教其计算方法）
            feedback_text = ""
            if feedback:
                reasons = feedback.get("reasons", {})
                feedback_text = f"""

【上一轮结果（仅供参考）】
- m = {feedback.get('m')}, anonymization = {feedback.get('anonymization')}
- 参与率 r = {feedback.get('participation_rate'):.4f}
- m0 = {feedback.get('m0'):.4f}
- 补偿成本 = {feedback.get('intermediary_cost'):.4f}
- 中介利润 = {feedback.get('intermediary_profit'):.4f}
- 参与者理由（逐条）: {reasons.get('participants')}
- 拒绝者理由（逐条）: {reasons.get('rejecters')}
"""

            prompt = f"""你是“数据中介”，你的目标是最大化你的期望利润。

【市场参数】
- 消费者数量 N = {market_params['N']}
- 消费者偏好 θ ~ Normal(均值 {market_params['mu_theta']:.2f}, 标准差 {market_params['sigma_theta']:.2f})
- 消费者隐私成本 τ ~ Normal(均值 {market_params['tau_mean']:.2f}, 标准差 {market_params['tau_std']:.2f})
- 数据结构：{market_params['data_structure']}

【你的决策（你先动）】
你要选择：
1) 补偿 m（范围 0 到 3）：你向每个参与者支付的金额
2) 匿名化策略：identified 或 anonymized

【消费者反应（随后）】
消费者会比较“参与的期望净收益”与其隐私成本 τ_i 来决定是否参与。
一般来说：m 越高，参与率越高；identified 可能降低参与激励（因个性化定价风险），anonymized 可能提高参与激励。

【你卖给商家的数据费 m0（关键）】
商家愿意为数据支付的价格 m0 由“数据带来的商家利润增量”决定：
m0 ≈ max(0, 期望[商家利润(有数据) − 商家利润(无数据)])
直觉：参与人数越多、数据越能支持个性化定价或改进产品，m0 越大，但边际收益递减。

【你的利润】
利润 = m0 − m × (参与人数)

【输出格式（必须严格遵守）】
只输出一行 JSON，不要输出任何额外文字：
{{"m": 数字,"anonymization":"identified" 或 "anonymized","reason":"50-100字"}}
{feedback_text}
请给出你的选择。
"""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **generate_args
                )
                
                answer = response.choices[0].message.content.strip()
                
                # 打印LLM的完整回答（用于调试和理解）
                print(f"  [中介策略] {answer[:100]}...")
                
                # 提取JSON
                json_match = re.search(r'\{[^}]+\}', answer)
                if json_match:
                    result = json.loads(json_match.group())
                    m = float(result['m'])
                    anon = result['anonymization']
                    
                    # 验证合法性
                    m = max(0.0, min(3.0, m))
                    if anon not in ['identified', 'anonymized']:
                        anon = 'anonymized'
                    
                    return m, anon
                else:
                    raise ValueError("无法解析JSON")
                    
            except Exception as e:
                print(f"⚠️ LLM调用失败: {e}，使用默认策略")
                # 失败时使用合理的默认值
                return 0.6, "anonymized"
        
        return llm_intermediary
    
    # ========================================================================
    # 创建LLM代理
    # ========================================================================
    print("\n创建LLM代理...")
    llm_consumer = create_llm_consumer(client, selected_model_config)
    llm_intermediary = create_llm_intermediary(client, selected_model_config)
    print("✅ LLM代理创建成功")
    
    # ========================================================================
    # 1. 初始化评估器
    # ========================================================================
    print("\n" + "=" * 70)
    print("步骤1: 加载Ground Truth")
    print("=" * 70)
    
    gt_path = "data/ground_truth/scenario_c_common_preferences_optimal.json"
    
    try:
        evaluator = ScenarioCEvaluator(gt_path)
        print(f"✅ 成功加载: {gt_path}")
        print(f"\n理论基准（配置A）:")
        print(f"  m* = {evaluator.gt_A['optimal_strategy']['m_star']:.4f}")
        print(f"  anonymization* = {evaluator.gt_A['optimal_strategy']['anonymization_star']}")
        print(f"  r* = {evaluator.gt_A['optimal_strategy']['r_star']:.4f}")
        print(f"  中介利润* = {evaluator.gt_A['optimal_strategy']['intermediary_profit_star']:.4f}")
        
    except FileNotFoundError:
        print(f"❌ 找不到Ground Truth文件: {gt_path}")
        print(f"\n请先运行以下命令生成Ground Truth:")
        print(f"  python -m src.scenarios.generate_scenario_c_gt")
        sys.exit(1)
    
    # ========================================================================
    # 2. 评估配置B（LLM消费者）
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"步骤2: 评估配置B（理性中介 × {model_name}消费者）")
    print("=" * 70)
    
    results_B = evaluator.evaluate_config_B(
        llm_consumer_agent=llm_consumer,
        verbose=True
    )
    
    # ========================================================================
    # 3. 评估配置C（LLM中介）
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"步骤3: 评估配置C（{model_name}中介 × 理性消费者）")
    print("=" * 70)
    
    # 修改：使用多轮学习版中介评估
    results_C = evaluator.evaluate_config_C_iterative(
        llm_intermediary_agent=llm_intermediary,
        rounds=20,
        verbose=True
    )
    
    # ========================================================================
    # 4. 评估配置D（双边LLM）
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"步骤4: 评估配置D（{model_name}中介 × {model_name}消费者）")
    print("=" * 70)
    
    # 修改：使用多轮学习版中介评估（配置D）
    results_D = evaluator.evaluate_config_D_iterative(
        llm_intermediary_agent=llm_intermediary,
        llm_consumer_agent=llm_consumer,
        rounds=20,
        verbose=True
    )
    
    # ========================================================================
    # 5. 生成综合报告
    # ========================================================================
    print("\n" + "=" * 70)
    print("步骤5: 生成综合报告")
    print("=" * 70)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation_results/scenario_c_{model_name}_{timestamp}.csv"
    df = evaluator.generate_report(
        results_B=results_B,
        results_C=results_C,
        results_D=results_D,
        output_path=output_path
    )
    
    print("\n报告预览:")
    print(df.to_string(index=False))
    
    # 6. 保存详细结果
    detailed_results = {
        "model": model_name,
        "timestamp": timestamp,
        "config_B": results_B,
        "config_C": results_C,
        "config_D": results_D,
    }
    
    output_json = f"evaluation_results/scenario_c_{model_name}_{timestamp}_detailed.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_json}")
    
    print("\n" + "=" * 70)
    print("✅ 评估完成！")
    print("=" * 70)
    print(f"\n📊 评估模型: {model_name}")
    print(f"📁 结果文件:")
    print(f"  • CSV报告: {output_path}")
    print(f"  • 详细JSON: {output_json}")
    print()
