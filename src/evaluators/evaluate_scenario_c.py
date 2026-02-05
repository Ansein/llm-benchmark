"""
场景C主评估器

支持4种配置的评估：
- 配置A：理性×理性（理论基准）
- 配置B：理性中介×LLM消费者
- 配置C：LLM中介×理性消费者
- 配置D：LLM中介×LLM消费者（多轮迭代学习）
- 配置D_FP：LLM中介×LLM消费者（虚拟博弈）

所有指标都是完全量化的、客观的。

========================================
运行方式
========================================

1. 【多轮迭代模式】（现有方法）
   运行配置B、C、D的完整评估：
   
   python -m src.evaluators.evaluate_scenario_c --mode iterative --model deepseek-v3.2
   
   # 自定义轮数（默认20轮）
   python -m src.evaluators.evaluate_scenario_c --mode iterative --model deepseek-v3.2 --rounds 30
   
   输出：
   - evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_YYYYMMDD_HHMMSS.csv
   - evaluation_results/scenario_c/scenario_c_common_preferences_deepseek-v3.2_YYYYMMDD_HHMMSS_detailed.json

2. 【虚拟博弈模式】（新增）
   只运行指定配置的虚拟博弈版本：
   
   # 一次运行所有三个配置（推荐）
   python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config all --model deepseek-v3.2
   
   # 单独运行配置B_FP：理性中介 × LLM消费者（消费者学习）
   python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config B --model deepseek-v3.2
   
   # 单独运行配置C_FP：LLM中介 × 理性消费者（中介学习）
   python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config C --model deepseek-v3.2
   
   # 单独运行配置D_FP：LLM中介 × LLM消费者（双方学习，默认）
   python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config D --model deepseek-v3.2
   
   # 自定义参数（默认50轮，信念窗口10）
   python -m src.evaluators.evaluate_scenario_c \
       --mode fp \
       --fp_config all \
       --model deepseek-v3.2 \
       --rounds 50 \
       --belief_window 10
   
   输出（每个配置一个目录）：
   - evaluation_results/scenario_c/fp_configB_deepseek-v3.2/eval_YYYYMMDD_HHMMSS.json
   - evaluation_results/scenario_c/fp_configC_deepseek-v3.2/eval_YYYYMMDD_HHMMSS.json
   - evaluation_results/scenario_c/fp_configD_deepseek-v3.2/eval_YYYYMMDD_HHMMSS.json
   - 对应的可视化图表（_profit_rate.png 和 _strategy_evolution.png）

3. 【为已有结果生成可视化】
   从已保存的FP结果JSON文件生成图表：
   
   # 单个文件
   python -m src.evaluators.evaluate_scenario_c --visualize evaluation_results/scenario_c/fp_deepseek-v3.2/eval_YYYYMMDD_HHMMSS.json
   
   # 整个目录
   python -m src.evaluators.evaluate_scenario_c --visualize evaluation_results/scenario_c/fp_deepseek-v3.2/
   
   # 多个路径
   python -m src.evaluators.evaluate_scenario_c --visualize path1.json path2/ path3/*.json

========================================
参数说明
========================================

--mode: 运行模式
  - iterative: 现有多轮迭代学习（运行B+C+D）
  - fp: 虚拟博弈（只运行指定配置的FP版本）

--fp_config: 虚拟博弈的配置选择（仅在mode=fp时有效）
  - all: 一次运行所有三个配置（推荐）
  - B: 配置B_FP（理性中介 × LLM消费者）- 测试消费者学习能力
  - C: 配置C_FP（LLM中介 × 理性消费者）- 测试中介学习能力
  - D: 配置D_FP（LLM中介 × LLM消费者）- 测试双方学习能力（默认）

--model: 模型配置名称
  默认: deepseek-v3.2
  可选: deepseek-v3.2, gpt-4.1-mini 等（见configs/model_configs.json）

--rounds: 多轮学习轮数
  - iterative模式默认: 20轮
  - fp模式默认: 50轮

--belief_window: 虚拟博弈信念窗口大小
  默认: 10（使用最近10轮历史）

--output-dir: 输出目录
  默认: evaluation_results/scenario_c

--visualize: 可视化模式
  为已有JSON文件生成图表（支持文件、目录、通配符）

========================================
虚拟博弈配置对比
========================================

| 配置     | 中介类型   | 消费者类型 | 中介学习 | 消费者学习 | 测试目标          |
|---------|-----------|-----------|---------|-----------|------------------|
| B_FP    | 理性      | LLM       | ❌      | ✅        | LLM消费者学习能力 |
| C_FP    | LLM       | 理性      | ✅      | ❌        | LLM中介学习能力   |
| D_FP    | LLM       | LLM       | ✅      | ✅        | 双方学习与均衡    |

========================================
Iterative vs FP 对比
========================================

| 特性           | Iterative（现有）      | FP（虚拟博弈）           |
|----------------|----------------------|-------------------------|
| 中介学习       | 按利润排序反馈       | 按时间序列历史趋势      |
| 消费者学习     | 无历史               | 观察最近N轮参与频率     |
| 收敛检测       | 无                   | 策略+参与集合稳定       |
| 默认轮数       | 20轮                 | 50轮（可提前收敛）      |
| 运行配置       | B+C+D                | B_FP或C_FP或D_FP      |
| 输出           | CSV报告              | JSON+可视化图表         |

========================================
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    compute_interaction_metrics,
    compute_consumer_metrics
)


# ========================================================================
# 辅助函数：带重试的API调用（全局函数，所有函数都可使用）
# ========================================================================
def call_llm_with_retry(client, model_name, messages, generate_args, max_attempts=5):
    """带重试机制的LLM API调用"""
    import time
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generate_args
            )
            return response
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 5  # 5秒、10秒、15秒、20秒
                print(f"⚠️ API调用失败 (尝试 {attempt+1}/{max_attempts}): {str(e)[:100]}")
                print(f"   等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"❌ API调用失败，已重试{max_attempts}次: {str(e)}")
                raise


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
                'data_structure': self.params_base['data_structure'],
            }
            if 's' in sample_data:
                consumer['s_i'] = float(sample_data['s'][i])
            
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
    ) -> Tuple[float, str, str, str]:
        """安全调用中介代理，兼容是否支持反馈/历史，并返回理由与原始输出"""
        if callable(llm_intermediary_agent):
            try:
                result = llm_intermediary_agent(
                    market_params=market_params,
                    feedback=feedback,
                    history=history
                )
            except TypeError:
                result = llm_intermediary_agent(market_params=market_params)
        else:
            result = llm_intermediary_agent.choose_strategy(
                market_params=market_params,
                feedback=feedback,
                history=history
            )

        if isinstance(result, tuple):
            if len(result) == 4:
                return result[0], result[1], str(result[2]), str(result[3])
            if len(result) == 3:
                return result[0], result[1], str(result[2]), ""
            if len(result) == 2:
                return result[0], result[1], "", ""
        return float(result), "anonymized", "", ""
    
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
            # 处理m_star（可能是向量）
            if isinstance(m_star, (list, np.ndarray)):
                m_mean = self.gt_A['optimal_strategy'].get('m_star_mean', np.mean(m_star))
                print(f"\n理论最优策略: m*=向量(均值={m_mean:.4f}), {anon_star}")
            else:
                print(f"\n理论最优策略: m*={m_star:.4f}, {anon_star}")
            print(f"理论参与率: r*={r_star:.4f}")
        
        # 2. 获取消费者数据
        consumers = self._get_sample_consumers()
        N = self.params_base['N']
        
        # 3. LLM决策
        if verbose:
            print(f"\n正在收集{N}个LLM消费者的决策...")
        
        llm_decisions = []
        for i, consumer_params in enumerate(consumers):
            # 处理m_star（可能是向量或标量）
            if isinstance(m_star, (list, np.ndarray)):
                m_i = m_star[i]  # 使用消费者i的个性化补偿
            else:
                m_i = m_star  # 统一补偿
            
            # 调用LLM代理
            if callable(llm_consumer_agent):
                decision = llm_consumer_agent(
                    consumer_params=consumer_params,
                    m=m_i,
                    anonymization=anon_star
                )
            else:
                decision = llm_consumer_agent.decide(
                    consumer_params=consumer_params,
                    m=m_i,
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
            "consumer": compute_consumer_metrics(
                llm_decisions=llm_decisions,
                theory_decisions=theory_decisions,
                outcome_llm=outcome_llm_dict,
                outcome_theory=outcome_theory
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
    
    def evaluate_config_B_fictitious_play(
        self,
        llm_client,
        model_config: Dict,
        max_rounds: int = 50,
        belief_window: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        配置B（虚拟博弈版）：理性中介 × LLM消费者（Fictitious Play）
        
        中介：始终使用理论最优策略（m*, anon*）
        消费者：基于历史学习其他消费者的参与概率
        
        Args:
            llm_client: OpenAI格式的LLM客户端
            model_config: 模型配置字典
            max_rounds: 最大轮数
            belief_window: 信念窗口大小
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        import re
        
        model_name = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        if verbose:
            print("\n" + "="*70)
            print("配置B（虚拟博弈）：理性中介 × LLM消费者 - Fictitious Play")
            print(f"最大轮数: {max_rounds}, 信念窗口: {belief_window}")
            print("="*70)
        
        # 理论最优策略（固定不变）
        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        r_star = self.gt_A['optimal_strategy']['r_star']
        
        if verbose:
            # 处理m_star（可能是向量）
            if isinstance(m_star, (list, np.ndarray)):
                m_mean = self.gt_A['optimal_strategy'].get('m_star_mean', np.mean(m_star))
                print(f"\n理性中介策略: m*=向量(均值={m_mean:.4f}), {anon_star}")
            else:
                print(f"\n理性中介策略: m*={m_star:.4f}, {anon_star}")
        
        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }
        
        history: List[Dict] = []
        consumers = self._get_sample_consumers()
        N = market_params['N']
        
        # ===== 虚拟博弈迭代 =====
        for round_num in range(max_rounds):
            if verbose:
                print(f"\n{'='*70}")
                print(f"[轮次 {round_num + 1}/{max_rounds}]")
                print(f"{'='*70}")
            
            # 1. 计算消费者信念
            window = min(belief_window, round_num)
            consumer_belief_probs = self._compute_consumer_belief(history, window_size=window, N=N)
            
            # 2. 消费者决策（基于历史）
            llm_decisions: List[bool] = []
            
            for idx, consumer_params in enumerate(consumers):
                consumer_params['consumer_id'] = idx
                
                # 处理m_star（可能是向量或标量）
                if isinstance(m_star, (list, np.ndarray)):
                    m_i = m_star[idx]  # 使用消费者i的个性化补偿
                else:
                    m_i = m_star  # 统一补偿
                
                # 构建消费者FP提示词
                consumer_prompt = self.build_consumer_prompt_fp(
                    consumer_params=consumer_params,
                    m=m_i,
                    anonymization=anon_star,
                    history=history,
                    consumer_belief_probs=consumer_belief_probs,
                    belief_window=belief_window,
                    N=N
                )
                
                # 调用消费者LLM（使用重试机制）
                try:
                    response = call_llm_with_retry(
                        client=llm_client,
                        model_name=model_name,
                        messages=[{"role": "user", "content": consumer_prompt}],
                        generate_args=generate_args
                    )
                    answer = response.choices[0].message.content.strip()
                    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                    decision_line = ""
                    for ln in reversed(lines):
                        if ln.startswith("决策"):
                            decision_line = ln
                            break
                    
                    dl = decision_line.lower()
                    if ("拒绝" in decision_line) or ("no" in dl):
                        decision = False
                    elif ("参与" in decision_line) or ("yes" in dl):
                        decision = True
                    elif "拒绝" in answer:
                        decision = False
                    elif "参与" in answer:
                        decision = True
                    else:
                        decision = m_i > consumer_params['tau_i']
                    
                    llm_decisions.append(bool(decision))
                except Exception as e:
                    if verbose and idx < 5:
                        print(f"[WARN] 消费者{idx}决策失败: {e}")
                    llm_decisions.append(False)
            
            llm_decisions_arr = np.array(llm_decisions)
            r_llm = float(np.mean(llm_decisions_arr))
            participation_set = [i for i, d in enumerate(llm_decisions) if d]
            
            # 3. 计算市场结果
            params = ScenarioCParams(
                m=m_star,
                anonymization=anon_star,
                **self.params_base
            )
            rng = np.random.default_rng(self.params_base['seed'])
            consumer_data = generate_consumer_data(params, rng=rng)
            
            from src.scenarios.scenario_c_social_data import estimate_m0_mc
            
            def participation_rule(p, world, rng):
                return llm_decisions_arr
            
            m_0, _, _, _ = estimate_m0_mc(
                params=params,
                participation_rule=participation_rule,
                T=100,
                beta=1.0,
                seed=self.params_base['seed']
            )
            
            outcome = simulate_market_outcome(
                consumer_data,
                llm_decisions_arr,
                params,
                producer_info_mode="with_data",
                m0=m_0,
                rng=rng
            )
            
            # 4. 记录历史
            round_info = {
                "round": round_num + 1,
                "m": float(m_star),
                "anonymization": anon_star,
                "participation_rate": r_llm,
                "num_participants": int(np.sum(llm_decisions_arr)),
                "participation_set": participation_set,
                "m0": float(m_0),
                "intermediary_profit": float(outcome.intermediary_profit),
            }
            history.append(round_info)
            
            if verbose:
                print(f"参与率: {r_llm:.2%} ({round_info['num_participants']}/{N})")
                print(f"利润: {round_info['intermediary_profit']:.4f}")
            
            # 5. 检查收敛（参与集合稳定）
            if len(history) >= 3:
                last_3_sets = [
                    frozenset(h.get('participation_set', [])) 
                    for h in history[-3:]
                ]
                if len(set(last_3_sets)) == 1:
                    if verbose:
                        print(f"\n[提前收敛] 连续3轮参与集合不变，停止迭代")
                    break
        
        # ===== 最终分析 =====
        if verbose:
            print(f"\n{'='*70}")
            print(f"[虚拟博弈结束] 总轮数: {len(history)}")
            print(f"{'='*70}")
        
        # 简化的收敛分析
        participation_rate_traj = [h['participation_rate'] for h in history]
        profit_traj = [h['intermediary_profit'] for h in history]
        
        final_rate = participation_rate_traj[-1]
        final_profit = profit_traj[-1]
        
        # 检查是否收敛
        converged = False
        convergence_round = None
        if len(history) >= 3:
            last_3_sets = [frozenset(h.get('participation_set', [])) for h in history[-3:]]
            converged = len(set(last_3_sets)) == 1
            if converged:
                for i in range(2, len(history)):
                    check_sets = [frozenset(h.get('participation_set', [])) for h in history[i-2:i+1]]
                    if len(set(check_sets)) == 1:
                        convergence_round = i - 2
                        break
        
        if verbose:
            print(f"\n[收敛性分析]")
            print(f"是否收敛: {converged}")
            if converged and convergence_round is not None:
                print(f"收敛轮数: {convergence_round}")
            print(f"最终参与率: {final_rate:.2%} (理论最优 {r_star:.2%})")
            print(f"最终利润: {final_profit:.4f}")
            print(f"参与率误差: {abs(final_rate - r_star):.4f}")
        
        results = {
            "model_name": model_config.get('config_name', 'unknown'),
            "game_type": "fictitious_play",
            "config": "B_fictitious_play",
            "max_rounds": max_rounds,
            "actual_rounds": len(history),
            "belief_window": belief_window,
            "history": history,
            "convergence_analysis": {
                "converged": converged,
                "convergence_round": convergence_round,
                "participation_rate_trajectory": participation_rate_traj,
                "profit_trajectory": profit_traj,
                "final_participation_rate": final_rate,
                "final_profit": final_profit
            },
            "ground_truth": {
                "m_star": m_star,
                "anonymization_star": anon_star,
                "r_star": r_star
            },
            "final_strategy": {
                "participation_rate": final_rate,
                "profit": final_profit
            }
        }
        
        return results
    
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
            # 处理m_star（可能是向量）
            if isinstance(m_star, (list, np.ndarray)):
                m_mean = self.gt_A['optimal_strategy'].get('m_star_mean', np.mean(m_star))
                print(f"\n理论最优策略: m*=向量(均值={m_mean:.4f}), {anon_star}")
            else:
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
            'm_init': 1.0,
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
        # 处理m_llm（可能是向量）
        m_llm_for_cost = np.mean(m_llm) if isinstance(m_llm, (list, np.ndarray)) else m_llm
        cost_llm = m_llm_for_cost * result_llm.num_participants
        
        # 使用GT中的intermediary_cost，如果不存在则从m_star计算
        if 'intermediary_cost' in self.gt_A.get('data_transaction', {}):
            cost_theory = self.gt_A['data_transaction']['intermediary_cost']
        else:
            m_star_for_cost = np.mean(m_star) if isinstance(m_star, (list, np.ndarray)) else m_star
            cost_theory = m_star_for_cost * self.gt_A['optimal_strategy'].get('num_participants_expected', 0)
        
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
            'm_init': 1.0,
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
            m_llm, anon_llm, intermediary_reason, intermediary_raw = self._call_intermediary_agent(
                llm_intermediary_agent,
                market_params=market_params,
                feedback=feedback,
                history=history
            )

            if verbose:
                if intermediary_raw:
                    print(f"[中介策略] {intermediary_raw}")
                print(f"LLM选择: m={m_llm:.4f}, {anon_llm}")
                if intermediary_reason:
                    print(f"中介理由: {intermediary_reason}")

            # 计算理性消费者对该策略的反应
            try:
                result_llm = evaluate_intermediary_strategy(
                    m=m_llm,
                    anonymization=anon_llm,
                    params_base=self.params_base,
                    num_mc_samples=50,
                    max_iter=100,
                    tol=1e-3,
                    seed=self.params_base['seed']
                )
            except RuntimeError as e:
                # 修改：固定点不收敛时不中断整个评估，记录失败轮次并继续
                if verbose:
                    print(f"⚠️ 固定点未收敛，跳过该轮: {e}")
                round_info = {
                    "round": t,
                    "m": float(m_llm),
                    "anonymization": anon_llm,
                    "intermediary_reason": intermediary_reason,
                    "intermediary_raw": intermediary_raw,
                    "participation_rate": 0.0,
                    "num_participants": 0,
                    "m0": 0.0,
                    "intermediary_cost": 0.0,
                    "intermediary_profit": -1e9,
                    "reasons": {
                        "participants": [],
                        "rejecters": []
                    },
                    "converged": False,
                    "error": str(e)
                }
                history.append(round_info)
                if verbose:
                    print("本轮结果（未收敛）:")
                    print(f"  m={round_info['m']:.4f}, anonymization={round_info['anonymization']}")
                    if round_info.get("intermediary_reason"):
                        print(f"  中介理由: {round_info['intermediary_reason']}")
                    print(f"  r={round_info['participation_rate']:.4f}, num={round_info['num_participants']}")
                    print(f"  m0={round_info['m0']:.4f}, cost={round_info['intermediary_cost']:.4f}")
                    print(f"  profit={round_info['intermediary_profit']:.4f}")
                    print(f"  error={round_info.get('error')}")
                continue

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
                "intermediary_reason": intermediary_reason,
                "intermediary_raw": intermediary_raw,
                "participation_rate": float(r_given_llm),
                "num_participants": num_participants,
                "m0": float(result_llm.m_0),
                "intermediary_cost": float(result_llm.intermediary_cost),
                "intermediary_profit": float(profit_llm),
                "reasons": {
                    "participants": reasons_participants,
                    "rejecters": reasons_rejecters
                },
                "converged": True
            }
            history.append(round_info)

            if (best_round is None) or (profit_llm > best_round["intermediary_profit"]):
                best_round = round_info

            if verbose:
                print("本轮结果:")
                print(f"  m={round_info['m']:.4f}, anonymization={round_info['anonymization']}")
                if round_info.get("intermediary_reason"):
                    print(f"  中介理由: {round_info['intermediary_reason']}")
                print(f"  r={round_info['participation_rate']:.4f}, num={round_info['num_participants']}")
                print(f"  m0={round_info['m0']:.4f}, cost={round_info['intermediary_cost']:.4f}")
                print(f"  profit={round_info['intermediary_profit']:.4f}")

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

        # 处理m_llm（可能是向量）
        m_llm_for_cost = np.mean(m_llm) if isinstance(m_llm, (list, np.ndarray)) else m_llm
        cost_llm = m_llm_for_cost * result_best.num_participants
        
        # 使用GT中的intermediary_cost，如果不存在则从m_star计算
        if 'intermediary_cost' in self.gt_A.get('data_transaction', {}):
            cost_theory = self.gt_A['data_transaction']['intermediary_cost']
        else:
            m_star_for_cost = np.mean(m_star) if isinstance(m_star, (list, np.ndarray)) else m_star
            cost_theory = m_star_for_cost * self.gt_A['optimal_strategy'].get('num_participants_expected', 0)

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
            'm_init': 1.0,
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

            m_llm, anon_llm, intermediary_reason, intermediary_raw = self._call_intermediary_agent(
                llm_intermediary_agent,
                market_params=market_params,
                feedback=feedback,
                history=history
            )

            if verbose:
                if intermediary_raw:
                    print(f"[中介策略] {intermediary_raw}")
                print(f"LLM中介选择: m={m_llm:.4f}, {anon_llm}")
                if intermediary_reason:
                    print(f"中介理由: {intermediary_reason}")

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
                "intermediary_reason": intermediary_reason,
                "intermediary_raw": intermediary_raw,
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
                print("本轮结果:")
                print(f"  m={round_info['m']:.4f}, anonymization={round_info['anonymization']}")
                if round_info.get("intermediary_reason"):
                    print(f"  中介理由: {round_info['intermediary_reason']}")
                print(f"  r={round_info['participation_rate']:.4f}, num={round_info['num_participants']}")
                print(f"  m0={round_info['m0']:.4f}, cost={round_info['intermediary_cost']:.4f}")
                print(f"  profit={round_info['intermediary_profit']:.4f}")

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
            'gini_coefficient': outcome_best.gini_coefficient,
        }

        # 计算同策略下理性消费者基准（用于消费者指标）
        result_rational = evaluate_intermediary_strategy(
            m=m_llm,
            anonymization=anon_llm,
            params_base=self.params_base,
            num_mc_samples=50,
            max_iter=100,
            tol=1e-3,
            seed=self.params_base['seed']
        )
        theory_decisions = self._get_theory_decisions(
            float(result_rational.delta_u),
            consumers
        )
        outcome_rational_dict = {
            'social_welfare': result_rational.social_welfare,
            'consumer_surplus': result_rational.consumer_surplus,
            'producer_profit': result_rational.producer_profit_with_data,
            'intermediary_profit': result_rational.intermediary_profit,
            'gini_coefficient': result_rational.gini_coefficient
        }

        metrics = {
            "config": "D_llm_intermediary_llm_consumer_iterative",
            "strategy": {
                "m_llm": m_llm,
                "anon_llm": anon_llm,
                "r_llm": float(np.mean(llm_decisions_arr)),
            },
            "vs_theory": {
                "m_error": abs(m_llm - (np.mean(m_star) if isinstance(m_star, (list, np.ndarray)) else m_star)),
                "anon_match": int(anon_llm == anon_star),
                "r_error": abs(float(np.mean(llm_decisions_arr)) - self.gt_A['optimal_strategy']['r_star']),
            },
            "consumer": compute_consumer_metrics(
                llm_decisions=llm_decisions_arr,
                theory_decisions=theory_decisions,
                outcome_llm=outcome_best_dict,
                outcome_theory=outcome_rational_dict
            ),
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
    
    def evaluate_config_C_fictitious_play(
        self,
        llm_client,
        model_config: Dict,
        max_rounds: int = 50,
        belief_window: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        配置C（虚拟博弈版）：LLM中介 × 理性消费者（Fictitious Play）
        
        中介：基于历史学习参与率趋势
        消费者：理性决策（给定中介策略后做最优反应）
        
        Args:
            llm_client: OpenAI格式的LLM客户端
            model_config: 模型配置字典
            max_rounds: 最大轮数
            belief_window: 信念窗口大小
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        import re
        
        model_name = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        if verbose:
            print("\n" + "="*70)
            print("配置C（虚拟博弈）：LLM中介 × 理性消费者 - Fictitious Play")
            print(f"最大轮数: {max_rounds}, 信念窗口: {belief_window}")
            print("="*70)
        
        m_star = self.gt_A['optimal_strategy']['m_star']
        anon_star = self.gt_A['optimal_strategy']['anonymization_star']
        profit_star = self.gt_A['optimal_strategy']['intermediary_profit_star']
        r_star = self.gt_A['optimal_strategy']['r_star']
        
        market_params = {
            'N': self.params_base['N'],
            'mu_theta': self.params_base['mu_theta'],
            'sigma_theta': self.params_base['sigma_theta'],
            'tau_mean': self.params_base['tau_mean'],
            'tau_std': self.params_base['tau_std'],
            'data_structure': self.params_base['data_structure'],
        }
        
        history: List[Dict] = []
        consumers = self._get_sample_consumers()
        
        # ===== 虚拟博弈迭代 =====
        for round_num in range(max_rounds):
            if verbose:
                print(f"\n{'='*70}")
                print(f"[轮次 {round_num + 1}/{max_rounds}]")
                print(f"{'='*70}")
            
            # 1. 计算中介信念
            window = min(belief_window, round_num)
            intermediary_belief = self._compute_intermediary_belief(history, window_size=window)
            
            # 2. 中介决策（基于历史）
            intermediary_prompt = self.build_intermediary_prompt_fp(
                market_params=market_params,
                history=history,
                belief_stats=intermediary_belief,
                belief_window=belief_window
            )
            
            # 调用中介LLM（使用重试机制）
            try:
                response = call_llm_with_retry(
                    client=llm_client,
                    model_name=model_name,
                    messages=[{"role": "user", "content": intermediary_prompt}],
                    generate_args=generate_args
                )
                answer = response.choices[0].message.content.strip()
                json_match = re.search(r'\{[^}]+\}', answer)
                if json_match:
                    answer = json_match.group(0)
                obj = json.loads(answer)
                m_llm = float(obj.get("m", 0.5))
                anon_llm = obj.get("anonymization", "anonymized")
                intermediary_reason = str(obj.get("reason", "")).strip()
            except Exception as e:
                print(f"[WARN] 中介决策失败: {e}，使用默认策略")
                m_llm = 0.5
                anon_llm = "anonymized"
                intermediary_reason = "调用失败"
            
            if verbose:
                print(f"中介选择: m={m_llm:.4f}, {anon_llm}")
                if intermediary_reason:
                    print(f"中介理由: {intermediary_reason}")
            
            # 3. 理性消费者决策
            # 使用理论最优决策规则
            params = ScenarioCParams(
                m=m_llm,
                anonymization=anon_llm,
                **self.params_base
            )
            rng = np.random.default_rng(self.params_base['seed'])
            consumer_data = generate_consumer_data(params, rng=rng)
            
            from src.scenarios.scenario_c_social_data import estimate_m0_mc
            
            # 获取tau值（从consumers列表）
            tau_array = np.array([c['tau_i'] for c in consumers])
            
            # 理性决策：基于delta_u
            if anon_llm == "identified":
                # identified情况：参与门槛略高
                delta_u = m_llm - 0.1  # 简化估算，考虑个性化定价风险
                
                def rational_participation_rule(p, world, rng):
                    return tau_array <= delta_u
                
                m_0, _, _, _ = estimate_m0_mc(
                    params=params,
                    participation_rule=rational_participation_rule,
                    T=100,
                    beta=1.0,
                    seed=self.params_base['seed']
                )
                
                rational_decisions = rational_participation_rule(None, consumer_data, rng)
            else:
                # anonymized情况：参与门槛为m
                delta_u = m_llm
                
                def rational_participation_rule(p, world, rng):
                    return tau_array <= delta_u
                
                m_0, _, _, _ = estimate_m0_mc(
                    params=params,
                    participation_rule=rational_participation_rule,
                    T=100,
                    beta=1.0,
                    seed=self.params_base['seed']
                )
                
                rational_decisions = rational_participation_rule(None, consumer_data, rng)
            
            r_rational = float(np.mean(rational_decisions))
            participation_set = [i for i, d in enumerate(rational_decisions) if d]
            
            outcome = simulate_market_outcome(
                consumer_data,
                rational_decisions,
                params,
                producer_info_mode="with_data",
                m0=m_0,
                rng=rng
            )
            
            # 4. 记录历史
            round_info = {
                "round": round_num + 1,
                "m": float(m_llm),
                "anonymization": anon_llm,
                "intermediary_reason": intermediary_reason,
                "participation_rate": r_rational,
                "num_participants": int(np.sum(rational_decisions)),
                "participation_set": participation_set,
                "m0": float(m_0),
                "intermediary_profit": float(outcome.intermediary_profit),
            }
            history.append(round_info)
            
            if verbose:
                print(f"理性消费者参与率: {r_rational:.2%} ({round_info['num_participants']}/{market_params['N']})")
                print(f"中介利润: {round_info['intermediary_profit']:.4f}")
            
            # 5. 检查收敛
            if self._check_fp_convergence(history, threshold=3):
                if verbose:
                    print(f"\n[提前收敛] 连续3轮策略和参与集合不变，停止迭代")
                break
        
        # ===== 最终分析 =====
        if verbose:
            print(f"\n{'='*70}")
            print(f"[虚拟博弈结束] 总轮数: {len(history)}")
            print(f"{'='*70}")
        
        # 收敛性分析
        convergence_analysis = self._analyze_fp_convergence(history, self.gt_A['optimal_strategy'])
        
        if verbose:
            print(f"\n[收敛性分析]")
            print(f"是否收敛: {convergence_analysis['converged']}")
            if convergence_analysis['converged']:
                print(f"收敛轮数: {convergence_analysis['convergence_round']}")
            print(f"最终策略: m={convergence_analysis['final_m']:.4f}, {convergence_analysis['final_anonymization']}")
            print(f"最终参与率: {convergence_analysis['final_participation_rate']:.2%}")
            print(f"最终利润: {convergence_analysis['final_profit']:.4f} (理论最优 {profit_star:.4f})")
            print(f"与理论最优对比:")
            print(f"  m误差: {convergence_analysis['similarity_to_optimal']['m_error']:.4f}")
            print(f"  anon匹配: {convergence_analysis['similarity_to_optimal']['anon_match']}")
            print(f"  利润比率: {convergence_analysis['similarity_to_optimal']['profit_ratio']:.2%}")
        
        results = {
            "model_name": model_config.get('config_name', 'unknown'),
            "game_type": "fictitious_play",
            "config": "C_fictitious_play",
            "max_rounds": max_rounds,
            "actual_rounds": len(history),
            "belief_window": belief_window,
            "history": history,
            "convergence_analysis": convergence_analysis,
            "ground_truth": {
                "m_star": m_star,
                "anonymization_star": anon_star,
                "profit_star": profit_star,
                "r_star": r_star
            },
            "final_strategy": {
                "m": convergence_analysis['final_m'],
                "anonymization": convergence_analysis['final_anonymization'],
                "participation_rate": convergence_analysis['final_participation_rate'],
                "profit": convergence_analysis['final_profit']
            }
        }
        
        return results
    
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
            'm_init': 1.0,
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
        for i, consumer_params in enumerate(consumers):
            # 处理m_llm（可能是向量或标量）
            if isinstance(m_llm, (list, np.ndarray)):
                m_i = m_llm[i]  # 使用消费者i的个性化补偿
            else:
                m_i = m_llm  # 统一补偿
            
            if callable(llm_consumer_agent):
                decision = llm_consumer_agent(
                    consumer_params=consumer_params,
                    m=m_i,
                    anonymization=anon_llm
                )
            else:
                decision = llm_consumer_agent.decide(
                    consumer_params=consumer_params,
                    m=m_i,
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
            'gini_coefficient': outcome_D.gini_coefficient,
        }

        # 同策略下理性消费者基准（用于消费者指标）
        result_rational = evaluate_intermediary_strategy(
            m=m_llm,
            anonymization=anon_llm,
            params_base=self.params_base,
            num_mc_samples=50,
            max_iter=100,
            tol=1e-3,
            seed=self.params_base['seed']
        )
        theory_decisions = self._get_theory_decisions(
            float(result_rational.delta_u),
            consumers
        )
        outcome_rational_dict = {
            'social_welfare': result_rational.social_welfare,
            'consumer_surplus': result_rational.consumer_surplus,
            'producer_profit': result_rational.producer_profit_with_data,
            'intermediary_profit': result_rational.intermediary_profit,
            'gini_coefficient': result_rational.gini_coefficient
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
            "consumer": compute_consumer_metrics(
                llm_decisions=llm_decisions,
                theory_decisions=theory_decisions,
                outcome_llm=outcome_D_dict,
                outcome_theory=outcome_rational_dict
            ),
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
    
    # ========================================================================
    # 虚拟博弈（Fictitious Play）相关方法
    # ========================================================================
    
    def _compute_intermediary_belief(
        self,
        history: List[Dict],
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        计算中介基于历史的信念（参与率趋势）
        
        Args:
            history: 历史记录
            window_size: 窗口大小
        
        Returns:
            {"avg_participation_rate": float,
             "avg_rate_identified": float,
             "avg_rate_anonymized": float}
        """
        if len(history) == 0:
            return {
                "avg_participation_rate": 0.5,
                "avg_rate_identified": 0.5,
                "avg_rate_anonymized": 0.5
            }
        
        recent_history = history[-window_size:]
        
        all_rates = [h['participation_rate'] for h in recent_history]
        identified_rates = [h['participation_rate'] for h in recent_history if h['anonymization'] == 'identified']
        anonymized_rates = [h['participation_rate'] for h in recent_history if h['anonymization'] == 'anonymized']
        
        return {
            "avg_participation_rate": float(np.mean(all_rates)) if all_rates else 0.5,
            "avg_rate_identified": float(np.mean(identified_rates)) if identified_rates else None,
            "avg_rate_anonymized": float(np.mean(anonymized_rates)) if anonymized_rates else None
        }
    
    def _compute_consumer_belief(
        self,
        history: List[Dict],
        window_size: int = 10,
        N: int = 100
    ) -> Dict[int, float]:
        """
        计算消费者基于历史的信念（其他消费者的参与概率）
        
        Args:
            history: 历史记录
            window_size: 窗口大小
            N: 消费者总数
        
        Returns:
            {consumer_id: 参与概率}
        """
        if len(history) == 0:
            return {i: 0.5 for i in range(N)}
        
        recent_history = history[-window_size:]
        
        belief_probs = {}
        for consumer_id in range(N):
            participate_count = sum(
                1 for h in recent_history 
                if consumer_id in h.get('participation_set', [])
            )
            belief_probs[consumer_id] = participate_count / len(recent_history)
        
        return belief_probs
    
    def _check_fp_convergence(
        self,
        history: List[Dict],
        threshold: int = 3
    ) -> bool:
        """
        检查FP是否收敛
        
        收敛条件：连续threshold轮
        - m变化 < 0.01
        - anon不变
        - 参与集合不变
        
        Args:
            history: 历史记录
            threshold: 稳定轮数阈值
        
        Returns:
            是否收敛
        """
        if len(history) < threshold:
            return False
        
        recent = history[-threshold:]
        
        # 检查m变化
        m_values = [h['m'] for h in recent]
        m_stable = all(abs(m_values[i] - m_values[i+1]) < 0.01 for i in range(len(m_values)-1))
        
        # 检查anon不变
        anon_values = [h['anonymization'] for h in recent]
        anon_stable = len(set(anon_values)) == 1
        
        # 检查参与集合不变
        participation_sets = [
            frozenset(h.get('participation_set', []))
            for h in recent
        ]
        participation_stable = len(set(participation_sets)) == 1
        
        return m_stable and anon_stable and participation_stable
    
    def _analyze_fp_convergence(
        self,
        history: List[Dict],
        gt_optimal: Dict
    ) -> Dict[str, Any]:
        """
        分析FP的收敛性
        
        Args:
            history: 完整历史
            gt_optimal: 理论最优策略
        
        Returns:
            收敛分析结果
        """
        n_rounds = len(history)
        
        # 提取轨迹
        m_trajectory = [h['m'] for h in history]
        anon_trajectory = [h['anonymization'] for h in history]
        participation_rate_trajectory = [h['participation_rate'] for h in history]
        profit_trajectory = [h.get('intermediary_profit', 0) for h in history]
        
        # 检测收敛轮数
        convergence_round = None
        for i in range(2, n_rounds):
            if self._check_fp_convergence(history[:i+1], threshold=3):
                convergence_round = i - 2  # 从这一轮开始稳定
                break
        
        # 计算稳定性
        if n_rounds >= 10:
            last_10_participation = participation_rate_trajectory[-10:]
            participation_stability = 1.0 - np.std(last_10_participation)
        else:
            participation_stability = 0.0
        
        # 与理论最优对比
        final_m = m_trajectory[-1]
        final_anon = anon_trajectory[-1]
        final_profit = profit_trajectory[-1]
        
        m_star = gt_optimal['m_star']
        anon_star = gt_optimal['anonymization_star']
        profit_star = gt_optimal['intermediary_profit_star']
        
        return {
            "converged": convergence_round is not None,
            "convergence_round": convergence_round,
            "participation_stability": float(participation_stability),
            "m_trajectory": m_trajectory,
            "anon_trajectory": anon_trajectory,
            "participation_rate_trajectory": participation_rate_trajectory,
            "profit_trajectory": profit_trajectory,
            "final_m": final_m,
            "final_anonymization": final_anon,
            "final_participation_rate": participation_rate_trajectory[-1],
            "final_profit": final_profit,
            "similarity_to_optimal": {
                "m_error": abs(final_m - m_star),
                "anon_match": final_anon == anon_star,
                "profit_ratio": final_profit / profit_star if profit_star > 0 else 0
            }
        }
    
    def build_intermediary_prompt_fp(
        self,
        market_params: Dict,
        history: List[Dict],
        belief_stats: Dict,
        belief_window: int = 10
    ) -> str:
        """构建中介的FP提示词（隐式信念）"""
        
        recent_history = history[-belief_window:] if history else []
        
        # 构建历史文本
        if len(recent_history) == 0:
            history_text = """
【历史记录】
这是第一轮，暂无历史记录。你可以假设大约50%的消费者会参与。"""
        else:
            # 展示历史（按时间顺序）
            history_lines = []
            for h in recent_history:
                history_lines.append(
                    f"- 轮次{h['round']}: "
                    f"m={h['m']:.3f}, {h['anonymization']}, "
                    f"参与率={h['participation_rate']:.1%}, "
                    f"利润={h.get('intermediary_profit', 0):.3f}"
                )
            
            # 计算统计趋势
            trend_text = "\n【参与率趋势】"
            if belief_stats.get('avg_rate_identified') is not None:
                trend_text += f"\n- identified 策略下平均参与率: {belief_stats['avg_rate_identified']:.1%}"
            if belief_stats.get('avg_rate_anonymized') is not None:
                trend_text += f"\n- anonymized 策略下平均参与率: {belief_stats['avg_rate_anonymized']:.1%}"
            
            history_text = f"""
【历史记录】（最近{len(recent_history)}轮）
{chr(10).join(history_lines)}
{trend_text}"""
        
        # 构建完整提示词
        prompt = f"""你是"数据中介"，目标是最大化你的利润。

【市场参数】
- 消费者数量 N = {market_params['N']}
- 消费者偏好 θ ~ Normal(均值 {market_params['mu_theta']:.2f}, 标准差 {market_params['sigma_theta']:.2f})
- 消费者隐私成本 τ ~ Normal(均值 {market_params['tau_mean']:.2f}, 标准差 {market_params['tau_std']:.2f})
- 数据结构：{market_params['data_structure']}

{history_text}

【你的决策】
选择本轮策略：
1) 补偿金额 m（建议范围 0.1 到 2.0）
2) 匿名化策略：identified 或 anonymized

【机制说明】
- 消费者会权衡"补偿收益"与"隐私成本+价格歧视风险"
- identified：商家可识别消费者，能精准定价，但消费者参与意愿可能降低
- anonymized：商家只能统一定价，消费者更可能参与
- 你的利润 = m0（商家支付）− m × 参与人数

【输出格式】
只输出一行JSON，不要额外文字：
{{"m": 数字, "anonymization": "identified"或"anonymized", "reason": "50-100字理由"}}

请基于历史记录，选择本轮策略以最大化利润。
"""
        return prompt
    
    def build_consumer_prompt_fp(
        self,
        consumer_params: Dict,
        m: float,
        anonymization: str,
        history: List[Dict],
        consumer_belief_probs: Dict[int, float],
        belief_window: int = 10,
        N: int = 100
    ) -> str:
        """构建消费者的FP提示词（隐式信念）"""
        
        consumer_id = consumer_params.get('consumer_id', -1)
        recent_history = history[-belief_window:] if history else []
        
        # 构建历史文本
        if len(recent_history) == 0:
            history_text = """
【历史观察】
这是第一轮，暂无历史记录。你可以假设其他消费者大约各有50%的概率参与。"""
        else:
            # 展示历史参与情况
            history_lines = []
            for h in recent_history:
                participants = h.get('participation_set', [])
                # 只显示前10个参与者
                participant_str = ', '.join(map(str, sorted(participants)[:10]))
                if len(participants) > 10:
                    participant_str += '...'
                history_lines.append(
                    f"- 轮次{h['round']}: "
                    f"m={h['m']:.2f}, {h['anonymization']}, "
                    f"参与者={{{participant_str}}}"
                )
            
            # 计算其他消费者的参与频率（只显示前10个）
            freq_lines = []
            consumer_ids = sorted(consumer_belief_probs.keys())
            if consumer_id in consumer_ids:
                consumer_ids.remove(consumer_id)  # 不显示自己
            
            for cid in consumer_ids[:10]:
                freq = consumer_belief_probs[cid]
                freq_lines.append(f"- 消费者{cid}: {freq:.0%}")
            if len(consumer_ids) > 10:
                freq_lines.append(f"- ...（共{len(consumer_ids)}个其他消费者）")
            
            history_text = f"""
【历史观察】（最近{len(recent_history)}轮）
{chr(10).join(history_lines)}

【其他消费者的参与频率】
{chr(10).join(freq_lines)}"""
        
        # 构建完整提示词
        data_structure = consumer_params.get('data_structure', 'common_preferences')
        theta_i = consumer_params['theta_i']
        tau_i = consumer_params['tau_i']
        s_i = consumer_params.get('s_i', None)
        
        signal_text = f"你的私人信号 s_i = {s_i:.2f}" if s_i is not None else "你的私人信号 s_i 未提供"
        
        if data_structure == "common_preferences":
            structure_text = "共同偏好：所有消费者真实偏好相同（记为 θ），你的信号满足 s_i = θ + 个体噪声。"
        else:
            structure_text = "共同经历：每个消费者真实偏好不同（记为 θ_i），但信号含共同冲击，s_i = θ_i + ε（ε对所有人相同）。"
        
        prompt = f"""你是消费者，需要决定是否参与数据分享计划。目标是最大化你的期望净效用。

【本轮条件】
- 补偿金额 m = {m:.2f}
- 匿名化策略：{anonymization}
  * identified：商家可能将你的数据与身份绑定，更容易对你做个性化定价（对高偏好者可能不利）
  * anonymized：商家只能利用匿名统计信息，通常更难针对个人定价

【你的参数】
- 偏好强度 θ_i = {theta_i:.2f}（越大表示你越喜欢该产品）
- 隐私成本 τ_i = {tau_i:.2f}（参与会带来的隐私损失成本）
- {signal_text}

【数据结构】
- {structure_text}

{history_text}

【决策要求】
判断参与是否值得。考虑要点：
1) 补偿 m 是参与的直接收益
2) identified 可能导致对你更不利的个性化定价风险（尤其当 θ_i 较高时）
3) anonymized 个性化定价风险较小
4) 参与会产生隐私成本 τ_i
5) 基于历史观察，考虑其他消费者的行为模式

【输出格式】
请按以下格式回答：
第1行：你的决策理由（50-100字）
第2行：决策：参与 或 决策：拒绝
"""
        return prompt
    
    def evaluate_config_D_fictitious_play(
        self,
        llm_client,  # 原始LLM client（OpenAI格式）
        model_config: Dict,  # 模型配置
        max_rounds: int = 50,
        belief_window: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        配置D（虚拟博弈版）：LLM中介 × LLM消费者（Fictitious Play）
        
        注意：此方法接受原始LLM client，直接使用FP提示词
        
        Args:
            llm_client: OpenAI格式的LLM客户端
            model_config: 模型配置字典
            max_rounds: 最大轮数
            belief_window: 信念窗口大小
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        import re
        
        model_name = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        if verbose:
            print("\n" + "="*70)
            print("配置D（虚拟博弈）：LLM中介 × LLM消费者 - Fictitious Play")
            print(f"最大轮数: {max_rounds}, 信念窗口: {belief_window}")
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
        
        # ===== 虚拟博弈迭代 =====
        for round_num in range(max_rounds):
            if verbose:
                print(f"\n{'='*70}")
                print(f"[轮次 {round_num + 1}/{max_rounds}]")
                print(f"{'='*70}")
            
            # 1. 计算中介信念
            window = min(belief_window, round_num)
            intermediary_belief = self._compute_intermediary_belief(history, window_size=window)
            
            # 2. 中介决策
            intermediary_prompt = self.build_intermediary_prompt_fp(
                market_params=market_params,
                history=history,
                belief_stats=intermediary_belief,
                belief_window=belief_window
            )
            
            # 调用中介LLM（使用FP提示词+重试机制）
            try:
                response = call_llm_with_retry(
                    client=llm_client,
                    model_name=model_name,
                    messages=[{"role": "user", "content": intermediary_prompt}],
                    generate_args=generate_args
                )
                answer = response.choices[0].message.content.strip()
                json_match = re.search(r'\{[^}]+\}', answer)
                if json_match:
                    answer = json_match.group(0)
                obj = json.loads(answer)
                m_llm = float(obj.get("m", 0.5))
                anon_llm = obj.get("anonymization", "anonymized")
                intermediary_reason = str(obj.get("reason", "")).strip()
            except Exception as e:
                print(f"[WARN] 中介决策失败: {e}，使用默认策略")
                m_llm = 0.5
                anon_llm = "anonymized"
                intermediary_reason = "调用失败"
            
            if verbose:
                print(f"中介选择: m={m_llm:.4f}, {anon_llm}")
                if intermediary_reason:
                    print(f"中介理由: {intermediary_reason}")
            
            # 3. 计算消费者信念
            N = market_params['N']
            consumer_belief_probs = self._compute_consumer_belief(history, window_size=window, N=N)
            
            # 4. 消费者决策
            consumers = self._get_sample_consumers()
            llm_decisions: List[bool] = []
            
            for idx, consumer_params in enumerate(consumers):
                consumer_params['consumer_id'] = idx  # 添加ID
                
                # 构建消费者FP提示词
                consumer_prompt = self.build_consumer_prompt_fp(
                    consumer_params=consumer_params,
                    m=m_llm,
                    anonymization=anon_llm,
                    history=history,
                    consumer_belief_probs=consumer_belief_probs,
                    belief_window=belief_window,
                    N=N
                )
                
                # 调用消费者LLM（使用FP提示词+重试机制）
                try:
                    response = call_llm_with_retry(
                        client=llm_client,
                        model_name=model_name,
                        messages=[{"role": "user", "content": consumer_prompt}],
                        generate_args=generate_args
                    )
                    answer = response.choices[0].message.content.strip()
                    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                    decision_line = ""
                    for ln in reversed(lines):
                        if ln.startswith("决策"):
                            decision_line = ln
                            break
                    
                    dl = decision_line.lower()
                    if ("拒绝" in decision_line) or ("no" in dl):
                        decision = False
                    elif ("参与" in decision_line) or ("yes" in dl):
                        decision = True
                    elif "拒绝" in answer:
                        decision = False
                    elif "参与" in answer:
                        decision = True
                    else:
                        # 默认根据m vs tau判断
                        decision = m_llm > consumer_params['tau_i']
                    
                    llm_decisions.append(bool(decision))
                except Exception as e:
                    if verbose and idx < 5:  # 只打印前几个错误
                        print(f"[WARN] 消费者{idx}决策失败: {e}")
                    llm_decisions.append(False)
            
            llm_decisions_arr = np.array(llm_decisions)
            r_llm = float(np.mean(llm_decisions_arr))
            participation_set = [i for i, d in enumerate(llm_decisions) if d]
            
            # 5. 计算市场结果
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
            
            # 6. 记录历史
            round_info = {
                "round": round_num + 1,
                "m": float(m_llm),
                "anonymization": anon_llm,
                "intermediary_reason": intermediary_reason,
                "participation_rate": r_llm,
                "num_participants": int(np.sum(llm_decisions_arr)),
                "participation_set": participation_set,
                "m0": float(m_0_D),
                "intermediary_cost": float(m_llm * np.sum(llm_decisions_arr)),
                "intermediary_profit": float(outcome_D.intermediary_profit),
            }
            history.append(round_info)
            
            if verbose:
                print(f"参与率: {r_llm:.2%} ({round_info['num_participants']}/{N})")
                print(f"利润: {round_info['intermediary_profit']:.4f}")
            
            # 7. 检查收敛
            if self._check_fp_convergence(history, threshold=3):
                if verbose:
                    print(f"\n[提前收敛] 连续3轮策略和参与集合不变，停止迭代")
                break
        
        # ===== 最终分析 =====
        if verbose:
            print(f"\n{'='*70}")
            print(f"[虚拟博弈结束] 总轮数: {len(history)}")
            print(f"{'='*70}")
        
        # 收敛性分析
        convergence_analysis = self._analyze_fp_convergence(history, self.gt_A['optimal_strategy'])
        
        if verbose:
            print(f"\n[收敛性分析]")
            print(f"是否收敛: {convergence_analysis['converged']}")
            if convergence_analysis['converged']:
                print(f"收敛轮数: {convergence_analysis['convergence_round']}")
            print(f"最终策略: m={convergence_analysis['final_m']:.4f}, {convergence_analysis['final_anonymization']}")
            print(f"最终参与率: {convergence_analysis['final_participation_rate']:.2%}")
            print(f"最终利润: {convergence_analysis['final_profit']:.4f} (理论最优 {profit_star:.4f})")
            print(f"与理论最优对比:")
            print(f"  m误差: {convergence_analysis['similarity_to_optimal']['m_error']:.4f}")
            print(f"  anon匹配: {convergence_analysis['similarity_to_optimal']['anon_match']}")
            print(f"  利润比率: {convergence_analysis['similarity_to_optimal']['profit_ratio']:.2%}")
        
        # 构造结果
        results = {
            "model_name": model_config.get('config_name', 'unknown'),
            "game_type": "fictitious_play",
            "config": "D_fictitious_play",
            "max_rounds": max_rounds,
            "actual_rounds": len(history),
            "belief_window": belief_window,
            "history": history,
            "convergence_analysis": convergence_analysis,
            "ground_truth": {
                "m_star": m_star,
                "anonymization_star": anon_star,
                "profit_star": profit_star,
                "r_star": self.gt_A['optimal_strategy']['r_star']
            },
            "final_strategy": {
                "m": convergence_analysis['final_m'],
                "anonymization": convergence_analysis['final_anonymization'],
                "participation_rate": convergence_analysis['final_participation_rate'],
                "profit": convergence_analysis['final_profit']
            }
        }
        
        return results
    
    def _visualize_fictitious_play(self, results: Dict, output_path: str):
        """为FP结果生成可视化（与场景B类似）"""
        from pathlib import Path
        
        output_dir = Path(output_path).parent
        base_name = Path(output_path).stem
        
        try:
            history = results.get("history", [])
            conv_analysis = results.get("convergence_analysis", {})
            
            if not history:
                print("[WARN] 没有历史数据，跳过可视化")
                return
            
            # === 可视化1：利润与参与率曲线 ===
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            profit_traj = conv_analysis.get("profit_trajectory", [])
            participation_traj = conv_analysis.get("participation_rate_trajectory", [])
            
            if profit_traj:
                rounds = list(range(1, len(profit_traj) + 1))
                
                # 主轴：利润
                ax1.plot(rounds, profit_traj, 'b-o', linewidth=2, markersize=4, label='Intermediary Profit')
                ax1.set_xlabel('Round', fontsize=12, fontfamily='Times New Roman')
                ax1.set_ylabel('Intermediary Profit', color='b', fontsize=12, fontfamily='Times New Roman')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True, alpha=0.3)
                
                # 设置刻度字体
                for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                    label.set_fontfamily('Times New Roman')
                
                # 次轴：参与率
                if participation_traj:
                    ax2 = ax1.twinx()
                    ax2.plot(rounds, participation_traj, 'r-s', linewidth=2, markersize=4, 
                            alpha=0.7, label='Participation Rate')
                    ax2.set_ylabel('Participation Rate', color='r', fontsize=12, fontfamily='Times New Roman')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.set_ylim([0, 1])
                    
                    # 设置次轴刻度字体
                    for label in ax2.get_yticklabels():
                        label.set_fontfamily('Times New Roman')
                
                # 标注收敛点
                if conv_analysis.get("converged"):
                    conv_round = conv_analysis.get("convergence_round")
                    if conv_round and conv_round < len(profit_traj):
                        ax1.axvline(x=conv_round + 1, color='g', linestyle='--', 
                                   alpha=0.5, label=f'Convergence (Round {conv_round + 1})')
                
                # 图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                if participation_traj:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', prop={'family': 'Times New Roman'})
                else:
                    legend = ax1.legend(loc='best', prop={'family': 'Times New Roman'})
                
                ax1.set_title('Fictitious Play: Profit and Participation Rate Evolution', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
                
                plt.tight_layout()
                fig1_path = output_dir / f"{base_name}_profit_rate.png"
                plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
                plt.close(fig1)
                print(f"[图表] 利润曲线已保存到: {fig1_path}")
            
            # === 可视化2：策略演化热力图 ===
            # 简化版：只显示m和参与率的演化
            fig2, (ax_m, ax_anon) = plt.subplots(2, 1, figsize=(12, 8))
            
            m_traj = conv_analysis.get("m_trajectory", [])
            anon_traj = conv_analysis.get("anon_trajectory", [])
            
            if m_traj:
                rounds = list(range(1, len(m_traj) + 1))
                
                # m的演化
                ax_m.plot(rounds, m_traj, 'b-o', linewidth=2, markersize=3)
                ax_m.set_xlabel('Round', fontsize=12, fontfamily='Times New Roman')
                ax_m.set_ylabel('Compensation Amount m', fontsize=12, fontfamily='Times New Roman')
                ax_m.grid(True, alpha=0.3)
                ax_m.set_title('Compensation Amount Evolution', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
                
                # 设置刻度字体
                for label in ax_m.get_xticklabels() + ax_m.get_yticklabels():
                    label.set_fontfamily('Times New Roman')
                
                # 标注理论最优
                m_star = results['ground_truth']['m_star']
                ax_m.axhline(y=m_star, color='r', linestyle='--', alpha=0.5, label=f'Theoretical Optimal m={m_star:.3f}')
                legend = ax_m.legend(prop={'family': 'Times New Roman'})
                
                # anonymization策略演化（用颜色块表示）
                anon_values = [1 if a == 'identified' else 0 for a in anon_traj]
                anon_matrix = np.array(anon_values).reshape(1, -1)
                
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(['#90EE90', '#FF6B6B'])
                im = ax_anon.imshow(anon_matrix, cmap=cmap, 
                                   aspect='auto', interpolation='none')
                ax_anon.set_xlabel('Round', fontsize=12, fontfamily='Times New Roman')
                ax_anon.set_ylabel('Anonymization Strategy', fontsize=12, fontfamily='Times New Roman')
                ax_anon.set_yticks([0])
                ax_anon.set_yticklabels(['Strategy'], fontfamily='Times New Roman')
                ax_anon.set_title('Anonymization Strategy Evolution (Green=Anonymized, Red=Identified)', 
                                 fontsize=12, fontweight='bold', fontfamily='Times New Roman')
                
                # 设置刻度字体
                for label in ax_anon.get_xticklabels() + ax_anon.get_yticklabels():
                    label.set_fontfamily('Times New Roman')
                
                plt.tight_layout()
                fig2_path = output_dir / f"{base_name}_strategy_evolution.png"
                plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print(f"[图表] 策略演化图已保存到: {fig2_path}")
            
        except Exception as e:
            print(f"[WARN] 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
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
            "Intermediary Profit Loss (%)": 0.0,
            "Individual Accuracy": np.nan,
            "TP": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "Consumer Surplus Gap": np.nan,
            "Gini Consumer Surplus": np.nan,
        }
        rows.append(row_A)
        
        # 配置B
        if results_B:
            profit_theory = self.gt_A['equilibrium']['intermediary_profit']
            profit_B = results_B['market']['intermediary_profit_llm']
            profit_loss_B = (
                (profit_theory - profit_B) / profit_theory if profit_theory != 0 else 0.0
            )
            confusion_B = results_B["consumer"]["decision_confusion_matrix"]
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
                "Intermediary Profit Loss (%)": profit_loss_B * 100,
                "Individual Accuracy": results_B["consumer"]["individual_accuracy"],
                "TP": confusion_B["TP"],
                "TN": confusion_B["TN"],
                "FP": confusion_B["FP"],
                "FN": confusion_B["FN"],
                "Consumer Surplus Gap": results_B["consumer"]["consumer_surplus_gap"],
                "Gini Consumer Surplus": results_B["consumer"]["gini_consumer_surplus"],
            }
            rows.append(row_B)
        
        # 配置C
        if results_C:
            profit_theory = self.gt_A['equilibrium']['intermediary_profit']
            profit_C = results_C['profit']['profit_llm']
            profit_loss_C = (
                (profit_theory - profit_C) / profit_theory if profit_theory != 0 else 0.0
            )
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
                "Intermediary Profit Loss (%)": profit_loss_C * 100,
                "Individual Accuracy": np.nan,
                "TP": np.nan,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "Consumer Surplus Gap": np.nan,
                "Gini Consumer Surplus": np.nan,
            }
            rows.append(row_C)
        
        # 配置D
        if results_D:
            profit_theory = self.gt_A['equilibrium']['intermediary_profit']
            profit_D = results_D['market']['intermediary_profit_llm']
            profit_loss_D = (
                (profit_theory - profit_D) / profit_theory if profit_theory != 0 else 0.0
            )
            confusion_D = results_D["consumer"]["decision_confusion_matrix"]
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
                "Intermediary Profit Loss (%)": profit_loss_D * 100,
                "Individual Accuracy": results_D["consumer"]["individual_accuracy"],
                "TP": confusion_D["TP"],
                "TN": confusion_D["TN"],
                "FP": confusion_D["FP"],
                "FN": confusion_D["FN"],
                "Consumer Surplus Gap": results_D["consumer"]["consumer_surplus_gap"],
                "Gini Consumer Surplus": results_D["consumer"]["gini_consumer_surplus"],
            }
            rows.append(row_D)
        
        df = pd.DataFrame(rows)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n报告已保存到: {output_path}")
        
        return df


def run_scenario_c_evaluation(
    model_config_name: str, 
    rounds: int = 20, 
    mode: str = "iterative",
    fp_config: str = "D",
    belief_window: int = 10,
    output_dir: str = "evaluation_results/scenario_c"
) -> Dict[str, Any]:
    """
    运行场景C评估（支持指定模型与学习轮数）
    
    Args:
        model_config_name: configs/model_configs.json 中的 config_name
        rounds: LLM中介多轮学习轮数
        output_dir: 输出目录（默认为 evaluation_results/scenario_c）
    
    Returns:
        汇总结果（包含输出文件路径）
    """
    import re
    from openai import OpenAI
    
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
        raise
    
    selected_model_config = None
    for config in model_configs:
        if config['config_name'] == model_config_name:
            selected_model_config = config
            break
    
    if selected_model_config is None:
        print(f"❌ 找不到模型配置: {model_config_name}")
        print(f"可用模型: {[cfg['config_name'] for cfg in model_configs]}")
        raise ValueError(f"找不到模型配置: {model_config_name}")
    
    model_name = selected_model_config['config_name']
    print(f"\n🎯 选择模型: {model_name}")
    
    # 创建OpenAI客户端（添加超时和重试配置）
    from openai import OpenAI
    import httpx
    
    client = OpenAI(
        api_key=selected_model_config['api_key'],
        timeout=httpx.Timeout(120.0, connect=30.0),  # 总超时120秒，连接超时30秒
        max_retries=3,  # 最多重试3次
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
            data_structure = consumer_params.get('data_structure', 'common_preferences')
            s_i = consumer_params.get('s_i', None)
            signal_text = f"你的私人信号 s_i = {s_i:.2f}" if s_i is not None else "你的私人信号 s_i 未提供"
            if data_structure == "common_preferences":
                structure_text = (
                    "共同偏好：所有消费者真实偏好相同（记为 θ），"
                    "你的信号满足 s_i = θ + 个体噪声。"
                )
            else:
                structure_text = (
                    "共同经历：每个消费者真实偏好不同（记为 θ_i），但信号含共同冲击，"
                    "s_i = θ_i + ε（ε对所有人相同）。"
                )
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

【你的私人信号】
- {signal_text}

【数据结构说明】
- 当前结构：{data_structure}
- {structure_text}

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
            response = call_llm_with_retry(
                client=client,
                model_name=model_name,
                messages=[{"role": "user", "content": prompt}],
                generate_args=generate_args
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
            try:
                decision, reason, answer = _call_llm_consumer(
                    consumer_params=consumer_params,
                    m=m,
                    anonymization=anonymization
                )
                return bool(decision), reason
            except Exception as e:
                print(f"⚠️ LLM调用失败: {e}")
                # 失败时使用简单启发式，并返回可记录的简短理由
                fallback_decision = bool(m > consumer_params['tau_i'])
                fallback_reason = "调用失败，使用m与τ的启发式判断"
                return fallback_decision, fallback_reason

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
            m_prev = market_params.get("m_init", 1.0)
            history_text = ""
            if feedback:
                reasons = feedback.get("reasons", {})
                m_prev = float(feedback.get("m", m_prev))
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
            # 修改：历史记忆管理（按利润排序，便于观察趋势）
            if history:
                try:
                    sorted_history = sorted(
                        history,
                        key=lambda x: float(x.get("intermediary_profit", 0.0)),
                        reverse=True
                    )
                except Exception:
                    sorted_history = history
                history_text = "\n【历史记忆（按利润从高到低）】\n"
                for h in sorted_history:
                    history_text += (
                        f"- m={h.get('m'):.3f}, "
                        f"anon={h.get('anonymization')}, "
                        f"r={h.get('participation_rate'):.3f}, "
                        f"profit={h.get('intermediary_profit'):.3f}\n"
                    )
            
            prompt = f"""你是“数据中介”，你的目标是最大化你的期望利润。

【市场参数】
- 消费者数量 N = {market_params['N']}
- 消费者偏好 θ ~ Normal(均值 {market_params['mu_theta']:.2f}, 标准差 {market_params['sigma_theta']:.2f})
- 消费者隐私成本 τ ~ Normal(均值 {market_params['tau_mean']:.2f}, 标准差 {market_params['tau_std']:.2f})
- 数据结构：{market_params['data_structure']}

【你的决策（你先动）】
你要选择：
1) 本轮补偿的改变量 Δm（范围 -0.5 到 0.5）
2) 匿名化策略：identified 或 anonymized

【消费者反应（随后）】
消费者会比较“参与的期望净收益”与其隐私成本 τ_i 来决定是否参与。
你需要权衡：identified 策略下，数据可识别性/定价精度提高，但可能降低消费者参与率；anonymized 策略下，参与率可能提高，但个性化定价能力下降。请自行权衡以最大化利润。

【你卖给商家的数据费 m0（关键）】
商家愿意为数据支付的价格 m0 由“数据带来的商家利润增量”决定：
m0 ≈ max(0, 期望[商家利润(有数据) − 商家利润(无数据)])
直觉：参与人数越多、数据越能支持个性化定价或改进产品，m0 越大，但边际收益递减。

【你的利润】
利润 = m0 − m × (参与人数)

【你的目标】
你的目标是最大化利润。请根据历史记忆，以利润最高的历史策略为基准进行微调，努力提高你的利润。

【输出格式（必须严格遵守）】
只输出一行 JSON，不要输出任何额外文字：
{{"delta_m": 数字,"anonymization":"identified" 或 "anonymized","reason":"50-100字"}}
{feedback_text}
{history_text}
上一轮 m = {m_prev:.2f}，本轮 m = max(m_prev + Δm, 0)（最小值为0，无最大值限制）。
请给出你的选择。
"""
            response = call_llm_with_retry(
                client=client,
                model_name=model_name,
                messages=[{"role": "user", "content": prompt}],
                generate_args=generate_args
            )
            answer = response.choices[0].message.content.strip()
            raw_answer = answer
            json_match = re.search(r'\{[^}]+\}', answer)
            if json_match:
                answer = json_match.group(0)
            try:
                obj = json.loads(answer)
            except Exception:
                obj = {"delta_m": 0.0, "anonymization": "anonymized", "reason": "解析失败"}
            delta_m = float(obj.get("delta_m", 0.0))
            anonymization = obj.get("anonymization", "anonymized")
            reason = str(obj.get("reason", "")).strip()
            # 修改：解除m的上限限制，只保留最小值0
            m_current = max(0.0, m_prev + delta_m)
            return m_current, anonymization, reason, raw_answer
        
        return llm_intermediary
    
    llm_consumer = create_llm_consumer(client, selected_model_config)
    llm_intermediary = create_llm_intermediary(client, selected_model_config)
    print("✅ LLM代理创建成功")
    
    # ========================================================================
    # 1. 初始化评估器（只评估共同偏好，跳过共同经历以节省时间）
    # ========================================================================
    gt_jobs = [
        ("common_preferences", "data/ground_truth/scenario_c_common_preferences_optimal.json"),
        # 注释掉common_experience以节省时间（如需要可以取消注释）
        # ("common_experience", "data/ground_truth/scenario_c_common_experience_optimal.json"),
    ]
    
    summary = {
        "model": model_name,
        "rounds": rounds,
        "outputs": []
    }
    
    for gt_tag, gt_path in gt_jobs:
        print("\n" + "=" * 70)
        print(f"步骤1: 加载Ground Truth（{gt_tag}）")
        print("=" * 70)
        
        try:
            evaluator = ScenarioCEvaluator(gt_path)
            print(f"✅ 成功加载: {gt_path}")
            print(f"\n理论基准（配置A）:")
            
            # 处理m_star（可能是标量或向量）
            m_star = evaluator.gt_A['optimal_strategy']['m_star']
            if isinstance(m_star, (list, np.ndarray)):
                m_mean = evaluator.gt_A['optimal_strategy'].get('m_star_mean', np.mean(m_star))
                m_std = evaluator.gt_A['optimal_strategy'].get('m_star_std', np.std(m_star))
                print(f"  m* = 向量 (均值={m_mean:.4f}, std={m_std:.4f})")
            else:
                print(f"  m* = {m_star:.4f}")
            
            print(f"  anonymization* = {evaluator.gt_A['optimal_strategy']['anonymization_star']}")
            print(f"  r* = {evaluator.gt_A['optimal_strategy']['r_star']:.4f}")
            print(f"  中介利润* = {evaluator.gt_A['optimal_strategy']['intermediary_profit_star']:.4f}")
            
        except FileNotFoundError:
            print(f"❌ 找不到Ground Truth文件: {gt_path}")
            print(f"\n请先运行以下命令生成 Ground Truth:")
            print(f"  python -m src.scenarios.generate_scenario_c_gt")
            raise
        
        # FP模式：根据fp_config选择运行哪个配置
        if mode == "fp":
            # 确定要运行的配置列表
            if fp_config == "all":
                configs_to_run = ["B", "C", "D"]
            else:
                configs_to_run = [fp_config]
            
            fp_results = {}  # 存储所有FP结果
            
            for config in configs_to_run:
                print("\n" + "=" * 70)
                print(f"步骤{2 if len(configs_to_run) == 1 else f'2-{chr(65 + configs_to_run.index(config))}'}: 虚拟博弈 - 配置{config}_FP")
                print("=" * 70)
                
                if config == "B":
                    # 配置B_FP: 理性中介 × LLM消费者
                    result = evaluator.evaluate_config_B_fictitious_play(
                        llm_client=client,
                        model_config=selected_model_config,
                        max_rounds=rounds if rounds != 20 else 50,
                        belief_window=belief_window,
                        verbose=True
                    )
                elif config == "C":
                    # 配置C_FP: LLM中介 × 理性消费者
                    result = evaluator.evaluate_config_C_fictitious_play(
                        llm_client=client,
                        model_config=selected_model_config,
                        max_rounds=rounds if rounds != 20 else 50,
                        belief_window=belief_window,
                        verbose=True
                    )
                else:  # config == "D"
                    # 配置D_FP: LLM中介 × LLM消费者
                    result = evaluator.evaluate_config_D_fictitious_play(
                        llm_client=client,
                        model_config=selected_model_config,
                        max_rounds=rounds if rounds != 20 else 50,
                        belief_window=belief_window,
                        verbose=True
                    )
                
                fp_results[config] = result
            
            results_B = None
            results_C = None
            results_D = None
        else:
            # Iterative模式：运行配置B、C、D
            print("\n" + "=" * 70)
            print(f"步骤2: 评估配置B（理性中介 × {model_name}消费者）")
            print("=" * 70)
            results_B = evaluator.evaluate_config_B(
                llm_consumer_agent=llm_consumer,
                verbose=True
            )
            
            print("\n" + "=" * 70)
            print(f"步骤3: 评估配置C（{model_name}中介 × 理性消费者）")
            print("=" * 70)
            results_C = evaluator.evaluate_config_C_iterative(
                llm_intermediary_agent=llm_intermediary,
                rounds=rounds,
                verbose=True
            )
            
            print("\n" + "=" * 70)
            print(f"步骤4: 评估配置D（{model_name}中介 × {model_name}消费者）")
            print("=" * 70)
            
            # 现有多轮迭代模式
            results_D = evaluator.evaluate_config_D_iterative(
                llm_intermediary_agent=llm_intermediary,
                llm_consumer_agent=llm_consumer,
                rounds=rounds,
                verbose=True
            )
        
        print("\n" + "=" * 70)
        print(f"步骤{3 if mode == 'fp' else 5}: 保存结果（{gt_tag}）")
        print("=" * 70)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据模式选择输出目录
        if mode == "fp":
            # FP模式：为每个配置保存结果
            for config, result in fp_results.items():
                print(f"\n[配置{config}_FP]")
                config_suffix = f"config{config}"
                final_output_dir = f"{output_dir}/fp_{config_suffix}_{model_name}"
                Path(final_output_dir).mkdir(parents=True, exist_ok=True)
                
                # FP模式只保存详细JSON（不生成CSV报告）
                output_json = f"{final_output_dir}/eval_{timestamp}.json"
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"  结果已保存到: {output_json}")
                
                # 生成可视化
                evaluator._visualize_fictitious_play(result, output_json)
                
                print(f"  可视化:")
                print(f"    • 利润曲线: {Path(output_json).stem}_profit_rate.png")
                print(f"    • 策略演化: {Path(output_json).stem}_strategy_evolution.png")
                
                summary["outputs"].append({
                    "gt_tag": gt_tag,
                    "mode": "fictitious_play",
                    "config": config,
                    "detailed_json": output_json
                })
            
            print("\n" + "=" * 70)
            if fp_config == "all":
                print(f"✅ 虚拟博弈评估完成（所有配置，{gt_tag}）！")
            else:
                print(f"✅ 虚拟博弈评估完成（配置{fp_config}，{gt_tag}）！")
            print("=" * 70)
            print(f"\n📊 评估模型: {model_name}")
            print(f"📊 运行配置: {', '.join(fp_results.keys())}")
            print()
        else:
            # Iterative模式：保持原有逻辑
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_dir}/scenario_c_{gt_tag}_{model_name}_{timestamp}.csv"
            df = evaluator.generate_report(
                results_B=results_B,
                results_C=results_C,
                results_D=results_D,
                output_path=output_path
            )
            
            print("\n报告预览:")
            print(df.to_string(index=False))
            
            detailed_results = {
                "model": model_name,
                "timestamp": timestamp,
                "gt_tag": gt_tag,
                "config_B": results_B,
                "config_C": results_C,
                "config_D": results_D,
                "report_rows": df.to_dict(orient="records")
            }
            
            output_json = f"{output_dir}/scenario_c_{gt_tag}_{model_name}_{timestamp}_detailed.json"
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {output_json}")
            
            print("\n" + "=" * 70)
            print(f"✅ 评估完成（{gt_tag}）！")
            print("=" * 70)
            print(f"\n📊 评估模型: {model_name}")
            print(f"📁 结果文件:")
            print(f"  • CSV报告: {output_path}")
            print(f"  • 详细JSON: {output_json}")
            print()
            
            summary["outputs"].append({
                "gt_tag": gt_tag,
                "mode": "iterative",
                "csv_report": output_path,
                "detailed_json": output_json
            })
    
    return summary


# ============================================================================
# 直接运行示例
# ============================================================================

if __name__ == "__main__":
    """
    直接运行评估器
    
    使用configs/model_configs.json中配置的真实LLM模型
    """
    import argparse
    import io
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="场景C评估器（真实LLM）")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2",
        help="模型配置名称（configs/model_configs.json 中的 config_name）"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="多轮学习轮数（iterative模式默认20，fp模式默认50）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="iterative",
        choices=["iterative", "fp"],
        help="博弈模式：iterative=现有多轮学习，fp=虚拟博弈"
    )
    parser.add_argument(
        "--fp_config",
        type=str,
        default="D",
        choices=["B", "C", "D", "all"],
        help="FP模式下运行的配置：B=理性中介×LLM消费者，C=LLM中介×理性消费者，D=LLM×LLM，all=运行所有三个配置"
    )
    parser.add_argument(
        "--belief_window",
        type=int,
        default=10,
        help="虚拟博弈信念窗口大小"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/scenario_c",
        help="输出目录（默认: evaluation_results/scenario_c）"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        nargs='+',
        help="为已有JSON文件生成可视化（支持文件路径或目录）"
    )
    args = parser.parse_args()
    
    # ===== 可视化模式 =====
    if args.visualize:
        print(f"\n{'='*70}")
        print(f"[可视化模式] 从已有结果生成图表")
        print(f"{'='*70}")
        
        import glob
        
        # 收集所有JSON文件
        json_files = []
        for path_pattern in args.visualize:
            path_obj = Path(path_pattern)
            
            if path_obj.is_file() and path_obj.suffix == '.json':
                json_files.append(path_obj)
            elif path_obj.is_dir():
                json_files.extend(path_obj.glob('*.json'))
            elif '*' in str(path_pattern):
                json_files.extend([Path(p) for p in glob.glob(path_pattern)])
            else:
                print(f"[WARN] 无效路径: {path_pattern}")
        
        if not json_files:
            print("[ERROR] 未找到任何JSON文件")
            sys.exit(1)
        
        print(f"\n找到 {len(json_files)} 个JSON文件\n")
        
        # 创建临时评估器（用于访问可视化方法）
        temp_evaluator = ScenarioCEvaluator("data/ground_truth/scenario_c_common_preferences_optimal.json")
        
        # 为每个JSON文件生成可视化
        for json_path in json_files:
            try:
                print(f"处理: {json_path}")
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 检查是否是FP结果
                if results.get("game_type") != "fictitious_play":
                    print(f"  [SKIP] 不是虚拟博弈结果，跳过")
                    continue
                
                # 生成可视化
                temp_evaluator._visualize_fictitious_play(results, str(json_path))
                print(f"  [OK] 可视化生成成功\n")
                
            except Exception as e:
                print(f"  [ERROR] 处理失败: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"可视化完成！")
        print(f"{'='*70}")
        sys.exit(0)
    
    # ===== 正常运行模式 =====
    run_scenario_c_evaluation(
        model_config_name=args.model, 
        rounds=args.rounds, 
        mode=args.mode,
        fp_config=args.fp_config,
        belief_window=args.belief_window,
        output_dir=args.output_dir
    )