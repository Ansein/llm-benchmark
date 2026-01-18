"""
场景C的LLM评估器 - 社会数据外部性
评估LLM在"The Economics of Social Data"场景下的参与决策能力

核心机制:
1. 社会数据外部性 - 个人数据对他人有预测价值
2. 搭便车问题 - 拒绝者仍能从参与者数据中学习
3. 匿名化保护 - 匿名化阻止个性化定价
4. 参与决策 - 权衡补偿、学习收益与隐私风险

博弈时序:
1. 阶段0: 生成数据结构（公共知识）
2. 阶段1: 中介发布合同（补偿m，匿名化政策）
3. 阶段2: 消费者同时决策（看不到他人决策）
4. 阶段3: 市场交易与结算
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# 支持直接运行和模块导入
try:
    from .llm_client import LLMClient, create_llm_client
    from src.scenarios.scenario_c_social_data import (
        ScenarioCParams, ConsumerData, MarketOutcome,
        generate_consumer_data, simulate_market_outcome,
        compute_posterior_mean_consumer
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.evaluators.llm_client import LLMClient, create_llm_client
    from src.scenarios.scenario_c_social_data import (
        ScenarioCParams, ConsumerData, MarketOutcome,
        generate_consumer_data, simulate_market_outcome,
        compute_posterior_mean_consumer
    )


class ScenarioCEvaluator:
    """场景C评估器"""
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        ground_truth_path: str = "data/ground_truth/scenario_c_result.json"
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
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        
        # 重建params
        params_dict = self.gt_data["params"]
        self.params = ScenarioCParams(**params_dict)
        
        # 重建data
        self.data = ConsumerData(
            w=np.array(self.gt_data["data"]["w"]),
            s=np.array(self.gt_data["data"]["s"]),
            e=np.zeros(self.params.N),  # 不需要e的值
            theta=self.gt_data["data"]["theta"],
            epsilon=self.gt_data["data"]["epsilon"]
        )
        
        self.gt_outcome = self.gt_data["outcome"]
        self.gt_participation_rate = self.gt_data["rational_participation_rate"]
    
    def _build_prompt_for_consumer(
        self, 
        consumer_id: int, 
        current_participation: np.ndarray,
        provide_feedback: bool = False,
        previous_decision: Optional[str] = None
    ) -> str:
        """
        为消费者i构建决策提示
        
        Args:
            consumer_id: 消费者ID
            current_participation: 当前其他消费者的参与状态
            provide_feedback: 是否提供反馈（用于迭代决策）
            previous_decision: 之前的决策（如果有）
            
        Returns:
            提示文本
        """
        params = self.params
        s_i = self.data.s[consumer_id]
        
        # 数据结构描述
        if params.data_structure == "common_preferences":
            data_structure_desc = """**共同偏好型（Common Preferences）**
所有消费者对产品的真实价值相近，但各自的初步评估有随机误差。
多人的数据可以通过平均滤掉噪声，更准确估计产品的真实价值。"""
        else:  # common_experience
            data_structure_desc = """**共同经历型（Common Experience）**
每个消费者的真实偏好不同，但都受到相同的市场信息噪声影响。
多人的数据可以识别并过滤共同噪声，更准确估计各自的真实偏好。"""
        
        # 匿名化政策描述
        if params.anonymization == "identified":
            anonymization_desc = """**实名制（Identified）**
- 你的数据会与你的身份关联
- 生产者将知道每个消费者的信号
- 生产者可以对每个人设置不同的价格（个性化定价）"""
            
            participation_cost = "生产者会看到你的信号，可能对你进行精准的价格歧视"
            rejection_protection = "生产者无法用你的数据对你定价"
        else:  # anonymized
            anonymization_desc = """**匿名化（Anonymized）**
- 你的数据会被打乱身份标识
- 生产者只能看到信号集合，无法识别谁是谁
- 生产者只能对所有人设置统一价格"""
            
            participation_cost = "生产者会看到你的信号，但不知道是你的，所有人将面临相同价格"
            rejection_protection = "所有人面临统一价格（无论是否参与）"
        
        # 当前参与情况
        num_current_participants = int(np.sum(current_participation))
        participation_info = f"当前已有 {num_current_participants} 人参与数据共享"
        
        # 学习收益说明
        learning_benefit = f"""
✓ 如果你参与：
  • 你会获得 {params.m:.2f} 的补偿
  • 你可以看到其他参与者的数据（包括你自己的）
  • 帮助你更准确判断产品是否适合你
  • {participation_cost}

✗ 如果你拒绝：
  • 你不会获得补偿
  • 你仍可以看到其他参与者的数据（搭便车）
  • 也能提高判断准确性，但可能不如参与时准确
  • {rejection_protection}
"""
        
        # 反馈信息（如果需要）
        feedback = ""
        if provide_feedback and previous_decision:
            feedback = f"\n【上一轮决策】\n你之前选择了：{previous_decision}\n请重新考虑你的决策。\n"
        
        prompt = f"""你是 {params.N} 个消费者中的消费者 {consumer_id}。

【产品与市场】
市场上有一个产品，你对它的初步评估（信号）是：**{s_i:.2f}**
（这个评估可能有噪声，真实价值范围通常在 {params.mu_theta - 2*params.sigma_theta:.1f} 到 {params.mu_theta + 2*params.sigma_theta:.1f} 之间）

【数据环境】
{data_structure_desc}

【数据中介的提议】
一个数据中介愿意支付你 **{params.m:.2f} 元** 来获取你的信号数据。

数据政策：
{anonymization_desc}

【当前状况】
{participation_info}

【你需要权衡】
{learning_benefit}

【关键事实】
• 共有 {params.N} 个消费者同时独立决策（你看不到别人的选择）
• 生产者会根据获得的数据信息来定价
• **即使你不参与，你仍可能看到其他参与者的匿名数据并从中学习（搭便车）**
• 参与的主要代价是：{'可能被精准价格歧视' if params.anonymization == 'identified' else '几乎没有隐私代价（因为匿名化）'}

{feedback}
请仔细权衡参与的收益（补偿 + 学习）与代价（可能的价格歧视），做出你的选择。

**请以JSON格式回答：**
```json
{{
  "decision": "accept" 或 "reject",
  "reasoning": "你的理由（1-2句话）"
}}
```

只返回JSON，不要有其他内容。"""
        
        return prompt
    
    def _call_llm_for_consumer(
        self, 
        consumer_id: int, 
        current_participation: np.ndarray,
        num_trials: int = 3
    ) -> Tuple[bool, str, List[str]]:
        """
        调用LLM为消费者做决策
        
        Args:
            consumer_id: 消费者ID
            current_participation: 当前参与状态
            num_trials: 重复次数（多数投票）
            
        Returns:
            (decision, reasoning, all_decisions)
            decision: True表示参与
            reasoning: 决策理由
            all_decisions: 所有试验的决策列表
        """
        prompt = self._build_prompt_for_consumer(consumer_id, current_participation)
        
        all_decisions = []
        all_reasonings = []
        
        for trial in range(num_trials):
            try:
                # 调用LLM并解析JSON响应
                messages = [{"role": "user", "content": prompt}]
                result = self.llm_client.generate_json(messages)
                
                decision_str = result.get("decision", "").lower()
                reasoning = result.get("reasoning", "")
                
                if "accept" in decision_str:
                    all_decisions.append("accept")
                elif "reject" in decision_str:
                    all_decisions.append("reject")
                else:
                    print(f"  警告: 消费者{consumer_id}试验{trial+1}返回无效决策: {decision_str}, 默认为reject")
                    all_decisions.append("reject")
                
                all_reasonings.append(reasoning)
                
            except (json.JSONDecodeError, Exception) as e:
                print(f"  警告: 消费者{consumer_id}试验{trial+1}失败: {e}")
                all_decisions.append("reject")  # 默认拒绝
                all_reasonings.append(f"调用失败: {str(e)}")
        
        # 多数投票
        accept_count = all_decisions.count("accept")
        reject_count = all_decisions.count("reject")
        
        final_decision = accept_count > reject_count
        final_reasoning = all_reasonings[0] if all_reasonings else "无理由"
        
        return final_decision, final_reasoning, all_decisions
    
    def evaluate(
        self, 
        max_iterations: int = 10, 
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """
        评估LLM在场景C下的决策能力
        
        使用固定点迭代找到LLM均衡:
        1. 初始化：无人参与
        2. 迭代：随机顺序遍历消费者，每人基于当前状态决策
        3. 收敛：无人改变决策
        
        Args:
            max_iterations: 最大迭代次数
            num_trials: 每个决策的重复次数
            
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"场景C LLM评估 - {self.llm_client.model_name}")
        print(f"{'='*60}")
        print(f"数据结构: {self.params.data_structure}")
        print(f"匿名化: {self.params.anonymization}")
        print(f"补偿: {self.params.m:.2f}")
        print(f"消费者数量: {self.params.N}")
        
        # 初始化：无人参与
        current_participation = np.zeros(self.params.N, dtype=bool)
        participation_history = [current_participation.copy()]
        
        converged = False
        final_iteration = 0
        
        # 记录每个消费者的决策历史
        decision_logs = {i: [] for i in range(self.params.N)}
        
        for iteration in range(max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            changed = False
            consumer_order = np.random.permutation(self.params.N)
            
            for consumer_id in consumer_order:
                # 调用LLM决策
                decision, reasoning, all_decisions = self._call_llm_for_consumer(
                    consumer_id, current_participation, num_trials
                )
                
                # 记录
                decision_logs[consumer_id].append({
                    "iteration": iteration + 1,
                    "decision": decision,
                    "reasoning": reasoning,
                    "all_trials": all_decisions
                })
                
                # 检查是否改变
                if decision != current_participation[consumer_id]:
                    changed = True
                    current_participation[consumer_id] = decision
                    action = "参与" if decision else "退出"
                    print(f"  消费者{consumer_id}: {action} (理由: {reasoning[:50]}...)")
            
            participation_history.append(current_participation.copy())
            
            # 检查收敛
            if not changed:
                converged = True
                final_iteration = iteration + 1
                print(f"\n✅ 达到均衡! (迭代{final_iteration}次)")
                break
        
        if not converged:
            final_iteration = max_iterations
            print(f"\n⚠️  未收敛: 达到最大迭代次数{max_iterations}")
        
        # 计算LLM均衡下的市场结果
        llm_outcome = simulate_market_outcome(
            self.data, current_participation, self.params
        )
        
        # 计算偏差
        deviations = self._compute_deviations(llm_outcome)
        
        # 计算标签
        labels = self._compute_labels(llm_outcome)
        
        # 打印结果
        self._print_results(llm_outcome, deviations, labels, converged, final_iteration)
        
        # 构建返回结果
        result = {
            "model_name": self.llm_client.model_name,
            "scenario": "C",
            "converged": converged,
            "iterations": final_iteration,
            "llm_participation": current_participation.tolist(),
            "llm_participation_rate": float(llm_outcome.participation_rate),
            "gt_participation_rate": float(self.gt_participation_rate),
            "participation_history": [p.tolist() for p in participation_history],
            "decision_logs": decision_logs,
            "metrics": {
                "llm": {
                    "participation_rate": float(llm_outcome.participation_rate),
                    "num_participants": int(llm_outcome.num_participants),
                    "consumer_surplus": float(llm_outcome.consumer_surplus),
                    "producer_profit": float(llm_outcome.producer_profit),
                    "social_welfare": float(llm_outcome.social_welfare),
                    "gini_coefficient": float(llm_outcome.gini_coefficient),
                    "price_variance": float(llm_outcome.price_variance),
                    "price_discrimination_index": float(llm_outcome.price_discrimination_index),
                    "acceptor_avg_utility": float(llm_outcome.acceptor_avg_utility),
                    "rejecter_avg_utility": float(llm_outcome.rejecter_avg_utility),
                    "learning_quality_participants": float(llm_outcome.learning_quality_participants),
                    "learning_quality_rejecters": float(llm_outcome.learning_quality_rejecters),
                },
                "ground_truth": self.gt_outcome,
                "deviations": deviations
            },
            "labels": labels,
            "params": self.params.to_dict()
        }
        
        return result
    
    def _compute_deviations(self, llm_outcome: MarketOutcome) -> Dict[str, float]:
        """计算LLM结果与Ground Truth的偏差"""
        gt = self.gt_outcome
        
        deviations = {
            "participation_rate_mae": abs(llm_outcome.participation_rate - gt["participation_rate"]),
            "consumer_surplus_mae": abs(llm_outcome.consumer_surplus - gt["consumer_surplus"]),
            "producer_profit_mae": abs(llm_outcome.producer_profit - gt["producer_profit"]),
            "social_welfare_mae": abs(llm_outcome.social_welfare - gt["social_welfare"]),
            "gini_mae": abs(llm_outcome.gini_coefficient - gt["gini_coefficient"]),
            "price_discrimination_mae": abs(llm_outcome.price_discrimination_index - gt["price_discrimination_index"]),
        }
        
        return deviations
    
    def _compute_labels(self, llm_outcome: MarketOutcome) -> Dict[str, Any]:
        """计算标签（用于分类评估）"""
        llm_rate = llm_outcome.participation_rate
        gt_rate = self.gt_participation_rate
        
        # 参与率分桶
        def bucket(rate):
            if rate < 0.33:
                return "low"
            elif rate < 0.67:
                return "medium"
            else:
                return "high"
        
        llm_bucket = bucket(llm_rate)
        gt_bucket = bucket(gt_rate)
        bucket_match = llm_bucket == gt_bucket
        
        # 方向标签：是否过度参与或过度拒绝
        rate_diff = llm_rate - gt_rate
        if abs(rate_diff) < 0.1:
            direction = "match"
        elif rate_diff > 0:
            direction = "over_participation"
        else:
            direction = "under_participation"
        
        return {
            "llm_participation_bucket": llm_bucket,
            "gt_participation_bucket": gt_bucket,
            "bucket_match": bucket_match,
            "direction": direction,
            "rate_difference": float(rate_diff)
        }
    
    def _print_results(
        self, 
        llm_outcome: MarketOutcome, 
        deviations: Dict[str, float],
        labels: Dict[str, Any],
        converged: bool,
        iterations: int
    ):
        """打印评估结果"""
        gt = self.gt_outcome
        
        print(f"\n{'='*60}")
        print(f"评估结果")
        print(f"{'='*60}")
        
        print(f"\n【收敛情况】")
        print(f"  收敛: {'✅' if converged else '❌'}")
        print(f"  迭代次数: {iterations}")
        
        print(f"\n【参与情况比较】")
        print(f"  LLM参与率:  {llm_outcome.participation_rate:.2%} ({llm_outcome.num_participants}/{self.params.N})")
        print(f"  理论参与率: {self.gt_participation_rate:.2%}")
        print(f"  偏差:       {deviations['participation_rate_mae']:.2%}")
        
        print(f"\n【关键指标】")
        metrics = [
            ("消费者剩余", llm_outcome.consumer_surplus, gt["consumer_surplus"], deviations["consumer_surplus_mae"]),
            ("生产者利润", llm_outcome.producer_profit, gt["producer_profit"], deviations["producer_profit_mae"]),
            ("社会福利", llm_outcome.social_welfare, gt["social_welfare"], deviations["social_welfare_mae"]),
            ("Gini系数", llm_outcome.gini_coefficient, gt["gini_coefficient"], deviations["gini_mae"]),
            ("价格歧视指数", llm_outcome.price_discrimination_index, gt["price_discrimination_index"], deviations["price_discrimination_mae"]),
        ]
        
        for name, llm_val, gt_val, mae in metrics:
            print(f"  {name:12s}: LLM={llm_val:8.4f}  |  GT={gt_val:8.4f}  |  MAE={mae:8.4f}")
        
        print(f"\n【标签一致性】")
        print(f"  参与率分桶: LLM={labels['llm_participation_bucket']:6s}  |  GT={labels['gt_participation_bucket']:6s}  |  {'✅' if labels['bucket_match'] else '❌'}")
        print(f"  方向标签:   {labels['direction']}")
        
        print(f"\n【学习质量】")
        print(f"  参与者学习误差: {llm_outcome.learning_quality_participants:.4f}")
        print(f"  拒绝者学习误差: {llm_outcome.learning_quality_rejecters:.4f}")


def run_scenario_c_evaluation(
    model_name: str = "gpt-4.1-mini",
    ground_truth_path: str = "data/ground_truth/scenario_c_result.json",
    output_path: Optional[str] = None,
    max_iterations: int = 10,
    num_trials: int = 3
) -> Dict[str, Any]:
    """
    运行场景C的评估
    
    Args:
        model_name: 模型配置名称（如 "gpt-4.1-mini"）
        ground_truth_path: Ground Truth文件路径
        output_path: 输出文件路径（可选）
        max_iterations: 最大迭代次数
        num_trials: 每个决策的重复次数
        
    Returns:
        评估结果字典
    """
    # 创建LLM客户端
    llm_client = create_llm_client(model_name)
    
    # 创建评估器
    evaluator = ScenarioCEvaluator(llm_client, ground_truth_path)
    
    # 运行评估
    result = evaluator.evaluate(max_iterations, num_trials)
    
    # 保存结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 评估结果已保存到: {output_path}")
    
    return result


# 示例使用
if __name__ == "__main__":
    import sys
    
    # 设置编码
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 运行评估
    result = run_scenario_c_evaluation(
        model_name="gpt-4.1-mini",
        ground_truth_path="data/ground_truth/scenario_c_result.json",
        output_path="evaluation_results/eval_scenario_C_gpt-4.1-mini.json",
        max_iterations=10,
        num_trials=3
    )
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
