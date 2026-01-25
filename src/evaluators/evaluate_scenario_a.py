"""
场景A的LLM评估器
评估LLM在个性化定价与隐私选择场景下的决策能力
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from .llm_client import LLMClient
from src.scenarios.scenario_a_personalization import ScenarioAParams, solve_for_D


class ScenarioAEvaluator:
    """场景A评估器"""
    
    def __init__(self, llm_client: LLMClient, ground_truth_path: str = "data/ground_truth/scenario_a_result.json"):
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
        
        self.params = ScenarioAParams(**self.gt_data["params"])
        self.gt_numeric = self.gt_data["gt_numeric"]
        self.gt_labels = self.gt_data["gt_labels"]
    
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个经济学专家，擅长分析市场机制和隐私外部性问题。
你需要在给定的场景下做出理性决策，并解释你的推理过程。
请严格按照JSON格式输出，不要包含任何额外的文本。"""
    
    def build_disclosure_prompt(self, consumer_id: int, current_disclosure_set: List[int]) -> str:
        """
        构建披露决策提示（针对单个消费者）
        
        Args:
            consumer_id: 消费者ID
            current_disclosure_set: 当前其他消费者的披露集合
        
        Returns:
            提示文本
        """
        theta_i = self.params.theta[consumer_id]
        c_privacy_i = self.params.c_privacy[consumer_id]
        n = self.params.n
        
        prompt = f"""
# 场景描述：个性化定价与隐私选择

你是消费者 {consumer_id}，正在考虑是否向平台披露你的个人数据。

## 你的信息
- 你对产品的真实愿付（willingness to pay）: {theta_i:.2f}
- 你的隐私成本（披露数据会带来的心理成本）: {c_privacy_i:.3f}
- 总共有 {n} 个消费者（包括你）

## 市场规则
1. **披露数据的消费者**：平台会识别你的愿付，并向你收取个性化价格 p_i = {theta_i:.2f}
   - 你的购买效用 = {theta_i:.2f} - {theta_i:.2f} - {c_privacy_i:.3f} = {-c_privacy_i:.3f}

2. **不披露数据的消费者**：平台无法识别你，只能对所有未披露者收取统一价格 p_uniform
   - 你的购买效用 = {theta_i:.2f} - p_uniform（如果你选择购买的话）
   - 是否购买取决于：{theta_i:.2f} >= p_uniform

3. **关键点**：统一价格 p_uniform 取决于有多少人披露数据
   - 披露的人越多，平台对未披露者的信息越少，统一价格可能会变化

## 当前情况
- 其他消费者中，有 {len(current_disclosure_set)} 人选择了披露数据
- 你需要决定：是否披露数据？

## 决策任务
请输出你的决策，格式为JSON：
{{
  "decision": 0 或 1（0=不披露，1=披露），
  "rationale": "简短解释你的推理过程（可选）"
}}

请只输出JSON，不要包含其他文本。
"""
        return prompt
    
    def query_llm_disclosure_decision(
        self, 
        consumer_id: int, 
        current_disclosure_set: List[int],
        num_trials: int = 3
    ) -> Tuple[int, List[int]]:
        """
        查询LLM的披露决策
        
        Args:
            consumer_id: 消费者ID
            current_disclosure_set: 当前披露集合
            num_trials: 重复查询次数（用于评估稳定性）
        
        Returns:
            (多数投票结果, 所有试验的决策列表)
        """
        prompt = self.build_disclosure_prompt(consumer_id, current_disclosure_set)
        
        decisions = []
        for trial in range(num_trials):
            try:
                response = self.llm_client.generate_json([
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": prompt}
                ])
                
                decision = int(response["decision"])
                if decision not in [0, 1]:
                    print(f"  ⚠️  试验{trial+1}: 无效决策 {decision}，默认为0")
                    decision = 0
                
                decisions.append(decision)
                
            except Exception as e:
                print(f"  ⚠️  试验{trial+1}失败: {e}")
                decisions.append(0)  # 失败时默认不披露
        
        # 多数投票
        final_decision = 1 if sum(decisions) > len(decisions) / 2 else 0
        return final_decision, decisions
    
    def simulate_llm_equilibrium(self, num_trials: int = 3, max_iterations: int = 10) -> Dict[str, Any]:
        """
        模拟LLM代理达到的披露均衡
        
        策略：
        1. 从空集合开始
        2. 每轮让每个消费者（随机顺序）重新决策
        3. 重复直到收敛或达到最大迭代次数
        
        Args:
            num_trials: 每个决策的重复次数
            max_iterations: 最大迭代次数
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"🤖 开始模拟LLM均衡 (模型: {self.llm_client.config_name})")
        print(f"{'='*60}")
        
        n = self.params.n
        disclosure_set = set()  # 当前披露集合
        
        # 追踪收敛过程
        history = []
        
        for iteration in range(max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            print(f"当前披露集合: {sorted(disclosure_set)}")
            
            # 随机顺序遍历消费者
            consumers = list(range(n))
            np.random.shuffle(consumers)
            
            changed = False
            for consumer_id in consumers:
                # 其他人的披露集合
                others_disclosure = sorted([i for i in disclosure_set if i != consumer_id])
                
                # 查询LLM决策
                print(f"\n  消费者 {consumer_id}: ", end="")
                decision, trials = self.query_llm_disclosure_decision(
                    consumer_id, 
                    others_disclosure,
                    num_trials=num_trials
                )
                
                print(f"决策={decision}, 试验结果={trials}")
                
                # 更新披露集合
                if decision == 1 and consumer_id not in disclosure_set:
                    disclosure_set.add(consumer_id)
                    changed = True
                    print(f"    ✅ 消费者{consumer_id}加入披露集合")
                elif decision == 0 and consumer_id in disclosure_set:
                    disclosure_set.remove(consumer_id)
                    changed = True
                    print(f"    ❌ 消费者{consumer_id}离开披露集合")
            
            # 记录历史
            history.append(sorted(disclosure_set))
            
            # 检查收敛
            if not changed:
                print(f"\n✅ 在第{iteration + 1}轮达到收敛！")
                break
        
        # 计算LLM均衡下的结果
        llm_disclosure_set = sorted(disclosure_set)
        llm_outcome = solve_for_D(self.params, set(llm_disclosure_set))
        
        # 与ground truth比较
        gt_disclosure_set = sorted(self.gt_numeric["eq_disclosure_set"])
        gt_profit = self.gt_numeric["eq_profit"]
        gt_CS = self.gt_numeric["eq_CS"]
        gt_W = self.gt_numeric["eq_W"]
        
        # 计算偏差
        results = {
            "model_name": self.llm_client.config_name,
            "llm_disclosure_set": llm_disclosure_set,
            "gt_disclosure_set": gt_disclosure_set,
            "convergence_history": history,
            "converged": len(history) < max_iterations,
            "iterations": len(history),
            "metrics": {
                "llm": {
                    "profit": llm_outcome.total_profit,
                    "consumer_surplus": llm_outcome.consumer_surplus,
                    "welfare": llm_outcome.welfare,
                    "disclosure_rate": len(llm_disclosure_set) / n
                },
                "ground_truth": {
                    "profit": gt_profit,
                    "consumer_surplus": gt_CS,
                    "welfare": gt_W,
                    "disclosure_rate": len(gt_disclosure_set) / n
                },
                "deviations": {
                    "profit_mae": abs(llm_outcome.total_profit - gt_profit),
                    "cs_mae": abs(llm_outcome.consumer_surplus - gt_CS),
                    "welfare_mae": abs(llm_outcome.welfare - gt_W),
                    "disclosure_rate_mae": abs(len(llm_disclosure_set) / n - len(gt_disclosure_set) / n)
                }
            },
            "labels": {
                "llm_disclosure_rate_bucket": self._bucket_disclosure_rate(len(llm_disclosure_set) / n),
                "gt_disclosure_rate_bucket": self.gt_labels["disclosure_rate_bucket"],
                "llm_over_disclosure": 1 if len(llm_disclosure_set) > len(self.gt_numeric["fb_disclosure_set"]) else 0,
                "gt_over_disclosure": self.gt_labels["over_disclosure"]
            }
        }
        
        return results
    
    def _bucket_disclosure_rate(self, rate: float) -> str:
        """将披露率分桶"""
        if rate < 0.33:
            return "low"
        elif rate < 0.67:
            return "medium"
        else:
            return "high"
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"📊 评估结果摘要")
        print(f"{'='*60}")
        
        print(f"\n【披露集合比较】")
        print(f"  LLM均衡: {results['llm_disclosure_set']}")
        print(f"  理论均衡: {results['gt_disclosure_set']}")
        print(f"  收敛情况: {'✅ 已收敛' if results['converged'] else '❌ 未收敛'} (迭代{results['iterations']}次)")
        
        print(f"\n【关键指标】")
        llm_m = results['metrics']['llm']
        gt_m = results['metrics']['ground_truth']
        dev_m = results['metrics']['deviations']
        
        print(f"  平台利润:     LLM={llm_m['profit']:.3f}  |  GT={gt_m['profit']:.3f}  |  MAE={dev_m['profit_mae']:.3f}")
        print(f"  消费者剩余:   LLM={llm_m['consumer_surplus']:.3f}  |  GT={gt_m['consumer_surplus']:.3f}  |  MAE={dev_m['cs_mae']:.3f}")
        print(f"  社会福利:     LLM={llm_m['welfare']:.3f}  |  GT={gt_m['welfare']:.3f}  |  MAE={dev_m['welfare_mae']:.3f}")
        print(f"  披露率:       LLM={llm_m['disclosure_rate']:.2%}  |  GT={gt_m['disclosure_rate']:.2%}  |  MAE={dev_m['disclosure_rate_mae']:.2%}")
        
        print(f"\n【标签一致性】")
        llm_l = results['labels']
        print(f"  披露率分桶:   LLM={llm_l['llm_disclosure_rate_bucket']}  |  GT={llm_l['gt_disclosure_rate_bucket']}  |  {'✅' if llm_l['llm_disclosure_rate_bucket'] == llm_l['gt_disclosure_rate_bucket'] else '❌'}")
        print(f"  过度披露:     LLM={llm_l['llm_over_disclosure']}  |  GT={llm_l['gt_over_disclosure']}  |  {'✅' if llm_l['llm_over_disclosure'] == llm_l['gt_over_disclosure'] else '❌'}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {output_path}")


def main():
    """测试评估器"""
    from .llm_client import create_llm_client
    import os
    
    # 创建LLM客户端
    llm_client = create_llm_client("deepseek-v3.2")
    
    # 创建评估器
    evaluator = ScenarioAEvaluator(llm_client)
    
    # 运行评估
    results = evaluator.simulate_llm_equilibrium(num_trials=3, max_iterations=5)
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)
    
    # 创建场景A专属输出目录
    output_dir = "evaluation_results/scenario_a"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    evaluator.save_results(results, f"{output_dir}/eval_scenario_a_{llm_client.config_name}.json")


if __name__ == "__main__":
    main()
