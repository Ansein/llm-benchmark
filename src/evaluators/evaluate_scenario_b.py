"""
场景B的LLM评估器
评估LLM在"Too Much Data"场景下的决策能力（推断外部性）
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple

# 支持直接运行和模块导入
try:
    from .llm_client import LLMClient
    from src.scenarios.scenario_b_too_much_data import ScenarioBParams, calculate_leakage, calculate_outcome
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.evaluators.llm_client import LLMClient
    from src.scenarios.scenario_b_too_much_data import ScenarioBParams, calculate_leakage, calculate_outcome


class ScenarioBEvaluator:
    """场景B评估器"""
    
    def __init__(self, llm_client: LLMClient, ground_truth_path: str = "data/ground_truth/scenario_b_result.json"):
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
        
        # 重建params（需要转换Sigma为numpy数组）
        params_dict = self.gt_data["params"].copy()
        params_dict["Sigma"] = np.array(params_dict["Sigma"])
        self.params = ScenarioBParams(**params_dict)
        self.gt_numeric = self.gt_data["gt_numeric"]
        self.gt_labels = self.gt_data["gt_labels"]
    
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个经济学专家，擅长分析数据市场和隐私外部性问题。
你需要理解"推断外部性"（inference externality）的概念：即使你不分享数据，平台也可以通过其他人的数据推断你的信息。
请严格按照JSON格式输出，不要包含任何额外的文本。"""
    
    def build_sharing_prompt(self, user_id: int, last_round_broadcast: Dict[str, Any]) -> str:
        """
        并行博弈提示词：只给机制说明，不给引导信息和计算结果
        
        Args:
            user_id: 用户ID
            last_round_broadcast: 上一轮的广播信息
        
        Returns:
            提示文本
        """
        v_i = self.params.v[user_id]
        n = self.params.n
        rho = self.params.rho
        sigma_noise_sq = self.params.sigma_noise_sq
        
        # 上一轮广播信息
        last_share_set = last_round_broadcast.get("share_set", [])
        last_share_rate = last_round_broadcast.get("share_rate", 0.0)
        
        # === 计算本轮如果用户分享，平台会支付的补偿价格 p_i ===
        # 假设其他用户保持上一轮的分享决策
        current_S = set(last_share_set)
        
        # 计算用户i分享后的泄露
        S_with_i = current_S | {user_id}
        from src.scenarios.scenario_b_too_much_data import calculate_leakage
        leakage_with_i = calculate_leakage(S_with_i, self.params.Sigma, sigma_noise_sq)
        leak_i_with = leakage_with_i[user_id]
        
        # 计算用户i不分享时的泄露（基础泄露）
        S_without_i = current_S - {user_id}
        leakage_without_i = calculate_leakage(S_without_i, self.params.Sigma, sigma_noise_sq)
        leak_i_without = leakage_without_i[user_id]
        
        # 边际信息价值 ΔI_i
        marginal_info_value = max(0.0, leak_i_with - leak_i_without)
        
        # 补偿价格 p_i = v_i × ΔI_i
        compensation_price = v_i * marginal_info_value
        
        prompt = f"""
# 场景：数据市场与推断外部性

你是用户 {user_id}，正在参与一个数据市场博弈。  
在这一轮中，所有用户将**同时决定是否分享数据**。  
在你做决定时，你**不知道其他用户在本轮会如何选择**，也无法看到任何其他用户本轮的决定。

## 基本参数
- 你的隐私偏好：v[{user_id}] = {v_i:.3f}
  （所有用户的v都从[0.3, 1.2]范围均匀抽样，你可以据此判断自己的相对水平）
- 类型相关系数：ρ = {rho:.2f}
- 观测噪声：σ² = {sigma_noise_sq}

（不同用户的隐私偏好存在差异，但你无法观察到其他用户的具体v值。）

## 推断外部性机制

**核心概念**：即使你不分享数据，平台也能通过贝叶斯更新，利用其他人的数据推断你的类型。

**关键因素**：
1. **类型相关性**（ρ）：你的类型与其他用户的相关程度
   - ρ越高 → 其他人的数据越能揭示你的信息
   
2. **已分享数据量**：已经分享的用户越多
   - 推断越准确 → 你的泄露越多
   
3. **观测噪声**（σ²）：平台观测数据的准确度
   - σ²越小 → 推断越准确

**泄露机制**：
- 你的信息泄露 = 平台对你类型的不确定性减少量
- 不确定性通过贝叶斯后验方差衡量

## 分享与补偿机制

**本轮补偿价格**（基于上一轮情况）：
- 如果你选择分享，平台将支付补偿：**p_i = {compensation_price:.4f}**
  
- 这个价格是根据以下公式计算的：
  - p_i = v_i × ΔI_i
  - v_i = {v_i:.3f}（你的隐私偏好）
  - ΔI_i = {marginal_info_value:.4f}（边际信息价值）
  
- **边际信息价值** ΔI_i 是你分享数据带来的额外信息：
  - 基于上一轮分享集合 {last_share_set}（这些用户选择了分享）
  - 衡量你的数据在当前情况下的边际贡献
  - 如果已有很多人分享，你的边际贡献会减少（次模性）

**决策权衡（基于上一轮状态的估算）**：

如果**假设其他用户保持上一轮的决策不变**：

- 如果你**不分享**：
  - 基础泄露：{leak_i_without:.4f}
  - 获得补偿：0
  - 估算净效用：-{v_i * leak_i_without:.4f}
  
- 如果你**分享**：
  - 总泄露：{leak_i_with:.4f}
  - 获得补偿：{compensation_price:.4f}
  - 估算净效用：{compensation_price - v_i * leak_i_with:.4f}

**重要警告 - 这只是静态估算！**

⚠️ **实际情况会更复杂**：
1. **其他用户也在做决策**：他们可能同时改变选择，导致实际的分享集合与上轮不同
2. **补偿价格会动态变化**：
   - 如果更多人分享 → 你的边际价值↓ → 补偿↓
   - 如果更少人分享 → 你的边际价值↑ → 补偿↑
3. **次模性效应**：已经有很多人分享时，额外分享的边际价值会递减
4. **需要战略思考**：不要只看当前数值，要考虑这是一个动态博弈

**理性决策建议**：
- 不要机械地比较两个数值就决定
- 考虑其他用户的可能反应
- 思考分享集合的稳定性（是否接近均衡）
- 评估自己的隐私偏好在群体中的相对位置


你的目标是：  
**在理解推断外部性和补偿机制的基础上，做出战略性决策，使你的长期净效用最大化。**

这不是简单的一次性决策，而是一个会收敛到均衡的动态过程。你需要思考：
- 什么样的分享集合是稳定的（均衡）？
- 在那个均衡中，你是否应该分享？
- 你的决策应该帮助系统收敛，而不是制造震荡

## 上一轮的公共信息（广播）

平台在上一轮结束后公布了以下信息：

- **上一轮分享集合**：{last_share_set}
  - 这是一个用户ID列表，表示上一轮**选择分享数据的用户**
  - 例如 [0, 2, 5] 表示用户0、用户2、用户5选择了分享
  - 集合中有 {len(last_share_set)} 个用户选择了分享
  
- **上一轮分享率**：{last_share_rate:.1%}
  - 这是分享用户占总用户数的比例
  
**重要说明**：
- 这只是上一轮的历史结果，仅供参考
- **本轮其他用户可能会改变决策**，你无法预知
- 本轮结束前不会有新的公共信息更新
- 但你可以基于这个历史信息推测大致的市场状态

## 你的任务

基于上述机制与补偿规则，进行**战略性思考**并判断是否分享数据。

**思考框架**：

1. **评估市场状态**：
   - 当前有多少人在分享？分享率是高是低？
   - 这个状态看起来稳定吗，还是可能大幅变化？

2. **理解推断外部性的核心**：
   - 不分享也会有基础泄露（这是关键！）
   - 分享的真正成本是**边际泄露**（总泄露 - 基础泄露）
   - 补偿应该覆盖的是边际成本，而非总成本

3. **考虑动态效应**：
   - 如果很多人都这样想，会发生什么？
   - 分享集合会如何演化？
   - 你的决策应该指向一个稳定的均衡

4. **评估自己的位置**：
   - 你的隐私偏好 v = {v_i:.3f} 在 [0.3, 1.2] 范围内处于什么水平？
   - 隐私偏好较低的用户更可能从分享中获益
   - 隐私偏好较高的用户更谨慎

5. **参考但不盲从数值**：
   - 上述净效用估算提供了**一个参考方向**
   - 但它基于"其他人不变"的假设，这在动态博弈中不成立
   - 真正重要的是理解机制，而非机械比较数值

## 输出格式

{{
  "decision": 0或1（0=不分享，1=分享），
  "rationale": "你的推理过程（100-150字，说明你如何理解推断外部性、补偿机制，并做出决策）"
}}
"""
        return prompt
    
    def query_llm_sharing_decision(
        self, 
        user_id: int, 
        last_round_broadcast: Dict[str, Any],
        num_trials: int = 1
    ) -> Tuple[int, str]:
        """
        查询LLM的分享决策（并行博弈模式）
        
        Args:
            user_id: 用户ID
            last_round_broadcast: 上一轮的广播信息
            num_trials: 重复查询次数（默认1次，节省成本）
        
        Returns:
            (决策, 推理说明)
        """
        prompt = self.build_sharing_prompt(user_id, last_round_broadcast)
        
        decisions = []
        rationales = []
        
        for trial in range(num_trials):
            retry_count = 0
            max_retries = 1  # 失败时重试一次
            
            while retry_count <= max_retries:
                try:
                    response = self.llm_client.generate_json([
                        {"role": "system", "content": self.build_system_prompt()},
                        {"role": "user", "content": prompt}
                    ])
                    
                    # 容错解析decision
                    raw_decision = response.get("decision", 0)
                    decision = self._parse_decision(raw_decision)
                    
                    if decision not in [0, 1]:
                        print(f"  ⚠️  用户{user_id} 试验{trial+1}: 无效决策 {decision}，默认为0")
                        decision = 0
                    
                    decisions.append(decision)
                    rationales.append(response.get("rationale", ""))
                    break  # 成功，退出重试循环
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"  ⚠️  用户{user_id} 试验{trial+1}失败（已重试{max_retries}次）: {e}")
                        decisions.append(0)  # 失败时默认不分享
                        rationales.append("")
                    else:
                        print(f"  ⚠️  用户{user_id} 试验{trial+1}失败，重试中...")
        
        # 多数投票
        final_decision = 1 if sum(decisions) > len(decisions) / 2 else 0
        final_rationale = rationales[0] if rationales else ""
        
        return final_decision, final_rationale
    
    def _parse_decision(self, raw_decision) -> int:
        """
        容错解析LLM的决策输出
        
        Args:
            raw_decision: LLM输出的原始决策值
        
        Returns:
            解析后的决策（0或1）
        """
        if isinstance(raw_decision, str):
            raw = raw_decision.strip().lower()
            if raw in ["1", "分享", "share", "yes", "true"]:
                return 1
            elif raw in ["0", "不分享", "not_share", "no", "false"]:
                return 0
            else:
                # 尝试转换为整数
                try:
                    return int(raw_decision)
                except:
                    return 0
        elif isinstance(raw_decision, bool):
            return 1 if raw_decision else 0
        else:
            return int(raw_decision)
    
    def simulate_llm_equilibrium(self, num_trials: int = 1, max_rounds: int = 15) -> Dict[str, Any]:
        """
        模拟LLM代理达到的分享均衡（并行博弈+广播）
        
        Args:
            num_trials: 每个决策的重复次数（默认1次）
            max_rounds: 最大轮数
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"🤖 开始并行博弈模拟 (模型: {self.llm_client.config_name})")
        print(f"{'='*60}")
        
        n = self.params.n
        
        # 初始广播信息
        last_round_broadcast = {
            "round": 0,
            "share_set": [],
            "share_rate": 0.0
        }
        
        # 追踪收敛过程
        history = []
        rationales_history = []  # 记录推理过程
        
        converged = False
        cycle_detected = False
        
        for round_num in range(max_rounds):
            print(f"\n{'='*60}")
            print(f"📢 第 {round_num + 1} 轮博弈")
            print(f"{'='*60}")
            print(f"上一轮广播: 分享集合={last_round_broadcast['share_set']}, 分享率={last_round_broadcast['share_rate']:.1%}")
            
            # 第1步：串行模拟并行决策（所有用户看到相同的广播信息）
            round_decisions = {}
            round_rationales = {}
            
            print(f"\n收集所有用户决策...")
            for user_id in range(n):
                decision, rationale = self.query_llm_sharing_decision(
                    user_id,
                    last_round_broadcast,
                    num_trials=num_trials
                )
                round_decisions[user_id] = decision
                round_rationales[user_id] = rationale
                
                print(f"  用户{user_id}: 决策={decision}")
            
            # 第2步：广播结果（选项1：最小广播）
            current_share_set = sorted([i for i in range(n) if round_decisions[i] == 1])
            current_broadcast = {
                "round": round_num + 1,
                "share_set": current_share_set,
                "share_rate": len(current_share_set) / n
            }
            
            print(f"\n📢 广播结果: {current_share_set} (分享率: {current_broadcast['share_rate']:.1%})")
            
            # 记录历史
            history.append(current_share_set)
            rationales_history.append(round_rationales)
            
            # 第3步：检查收敛（连续2轮不变）
            if len(history) >= 2 and history[-1] == history[-2]:
                print(f"\n✅ 在第{round_num + 1}轮达到收敛！")
                converged = True
                break
            
            # 第4步：检测2-cycle振荡（ABAB模式）
            if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
                print(f"\n⚠️  检测到2-cycle振荡，选择更优结果作为稳定输出")
                cycle_detected = True
                
                # 计算两个状态的结果
                set_a = set(history[-1])
                set_b = set(history[-2])
                outcome_a = calculate_outcome(set_a, self.params)
                outcome_b = calculate_outcome(set_b, self.params)
                
                # 选择平台利润更高的状态（也可以选社会福利更高）
                if outcome_a["profit"] >= outcome_b["profit"]:
                    print(f"   选择状态A: {history[-1]} (利润={outcome_a['profit']:.4f})")
                    # 保持当前history[-1]
                else:
                    print(f"   选择状态B: {history[-2]} (利润={outcome_b['profit']:.4f})")
                    # 用状态B替换最后一个
                    history[-1] = history[-2]
                
                converged = True
                break
            
            # 更新广播信息
            last_round_broadcast = current_broadcast
        
        # 计算LLM均衡下的结果
        llm_share_set = history[-1] if history else []
        llm_outcome = calculate_outcome(set(llm_share_set), self.params)
        
        # 与ground truth比较
        gt_share_set = sorted(self.gt_numeric["eq_share_set"])
        gt_profit = self.gt_numeric["eq_profit"]
        gt_W = self.gt_numeric["eq_W"]
        gt_total_leakage = self.gt_numeric["eq_total_leakage"]
        
        # 计算Jaccard相似度
        jaccard_sim = self._jaccard_similarity(set(llm_share_set), set(gt_share_set))
        
        # 计算偏差（指标1：均衡质量）
        results = {
            "model_name": self.llm_client.config_name,
            "llm_share_set": llm_share_set,
            "gt_share_set": gt_share_set,
            "convergence_history": history,
            "rationales_history": rationales_history,  # 保存推理过程
            "converged": converged,
            "cycle_detected": cycle_detected,
            "rounds": len(history),
            "iterations": len(history),  # 为兼容性添加，与rounds相同
            "equilibrium_quality": {
                "share_set_similarity": jaccard_sim,
                "share_rate_error": abs(len(llm_share_set) / n - len(gt_share_set) / n),
                "welfare_mae": abs(llm_outcome["welfare"] - gt_W),
                "profit_mae": abs(llm_outcome["profit"] - gt_profit),
                # 使用与GT对齐的判定标准（而非固定阈值）
                "correct_equilibrium": 1 if jaccard_sim >= 0.6 else 0,
                "equilibrium_type": "good" if jaccard_sim >= 0.6 else "bad"
            },
            "metrics": {
                "llm": {
                    "profit": llm_outcome["profit"],
                    "welfare": llm_outcome["welfare"],
                    "total_leakage": llm_outcome["total_leakage"],
                    "share_rate": len(llm_share_set) / n
                },
                "ground_truth": {
                    "profit": gt_profit,
                    "welfare": gt_W,
                    "total_leakage": gt_total_leakage,
                    "share_rate": len(gt_share_set) / n
                },
                "deviations": {
                    "profit_mae": abs(llm_outcome["profit"] - gt_profit),
                    "welfare_mae": abs(llm_outcome["welfare"] - gt_W),
                    "total_leakage_mae": abs(llm_outcome["total_leakage"] - gt_total_leakage),
                    "share_rate_mae": abs(len(llm_share_set) / n - len(gt_share_set) / n)
                }
            },
            "labels": {
                "llm_leakage_bucket": self._bucket_share_rate(len(llm_share_set) / n),
                "gt_leakage_bucket": self.gt_labels.get("leakage_bucket", "unknown"),
                "llm_over_sharing": 1 if len(llm_share_set) > len(gt_share_set) else 0,
                "gt_over_sharing": self.gt_labels.get("over_sharing", 0)
            }
        }
        
        return results
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """计算Jaccard相似度"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _bucket_share_rate(self, rate: float) -> str:
        """将分享率分桶"""
        if rate < 0.3:
            return "low"
        elif rate < 0.7:
            return "medium"
        else:
            return "high"
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"📊 评估结果摘要")
        print(f"{'='*60}")
        
        print(f"\n【分享集合比较】")
        print(f"  LLM均衡: {results['llm_share_set']}")
        print(f"  理论均衡: {results['gt_share_set']}")
        
        convergence_status = "✅ 已收敛"
        if not results['converged']:
            convergence_status = "❌ 未收敛"
        elif results.get('cycle_detected', False):
            convergence_status = "⚠️  已收敛（2-cycle）"
        
        print(f"  收敛情况: {convergence_status} (共{results['rounds']}轮)")
        
        print(f"\n【均衡质量指标】")
        eq_quality = results['equilibrium_quality']
        print(f"  集合相似度(Jaccard): {eq_quality['share_set_similarity']:.3f}")
        print(f"  分享率误差:          {eq_quality['share_rate_error']:.2%}")
        print(f"  福利偏差(MAE):       {eq_quality['welfare_mae']:.4f}")
        print(f"  利润偏差(MAE):       {eq_quality['profit_mae']:.4f}")
        print(f"  均衡类型:            {eq_quality['equilibrium_type']}")
        print(f"  是否正确均衡:        {'✅' if eq_quality['correct_equilibrium'] == 1 else '❌'}")
        
        print(f"\n【关键指标对比】")
        llm_m = results['metrics']['llm']
        gt_m = results['metrics']['ground_truth']
        dev_m = results['metrics']['deviations']
        
        print(f"  平台利润:     LLM={llm_m['profit']:.4f}  |  GT={gt_m['profit']:.4f}  |  MAE={dev_m['profit_mae']:.4f}")
        print(f"  社会福利:     LLM={llm_m['welfare']:.4f}  |  GT={gt_m['welfare']:.4f}  |  MAE={dev_m['welfare_mae']:.4f}")
        print(f"  总泄露量:     LLM={llm_m['total_leakage']:.4f}  |  GT={gt_m['total_leakage']:.4f}  |  MAE={dev_m['total_leakage_mae']:.4f}")
        print(f"  分享率:       LLM={llm_m['share_rate']:.2%}  |  GT={gt_m['share_rate']:.2%}  |  MAE={dev_m['share_rate_mae']:.2%}")
        
        print(f"\n【收敛轨迹】")
        for i, share_set in enumerate(results['convergence_history']):
            print(f"  第{i+1}轮: {share_set} (分享率: {len(share_set)/self.params.n:.2%})")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {output_path}")


def main():
    """测试评估器"""
    try:
        from .llm_client import create_llm_client
    except ImportError:
        from src.evaluators.llm_client import create_llm_client
    
    # 创建LLM客户端
    llm_client = create_llm_client("gpt-4.1-mini") # 仅为测试示例
    
    # 创建评估器
    evaluator = ScenarioBEvaluator(llm_client)
    
    # 运行评估（并行博弈模式）
    results = evaluator.simulate_llm_equilibrium(num_trials=1, max_rounds=15)
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)
    
    # 保存结果
    evaluator.save_results(results, f"evaluation_results/eval_scenario_B_{llm_client.config_name}.json")


if __name__ == "__main__":
    main()
