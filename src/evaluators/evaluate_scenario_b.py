"""
场景B的LLM评估器 - 静态博弈版本
评估LLM在"Too Much Data"场景下的决策能力（推断外部性）

博弈时序：
1. 阶段0：生成相关结构与隐私偏好（公共知识）
2. 阶段1：平台报价（统一价或个性化价）
3. 阶段2：用户同时决策（基于信念，看不到他人决策）
4. 阶段3：结算（计算泄露、效用、利润）
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Set

# 支持直接运行和模块导入
try:
    from .llm_client import LLMClient
    from src.scenarios.scenario_b_too_much_data import (
        ScenarioBParams, calculate_leakage, calculate_outcome, calculate_outcome_with_prices,
        compute_posterior_covariance, solve_stackelberg_personalized
    )
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.evaluators.llm_client import LLMClient
    from src.scenarios.scenario_b_too_much_data import (
        ScenarioBParams, calculate_leakage, calculate_outcome, calculate_outcome_with_prices,
        compute_posterior_covariance, solve_stackelberg_personalized
    )


class ScenarioBEvaluator:
    """场景B评估器（静态博弈版本）"""
    
    def __init__(self, llm_client: LLMClient, ground_truth_path: str = "data/ground_truth/scenario_b_result.json", use_theory_platform: bool = True):
        """
        初始化评估器
        
        Args:
            llm_client: LLM客户端
            ground_truth_path: ground truth文件路径
        """
        self.llm_client = llm_client
        self.use_theory_platform = use_theory_platform
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
        
        # 计算并缓存相关性结构摘要（用于提示词）
        self.correlation_summaries = self._compute_correlation_summaries()
    
    def _compute_correlation_summaries(self) -> Dict[int, Dict[str, Any]]:
        """
        计算每个用户的相关性结构摘要（压缩表示，用于提示词）
        
        Returns:
            {user_id: {
                "mean_corr": 平均相关系数,
                "topk_neighbors": [(neighbor_id, corr), ...],  # 按相关性降序
                "strong_neighbors_count": 强相关邻居数量(corr > 阈值)
            }}
        """
        n = self.params.n
        Sigma = self.params.Sigma
        summaries = {}
        
        strong_corr_threshold = 0.5  # 定义"强相关"的阈值
        topk = min(3, n - 1)  # 最多显示3个最强相关邻居
        
        for i in range(n):
            # 提取用户i与其他人的相关系数
            corr_with_others = []
            for j in range(n):
                if i != j:
                    corr_ij = Sigma[i, j]
                    corr_with_others.append((j, corr_ij))
            
            # 排序（降序）
            corr_with_others.sort(key=lambda x: x[1], reverse=True)
            
            # 平均相关系数
            mean_corr = np.mean([c for _, c in corr_with_others])
            
            # TopK邻居
            topk_neighbors = corr_with_others[:topk]
            
            # 强相关邻居数量
            strong_count = sum(1 for _, c in corr_with_others if c > strong_corr_threshold)
            
            summaries[i] = {
                "mean_corr": float(mean_corr),
                "topk_neighbors": topk_neighbors,
                "strong_neighbors_count": strong_count
            }
        
        return summaries
    
    def build_system_prompt_user(self) -> str:
        """构建用户的系统提示"""
        return """你是理性经济主体，目标是在不确定他人行为的情况下最大化你的期望效用。
你必须输出严格JSON格式，不要包含任何额外的文本。"""
    
    def build_platform_pricing_prompt(self, pricing_mode: str = "uniform") -> str:
        """
        [已废弃] 构建平台报价提示词
        
        注意：平台报价现在完全由理论求解器决定（solve_stackelberg_personalized），
              基于利润最大化原则计算激励相容的个性化价格。
              此方法保留仅供参考。
        """
        n = self.params.n
        rho = self.params.rho
        sigma_noise_sq = self.params.sigma_noise_sq
        v_min, v_max = 0.3, 1.2
        
        # 构造相关性摘要列表（简版，给平台看）
        corr_summary_list = []
        for i in range(n):
            summary = self.correlation_summaries[i]
            corr_summary_list.append({
                "user_id": i,
                "mean_corr": f"{summary['mean_corr']:.2f}",
                "strong_neighbors_count": summary['strong_neighbors_count']
            })
        
        if pricing_mode == "uniform":
            prompt = f"""
# 场景：数据市场平台定价（统一价格版本）

你是平台。在这一轮你将对所有用户给出**同一报价 p**（take-it-or-leave-it）。
用户将**同时决定是否分享**。你不知道每个用户的隐私偏好实现值，但知道其先验分布与外部性结构摘要。

## 公共信息

**市场规模**：
- 用户总数 n = {n}

**隐私偏好分布**（先验）：
- 所有用户的隐私偏好 v 均匀分布在 [{v_min}, {v_max}]
- 你无法观察到每个用户的具体 v 值

**相关性结构摘要**：
- 总体相关强度（等相关系数）：ρ = {rho:.2f}
- 观测噪声方差：σ² = {sigma_noise_sq}

**用户相关性分布**：
{json.dumps(corr_summary_list, indent=2, ensure_ascii=False)}

## 推断外部性机制

**核心**：用户类型相关，你买到部分数据后可以推断其他人的信息。

**泄露信息量 I_i**：
- 给定分享集合 S，对每个用户 i（包括不分享者），平台对其类型的推断精度提升量
- 通过贝叶斯更新计算：I_i(S) = Var(X_i) - Var(X_i | S)
- **关键外部性**：即使用户 i 不分享，只要其他人分享，平台也能通过相关性推断 i 的信息

**次模性**：
- 分享的人越多，新增一个用户的边际信息价值越低
- 这意味着：价格应平衡"吸引足够多人分享"与"避免过度支付"

## 你的利润函数

**利润 = 总信息价值 - 总支付**：
- U = Σ_i I_i(shares) - (#shares) × p

其中：
- Σ_i I_i(shares)：从所有用户（包括不分享者）获得的总信息价值
- (#shares) × p：支付给分享用户的总费用

**用户决策**：
- 用户 i 会比较分享与不分享的期望效用
- 效用：u_i = share × p - v_i × I_i(share, others)
- 用户基于先验与相关结构形成对"他人分享比例"的信念

## 你的任务

选择一个统一报价 p，使你的**期望利润最大化**。

**思考要点**：
1. **预期分享率**：给定价格 p，你认为会有多少比例的用户选择分享？
   - v 较低的用户更可能分享（隐私成本低）
   - 相关性越强，边际泄露越小（次模性），用户分享意愿可能下降
   
2. **价格权衡**：
   - p 太低 → 分享率低 → 信息少
   - p 太高 → 成本高 → 利润低
   
3. **外部性考虑**：
   - 高相关性 ρ={rho:.2f} 意味着：买到部分数据就能推断很多人
   - 这降低了用户分享的边际价值，但提升了你的总价值

## 输出格式

请输出严格JSON：
{{
  "uniform_price": 一个非负数（你设定的统一报价）,
  "belief_share_rate": 0到1之间的小数（你预期的分享比例）,
  "reason": "简要说明定价逻辑（不超过200字）"
}}
"""
        else:
            # TODO: 个性化价格版本（P2）
            prompt = "个性化价格版本暂未实现"
        
        return prompt
    
    def build_user_decision_prompt(self, user_id: int, price: float) -> str:
        """
        构建用户决策提示词（阶段2：用户同时决策）
        
        Args:
            user_id: 用户ID
            price: 平台给出的报价
        
        Returns:
            提示文本
        """
        v_i = self.params.v[user_id]
        n = self.params.n
        rho = self.params.rho
        sigma_noise_sq = self.params.sigma_noise_sq
        v_min, v_max = 0.3, 1.2
        v_mean = (v_min + v_max) / 2
        
        # 获取该用户的相关性摘要
        corr_summary = self.correlation_summaries[user_id]
        mean_corr = corr_summary["mean_corr"]
        topk_neighbors = corr_summary["topk_neighbors"]
        strong_neighbors_count = corr_summary["strong_neighbors_count"]
        
        # 格式化TopK邻居信息
        neighbors_str = ", ".join([f"用户{j}(相关系数={c:.2f})" for j, c in topk_neighbors])
        
        # 判断用户v在分布中的相对位置
        if v_i < v_mean - 0.2:
            v_level = "低"
            v_description = "偏低"
        elif v_i < v_mean + 0.2:
            v_level = "中"
            v_description = "中等"
        else:
            v_level = "高"
            v_description = "偏高"
        
        prompt = f"""
# 场景：数据市场静态博弈（推断外部性）

你是用户 {user_id}，正在参与一个**一次性的数据市场决策**。

## 基本信息

**你的私有信息**：
- 你的隐私偏好：v[{user_id}] = {v_i:.3f}
- 平台给你的个性化报价：p[{user_id}] = {price:.4f}
  （注意：每个用户的报价可能不同，这是平台根据你的预期贡献定制的价格）

**公共知识**（所有人都知道）：
- 用户总数：n = {n}
- 类型相关系数：ρ = {rho:.2f}
  （你的类型与其他用户的类型相关，相关系数为 {rho:.2f}）
- 观测噪声：σ² = {sigma_noise_sq}
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
  （你的 v = {v_i:.3f}，相对位置：{v_description}，属于{v_level}隐私偏好群体）

**你的相关性结构**：
- 你与其他人的平均相关系数：{mean_corr:.2f}
- 你最强相关的邻居：{neighbors_str}
- 强相关邻居数量（相关系数 > 0.5）：{strong_neighbors_count}

**你不知道的信息**：
- 其他用户的具体 v 值（你只知道分布）
- 其他用户会如何决策（因为是同时决策）

## 推断外部性机制

**关键概念**：即使你不分享数据，平台也能通过其他人的数据推断你的信息。

**泄露信息量 I_i(a)**：
- 给定分享集合 S，平台对你的推断精度提升量
- 通过贝叶斯更新计算：I_i(S) = σ_i² - Var(X_i | S)
- **核心外部性**：I_i 不仅取决于你是否分享，还取决于其他人是否分享

**你的效用函数**：
- 如果你**分享**：u_i = p_i - v_i × I_i(你分享, 其他人的决策)
- 如果你**不分享**：u_i = 0 - v_i × I_i(你不分享, 其他人的决策)

**关键洞察**：
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息）
- 分享的真正成本是**边际泄露** = I_i(分享) - I_i(不分享)
- 补偿价格 p_i 旨在覆盖你的边际隐私损失

## 理性预期决策框架

因为你不知道其他人会如何选择，你需要：

**1. 基于分布推测其他人的行为**：
- v 值较低的用户更可能分享（隐私成本低）
- v 值较高的用户更不可能分享（隐私成本高）
- 你的 v = {v_i:.3f}，处于{v_level}水平

**2. 计算期望效用**：
- E[u_i | 分享] = E[p_i - v_i × I_i(1, a_{{-i}})]
- E[u_i | 不分享] = E[- v_i × I_i(0, a_{{-i}})]

**3. 理解次模性**：
- 分享的人越多，你的边际信息价值越低
- 基础泄露越高（别人分享多），你分享的边际泄露越小

**4. 做出最佳反应**：
- 如果 E[u_i | 分享] > E[u_i | 不分享]，则分享
- 否则不分享

## 你的任务

基于上述机制，在**不知道其他人具体决策**的情况下，通过**理性预期**判断是否分享数据。

**思考要点**：
1. 你的 v 值在分布中的位置如何？（v = {v_i:.3f}，属于{v_level}群体）
2. 预期会有多少比例的用户分享？
3. 在那个预期下，你分享的边际价值是多少？
4. 报价 p = {price:.4f} 能否覆盖你的边际隐私损失？
5. 相关性 ρ = {rho:.2f} 如何影响外部性？

## 输出格式

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "belief_share_rate": 0到1之间的小数（你认为其他人分享的比例）,
  "reason": "简要说明你的权衡与信念依据（不超过150字）"
}}
"""
        return prompt
    
    def _query_platform_pricing_deprecated(self, pricing_mode: str = "uniform", num_trials: int = 1) -> Dict[str, Any]:
        """
        [已废弃] 查询平台报价（LLM模式）
        
        注意：平台报价现在完全由理论求解器决定（solve_stackelberg_personalized），
        基于利润最大化原则计算激励相容的个性化价格。
        此方法保留仅供参考。
        """
        prompt = self.build_platform_pricing_prompt(pricing_mode)
        
        results = []
        
        for trial in range(num_trials):
            retry_count = 0
            max_retries = 1
            
            while retry_count <= max_retries:
                try:
                    response = self.llm_client.generate_json([
                        {"role": "system", "content": self.build_system_prompt_platform()},
                        {"role": "user", "content": prompt}
                    ])
                    
                    # 验证输出
                    if pricing_mode == "uniform":
                        price = float(response.get("uniform_price", 0.0))
                        if price < 0:
                            price = 0.0
                        result = {
                            "uniform_price": price,
                            "belief_share_rate": float(response.get("belief_share_rate", 0.5)),
                            "reason": response.get("reason", "")
                        }
                    else:
                        # TODO: 个性化价格版本
                        result = {}
                    
                    results.append(result)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"  [WARN] 平台报价失败（已重试{max_retries}次）: {e}")
                        # 默认值
                        if pricing_mode == "uniform":
                            results.append({
                                "uniform_price": 0.5,  # 默认中等价格
                                "belief_share_rate": 0.5,
                                "reason": "查询失败，使用默认值"
                            })
                    else:
                        print(f"  [WARN] 平台报价失败，重试中...")
        
        # 如果有多次试验，取平均（或使用其他聚合策略）
        if pricing_mode == "uniform":
            avg_price = np.mean([r["uniform_price"] for r in results])
            return {
                "uniform_price": avg_price,
                "belief_share_rate": results[0]["belief_share_rate"],
                "reason": results[0]["reason"]
            }
        else:
            return results[0] if results else {}
    
    def query_user_decision(
        self, 
        user_id: int, 
        price: float,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        查询用户决策（阶段2）
        
        Args:
            user_id: 用户ID
            price: 平台报价
            num_trials: 重复查询次数（多数投票）
        
        Returns:
            {
                "share": int (0或1),
                "belief_share_rate": float,
                "reason": str
            }
        """
        prompt = self.build_user_decision_prompt(user_id, price)
        
        decisions = []
        beliefs = []
        reasons = []
        
        for trial in range(num_trials):
            retry_count = 0
            max_retries = 1
            
            while retry_count <= max_retries:
                try:
                    response = self.llm_client.generate_json([
                        {"role": "system", "content": self.build_system_prompt_user()},
                        {"role": "user", "content": prompt}
                    ])
                    
                    # 容错解析
                    raw_share = response.get("share", 0)
                    share = self._parse_decision(raw_share)
                    
                    if share not in [0, 1]:
                        print(f"  [WARN] 用户{user_id} 试验{trial+1}: 无效决策 {share}，默认为0")
                        share = 0
                    
                    decisions.append(share)
                    beliefs.append(float(response.get("belief_share_rate", 0.5)))
                    reasons.append(response.get("reason", ""))
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"  [WARN] 用户{user_id} 试验{trial+1}失败（已重试{max_retries}次）: {e}")
                        decisions.append(0)
                        beliefs.append(0.5)
                        reasons.append("")
                    else:
                        print(f"  [WARN] 用户{user_id} 试验{trial+1}失败，重试中...")
        
        # 多数投票
        final_decision = 1 if sum(decisions) > len(decisions) / 2 else 0
        final_belief = np.mean(beliefs) if beliefs else 0.5
        final_reason = reasons[0] if reasons else ""
        
        return {
            "share": final_decision,
            "belief_share_rate": final_belief,
            "reason": final_reason
        }
    
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
    
    def simulate_static_game(self, num_trials: int = 1) -> Dict[str, Any]:
        """
        模拟静态博弈（两阶段：平台报价 → 用户同时决策）
        
        平台使用理论求解器计算个性化价格向量 p = [p_0, p_1, ..., p_{n-1}]，
        然后用户基于各自观察到的价格 p_i 同时做出分享决策。
        
        Args:
            num_trials: 每个决策的重复查询次数
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"[开始静态博弈模拟] 模型: {self.llm_client.config_name}")
        print(f"{'='*60}")
        
        n = self.params.n
        
        # ===== 阶段1：平台报价 =====
        print(f"\n{'='*60}")
        print(f"[阶段1] 平台报价")
        print(f"{'='*60}")
        
        
        # 平台定价：紧贴 Too Much Data（TMD）机制
        # 默认使用理论基线求解器（Stackelberg，个性化 take-it-or-leave-it 要约），而不是让平台LLM“自由输出价格”。
        if self.use_theory_platform:
            # 【性能优化】直接从预加载的ground truth获取价格，无需重新求解
            prices = self.gt_numeric["eq_prices"]
            theory_share_set = self.gt_numeric["eq_share_set"]
            theory_profit = self.gt_numeric["eq_profit"]
            solver_mode = self.gt_numeric.get("solver_mode", "exact")
            
            print(f"[优化] 使用预计算的理论最优价格（无需重新求解）")
            print(f"求解器模式: {solver_mode}")
            print(f"理论最优分享集合: {theory_share_set} (规模: {len(theory_share_set)}/{n})")
            print(f"个性化价格向量范围: [{min(prices):.4f}, {max(prices):.4f}]")
            print(f"价格向量统计: 均值={sum(prices)/n:.4f}, 非零价格数={sum(1 for p in prices if p > 0)}")
            # 均衡审计信息
            diag = self.gt_numeric.get("diagnostics", {})
            if diag:
                print(f"均衡裕度: min_margin_in={diag.get('min_margin_in'):.6f}, "
                      f"max_margin_out={diag.get('max_margin_out'):.6f}")
            
            # 记录平台信息（用于结果构造）
            platform_info = {
                "solver_mode": solver_mode,
                "theory_share_set": theory_share_set,
                "theory_profit": theory_profit,
                "prices": prices,
                "diagnostics": diag,
                "source": "precomputed_ground_truth"  # 标记来源
            }


# ===== 阶段2：用户同时决策 =====
        print(f"\n{'='*60}")
        print(f"[阶段2] 用户同时决策")
        print(f"{'='*60}")
        
        user_decisions = {}
        user_beliefs = {}
        user_reasons = {}
        
        print(f"\n收集所有用户决策（每个用户观察自己的个性化报价）...")
        for user_id in range(n):
            user_price = prices[user_id]
            decision_result = self.query_user_decision(user_id, user_price, num_trials=num_trials)
            user_decisions[user_id] = decision_result["share"]
            user_beliefs[user_id] = decision_result["belief_share_rate"]
            user_reasons[user_id] = decision_result["reason"]
            
            print(f"  用户{user_id}: price={user_price:.4f}, share={decision_result['share']}, "
                  f"belief={decision_result['belief_share_rate']:.2%}, v={self.params.v[user_id]:.3f}")
        
        # ===== 阶段3：结算 =====
        print(f"\n{'='*60}")
        print(f"[阶段3] 结算")
        print(f"{'='*60}")
        
        llm_share_set = sorted([i for i in range(n) if user_decisions[i] == 1])
        llm_outcome = calculate_outcome_with_prices(set(llm_share_set), self.params, prices)
        
        print(f"分享集合: {llm_share_set}")
        print(f"分享率: {len(llm_share_set) / n:.2%}")
        print(f"平台利润: {llm_outcome['profit']:.4f}")
        print(f"社会福利: {llm_outcome['welfare']:.4f}")
        
        # ===== 与Ground Truth比较 =====
        gt_share_set = sorted(self.gt_numeric["eq_share_set"])
        gt_profit = self.gt_numeric["eq_profit"]
        gt_W = self.gt_numeric["eq_W"]
        gt_total_leakage = self.gt_numeric["eq_total_leakage"]
        
        # 计算Jaccard相似度
        jaccard_sim = self._jaccard_similarity(set(llm_share_set), set(gt_share_set))
        
        # 构造结果
        results = {
            "model_name": self.llm_client.config_name,
            
            # 平台数据（个性化定价）
            "platform": platform_info,
            
            # 用户数据
            "users": {
                "decisions": user_decisions,
                "beliefs": user_beliefs,
                "reasons": user_reasons,
                "v_values": self.params.v
            },
            
            # 结果
            "llm_share_set": llm_share_set,
            "gt_share_set": gt_share_set,
            
            # 均衡质量指标
            "equilibrium_quality": {
                "share_set_similarity": jaccard_sim,
                "share_rate_error": abs(len(llm_share_set) / n - len(gt_share_set) / n),
                "welfare_mae": abs(llm_outcome["welfare"] - gt_W),
                "profit_mae": abs(llm_outcome["profit"] - gt_profit),
                "correct_equilibrium": 1 if jaccard_sim >= 0.6 else 0,
                "equilibrium_type": "good" if jaccard_sim >= 0.6 else "bad"
            },
            
            # 详细指标
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
            
            # 标签
            "labels": {
                "llm_leakage_bucket": self._bucket_share_rate(len(llm_share_set) / n),
                "gt_leakage_bucket": self.gt_labels.get("leakage_bucket", "unknown"),
                "llm_over_sharing": 1 if len(llm_share_set) > len(gt_share_set) else 0,
                "gt_over_sharing": self.gt_labels.get("over_sharing", 0)
            },
            
            # 信念一致性分析
            "belief_consistency": self._analyze_belief_consistency(user_beliefs, user_decisions)
        }
        
        return results
    
    def _analyze_belief_consistency(self, user_beliefs: Dict[int, float], user_decisions: Dict[int, int]) -> Dict[str, Any]:
        """
        分析用户信念与实际结果的一致性
        
        Args:
            user_beliefs: 每个用户对分享率的信念
            user_decisions: 每个用户的实际决策
        
        Returns:
            一致性分析结果
        """
        n = len(user_decisions)
        actual_share_rate = sum(user_decisions.values()) / n
        
        # 计算信念与实际的偏差
        belief_errors = []
        for user_id, belief in user_beliefs.items():
            error = abs(belief - actual_share_rate)
            belief_errors.append(error)
        
        return {
            "actual_share_rate": actual_share_rate,
            "mean_belief": np.mean(list(user_beliefs.values())),
            "mean_belief_error": np.mean(belief_errors),
            "max_belief_error": np.max(belief_errors),
            "belief_std": np.std(list(user_beliefs.values()))
        }
    
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
        print(f"[评估结果摘要]")
        print(f"{'='*60}")
        
        print(f"\n        【平台报价】")
        platform = results['platform']
        theory_share_set = platform.get("theory_share_set", [])
        prices = platform.get("prices", [])
        solver_mode = platform.get("solver_mode", "unknown")
        theory_profit = platform.get("theory_profit", 0.0)
        
        print(f"  求解器模式: {solver_mode}")
        print(f"  理论最优分享集合规模: {len(theory_share_set)}")
        print(f"  理论最优利润: {theory_profit:.4f}")
        if prices:
            print(f"  价格范围: [{min(prices):.4f}, {max(prices):.4f}]")
            print(f"  平均价格: {sum(prices)/len(prices):.4f}")
            print(f"  理论分享集合: {platform['theory_share_set']}")
        elif platform.get('mode') == 'llm_pricing':
            # LLM定价模式
            print(f"  定价模式: LLM定价")
            if 'uniform_price' in platform:
                print(f"  统一价格: {platform['uniform_price']:.4f}")
            if 'belief_share_rate' in platform:
                print(f"  平台预期分享率: {platform['belief_share_rate']:.2%}")
            if 'reason' in platform:
                print(f"  平台理由: {platform['reason'][:150]}...")
        else:
            # 兼容旧格式
            if 'uniform_price' in platform:
                print(f"  统一价格: {platform['uniform_price']:.4f}")
            if 'belief_share_rate' in platform:
                print(f"  平台预期分享率: {platform['belief_share_rate']:.2%}")
        
        print(f"\n【分享集合比较】")
        print(f"  LLM结果: {results['llm_share_set']}")
        print(f"  理论均衡: {results['gt_share_set']}")
        
        print(f"\n【均衡质量指标】")
        eq_quality = results['equilibrium_quality']
        print(f"  集合相似度(Jaccard): {eq_quality['share_set_similarity']:.3f}")
        print(f"  分享率误差:          {eq_quality['share_rate_error']:.2%}")
        print(f"  福利偏差(MAE):       {eq_quality['welfare_mae']:.4f}")
        print(f"  利润偏差(MAE):       {eq_quality['profit_mae']:.4f}")
        print(f"  均衡类型:            {eq_quality['equilibrium_type']}")
        print(f"  是否正确均衡:        {'[YES]' if eq_quality['correct_equilibrium'] == 1 else '[NO]'}")
        
        print(f"\n【关键指标对比】")
        llm_m = results['metrics']['llm']
        gt_m = results['metrics']['ground_truth']
        dev_m = results['metrics']['deviations']
        
        print(f"  平台利润:     LLM={llm_m['profit']:.4f}  |  GT={gt_m['profit']:.4f}  |  MAE={dev_m['profit_mae']:.4f}")
        print(f"  社会福利:     LLM={llm_m['welfare']:.4f}  |  GT={gt_m['welfare']:.4f}  |  MAE={dev_m['welfare_mae']:.4f}")
        print(f"  总泄露量:     LLM={llm_m['total_leakage']:.4f}  |  GT={gt_m['total_leakage']:.4f}  |  MAE={dev_m['total_leakage_mae']:.4f}")
        print(f"  分享率:       LLM={llm_m['share_rate']:.2%}  |  GT={gt_m['share_rate']:.2%}  |  MAE={dev_m['share_rate_mae']:.2%}")
        
        print(f"\n【信念一致性分析】")
        belief = results['belief_consistency']
        print(f"  实际分享率:         {belief['actual_share_rate']:.2%}")
        print(f"  平均信念分享率:     {belief['mean_belief']:.2%}")
        print(f"  平均信念误差:       {belief['mean_belief_error']:.2%}")
        print(f"  最大信念误差:       {belief['max_belief_error']:.2%}")
        print(f"  信念标准差:         {belief['belief_std']:.3f}")
        
        print(f"\n【用户决策分析】")
        users = results['users']
        n = len(users['decisions'])
        
        # 按v值分组分析
        v_low = [i for i in range(n) if users['v_values'][i] < 0.6]
        v_mid = [i for i in range(n) if 0.6 <= users['v_values'][i] < 0.9]
        v_high = [i for i in range(n) if users['v_values'][i] >= 0.9]
        
        for group_name, group_users in [("低v组", v_low), ("中v组", v_mid), ("高v组", v_high)]:
            if group_users:
                share_rate = sum(users['decisions'][i] for i in group_users) / len(group_users)
                avg_belief = np.mean([users['beliefs'][i] for i in group_users])
                print(f"  {group_name} (n={len(group_users)}): 分享率={share_rate:.2%}, 平均信念={avg_belief:.2%}")
    
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
    llm_client = create_llm_client("gpt-4.1-mini")  # 仅为测试示例
    
    # 创建评估器
    evaluator = ScenarioBEvaluator(llm_client)
    
    # 运行评估（静态博弈模式，使用个性化定价）
    results = evaluator.simulate_static_game(num_trials=1)
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)
    
    # 保存结果
    evaluator.save_results(results, f"evaluation_results/eval_scenario_B_{llm_client.config_name}.json")


if __name__ == "__main__":
    main()
