"""
场景B主评估器 - 推断外部性博弈

评估LLM在"Too Much Data"场景下的决策能力（推断外部性）

博弈时序：
1. 阶段0：生成相关结构与隐私偏好（公共知识）
2. 阶段1：平台报价（统一价或个性化价）
3. 阶段2：用户同时决策（基于信念，看不到他人决策）
4. 阶段3：结算（计算泄露、效用、利润）

支持两种博弈模式：
- 静态博弈：一次性决策（理论基准）
- 虚拟博弈：多轮迭代学习（测试LLM学习能力）

========================================
运行方式
========================================

1. 【静态博弈】（一次性决策，对比基准）
   
   python -m src.evaluators.evaluate_scenario_b --mode static --model deepseek-v3.2
   
   输出：
   - evaluation_results/static_deepseek-v3/eval_YYYYMMDD_HHMMSS.json

2. 【虚拟博弈】（多轮迭代学习，主要测试）
   
   python -m src.evaluators.evaluate_scenario_b --mode fp --model deepseek-v3.2
   
   # 自定义参数（默认50轮，信念窗口10，1次试验）
   python -m src.evaluators.evaluate_scenario_b \
       --mode fp \
       --model deepseek-v3.2 \
       --max_rounds 50 \
       --belief_window 10 \
       --num_trials 1
   
   输出：
   - evaluation_results/fp_deepseek-v3/eval_YYYYMMDD_HHMMSS.json
   - evaluation_results/fp_deepseek-v3/eval_YYYYMMDD_HHMMSS_share_rate.png
   - evaluation_results/fp_deepseek-v3/eval_YYYYMMDD_HHMMSS_strategy_heatmap.png

3. 【同时运行两种模式】
   
   python -m src.evaluators.evaluate_scenario_b --mode both --model deepseek-v3.2

4. 【为已有结果生成可视化】
   从已保存的FP结果JSON文件生成图表：
   
   # 单个文件
   python -m src.evaluators.evaluate_scenario_b --visualize evaluation_results/fp_deepseek-v3.2/eval_20260122_143530.json
   
   # 整个目录
   python -m src.evaluators.evaluate_scenario_b --visualize evaluation_results/fp_deepseek-v3.2/
   
   # 多个路径或通配符
   python -m src.evaluators.evaluate_scenario_b --visualize "evaluation_results/fp_*/eval_*.json"

========================================
参数说明
========================================

--mode: 运行模式
  - static: 静态博弈（一次性决策）
  - fp: 虚拟博弈（多轮学习）
  - both: 同时运行两种模式

--model: 模型配置名称
  默认: deepseek-v3.2
  可选: deepseek-v3.2, gpt-4.1-mini 等（见configs/model_configs.json）

--max_rounds: 虚拟博弈最大轮数
  默认: 50轮

--belief_window: 虚拟博弈信念窗口大小
  默认: 10（使用最近10轮历史）

--num_trials: 重复试验次数
  默认: 1次

--visualize: 可视化模式
  为已有JSON文件生成图表（支持文件、目录、通配符）

========================================
静态博弈 vs 虚拟博弈的区别
========================================

| 特性           | 静态博弈              | 虚拟博弈                |
|----------------|----------------------|-------------------------|
| 决策次数       | 1次                  | 最多50轮                |
| 历史信息       | 无                   | 观察最近10轮分享情况    |
| 信念更新       | 初始均匀分布         | 经验频率动态更新        |
| 收敛检测       | N/A                  | 连续3轮分享集合不变     |
| 提示词         | 简洁版               | 包含历史+概率估计       |
| 输出           | JSON                 | JSON + 可视化图表       |
| 测试目标       | 理解推断外部性       | 学习与收敛能力          |

========================================
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Set

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    
    
    def build_user_decision_prompt_fp(self, user_id: int, price: float, history: List[Dict[int, int]], belief_probs: Dict[int, float], current_round: int) -> str:
        """
        构建用户决策提示词（虚拟博弈版本）
        
        Args:
            user_id: 用户ID
            price: 平台给出的报价
            history: 历史记录（最近若干轮）
            belief_probs: 基于历史的其他用户分享概率估计
            current_round: 当前轮数
        
        Returns:
            提示文本
        """
        v_i = self.params.v[user_id]
        n = self.params.n
        rho = self.params.rho
        sigma_noise_sq = self.params.sigma_noise_sq
        v_min, v_max = 0.3, 1.2
        v_mean = (v_min + v_max) / 2
        
        # 判断用户v在分布中的相对位置
        if v_i < v_mean - 0.2:
            v_description = "偏低"
        elif v_i < v_mean + 0.2:
            v_description = "中等"
        else:
            v_description = "偏高"
        
        # 构建历史观察部分
        history_text = ""
        if len(history) == 0:
            # 第1轮，没有历史
            history_text = """【历史观察】
这是第一轮决策，暂无历史记录。你可以假设其他用户各有50%的概率选择分享。"""
        else:
            # 有历史记录
            history_lines = []
            for idx, round_decisions in enumerate(history):
                share_set = sorted([uid for uid, decision in round_decisions.items() if decision == 1])
                history_lines.append(f"- 轮次{idx+1}: {{{', '.join(map(str, share_set))}}}")
            
            # 构建概率估计
            prob_lines = []
            for uid in range(n):
                if uid != user_id:  # 不显示自己的
                    prob = belief_probs.get(uid, 0.5)
                    prob_lines.append(f"- 用户{uid}: {prob:.0%}")
            
            history_text = f"""【历史观察】
最近{len(history)}轮的分享情况：
{chr(10).join(history_lines)}

基于历史的分享概率估计：
{chr(10).join(prob_lines)}"""
        
        prompt = f"""
# 场景：数据市场决策（推断外部性）

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

## 公共知识：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
  你的信息与其他用户的信息相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

{history_text}

## 核心机制：
- **推断外部性**：泄露信息量不仅取决于你是否分享，还取决于其他人是否分享。任何人的分享都会增加所有人（包括不分享者）的信息泄露量。
- 如果你**分享**，你会得到来自平台的补偿 p = {price:.4f}，但你的信息会从间接部分泄露变成完全泄露
- 如果你**不分享**，你可以保护未间接泄露的那部分信息，但代价是无法得到补偿
- **次模性**：分享的人越多，你再分享带来的边际泄露越小（基础泄露越高，边际泄露越低）
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息），分享的真正成本是**边际泄露**带来的成本
- 补偿价格旨在覆盖你的边际隐私损失

## 决策框架：
- 隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 p vs 隐私成本 v × 边际泄露量
- 基于历史观察和其他用户的分享概率，估计你的期望效用

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
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

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

## 公共知识：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
  你的信息与其他用户的信息相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

## 核心机制：
- **推断外部性**：泄露信息量不仅取决于你是否分享，还取决于其他人是否分享。任何人的分享都会增加所有人（包括不分享者）的信息泄露量。
- 如果你**分享**，你会得到来自平台的补偿 p = {price:.4f}，但你的信息会从间接部分泄露变成完全泄露
- 如果你**不分享**，你可以保护未间接泄露的那部分信息，但代价是无法得到补偿
- **次模性**：分享的人越多，你再分享带来的边际泄露越小（基础泄露越高，边际泄露越低）
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息），分享的真正成本是**边际泄露**带来的成本
- 补偿价格旨在覆盖你的边际隐私损失

## 决策框架：
- 隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 p vs 隐私成本 v × 边际泄露量

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
        return prompt
    
    
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
    
    def _compute_belief_probs(self, history: List[Dict[int, int]], window_size: int = 10) -> Dict[int, float]:
        """
        基于历史计算每个用户的分享概率（经验频率）
        
        Args:
            history: 历史决策记录
            window_size: 窗口大小
        
        Returns:
            {user_id: 分享概率}
        """
        if len(history) == 0:
            # 初始信念：均匀分布
            return {i: 0.5 for i in range(self.params.n)}
        
        # 只看最近window_size轮
        recent_history = history[-window_size:]
        
        belief_probs = {}
        for user_id in range(self.params.n):
            share_count = sum(1 for round_decisions in recent_history if round_decisions.get(user_id, 0) == 1)
            belief_probs[user_id] = share_count / len(recent_history)
        
        return belief_probs
    
    def _check_convergence(self, history: List[Dict[int, int]], threshold: int = 3) -> bool:
        """
        检查是否收敛（连续threshold轮分享集合不变）
        
        Args:
            history: 历史决策记录
            threshold: 稳定轮数阈值
        
        Returns:
            是否收敛
        """
        if len(history) < threshold:
            return False
        
        # 检查最后threshold轮
        recent = history[-threshold:]
        share_sets = [frozenset([uid for uid, dec in round_decisions.items() if dec == 1]) for round_decisions in recent]
        
        # 所有集合是否相同
        return len(set(share_sets)) == 1
    
    def _compute_hamming_distance(self, decisions1: Dict[int, int], decisions2: Dict[int, int]) -> int:
        """计算两个决策向量的汉明距离"""
        diff = 0
        for uid in decisions1.keys():
            if decisions1[uid] != decisions2[uid]:
                diff += 1
        return diff
    
    def _analyze_convergence(self, history: List[Dict[int, int]], gt_share_set: set) -> Dict[str, Any]:
        """
        分析虚拟博弈的收敛性
        
        Args:
            history: 完整历史记录
            gt_share_set: 理论均衡分享集合
        
        Returns:
            收敛分析结果
        """
        n = self.params.n
        total_rounds = len(history)
        
        # 分享率轨迹
        share_rate_trajectory = [sum(dec.values()) / n for dec in history]
        
        # 与理论均衡的相似度轨迹
        similarity_trajectory = []
        for round_decisions in history:
            share_set = set([uid for uid, dec in round_decisions.items() if dec == 1])
            sim = self._jaccard_similarity(share_set, gt_share_set)
            similarity_trajectory.append(sim)
        
        # 汉明距离序列（相邻轮次）
        hamming_distances = []
        for i in range(1, len(history)):
            dist = self._compute_hamming_distance(history[i-1], history[i])
            hamming_distances.append(dist)
        
        # 检查收敛轮数（连续3轮不变）
        convergence_round = None
        for i in range(2, len(history)):
            if (hamming_distances[i-2] == 0 and 
                hamming_distances[i-1] == 0 and 
                i < len(hamming_distances) and
                hamming_distances[i] == 0):
                convergence_round = i - 1  # 从这一轮开始稳定
                break
        
        # 最后10轮稳定性
        last_10_hamming = hamming_distances[-10:] if len(hamming_distances) >= 10 else hamming_distances
        stability = 1.0 - (np.mean(last_10_hamming) / n) if last_10_hamming else 0.0
        
        # 检测周期震荡（简单版本：检查最后20轮是否有2-周期）
        oscillation_detected = False
        if len(history) >= 20:
            last_20 = history[-20:]
            share_sets = [frozenset([uid for uid, dec in rd.items() if dec == 1]) for rd in last_20]
            # 检查是否交替出现两个不同的集合
            unique_sets = list(set(share_sets))
            if len(unique_sets) == 2:
                # 检查是否交替
                pattern = [share_sets[i] == unique_sets[0] for i in range(len(share_sets))]
                alternating = all(pattern[i] != pattern[i+1] for i in range(len(pattern)-1))
                if alternating:
                    oscillation_detected = True
        
        # 最终分享集合
        final_share_set = set([uid for uid, dec in history[-1].items() if dec == 1])
        
        return {
            "converged": convergence_round is not None,
            "convergence_round": convergence_round,
            "final_stability": float(stability),
            "oscillation_detected": oscillation_detected,
            "avg_hamming_distance_last10": float(np.mean(last_10_hamming)) if last_10_hamming else 0.0,
            "share_rate_trajectory": share_rate_trajectory,
            "similarity_trajectory": similarity_trajectory,
            "hamming_distances": hamming_distances,
            "final_share_set": sorted(list(final_share_set)),
            "final_similarity_to_equilibrium": similarity_trajectory[-1] if similarity_trajectory else 0.0
        }
    
    def simulate_fictitious_play(self, max_rounds: int = 50, belief_window: int = 10, num_trials: int = 1) -> Dict[str, Any]:
        """
        模拟虚拟博弈（Fictitious Play）
        
        Args:
            max_rounds: 最大轮数
            belief_window: 信念窗口大小（使用最近N轮历史）
            num_trials: 每个决策的重复查询次数（建议为1以控制成本）
        
        Returns:
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"[开始虚拟博弈模拟] 模型: {self.llm_client.config_name}")
        print(f"最大轮数: {max_rounds}, 信念窗口: {belief_window}")
        print(f"{'='*60}")
        
        n = self.params.n
        
        # ===== 阶段0：平台报价（固定，使用理论最优价格）=====
        print(f"\n{'='*60}")
        print(f"[阶段0] 平台报价（固定）")
        print(f"{'='*60}")
        
        prices = self.gt_numeric["eq_prices"]
        theory_share_set = self.gt_numeric["eq_share_set"]
        theory_profit = self.gt_numeric["eq_profit"]
        
        print(f"使用理论最优价格")
        print(f"理论均衡分享集合: {theory_share_set} (规模: {len(theory_share_set)}/{n})")
        print(f"价格向量范围: [{min(prices):.4f}, {max(prices):.4f}]")
        
        # ===== 虚拟博弈迭代 =====
        history = []  # 记录每轮的决策 [{user_id: decision}, ...]
        
        for round_num in range(max_rounds):
            print(f"\n{'='*60}")
            print(f"[轮次 {round_num + 1}/{max_rounds}]")
            print(f"{'='*60}")
            
            # 计算当前信念（基于历史）
            window = min(belief_window, round_num)  # 实际窗口大小
            belief_probs = self._compute_belief_probs(history, window_size=window)
            
            # 用户同时决策
            round_decisions = {}
            for user_id in range(n):
                user_price = prices[user_id]
                
                # 获取历史（只传递窗口内的）
                recent_history = history[-window:] if window > 0 else []
                
                # 查询决策
                decision_result = self.query_user_decision_fp(
                    user_id, 
                    user_price, 
                    recent_history,
                    belief_probs,
                    round_num,
                    num_trials=num_trials
                )
                
                round_decisions[user_id] = decision_result["share"]
                
                if round_num % 10 == 0 or round_num < 5:  # 只在部分轮次打印详细信息
                    print(f"  用户{user_id}: share={decision_result['share']}, v={self.params.v[user_id]:.3f}")
            
            # 记录本轮结果
            history.append(round_decisions)
            
            # 计算本轮分享集合
            share_set = sorted([uid for uid, dec in round_decisions.items() if dec == 1])
            share_rate = len(share_set) / n
            print(f"本轮分享集合: {share_set} (分享率: {share_rate:.2%})")
            
            # 检查收敛
            if self._check_convergence(history, threshold=3):
                print(f"\n[提前收敛] 连续3轮分享集合不变，停止迭代")
                break
        
        # ===== 最终结算 =====
        print(f"\n{'='*60}")
        print(f"[虚拟博弈结束] 总轮数: {len(history)}")
        print(f"{'='*60}")
        
        final_decisions = history[-1]
        final_share_set = sorted([uid for uid, dec in final_decisions.items() if dec == 1])
        final_outcome = calculate_outcome_with_prices(set(final_share_set), self.params, prices)
        
        print(f"最终分享集合: {final_share_set}")
        print(f"最终分享率: {len(final_share_set) / n:.2%}")
        print(f"平台利润: {final_outcome['profit']:.4f}")
        print(f"社会福利: {final_outcome['welfare']:.4f}")
        
        # ===== 收敛性分析 =====
        gt_share_set = set(self.gt_numeric["eq_share_set"])
        convergence_analysis = self._analyze_convergence(history, gt_share_set)
        
        print(f"\n[收敛性分析]")
        print(f"是否收敛: {convergence_analysis['converged']}")
        if convergence_analysis['converged']:
            print(f"收敛轮数: {convergence_analysis['convergence_round']}")
        print(f"最终稳定性: {convergence_analysis['final_stability']:.3f}")
        print(f"是否检测到震荡: {convergence_analysis['oscillation_detected']}")
        print(f"最终与理论均衡的相似度: {convergence_analysis['final_similarity_to_equilibrium']:.3f}")
        
        # ===== 与Ground Truth比较 =====
        gt_profit = self.gt_numeric["eq_profit"]
        gt_W = self.gt_numeric["eq_W"]
        gt_total_leakage = self.gt_numeric["eq_total_leakage"]
        
        jaccard_sim = self._jaccard_similarity(set(final_share_set), gt_share_set)
        
        # 构造结果
        results = {
            "model_name": self.llm_client.config_name,
            "game_type": "fictitious_play",
            "max_rounds": max_rounds,
            "actual_rounds": len(history),
            "belief_window": belief_window,
            
            # 平台数据
            "platform": {
                "prices": prices,
                "theory_share_set": theory_share_set,
                "theory_profit": theory_profit,
                "source": "precomputed_ground_truth"
            },
            
            # 虚拟博弈历史
            "history": history,  # 完整历史
            
            # 收敛性分析
            "convergence_analysis": convergence_analysis,
            
            # 最终结果
            "final_share_set": final_share_set,
            "gt_share_set": sorted(list(gt_share_set)),
            
            # 均衡质量指标
            "equilibrium_quality": {
                "share_set_similarity": jaccard_sim,
                "share_rate_error": abs(len(final_share_set) / n - len(gt_share_set) / n),
                "welfare_mae": abs(final_outcome["welfare"] - gt_W),
                "profit_mae": abs(final_outcome["profit"] - gt_profit),
                "correct_equilibrium": 1 if jaccard_sim >= 0.6 else 0,
                "equilibrium_type": "good" if jaccard_sim >= 0.6 else "bad"
            },
            
            # 详细指标
            "metrics": {
                "final": {
                    "profit": final_outcome["profit"],
                    "welfare": final_outcome["welfare"],
                    "total_leakage": final_outcome["total_leakage"],
                    "share_rate": len(final_share_set) / n
                },
                "ground_truth": {
                    "profit": gt_profit,
                    "welfare": gt_W,
                    "total_leakage": gt_total_leakage,
                    "share_rate": len(gt_share_set) / n
                },
                "deviations": {
                    "profit_mae": abs(final_outcome["profit"] - gt_profit),
                    "welfare_mae": abs(final_outcome["welfare"] - gt_W),
                    "total_leakage_mae": abs(final_outcome["total_leakage"] - gt_total_leakage),
                    "share_rate_mae": abs(len(final_share_set) / n - len(gt_share_set) / n)
                }
            },
            
            # 标签
            "labels": {
                "final_leakage_bucket": self._bucket_share_rate(len(final_share_set) / n),
                "gt_leakage_bucket": self.gt_labels.get("leakage_bucket", "unknown"),
                "final_over_sharing": 1 if len(final_share_set) > len(gt_share_set) else 0,
                "gt_over_sharing": self.gt_labels.get("over_sharing", 0)
            }
        }
        
        return results
    
    def query_user_decision_fp(
        self, 
        user_id: int, 
        price: float,
        history: List[Dict[int, int]],
        belief_probs: Dict[int, float],
        current_round: int,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        查询用户决策（虚拟博弈版本）
        
        Args:
            user_id: 用户ID
            price: 平台报价
            history: 历史记录
            belief_probs: 信念概率
            current_round: 当前轮数
            num_trials: 重复查询次数
        
        Returns:
            {
                "share": int (0或1),
                "reason": str
            }
        """
        prompt = self.build_user_decision_prompt_fp(user_id, price, history, belief_probs, current_round)
        
        decisions = []
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
                        print(f"  [WARN] 用户{user_id} 轮次{current_round+1}: 无效决策 {share}，默认为0")
                        share = 0
                    
                    decisions.append(share)
                    reasons.append(response.get("reason", ""))
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"  [WARN] 用户{user_id} 轮次{current_round+1}失败: {e}")
                        decisions.append(0)
                        reasons.append("")
                    else:
                        print(f"  [WARN] 用户{user_id} 轮次{current_round+1}失败，重试中...")
        
        # 多数投票
        final_decision = 1 if sum(decisions) > len(decisions) / 2 else 0
        final_reason = reasons[0] if reasons else ""
        
        return {
            "share": final_decision,
            "reason": final_reason
        }
    
    def print_evaluation_summary_fp(self, results: Dict[str, Any]):
        """打印虚拟博弈评估摘要"""
        print(f"\n{'='*60}")
        print(f"[虚拟博弈评估结果摘要]")
        print(f"{'='*60}")
        
        print(f"\n【游戏设置】")
        print(f"  模型: {results['model_name']}")
        print(f"  最大轮数: {results['max_rounds']}")
        print(f"  实际轮数: {results['actual_rounds']}")
        print(f"  信念窗口: {results['belief_window']}")
        
        print(f"\n【平台报价】")
        platform = results['platform']
        prices = platform['prices']
        print(f"  价格范围: [{min(prices):.4f}, {max(prices):.4f}]")
        print(f"  平均价格: {sum(prices)/len(prices):.4f}")
        print(f"  理论分享集合: {platform['theory_share_set']}")
        
        print(f"\n【收敛性分析】")
        conv = results['convergence_analysis']
        print(f"  是否收敛: {'是' if conv['converged'] else '否'}")
        if conv['converged']:
            print(f"  收敛轮数: 第{conv['convergence_round']}轮")
        print(f"  最终稳定性: {conv['final_stability']:.3f}")
        print(f"  是否检测到震荡: {'是' if conv['oscillation_detected'] else '否'}")
        print(f"  最后10轮平均汉明距离: {conv['avg_hamming_distance_last10']:.2f}")
        print(f"  最终与理论均衡相似度: {conv['final_similarity_to_equilibrium']:.3f}")
        
        print(f"\n【分享集合比较】")
        print(f"  最终结果: {results['final_share_set']}")
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
        final_m = results['metrics']['final']
        gt_m = results['metrics']['ground_truth']
        dev_m = results['metrics']['deviations']
        
        print(f"  平台利润:     最终={final_m['profit']:.4f}  |  GT={gt_m['profit']:.4f}  |  MAE={dev_m['profit_mae']:.4f}")
        print(f"  社会福利:     最终={final_m['welfare']:.4f}  |  GT={gt_m['welfare']:.4f}  |  MAE={dev_m['welfare_mae']:.4f}")
        print(f"  总泄露量:     最终={final_m['total_leakage']:.4f}  |  GT={gt_m['total_leakage']:.4f}  |  MAE={dev_m['total_leakage_mae']:.4f}")
        print(f"  分享率:       最终={final_m['share_rate']:.2%}  |  GT={gt_m['share_rate']:.2%}  |  MAE={dev_m['share_rate_mae']:.2%}")
        
        print(f"\n【学习轨迹摘要】")
        traj = conv['share_rate_trajectory']
        print(f"  初始分享率: {traj[0]:.2%}")
        print(f"  最终分享率: {traj[-1]:.2%}")
        if len(traj) >= 10:
            print(f"  第10轮分享率: {traj[9]:.2%}")
        if len(traj) >= 25:
            print(f"  第25轮分享率: {traj[24]:.2%}")
        
        # 打印相似度轨迹的趋势
        sim_traj = conv['similarity_trajectory']
        print(f"\n【与理论均衡相似度演化】")
        print(f"  初始相似度: {sim_traj[0]:.3f}")
        print(f"  最终相似度: {sim_traj[-1]:.3f}")
        if len(sim_traj) >= 10:
            print(f"  第10轮相似度: {sim_traj[9]:.3f}")
    
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
        
        # 如果是虚拟博弈结果，自动生成可视化
        if results.get("game_type") == "fictitious_play":
            self._visualize_fictitious_play(results, output_path)
    
    def _visualize_fictitious_play(self, results: Dict[str, Any], json_path: str):
        """
        为虚拟博弈结果生成可视化图表
        
        Args:
            results: 评估结果字典
            json_path: JSON文件路径（用于确定输出目录）
        """
        import os
        from pathlib import Path
        
        # 确定输出目录（与JSON文件同目录）
        output_dir = Path(json_path).parent
        base_name = Path(json_path).stem
        
        try:
            # 提取数据
            history = results.get("history", [])
            conv_analysis = results.get("convergence_analysis", {})
            n = self.params.n
            
            if not history:
                print("[WARN] 没有历史数据，跳过可视化")
                return
            
            # === 可视化1：分享率曲线 ===
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            share_rate_traj = conv_analysis.get("share_rate_trajectory", [])
            similarity_traj = conv_analysis.get("similarity_trajectory", [])
            
            if share_rate_traj:
                rounds = list(range(1, len(share_rate_traj) + 1))
                
                # 主轴：分享率
                ax1.plot(rounds, share_rate_traj, 'b-o', linewidth=2, markersize=4, label='分享率')
                ax1.set_xlabel('轮次', fontsize=12)
                ax1.set_ylabel('分享率', color='b', fontsize=12)
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([0, 1])
                
                # 次轴：与理论均衡的相似度
                if similarity_traj:
                    ax2 = ax1.twinx()
                    ax2.plot(rounds, similarity_traj, 'r-s', linewidth=2, markersize=4, 
                            alpha=0.7, label='与均衡相似度')
                    ax2.set_ylabel('Jaccard相似度', color='r', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.set_ylim([0, 1])
                
                # 标注收敛点
                if conv_analysis.get("converged"):
                    conv_round = conv_analysis.get("convergence_round")
                    if conv_round and conv_round < len(share_rate_traj):
                        ax1.axvline(x=conv_round + 1, color='g', linestyle='--', 
                                   alpha=0.5, label=f'收敛点(第{conv_round + 1}轮)')
                
                # 图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                if similarity_traj:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                else:
                    ax1.legend(loc='best')
                
                ax1.set_title('虚拟博弈：分享率与收敛过程', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                fig1_path = output_dir / f"{base_name}_share_rate.png"
                plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
                plt.close(fig1)
                print(f"[图表] 分享率曲线已保存到: {fig1_path}")
            
            # === 可视化2：用户策略热力图 ===
            fig2, ax = plt.subplots(figsize=(max(12, len(history) * 0.3), max(6, n * 0.5)))
            
            # 构建策略矩阵：行=用户，列=轮次
            strategy_matrix = np.zeros((n, len(history)))
            for round_idx, round_decisions in enumerate(history):
                for user_id, decision in round_decisions.items():
                    # 确保user_id是整数（从JSON读取时可能是字符串）
                    user_id_int = int(user_id) if isinstance(user_id, str) else user_id
                    strategy_matrix[user_id_int, round_idx] = decision
            
            # 绘制热力图
            sns.heatmap(strategy_matrix, 
                       cmap=['#f0f0f0', '#2E86AB'],  # 0=浅灰，1=蓝色
                       cbar_kws={'label': '策略 (0=不分享, 1=分享)', 'ticks': [0, 1]},
                       linewidths=0.5,
                       linecolor='white',
                       square=False,
                       ax=ax)
            
            # 设置坐标轴
            ax.set_xlabel('轮次', fontsize=12)
            ax.set_ylabel('用户ID', fontsize=12)
            ax.set_title('虚拟博弈：用户策略演化热力图', fontsize=14, fontweight='bold')
            
            # 设置刻度
            ax.set_xticks(np.arange(0, len(history), max(1, len(history) // 20)) + 0.5)
            ax.set_xticklabels(np.arange(1, len(history) + 1, max(1, len(history) // 20)))
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_yticklabels(range(n))
            
            # 标注理论均衡分享集合
            gt_share_set = set(results.get("gt_share_set", []))
            if gt_share_set:
                # 在右侧添加标记
                for user_id in range(n):
                    if user_id in gt_share_set:
                        ax.text(len(history) + 0.5, user_id + 0.5, '★', 
                               ha='left', va='center', fontsize=12, color='red')
                
                ax.text(len(history) + 0.5, -1, '★=理论均衡', 
                       ha='left', va='center', fontsize=10, color='red', fontweight='bold')
            
            plt.tight_layout()
            fig2_path = output_dir / f"{base_name}_strategy_heatmap.png"
            plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"📊 策略热力图已保存到: {fig2_path}")
            
        except Exception as e:
            print(f"[WARN] 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """测试评估器"""
    import argparse
    import os
    from datetime import datetime
    from pathlib import Path
    import glob
    
    try:
        from .llm_client import create_llm_client
    except ImportError:
        from src.evaluators.llm_client import create_llm_client
    
    parser = argparse.ArgumentParser(description='场景B评估器')
    parser.add_argument('--model', type=str, default='deepseek-v3.2', help='LLM模型名称')
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'fp'], 
                        help='博弈模式：static=静态博弈，fp=虚拟博弈')
    parser.add_argument('--max_rounds', type=int, default=50, help='虚拟博弈最大轮数')
    parser.add_argument('--belief_window', type=int, default=10, help='虚拟博弈信念窗口大小')
    parser.add_argument('--num_trials', type=int, default=1, help='每个决策的重复查询次数')
    parser.add_argument('--visualize', type=str, nargs='+', help='为已有JSON文件生成可视化（支持文件路径或目录）')
    
    args = parser.parse_args()
    
    # ===== 可视化模式：直接从JSON生成图表 =====
    if args.visualize:
        print(f"\n{'='*60}")
        print(f"[可视化模式] 从已有结果生成图表")
        print(f"{'='*60}")
        
        # 收集所有JSON文件
        json_files = []
        for path_pattern in args.visualize:
            path_obj = Path(path_pattern)
            
            if path_obj.is_file() and path_obj.suffix == '.json':
                # 单个JSON文件
                json_files.append(path_obj)
            elif path_obj.is_dir():
                # 目录：查找所有JSON文件
                json_files.extend(path_obj.glob('*.json'))
            elif '*' in str(path_pattern):
                # 通配符模式
                json_files.extend([Path(p) for p in glob.glob(path_pattern)])
            else:
                print(f"[WARN] 无效路径: {path_pattern}")
        
        if not json_files:
            print("[ERROR] 未找到任何JSON文件")
            return
        
        print(f"\n找到 {len(json_files)} 个JSON文件\n")
        
        # 创建临时评估器（用于访问可视化方法和params）
        llm_client = create_llm_client(args.model)
        evaluator = ScenarioBEvaluator(llm_client)
        
        # 为每个JSON文件生成可视化
        for json_path in json_files:
            try:
                print(f"处理: {json_path}")
                
                # 读取JSON文件
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 检查是否是虚拟博弈结果
                if results.get("game_type") != "fictitious_play":
                    print(f"  [SKIP] 不是虚拟博弈结果，跳过")
                    continue
                
                # 生成可视化
                evaluator._visualize_fictitious_play(results, str(json_path))
                print(f"  ✓ 可视化生成成功\n")
                
            except Exception as e:
                print(f"  [ERROR] 处理失败: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"可视化完成！")
        print(f"{'='*60}")
        return
    
    # ===== 正常运行模式 =====
    # 创建LLM客户端
    llm_client = create_llm_client(args.model)
    
    # 创建评估器
    evaluator = ScenarioBEvaluator(llm_client)
    
    # 创建输出子文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results/{args.mode}_{llm_client.config_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.mode == 'static':
        # 运行静态博弈
        print(f"[模式] 静态博弈")
        results = evaluator.simulate_static_game(num_trials=args.num_trials)
        evaluator.print_evaluation_summary(results)
        output_path = f"{output_dir}/eval_{timestamp}.json"
        evaluator.save_results(results, output_path)
    
    elif args.mode == 'fp':
        # 运行虚拟博弈
        print(f"[模式] 虚拟博弈")
        results = evaluator.simulate_fictitious_play(
            max_rounds=args.max_rounds,
            belief_window=args.belief_window,
            num_trials=args.num_trials
        )
        evaluator.print_evaluation_summary_fp(results)
        output_path = f"{output_dir}/eval_{timestamp}.json"
        evaluator.save_results(results, output_path)


if __name__ == "__main__":
    main()
