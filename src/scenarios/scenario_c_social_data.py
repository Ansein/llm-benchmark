"""
场景C: The Economics of Social Data（社会数据的经济学）- 理论求解器

================================================================================
论文信息
================================================================================
标题：The Economics of Social Data
作者：Dirk Bergemann (Yale), Alessandro Bonatti (MIT Sloan), Tan Gan (Yale)
发表：2022年9月（RAND Journal of Economics）
核心问题：社会数据的外部性与匿名化政策的经济影响

论文链接：本代码基于论文理论框架实现

================================================================================
本文件的完整流程
================================================================================

【第1步】参数设置与数据结构定义
    - ScenarioCParams: 定义所有模型参数
      * 基础参数：N(消费者数), 数据结构, 匿名化政策
      * 分布参数：mu_theta, sigma_theta, sigma
      * 补偿参数：m(消费者), m_0(生产者)
      * 异质性参数：tau_mean, tau_std, tau_dist
      * 时序参数：participation_timing (ex_ante/ex_post)
    
    - ConsumerData: 存储生成的数据
      * w: 真实支付意愿（N,）
      * s: 观测信号（N,）
      * theta, epsilon: 结构参数（取决于数据结构）
    
    - MarketOutcome: 存储市场结果
      * 参与情况、价格、需求、效用
      * 福利指标：消费者剩余、生产者利润、中介利润、社会福利
      * 不平等指标：Gini系数、价格歧视指数

【第2步】数据生成（论文Section 3）
    generate_consumer_data(params) -> ConsumerData
    
    根据数据结构生成(w, s)：
    
    A. Common Preferences（共同偏好）- 论文式3.1
       真实偏好：w_i = θ  for all i（所有人相同）
       先验：θ ~ N(μ_θ, σ_θ²)
       信号：s_i = θ + σ·e_i，其中e_i ~ N(0,1) i.i.d.
       
       特点：
       - 所有人真实偏好相同
       - 信号噪声独立
       - 多人数据可以通过平均滤掉噪声，更准确估计θ
       - 即使实名制，后验估计趋于相同→无法有效歧视
    
    B. Common Experience（共同经历）- 论文式3.2
       真实偏好：w_i ~ N(μ_θ, σ_θ²) i.i.d.（每人不同）
       共同噪声：ε ~ N(0,1)
       信号：s_i = w_i + σ·ε for all i（共同冲击）
       
       特点：
       - 真实偏好异质
       - 所有人受相同噪声冲击（如误导性广告）
       - 多人数据可以识别并过滤共同噪声ε
       - 实名制下可以有效歧视

【第3步】贝叶斯后验估计（论文Section 3.3）
    
    A. 消费者后验估计
       compute_posterior_mean_consumer(s_i, X, params) -> μ_i
       
       消费者i的信息集：I_i = {s_i} ∪ X
       - s_i: 自己的私人信号
       - X: 数据库（参与者信号集合）
       
       Common Preferences下的精确贝叶斯更新（论文式3.3）：
       后验精度 = 先验精度 + 自己信号精度 + 他人信号总精度
       τ_post = τ_0 + τ_s + r·(N-1)·τ_s
       
       后验均值 = 各信息源加权平均
       μ_i = (τ_0·μ_θ + τ_s·s_i + r·(N-1)·τ_s·mean(X)) / τ_post
       
       Common Experience下的近似（论文附录A）：
       1) 估计共同噪声：ε̂ ≈ f(mean(X) - μ_θ)
       2) 过滤噪声：ŝ_i = s_i - σ·ε̂
       3) 结合先验：μ_i = g(ŝ_i, μ_θ)
    
    B. 生产者后验估计（核心区别！）
       compute_producer_posterior(data, participation, X, params) -> μ_producer
       
       这是匿名化机制的核心实现（论文Section 4）：
       
       实名（Identified）- 论文式4.1：
       - 生产者信息集：Y_0 = {(i, s_i) : i ∈ participants}
       - 对参与者i：知道s_i，可计算E[w_i | s_i, X]
       - 对拒绝者j：不知道s_j，只能用先验μ_θ
       - 结果：可以对参与者个性化定价
       
       匿名（Anonymized）- 论文式4.2：
       - 生产者信息集：Y_0 = {s_i : i ∈ participants}（无身份）
       - 只知道信号集合，无法匹配个体
       - 对所有人：μ_producer[i] = E[θ | X]（相同）
       - 结果：必须统一定价
       
       这正是论文命题2的关键：匿名化通过阻止个性化定价保护消费者

【第4步】生产者定价（论文Section 2.2）
    
    线性-二次模型下的最优定价（论文式2.3）：
    
    A. 个性化定价（实名）
       compute_optimal_price_personalized(μ_i, c) -> p_i*
       
       生产者最大化对i的利润：
       max_{p_i} (p_i - c)·q_i = (p_i - c)·max(μ_i - p_i, 0)
       
       一阶条件：μ_i - 2p_i + c = 0
       最优价格：p_i* = (μ_i + c) / 2
       最优需求：q_i* = (μ_i - c) / 2
    
    B. 统一定价（匿名）- 论文式4.3
       compute_optimal_price_uniform(μ_list, c) -> p*
       
       生产者最大化总利润：
       max_p Σ_i (p - c)·max(μ_i - p, 0)
       
       这是非平凡的优化问题（非凸），使用数值方法：
       - scipy.optimize.minimize_scalar
       - 搜索区间：[c, max(μ)]
       - 保证全局最优

【第5步】市场结果模拟（论文Section 2）
    simulate_market_outcome(data, participation, params) -> MarketOutcome
    
    完整的市场均衡计算：
    
    1) 收集参与者信号 → 形成数据库X
    2) 匿名化处理（如果需要shuffle）
    3) 消费者计算后验μ_consumers[i] = E[w_i | s_i, X]
    4) 生产者计算后验μ_producer[i] = E[w_i | Y_0]（关键区别！）
    5) 生产者定价：
       - 实名：p_i = (μ_producer[i] + c) / 2
       - 匿名：p = uniform_price(μ_producer)
    6) 消费者购买：q_i = max(μ_consumers[i] - p_i, 0)
    7) 计算效用（论文式2.1）：
       u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
       参与者额外获得补偿：u_i += m
    8) 计算福利指标：
       - 消费者剩余：CS = Σu_i
       - 生产者利润：PS = Σ(p_i - c)·q_i
       - 中介利润：IS = m_0 - m·N_参与
       - 社会福利：SW = CS + PS + IS
    9) 计算不平等指标：
       - Gini系数
       - 价格歧视指数 = max(p) - min(p)

【第6步】参与决策均衡（论文Section 5 + 我们的扩展）
    
    核心权衡（论文式5.1）：
    ΔU_i = E[u_i | 参与, r] - E[u_i | 拒绝, r] + m - τ_i
    
    消费者参与当且仅当：ΔU_i ≥ 0
    
    A. Ex Ante参与（论文标准时序）- 我们的主实现
       compute_rational_participation_rate_ex_ante(params) -> r*
       
       时序（论文Section 5.1）：
       (1) 中介发布合约(m, 匿名化政策)
       (2) 消费者在不知道(w,s)实现时决策  ← Ex Ante
       (3) 信号实现，按政策流动
       (4) 生产者定价，消费者购买
       
       期望效用计算（两层Monte Carlo）：
       外层：遍历可能的世界状态(w, s)
       内层：遍历可能的参与者集合
       E[u_i | a_i, r] = E_{w,s} E_{a_{-i}|r} [u_i(w, s, a, 信息流)]
       
       固定点方程（有异质性）：
       r* = P(τ_i ≤ ΔU(r*)) = F_τ(ΔU(r*))
       
       其中F_τ是隐私成本的累积分布函数：
       - tau_dist="normal": Φ((ΔU - μ_τ) / σ_τ)
       - tau_dist="uniform": (ΔU - a) / (b - a)
       - tau_dist="none": 1 if ΔU>0 else 0
    
    B. Ex Post参与（扩展/鲁棒性）
       compute_rational_participation_rate_ex_post(data, params) -> r*
       
       时序：
       (1) (w, s)实现
       (2) 消费者观察到s_i后决策  ← Ex Post
       (3) 数据流动，定价，购买
       
       基于特定realized (w,s)计算参与率
       注意：与论文时序不一致，仅用于对比分析

【第7步】Ground Truth生成
    generate_ground_truth(params) -> result_dict
    
    完整流程：
    1) 根据参与时序选择算法
       - Ex Ante: 先求均衡r*，再生成示例数据
       - Ex Post: 先生成数据，再求均衡r*
    
    2) 使用均衡参与率r*生成参与决策
       采样：participation[i] ~ Bernoulli(r*)
    
    3) 模拟市场结果
       调用simulate_market_outcome()
    
    4) 输出JSON格式结果：
       - params: 所有参数
       - data: w, s, theta, epsilon
       - rational_participation_rate: r*
       - outcome: 所有市场结果和福利指标
       - detailed_results: 价格、需求、效用细节

================================================================================
核心经济学原理（对应论文命题）
================================================================================

命题1（社会数据外部性）- 论文Proposition 1：
"个人i的数据s_i通过条件相关性预测他人j的行为"
→ 我们的实现：后验估计中使用participant_signals更新所有消费者信念

命题2（匿名化保护）- 论文Proposition 2：
"匿名化阻止个性化定价，当σ_θ²足够大时提高消费者剩余"
→ 我们的实现：
  - 实名：μ_producer[i]各不相同 → p_i个性化
  - 匿名：μ_producer[i]全相同 → p统一

命题3（搭便车）- 论文Proposition 3：
"信息完全披露下(Y_i = X)，拒绝者免费获得参与者数据"
→ 我们的实现：拒绝者在计算后验时也使用participant_signals

定理1（参与激励）- 论文Theorem 1：
"补偿m越高，参与率越高；匿名化可能提高参与率"
→ 我们的实现：固定点方程r* = F_τ(ΔU(r*, m, 匿名化))

================================================================================
我们的实现相对论文的简化与扩展
================================================================================

简化（为了Benchmark可行性）：
1. 固定信息披露政策：Y_0 = Y_i = X（论文研究最优披露，我们固定）
2. 均匀补偿m（论文允许m_i，我们简化）
3. 边际成本c=0（论文常见假设）
4. 消费者对称（除隐私成本τ_i）

扩展（为了LLM评估）：
1. Ex Ante vs Ex Post对比（论文只有Ex Ante）
2. 异质性参数化：τ_i ~ F_τ（论文讨论但未参数化）
3. 中介利润计算：m_0支持（论文隐含）
4. 完整福利分解（便于分析）

关键：我们的主结果（Ex Ante + 异质性）严格对齐论文时序和理论

================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, Callable, Union
from dataclasses import dataclass, asdict
import json
from scipy.optimize import minimize_scalar


@dataclass
class ScenarioCParams:
    """
    场景C参数配置类
    
    对应论文Section 2-4的模型参数设定
    
    ========================================================================
    基础参数
    ========================================================================
    """
    N: int  # 消费者数量（论文中的i = 1, ..., N）
    
    # 数据结构类型（论文Section 3）
    # - "common_preferences": 所有人真实偏好相同w_i=θ，噪声独立e_i ~ i.i.d.
    #   对应论文式3.1，适合建模"产品质量评估"场景
    # - "common_experience": 偏好异质w_i ~ 分布，共同噪声ε
    #   对应论文式3.2，适合建模"市场噪声影响"场景
    data_structure: Literal["common_preferences", "common_experience"]
    
    # 匿名化政策（论文Section 4的核心）
    # - "identified": 实名制，生产者知道(i, s_i)映射，可个性化定价
    #   对应论文式4.1，Y_0包含身份信息
    # - "anonymized": 匿名化，生产者只知道信号集合{s_i}，必须统一定价
    #   对应论文式4.2，Y_0不包含身份映射
    # 这是论文Proposition 2的关键：匿名化阻止价格歧视
    anonymization: Literal["identified", "anonymized"]
    
    """
    ========================================================================
    数据生成参数（对应论文Section 3的信息结构）
    ========================================================================
    """
    # 先验均值μ_θ（论文式3.1, 3.2）
    # - Common Preferences: θ的先验均值，E[θ] = μ_θ
    # - Common Experience: w_i的先验均值，E[w_i] = μ_θ
    # 典型值：5.0（表示平均支付意愿）
    mu_theta: float
    
    # 先验标准差σ_θ（论文式3.1, 3.2）
    # - Common Preferences: θ的不确定性，Var(θ) = σ_θ²
    # - Common Experience: w_i的异质性，Var(w_i) = σ_θ²
    # 典型值：1.0
    # 影响：σ_θ越大，数据学习价值越高（论文Proposition 1）
    sigma_theta: float
    
    # 噪声水平σ（论文式3.1, 3.2）
    # - Common Preferences: 信号噪声，s_i = θ + σ·e_i
    # - Common Experience: 共同噪声强度，s_i = w_i + σ·ε
    # 典型值：1.0
    # 影响：σ越大，单个信号越不准确，多人数据价值越高
    # 信噪比：SNR = σ_θ / σ
    sigma: float
    
    """
    ========================================================================
    支付参数（对应论文Section 5的参与激励）
    ========================================================================
    """
    # 中介向消费者支付的补偿m（论文式(4), (11)）
    # 
    # ⚠️ 重要修正：论文标准模型使用个性化补偿m_i（向量）
    # - 论文式(4): R = m0 − Σ^N_{i=1} m_i
    # - 论文式(11): m*_i = Ui((Si, X−i), X−i) − Ui((Si, X), X)
    # 
    # 类型支持：
    # - float: 统一补偿（简化版本，向后兼容）
    # - np.ndarray[N]: 个性化补偿（论文标准，推荐）
    # 
    # 消费者参与激励：ΔU = E[u|参与] - E[u|拒绝] + m_i - τ_i
    # 典型值：0.5-2.0（统一）或向量（个性化）
    # 影响：m越高，参与率越高（论文Theorem 1）
    m: Union[float, np.ndarray]
    
    # 注意：m_0（生产者向中介支付）不是参数，而是内生变量
    # m_0 = β × max(0, E[π_with_data - π_no_data])
    # 由 estimate_m0_mc() 函数计算
    # 在 IntermediaryOptimizationResult 和 GT 输出中返回
    
    """
    ========================================================================
    成本参数（论文Section 2.2）
    ========================================================================
    """
    # 边际成本c（论文式2.3）
    # 生产者生产单位产品的成本
    # 论文常见假设：c = 0（简化分析）
    # 定价公式：p_i* = (μ_i + c) / 2
    c: float = 0.0
    
    """
    ========================================================================
    异质性参数（我们的扩展，产生内点参与率）
    ========================================================================
    隐私成本/参与成本τ_i的分布参数
    
    论文背景：论文讨论了消费者异质性但未明确参数化
    学术标准：引入τ_i ~ F_τ是隐私经济学文献的常见做法
    （参考：Acquisti et al. 2016）
    
    作用：产生内点参与率r* ∈ (0,1)，便于LLM偏差分析
    无异质性时：r*∈{0,1}（角点解），难以衡量偏差
    """
    # 隐私成本均值μ_τ
    # 典型值：0.5（与补偿m同量纲）
    # 解释：平均而言，消费者需要多少补偿才愿意承担隐私风险
    tau_mean: float = 0.5
    
    # 隐私成本标准差σ_τ
    # 典型值：0.3
    # 解释：消费者隐私偏好的异质性程度
    tau_std: float = 0.3
    
    # 隐私成本分布类型
    # - "normal": τ_i ~ N(μ_τ, σ_τ²)，标准假设
    #   参与率：r* = Φ((ΔU - μ_τ) / σ_τ)，Φ是标准正态CDF
    # - "uniform": τ_i ~ U[μ_τ - √3·σ_τ, μ_τ + √3·σ_τ]
    #   参与率：r* = (ΔU - a) / (b - a)
    # - "none": 无异质性，所有人τ_i = 0
    #   参与率：r* = 1 if ΔU>0 else 0（角点解）
    tau_dist: Literal["normal", "uniform", "none"] = "none"
    
    """
    ========================================================================
    时序模式（关键！影响学术可信度）
    ========================================================================
    参与决策的时序设定
    
    这是GPT指出的"识别问题"（identification problem）核心！
    """
    # 参与决策时序
    # - "ex_ante": 论文标准时序（推荐，学术正确）
    #   消费者在不知道(w, s)实现时决策
    #   期望对所有随机性取平均（两层Monte Carlo）
    #   对应论文Section 5.1的合约时序
    #   
    # - "ex_post": 扩展/鲁棒性（与论文时序不一致）
    #   消费者看到realized (w, s)后决策
    #   仅用于对比分析，不应作为主结果
    #   
    # 学术影响：主结果必须用ex_ante，否则审稿人会质疑"求解了另一个模型"
    participation_timing: Literal["ex_ante", "ex_post"] = "ex_ante"
    
    """
    ========================================================================
    算法参数
    ========================================================================
    """
    # 后验估计方法（仅Common Experience用）
    # - "approx": 近似方法（快速，我们的默认）
    #   使用简化的共同噪声估计ε̂
    # - "exact": 精确贝叶斯更新（待实现）
    #   完整的高斯共轭先验更新
    posterior_method: Literal["exact", "approx"] = "approx"
    
    # 随机种子（可重现性）
    # 固定种子保证：相同参数 → 相同结果
    seed: int = 42
    
    def __post_init__(self):
        """
        后处理：自动转换m为向量格式
        
        向后兼容性：
        - 如果m是标量 → 自动扩展为N维向量（所有人相同补偿）
        - 如果m是向量 → 验证长度并转换为np.ndarray
        """
        if isinstance(self.m, (int, float)):
            # 统一补偿：自动扩展为向量
            self.m = np.full(self.N, float(self.m))
        else:
            # 个性化补偿：转换并验证
            self.m = np.array(self.m, dtype=float)
            if len(self.m) != self.N:
                raise ValueError(
                    f"个性化补偿m的长度({len(self.m)})必须等于消费者数N({self.N})"
                )
    
    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化），确保所有numpy数组转为JSON兼容格式"""
        # ✅ 确保m字段JSON可序列化
        if isinstance(self.m, np.ndarray):
            if np.all(self.m == self.m[0]):
                m_value = float(self.m[0])  # 统一补偿 -> 标量
            else:
                m_value = self.m.tolist()  # 个性化补偿 -> 列表
        else:
            m_value = float(self.m)
        
        return {
            'N': int(self.N),
            'data_structure': self.data_structure,
            'mu_theta': float(self.mu_theta),
            'sigma_theta': float(self.sigma_theta),
            'sigma': float(self.sigma),
            'anonymization': self.anonymization,
            'm': m_value,  # ✅ JSON可序列化
            'tau_mean': float(self.tau_mean),
            'tau_std': float(self.tau_std),
            'tau_dist': self.tau_dist,
            'participation_timing': self.participation_timing,
            'seed': int(self.seed) if self.seed is not None else None
        }


@dataclass
class ConsumerData:
    """消费者的数据（用于模拟）"""
    w: np.ndarray  # 真实支付意愿 (N,)
    s: np.ndarray  # 信号 (N,)
    e: np.ndarray  # 噪声成分 (N,)
    theta: Optional[float] = None  # 共同偏好（仅Common Preferences）
    epsilon: Optional[float] = None  # 共同噪声（仅Common Experience）


@dataclass
class MarketOutcome:
    """市场结果"""
    # 参与情况
    participation: np.ndarray  # (N,) bool数组，True表示参与
    participation_rate: float # 参与率
    num_participants: int # 参与者数量
    
    # 价格与数量
    prices: np.ndarray  # (N,) 每个消费者面临的价格
    quantities: np.ndarray  # (N,) 每个消费者的购买量
    
    # 后验估计
    mu_consumers: np.ndarray  # (N,) 消费者的后验期望
    mu_producer: np.ndarray  # (N,) 生产者对每个消费者的后验期望
    
    # 福利指标
    utilities: np.ndarray  # (N,) 每个消费者的效用（含补偿）
    consumer_surplus: float  # 消费者总剩余
    producer_profit: float  # 生产者利润（不含数据支付）
    intermediary_profit: float  # 中介利润 = m_0 - m*num_participants
    social_welfare: float  # 社会福利（含中介利润）
    
    # 学习质量
    learning_quality_participants: float  # 参与者的学习质量 mean|mu_i - w_i|
    learning_quality_rejecters: float  # 拒绝者的学习质量
    
    # 不平等指标
    gini_coefficient: float # 基尼系数
    acceptor_avg_utility: float # 接受者的平均效用
    rejecter_avg_utility: float # 拒绝者的平均效用
    
    # 价格歧视指标
    price_variance: float # 价格方差
    price_discrimination_index: float  # max(p) - min(p)


def generate_consumer_data(
    params: ScenarioCParams,
    rng: Optional[np.random.Generator] = None
) -> ConsumerData:
    """
    生成消费者数据（真实偏好和信号）
    
    Args:
        params: 场景参数
        rng: 随机数生成器（如果为None，使用params.seed创建）
        
    Returns:
        ConsumerData对象
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)
    
    N = params.N # 消费者数量
    
    if params.data_structure == "common_preferences": # 共同偏好假设
        # 所有消费者有相同的真实偏好θ
        theta = rng.normal(params.mu_theta, params.sigma_theta) # 真实偏好
        w = np.ones(N) * theta # 真实支付意愿
        
        # 独立噪声 ~ N(0, 1)
        e = rng.normal(0, 1, N)
        
        # 信号 = 真实支付意愿 + 噪声 * 噪声水平
        s = w + params.sigma * e
        
        return ConsumerData(w=w, s=s, e=e, theta=theta, epsilon=None)
    
    elif params.data_structure == "common_experience": # 共同经验假设
        # 每个消费者的真实偏好不同
        w = rng.normal(params.mu_theta, params.sigma_theta, N) # 真实偏好 ~ N(μ_theta, σ_theta²)
        
        # 共同噪声冲击
        epsilon = rng.normal(0, 1)
        e = np.ones(N) * epsilon
        
        # 信号
        s = w + params.sigma * e
        
        return ConsumerData(w=w, s=s, e=e, theta=None, epsilon=epsilon)
    
    else:
        raise ValueError(f"Unknown data structure: {params.data_structure}")


def _compute_ce_posterior_approx(
    s_i: float,
    participant_signals: np.ndarray,
    params: ScenarioCParams
) -> float:
    """
    Common Experience数据结构的后验估计（近似方法）
    
    使用简化的贝叶斯更新公式：
    1. 估计共同噪声 ε
    2. 过滤信号: s_i - σ·ε_hat
    3. 结合先验得到后验
    
    注意：这是近似方法，误差来源于对ε的点估计。
    完整的贝叶斯更新需要考虑ε的后验分布。
    
    Args:
        s_i: 消费者i的信号
        participant_signals: 参与者信号
        params: 参数
        
    Returns:
        后验期望 E[w_i | s_i, signals]
    """
    n = len(participant_signals) # 参与者数量
    signal_mean = np.mean(participant_signals) # 参与者信号的平均值
    
    # 估计共同噪声的后验期望
    # E[ε | signals] ≈ (1/(1 + n·σ²/σ_θ²)) · (mean(signals) - μ_θ) / σ
    epsilon_posterior_variance = 1 / (1 + n * params.sigma ** 2 / params.sigma_theta ** 2) # 共同噪声的后验方差
    epsilon_hat = epsilon_posterior_variance * (signal_mean - params.mu_theta) / params.sigma # 共同噪声的后验期望
    
    # 过滤噪声后的信号
    filtered_signal = s_i - params.sigma * epsilon_hat
    
    # 结合先验和过滤后的信号
    prior_precision = 1 / (params.sigma_theta ** 2) # 先验精度 = 1 / 先验方差
    # 过滤后信号的精度（近似）
    filtered_precision = 1 / (params.sigma ** 2 * epsilon_posterior_variance) # 过滤后信号的精度 = 1 / (噪声方差 * 共同噪声的后验方差)
    
    posterior_precision = prior_precision + filtered_precision # 后验精度 = 先验精度 + 过滤后信号的精度
    posterior_mean = (prior_precision * params.mu_theta + filtered_precision * filtered_signal) / posterior_precision # 后验期望 = (先验期望 + 过滤后信号的精度 * 过滤后信号) / 后验精度
    
    return posterior_mean # 返回后验期望


def compute_posterior_mean_consumer(
    s_i: float,
    participant_signals: np.ndarray,
    params: ScenarioCParams
) -> float:
    """
    计算消费者i的后验期望 E[w_i | s_i, participant_signals]
    
    ⚠️ 关键修正（P0-1）：消费者后验必须包含s_i（论文信息集 I_i={s_i}∪X）
    
    论文机制：消费者永远观察到自己的私人信号s_i，这是其信息优势的来源。
    
    处理participant_signals中的s_i：
    - 如果s_i在participant_signals中（消费者是参与者），需避免double count
    - 采用"从X中分离出s_i"的方式：X_others = X without {s_i}
    
    Args:
        s_i: 消费者i自己的信号（必须纳入后验！）
        participant_signals: 参与者的信号集合X（可能包括s_i）
        params: 场景参数
        
    Returns:
        后验期望 mu_i = E[w_i | s_i, X]
    """
    if params.data_structure == "common_preferences":
        # w_i = θ for all i （所有人共享同一个真实偏好θ）
        # 消费者信息集：I_i = {s_i} ∪ X
        # 后验：E[θ | s_i, X] = E[θ | s_i, X_{-i}]（X_{-i}是其他参与者信号）
        
        # 构造"其他参与者信号"（避免double count）
        # 方法：检查s_i是否在participant_signals中
        # 如果在，说明i是参与者，需要从X中移除s_i得到X_{-i}
        # 如果不在，说明i是拒绝者，X就是X_{-i}
        
        # 找到与s_i匹配的信号（容忍浮点误差）
        is_in_X = np.any(np.abs(participant_signals - s_i) < 1e-9)
        if is_in_X and len(participant_signals) > 1:
            # i是参与者，从X中移除s_i
            other_signals = participant_signals[np.abs(participant_signals - s_i) >= 1e-9]
        elif is_in_X and len(participant_signals) == 1:
            # i是唯一参与者，X_{-i}为空
            other_signals = np.array([])
        else:
            # i是拒绝者，X_{-i} = X
            other_signals = participant_signals
        
        # 精度（precision = 1/variance）
        tau_0 = 1 / (params.sigma_theta ** 2)  # 先验精度
        tau_s = 1 / (params.sigma ** 2)        # 信号精度
        
        # 后验精度 = 先验精度 + s_i精度 + 其他信号精度
        # 重要：s_i必须纳入！（无论i是否参与）
        n_others = len(other_signals)
        posterior_precision = tau_0 + tau_s + n_others * tau_s
        
        # 后验均值（共轭正态）
        posterior_mean = (
            tau_0 * params.mu_theta +      # 先验贡献
            tau_s * s_i +                   # s_i贡献（关键！）
            tau_s * np.sum(other_signals)   # 其他参与者贡献
        ) / posterior_precision
        
        return posterior_mean
    
    elif params.data_structure == "common_experience":
        # s_i = w_i + σ·ε （个体偏好 + 共同噪声）
        # 消费者信息集：I_i = {s_i} ∪ X
        # 需要从X估计共同噪声ε，然后用s_i推断w_i
        
        # 构造"其他参与者信号"（避免在估计ε时double count s_i）
        is_in_X = np.any(np.abs(participant_signals - s_i) < 1e-9)
        if is_in_X and len(participant_signals) > 1:
            other_signals = participant_signals[np.abs(participant_signals - s_i) >= 1e-9]
        elif is_in_X and len(participant_signals) == 1:
            other_signals = np.array([])
        else:
            other_signals = participant_signals
        
        # 如果没有其他参与者信号，只用s_i和先验
        if len(other_signals) == 0:
            # 只有s_i，无法估计ε，用先验收缩
            tau_0 = 1 / (params.sigma_theta ** 2)
            tau_s = 1 / (params.sigma ** 2)
            # s_i = w_i + σε, 但不知道ε，假设E[ε]=0
            # 后验：E[w_i | s_i] = (tau_0·μ_θ + tau_s·s_i) / (tau_0 + tau_s)
            return (tau_0 * params.mu_theta + tau_s * s_i) / (tau_0 + tau_s)
        
        # 根据参数选择后验估计方法
        if params.posterior_method == "exact":
            # 精确贝叶斯估计（更复杂但更准确）
            # TODO: 实现完整的高斯共轭先验更新
            # 目前回退到近似方法
            return _compute_ce_posterior_approx(s_i, other_signals, params)
        else:
            # 近似方法（计算效率高，用X_{-i}估计ε）
            return _compute_ce_posterior_approx(s_i, other_signals, params)
    
    else:
        raise ValueError(f"Unknown data structure: {params.data_structure}")


def compute_optimal_price_personalized(mu_i: float, c: float = 0.0) -> float:
    """
    计算个性化定价下的最优价格（闭式解）
    
    ========================================================================
    理论基础（论文Section 2.2）
    ========================================================================
    线性-二次模型下的单个消费者最优定价：
    
    需求函数：q_i(p) = max(μ_i - p, 0)
    其中μ_i是生产者对w_i的后验期望
    
    利润函数：π_i(p) = (p - c) · max(μ_i - p, 0)
    
    当μ_i > c时，内点解（q_i > 0）：
    一阶条件：dπ/dp = (μ_i - p) - (p - c) = 0
    解得：p_i* = (μ_i + c) / 2
    
    此时：
    - 最优需求：q_i* = (μ_i - c) / 2
    - 最优利润：π_i* = (μ_i - c)² / 4
    
    当μ_i ≤ c时，角点解：p_i* = c, q_i* = 0
    
    ========================================================================
    实名制（Identified）下的应用
    ========================================================================
    当anonymization="identified"时：
    - 生产者知道(i, s_i)映射
    - 可以对每个消费者i单独计算μ_i^{prod} = E[w_i | s_i, X]
    - 对每个i设置个性化价格p_i = (μ_i^{prod} + c) / 2
    - 这是论文Proposition 2的核心：实名制允许价格歧视
    
    ========================================================================
    Args:
        mu_i: 生产者对消费者i支付意愿的后验期望（μ_i^{prod}）
        c: 边际成本（论文常设c=0）
        
    Returns:
        最优价格 p_i* = (μ_i + c) / 2
    ========================================================================
    """
    return (mu_i + c) / 2


def compute_optimal_price_uniform(mu_list: List[float], c: float = 0.0) -> Tuple[float, float]:
    """
    计算统一定价下的最优价格（方法A：数值优化 - 推荐）
    
    ========================================================================
    理论基础（论文Section 4 + GPT求解器建议）
    ========================================================================
    匿名化下的统一定价问题：
    
    生产者必须对所有消费者设置同一价格p（无法个性化）
    
    需求函数：q_i(p) = max(μ_i - p, 0)
    其中μ_i是生产者对每个消费者的后验期望
    
    总利润函数：
    Π(p) = Σ_i (p - c) · max(μ_i - p, 0)
    
    目标：p* = argmax_{p ≥ c} Π(p)
    
    ========================================================================
    为什么这是"非平凡"的优化问题？
    ========================================================================
    关键难点：利润函数Π(p)是分段线性的（非凸、非平滑）
    
    误区（常见错误）：
    ❌ 错误想法："最优价格是某个μ_i/2"
    ❌ 错误做法：candidates = {c} ∪ {μ_i/2}，然后枚举
    
    为什么这是错的？
    - 个性化定价：对单个消费者i，p_i* = (μ_i + c) / 2 ✓
    - 统一定价：对异质消费者{μ_i}，p* ≠ 任何μ_i/2 ✗
    
    反例（GPT指出）：
    μ = [10, 6, 2], c = 0
    - 错误方法会试p ∈ {0, 5, 3, 1}
    - 真实最优可能是p* ≈ 4.5（不在候选集中）
    
    正确做法（两种）：
    A. 数值优化（本函数）- 最简单、最稳健
    B. 正确的分段枚举（见compute_optimal_price_uniform_piecewise）
    
    ========================================================================
    本函数实现：方法A - 数值优化
    ========================================================================
    使用scipy.optimize.minimize_scalar进行一维搜索
    
    优点：
    - 完全正确（保证全局最优）
    - 实现简单
    - 对任意μ分布都适用
    
    关键参数：
    - 搜索区间：[c, max(μ)]
      * 下界c：p < c无意义（亏本）
      * 上界max(μ)：p > max(μ)时所有q_i = 0，利润为0
    - method='bounded'：确保在区间内搜索
    
    ========================================================================
    匿名化（Anonymized）下的应用
    ========================================================================
    当anonymization="anonymized"时：
    - 生产者只知道信号集合{s_i}（无身份）
    - 所有消费者μ_i^{prod}相同（基于聚合统计）
    - 即使μ_i相同，也必须用统一价p（无法识别个体）
    - 这是论文Proposition 2的保护机制：匿名化阻止价格歧视
    
    ========================================================================
    Args:
        mu_list: 所有N个消费者的后验期望列表[μ_1, ..., μ_N]
                 注意：这是生产者的后验μ^{prod}，不是消费者的μ^{cons}
        c: 边际成本（默认0）
        
    Returns:
        (optimal_price, max_profit)
        - optimal_price: 最优统一价格p*
        - max_profit: 对应的最大利润Π(p*)
    
    ========================================================================
    实现细节
    ========================================================================
    """
    # 边界情况：无消费者
    if len(mu_list) == 0:
        return c, 0.0
    
    # 确定搜索上界
    max_mu = max(mu_list)
    
    # 定义利润函数
    # 注意：minimize_scalar是求最小值，所以要取负号
    def profit(p):
        # 对每个消费者计算需求量
        quantities = [max(mu_i - p, 0) for mu_i in mu_list]
        # 总利润 = Σ(p - c)·q_i
        total_profit = sum((p - c) * q for q in quantities)
        return -total_profit  # 负号：最大化利润 = 最小化负利润
    
    # 一维数值优化
    result = minimize_scalar(
        profit,
        bounds=(c, max_mu),
        method='bounded'  # 有界优化
    )
    
    optimal_price = result.x  # 最优价格
    max_profit = -result.fun  # 最大利润（还原符号）
    
    return optimal_price, max_profit


def compute_optimal_price_uniform_piecewise(mu_list: List[float], c: float = 0.0) -> float:
    """
    计算统一定价下的最优价格（方法B：正确的分段枚举 - 高效且精确）
    
    ========================================================================
    理论基础（GPT求解器建议 + 数理经济学标准做法）
    ========================================================================
    这是方法A（数值优化）的解析等价物。
    
    核心思想：利用利润函数的分段结构
    
    关键洞察：
    给定μ按降序排列：μ_(1) ≥ μ_(2) ≥ ... ≥ μ_(N)
    
    利润函数Π(p)在区间[μ_(k+1), μ_(k)]内是二次函数：
    - 在该区间内，恰好有k个消费者购买（q_i > 0）
    - 其余N-k个消费者不购买（q_i = 0）
    
    在第k段内的利润：
    Π_k(p) = (p - c) · Σ_{j=1}^k (μ_(j) - p)
           = (p - c) · [k · μ̄_{1:k} - k · p]
    
    其中μ̄_{1:k} = (1/k) Σ_{j=1}^k μ_(j)（前k个的平均值）
    
    ========================================================================
    正确的候选价格集合（与μ_i/2的区别！）
    ========================================================================
    ❌ 错误候选：{μ_i/2} - 这是个性化定价的最优解，不适用于统一定价
    
    ✅ 正确候选：对每个k = 1, ..., N
    
    1) 计算第k段的内点候选：
       p_k = (μ̄_{1:k} + c) / 2
       
       注意：这里用的是前k个的平均μ̄_{1:k}，不是单个μ_k！
    
    2) 检查p_k是否在有效区间[max(c, μ_(k+1)), μ_(k)]内：
       - 若在：p_k是该段候选
       - 若不在：边界（μ_(k)或μ_(k+1)）是该段候选
    
    3) 遍历所有k，计算每个候选的利润，取最大
    
    ========================================================================
    为什么这比方法A高效？
    ========================================================================
    - 方法A（数值）：O(M·N)，M是优化迭代次数（~20-50次）
    - 方法B（分段）：O(N log N + N) = O(N log N)（排序 + 线性扫描）
    
    对大规模N（如N>1000），方法B显著更快
    对小规模N（如N<100），两者差异不大，方法A更简单
    
    ========================================================================
    实现逻辑（详细步骤）
    ========================================================================
    """
    # 边界情况
    if len(mu_list) == 0:
        return c
    
    # 步骤1：降序排序μ_i
    mu_sorted = sorted(mu_list, reverse=True)  # μ_(1) ≥ μ_(2) ≥ ... ≥ μ_(N)
    N = len(mu_sorted)
    
    # 辅助函数：计算给定价格p的总利润
    def compute_profit(p):
        total = 0.0
        for mu_i in mu_sorted:
            if mu_i > p:  # 购买
                total += (p - c) * (mu_i - p)
            else:  # 不购买（后面的更不会买，可提前终止）
                break
        return total
    
    best_price = c
    best_profit = 0.0
    
    # 步骤2：遍历每个可能的购买人数k = 1, 2, ..., N
    prefix_sum = 0.0  # 累积和：Σ_{j=1}^k μ_(j)
    
    for k in range(1, N + 1):
        # 更新前缀和
        prefix_sum += mu_sorted[k - 1]  # 加上μ_(k)
        
        # 计算前k个的平均值
        mu_bar_k = prefix_sum / k
        
        # 第k段的内点候选价格
        p_k = (mu_bar_k + c) / 2
        
        # 确定第k段的有效区间
        upper_bound = mu_sorted[k - 1]  # μ_(k)
        if k == N:
            lower_bound = c  # 最后一段，下界是成本
        else:
            lower_bound = max(c, mu_sorted[k])  # μ_(k+1)，但不能低于成本
        
        # 将候选价格限制在有效区间内
        p_candidate = min(max(p_k, lower_bound), upper_bound)
        
        # 计算该候选的利润
        profit_candidate = compute_profit(p_candidate)
        
        # 更新最优解
        if profit_candidate > best_profit:
            best_profit = profit_candidate
            best_price = p_candidate
    
    return best_price


# ============================================================================
# 已废弃函数（保留用于对比说明）
# ============================================================================
def compute_optimal_price_uniform_efficient_DEPRECATED(mu_list: List[float], c: float = 0.0) -> float:
    """
    ❌ 已废弃：这是错误的实现（基于μ_i/2候选集）
    
    ========================================================================
    为什么这个函数是错误的？（保留作为教学示例）
    ========================================================================
    错误根源：混淆了个性化定价和统一定价
    
    个性化定价（每个i单独）：
    - p_i* = (μ_i + c) / 2 ✓ 正确
    - 这是单消费者优化问题的闭式解
    
    统一定价（所有i共同一个p）：
    - p* = (某个μ_i + c) / 2 ✗ 错误
    - 这是N消费者耦合优化问题，无简单闭式解
    
    反例：
    μ = [10, 8, 2], c = 0
    - 候选集: {0, 5, 4, 1}
    - p=5: 需求=[5,3,0], 利润=5·5+5·3=40
    - p=4: 需求=[6,4,0], 利润=4·6+4·4=40
    - 真实最优p*≈6: 需求=[4,2,0], 利润=6·4+6·2=36？不对...
    
    实际上这个例子比较复杂，但GPT已经指出：
    对异质消费者，最优统一价一般不在{μ_i/2}集合中
    
    正确做法：
    - 方法A：数值优化（compute_optimal_price_uniform）
    - 方法B：分段枚举（compute_optimal_price_uniform_piecewise）
    
    ========================================================================
    请勿使用本函数！仅保留用于对比和教学目的
    ========================================================================
    """
    raise DeprecationWarning(
        "此函数基于错误的μ_i/2候选集，请使用："
        "\n- compute_optimal_price_uniform (数值优化，推荐)"
        "\n- compute_optimal_price_uniform_piecewise (分段枚举，高效)"
    )


def compute_producer_posterior(
    data: ConsumerData,
    participation: np.ndarray,
    participant_signals: np.ndarray,
    params: ScenarioCParams
) -> np.ndarray:
    """
    计算生产者对每个消费者wᵢ的后验期望
    
    关键区别（这是匿名化机制的核心）:
    - 实名(identified): 生产者知道参与者的(i, sᵢ)映射，可以对参与者个性化定价
    - 匿名(anonymized): 生产者只知道信号集合（无身份），必须统一定价
    
    Args:
        data: 消费者数据
        participation: 参与决策
        participant_signals: 参与者的信号（可能已打乱）
        params: 场景参数
        
    Returns:
        (N,) 生产者对每个消费者的后验期望
    """
    N = params.N # 消费者数量
    mu_producer = np.full(N, params.mu_theta)  # 默认使用先验
    
    if params.anonymization == "identified":
        # 实名: 生产者可以看到 (i, sᵢ) 映射
        # 
        # ⚠️ 关键修正（P0-2）：拒绝者后验不应固定为先验！
        # 
        # 论文核心：社会数据外部性（Social Data Externality）
        # - 即使消费者i拒绝参与，生产者仍可利用其他参与者的信号X改善对i的预测
        # - 这是"搭便车"（free-riding）问题的根源：你不贡献数据，但别人的数据仍能帮助预测你
        # 
        # 正确处理：
        # - 参与者：用s_i + X计算个体后验（最精准）
        # - 拒绝者：用X计算改善的后验（无s_i，但比先验好）
        
        if len(participant_signals) == 0:
            # 无参与者：所有人使用先验（已初始化）
            pass
        else:
            if params.data_structure == "common_preferences":
                # Common Preferences: w_i = θ for all i
                # 拒绝者虽无s_i，但生产者可用X更新对θ的估计
                # E[θ | X] 适用于所有拒绝者
                tau_0 = 1 / (params.sigma_theta ** 2)
                tau_s = 1 / (params.sigma ** 2)
                n_participants = len(participant_signals)
                
                # 拒绝者的共同后验（只用X，无个体s_i）
                posterior_mean_rejecters = (
                    tau_0 * params.mu_theta +
                    tau_s * np.sum(participant_signals)
                ) / (tau_0 + n_participants * tau_s)
                
                # 分别处理参与者和拒绝者
                for i in range(N):
                    if participation[i]:
                        # 参与者: 用s_i + X计算个体后验
                        mu_producer[i] = compute_posterior_mean_consumer(
                            data.s[i], participant_signals, params
                        )
                    else:
                        # 拒绝者: 用X更新对θ的估计
                        mu_producer[i] = posterior_mean_rejecters
            
            else:  # common_experience
                # Common Experience: s_i = w_i + σ·ε
                # 生产者可用X估计共同冲击ε，改善对所有人的预测
                
                # 估计共同冲击（使用所有参与者信号）
                signal_mean = np.mean(participant_signals)
                n_participants = len(participant_signals)
                # 简化估计：epsilon_hat ≈ (signal_mean - mu_theta) / sigma
                # 更准确的版本需要考虑后验收缩
                epsilon_posterior_var = 1 / (1 + n_participants * params.sigma**2 / params.sigma_theta**2)
                epsilon_hat = epsilon_posterior_var * (signal_mean - params.mu_theta) / params.sigma
                
                # 对所有人的代表性预测（考虑共同冲击）
                # E[w | X] ≈ μ_θ + σ·E[ε|X]（忽略个体特异性）
                common_prediction = params.mu_theta + params.sigma * epsilon_hat
                
                for i in range(N):
                    if participation[i]:
                        # 参与者: 用s_i + X计算精准后验
                        mu_producer[i] = compute_posterior_mean_consumer(
                            data.s[i], participant_signals, params
                        )
                    else:
                        # 拒绝者: 用共同预测（基于ε估计）
                        # 比先验好，但不如参与者的个性化预测
                        mu_producer[i] = common_prediction
    
    else:  # 匿名化
        # 匿名: 生产者只能看到信号集合 {sᵢ : i ∈ participants}（无身份）
        # 无法识别哪个信号对应哪个消费者，因此无法个性化定价
        # 但可以用聚合统计改善预测！（对所有人统一后验）
        
        if len(participant_signals) > 0:
            if params.data_structure == "common_preferences":
                # Common Preferences: 可以用信号均值更新对θ的估计
                # E[θ | X] 对所有人相同
                mean_signal = np.mean(participant_signals)
                n_participants = len(participant_signals)
                
                # 后验期望（共轭正态更新）
                tau_X = n_participants / params.sigma**2
                tau_0 = 1 / params.sigma_theta**2
                mu_common = (tau_0 * params.mu_theta + tau_X * mean_signal) / (tau_0 + tau_X)
                mu_producer[:] = mu_common
            
            else:  # common_experience
                # ⚠️ 关键修正（P0-3）：匿名化下仍可学习！
                # 
                # Common Experience: s_i = w_i + σ·ε
                # 虽然无法识别个体，但生产者可以：
                # 1. 用信号集合的均值估计共同冲击ε
                # 2. 更新对"代表性个体"的预测
                # 3. 对所有人使用这个改善的统一预测
                # 
                # 这体现了数据的价值，即使在匿名化下！
                # 否则会人为压低匿名化的福利效应。
                
                signal_mean = np.mean(participant_signals)
                n_participants = len(participant_signals)
                
                # 估计共同冲击（贝叶斯更新，带收缩）
                # E[ε | X] = (1 / (1 + n·σ²/σ_θ²)) · (mean(X) - μ_θ) / σ
                epsilon_posterior_var = 1 / (1 + n_participants * params.sigma**2 / params.sigma_theta**2)
                epsilon_hat = epsilon_posterior_var * (signal_mean - params.mu_theta) / params.sigma
                
                # 代表性个体的后验均值
                # E[w | X] ≈ μ_θ + σ·E[ε|X]
                # 这比先验μ_θ更准确（利用了数据中的共同信息）
                mu_common = params.mu_theta + params.sigma * epsilon_hat
                
                # 进一步收缩到先验（避免小样本过拟合）
                # 加权：先验 vs 数据驱动的预测
                data_weight = n_participants / (n_participants + 1.0)
                mu_common_shrunk = (1 - data_weight) * params.mu_theta + data_weight * mu_common
                
                mu_producer[:] = mu_common_shrunk
        # 无参与者时，使用先验（已初始化）
    
    return mu_producer # 返回所有消费者的后验期望


def simulate_market_outcome(
    data: ConsumerData,
    participation: np.ndarray,
    params: ScenarioCParams,
    producer_info_mode: str = "with_data",
    m0: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> MarketOutcome:
    """
    模拟给定参与决策下的完整市场均衡
    
    ========================================================================
    理论框架（论文Section 2 + Section 4）
    ========================================================================
    这是整个模型的核心函数，实现了论文中的完整博弈序列：
    
    博弈序列（论文Figure 1）：
    (1) 自然产生(w, s)
    (2) 中介发布合约(m, 匿名化政策)
    (3) 消费者决策是否参与 → participation
    (4) 中介收集数据，形成数据库X
    (5) 中介根据政策处理数据（实名/匿名）
    (6) 信息披露：Y_0给生产者，Y_i给消费者
    (7) 生产者定价 → prices
    (8) 消费者购买 → quantities
    (9) 效用实现，补偿发放
    
    本函数实现步骤(4)-(9)，输入是(1)-(3)的结果
    
    ========================================================================
    关键经济学机制
    ========================================================================
    
    **信息不对称的核心**：
    - 消费者信息集：I_i = {s_i} ∪ X（有私人信号）
    - 生产者信息集：Y_0 = X或{(i,s_i)}（取决于匿名化）
    - 匿名化通过改变Y_0影响定价能力
    
    **社会数据外部性**：
    - 数据库X = {s_i : i ∈ participants}
    - 拒绝者i也能从X中学习（搭便车）
    - 参与者数量↑ → X信息含量↑ → 所有人后验精度↑
    
    **匿名化的双重作用**：
    - 阻止价格歧视（论文Proposition 2）
    - 但可能降低参与激励（论文Theorem 1）
    
    ========================================================================
    Args:
        data: ConsumerData - (w, s)的实现
              w: 真实支付意愿（消费者不知道，但最终用于结算）
              s: 观测信号（消费者和中介都知道）
        
        participation: (N,) bool数组
                       participation[i]=True表示i参与数据交易
                       这是LLM决策或理性均衡的结果
        
        params: ScenarioCParams - 所有模型参数
        
        producer_info_mode: str - 生产者信息模式（用于m_0计算）
                           "with_data" (默认): 生产者按政策获得中介数据Y_0
                           "no_data": 生产者无任何中介信息，Y_0=∅（计算基准）
                           用途：计算数据信息价值 m_0 = E[π_with] - E[π_no]
        
    Returns:
        MarketOutcome - 完整的市场结果和福利指标
    
    ========================================================================
    实现细节（逐步拆解）
    ========================================================================
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)
    
    N = params.N
    
    # ------------------------------------------------------------------------
    # 步骤1：数据收集与匿名化处理
    # 对应论文Section 4 - 中介的信息设计
    # ------------------------------------------------------------------------
    
    # 1.1 收集参与者信号，形成数据库X
    participant_indices = np.where(participation)[0]  # 参与者的索引
    participant_signals = data.s[participant_indices]  # 参与者的信号
    
    # 1.2 匿名化处理（论文核心机制）
    # 实名（Identified）：保留(i, s_i)映射
    # 匿名（Anonymized）：打乱映射，变成无序集合{s_i}
    if params.anonymization == "anonymized" and len(participant_signals) > 0:
        participant_signals = participant_signals.copy()
        rng.shuffle(participant_signals)  # 打乱顺序，破坏身份映射
        # 注意：这里的shuffle是概念性的，实际效果是生产者无法知道
        # 哪个信号对应哪个消费者
    
    # ------------------------------------------------------------------------
    # 步骤2：消费者后验估计（贝叶斯学习）
    # 对应论文Section 3.3 - Bayesian updating
    # ------------------------------------------------------------------------
    
    # 消费者i的信息集：I_i = {s_i, X}
    # - s_i：自己的私人信号（只有i知道）
    # - X：数据库（所有人都能看到，包括拒绝者！）
    #
    # 这体现了搭便车（free-riding）：
    # 拒绝者不贡献数据，但仍能从参与者数据中学习
    
    mu_consumers = np.zeros(N)
    for i in range(N):
        # 每个消费者计算E[w_i | s_i, X]
        mu_consumers[i] = compute_posterior_mean_consumer(
            data.s[i], participant_signals, params
        )
    
    # 关键洞察：即使是拒绝者（participation[i]=False），
    # 也能用participant_signals更新信念，这是数据外部性的体现
    
    # ------------------------------------------------------------------------
    # 步骤3：生产者后验估计（匿名化的关键！）
    # 对应论文Section 4 - Anonymization的经济效果
    # ------------------------------------------------------------------------
    
    # 生产者的信息集Y_0取决于两个因素：
    # (1) 是否有数据（producer_info_mode）
    # (2) 匿名化政策（anonymization）
    #
    # producer_info_mode == "no_data"（无数据基准）：
    # - Y_0 = ∅（无任何中介信息）
    # - 对所有人：μ_producer[i] = μ_θ（先验）
    # - 结果：必须统一定价，无法识别个体
    # - 用途：计算数据的信息价值 m_0 = E[π_with] - E[π_no]
    #
    # producer_info_mode == "with_data"（默认）：
    #   实名（Identified）：
    #   - Y_0 = {(i, s_i) : i ∈ participants}
    #   - 对参与者i：知道s_i，可计算E[w_i | s_i, X]（与消费者i相同）
    #   - 对拒绝者j：不知道s_j，只能用先验μ_θ
    #   - 结果：μ_producer异质，可以个性化定价
    #
    #   匿名（Anonymized）：
    #   - Y_0 = {s_i : i ∈ participants}（无身份）
    #   - 无法识别哪个信号对应哪个消费者
    #   - 对所有人：μ_producer[i] = E[θ | X]（相同）
    #   - 结果：μ_producer同质，必须统一定价
    #
    # 这正是论文Proposition 2的核心：
    # "匿名化通过阻止生产者识别个体，防止价格歧视"
    
    if producer_info_mode == "no_data":
        # 无数据基准：生产者只有先验信息
        mu_producer = np.full(N, params.mu_theta)
    elif producer_info_mode == "with_data":
        # 默认：生产者按政策获得中介数据
        mu_producer = compute_producer_posterior(
            data, participation, participant_signals, params
        )
    else:
        raise ValueError(f"Unknown producer_info_mode: {producer_info_mode}")
    
    # ------------------------------------------------------------------------
    # 步骤4：生产者定价（最优定价策略）
    # 对应论文Section 2.2 + Section 4
    # ------------------------------------------------------------------------
    
    # 生产者目标：max Π = Σ_i (p_i - c) · q_i
    # 约束条件：取决于（1）匿名化政策 （2）是否有数据
    
    prices = np.zeros(N)
    
    # 无数据基准：强制统一定价（因为所有人信息相同）
    if producer_info_mode == "no_data":
        # 即使params.anonymization="identified"，无数据下也必须统一定价
        # 因为所有人的μ_producer都是先验μ_θ
        p_uniform, _ = compute_optimal_price_uniform(
            mu_producer.tolist(), params.c
        )
        prices[:] = p_uniform
    
    elif params.anonymization == "identified":
        # ----------------------------------------------------------------
        # 实名制：个性化定价（Third-degree price discrimination）
        # ----------------------------------------------------------------
        # 生产者知道每个i的后验μ_producer[i]（可能异质）
        # 可以对每个i单独优化：max_{p_i} (p_i - c)·(μ_i - p_i)
        # 闭式解：p_i* = (μ_producer[i] + c) / 2
        
        for i in range(N):
            prices[i] = compute_optimal_price_personalized(
                mu_producer[i], params.c
            )
        
        # 结果：参与者可能面临不同价格（如果μ_producer[i]异质）
        # 拒绝者面临基于先验的价格p_j = (μ_θ + c) / 2
        
    else:
        # ----------------------------------------------------------------
        # 匿名化：统一定价（Uniform pricing）
        # ----------------------------------------------------------------
        # 生产者无法识别个体，必须对所有人设同一价格p
        # 优化问题：max_p Σ_i (p - c)·max(μ_producer[i] - p, 0)
        #
        # 注意：即使匿名化下μ_producer[i]全相同，定价问题仍是
        # "统一价"而非"个性化价"，因为无法针对特定消费者调价
        #
        # 关键：这是非平凡优化，不能用μ/2公式！
        # 正确方法：数值优化（方法A）或分段枚举（方法B）
        
        p_uniform, _ = compute_optimal_price_uniform(
            mu_producer.tolist(), params.c
        )
        prices[:] = p_uniform
        
        # 结果：所有人面临相同价格
        # 这是匿名化保护消费者剩余的关键机制（论文Proposition 2）
    
    # ------------------------------------------------------------------------
    # 步骤5：消费者购买决策（理性需求）
    # 对应论文Section 2.1 - Consumer demand
    # ------------------------------------------------------------------------
    
    # 消费者i的优化问题：
    # max_{q_i} u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
    #
    # 但消费者不知道真实w_i，只能用后验μ_consumers[i]
    # 期望效用最大化：
    # max_{q_i} E[u_i | I_i] = μ_consumers[i]·q_i - p_i·q_i - 0.5·q_i²
    #
    # 一阶条件：μ_consumers[i] - p_i - q_i = 0
    # 最优需求：q_i* = max(μ_consumers[i] - p_i, 0)
    
    quantities = np.maximum(mu_consumers - prices, 0)
    
    # 非负约束：如果μ_i < p_i，则q_i = 0（不购买）
    
    # ------------------------------------------------------------------------
    # 步骤6：效用实现（用真实w_i结算）
    # 对应论文Section 2.1 + Section 5
    # ------------------------------------------------------------------------
    
    # 消费者的实现效用（论文式2.1）：
    # u_i = w_i·q_i - p_i·q_i - 0.5·q_i²
    #
    # 注意：这里用真实w_i计算，而非后验μ_i
    # 这体现了学习的价值：如果μ_i ≈ w_i，则需求q_i接近真实最优
    
    utilities = data.w * quantities - prices * quantities - 0.5 * quantities ** 2
    
    # 参与补偿（论文Section 5 - Participation incentive）
    # 参与者额外获得中介支付的补偿m_i（论文式(11)）
    # 这是激励参与的直接手段
    # 
    # 重要：这里不会产生"double counting"
    # 因为utilities[participation]只被加一次m
    # （之前的P0-1修复）
    # 
    # ✅ 支持个性化补偿：每个消费者i获得m[i]
    # - 如果m是统一补偿（通过__post_init__扩展为向量），所有人获得相同补偿
    # - 如果m是个性化补偿，每个人获得不同补偿
    utilities[participation] += params.m[participation]
    
    # ------------------------------------------------------------------------
    # 步骤7：福利指标计算
    # 对应论文Section 5 - Welfare analysis
    # ------------------------------------------------------------------------
    
    # 7.1 消费者剩余（Consumer Surplus）
    # CS = Σ_i u_i（包含补偿）
    # 这是消费者的总福利，包括：
    # - 产品消费的效用（w_i·q_i - 0.5·q_i²）
    # - 支付的成本（-p_i·q_i）
    # - 参与者获得的补偿（+m）
    consumer_surplus = np.sum(utilities)
    
    # 7.2 生产者利润（Producer Profit）
    # PS = Σ_i (p_i - c)·q_i
    # 这是生产者从产品销售中获得的利润
    # 不包括向中介支付的m_0（那部分是转移支付）
    producer_profit = np.sum((prices - params.c) * quantities)
    
    # 7.3 中介利润（Intermediary Profit）
    # R = m_0 - Σ^N_{i=1} m_i·a_i （论文式(4)）
    #
    # 收入：m_0（生产者向中介支付）
    # - 可以是固定费用
    # - 或生产者利润提升的某个比例（我们的扩展）
    # - 论文隐含假设中介能获得部分生产者剩余
    #
    # 支出：Σ m_i·a_i（中介向参与者支付）
    # - 激励参与的成本
    # - m_i是每个消费者i的补偿（论文标准）
    # - a_i = 1 if 参与, 0 if 拒绝
    #
    # ✅ 支持个性化补偿：
    # - 统一补偿：Σ m·a_i = m·Σa_i = m·N_participants
    # - 个性化补偿：Σ m_i·a_i（只对参与者求和）
    #
    # 净利润：R = 收入 - 支出
    # m0作为显式参数传入（默认0.0）
    # 理论求解器会传入estimate_m0_mc计算的内生m_0
    
    num_participants = int(np.sum(participation))
    intermediary_cost = np.sum(params.m[participation])  # ✅ 只对参与者求和
    intermediary_profit = m0 - intermediary_cost
    
    # 7.4 社会福利（Social Welfare）
    # SW = CS + PS + IS
    #
    # 重要性质：
    # - 如果m_0 = 0，则SW = CS + PS - m·N
    # - 补偿m是转移支付：从中介转移到消费者
    # - 不影响社会总福利（如果m_0足够大抵消支出）
    # - 但影响福利分配（CS ↑, IS ↓）
    #
    # 论文关注点：
    # - 匿名化如何影响SW（通过改变定价和参与）
    # - 数据外部性如何影响SW（通过学习质量）
    
    social_welfare = consumer_surplus + producer_profit + intermediary_profit
    
    # ------------------------------------------------------------------------
    # 步骤8：学习质量指标
    # 衡量数据外部性的学习效果
    # ------------------------------------------------------------------------
    
    # 学习误差 = |后验估计 - 真实值|
    # 越小表示学习越准确
    learning_errors = np.abs(mu_consumers - data.w)
    
    # 参与者的学习质量
    # 参与者贡献了数据，理论上应该学习更准确？
    # 但在我们的设定中（Y_i = X），参与者和拒绝者看到相同数据库
    # 区别只在于参与者的s_i在X中（自己的信号被包含）
    if np.any(participation):
        learning_quality_participants = np.mean(learning_errors[participation])
    else:
        learning_quality_participants = np.mean(learning_errors)  # 无参与者时所有人一样
    
    # 拒绝者的学习质量
    # 搭便车效应：拒绝者不贡献数据，但仍能从X中学习
    # 这是数据外部性的直接体现
    if np.any(~participation):
        learning_quality_rejecters = np.mean(learning_errors[~participation])
    else:
        learning_quality_rejecters = 0.0
    
    # 理论预期：
    # - 参与率↑ → |X|↑ → 所有人学习质量↑（外部性）
    # - Common Preferences：参与者和拒绝者学习质量应相近
    # - Common Experience：参与者可能略好（自己信号被包含）
    
    # ------------------------------------------------------------------------
    # 步骤9：不平等指标
    # 衡量福利分配的公平性
    # ------------------------------------------------------------------------
    
    # 基尼系数（Gini Coefficient）
    # 范围[0, 1]，0表示完全平等，1表示完全不平等
    # 用于衡量效用分配的不平等程度
    gini_coefficient = compute_gini(utilities)
    
    # 参与者平均效用
    if np.any(participation):
        acceptor_avg_utility = np.mean(utilities[participation])
    else:
        acceptor_avg_utility = 0.0
    
    # 拒绝者平均效用
    if np.any(~participation):
        rejecter_avg_utility = np.mean(utilities[~participation])
    else:
        rejecter_avg_utility = 0.0
    
    # 理论分析：
    # - 如果acceptor_avg > rejecter_avg：参与是有利可图的（补偿m足够大）
    # - 如果acceptor_avg < rejecter_avg：拒绝者搭便车成功（m太小或价格歧视严重）
    # - 匿名化应该缩小两者差距（减少价格歧视对参与者的伤害）
    
    # ------------------------------------------------------------------------
    # 步骤10：价格歧视指标
    # 衡量匿名化政策的效果
    # ------------------------------------------------------------------------
    
    # 价格方差
    # Var(prices) = 0 表示统一定价（匿名化效果）
    # Var(prices) > 0 表示价格歧视（实名制效果）
    price_variance = np.var(prices)
    
    # 价格歧视指数 = max(p) - min(p)
    # 0表示完全统一定价
    # 越大表示价格歧视越严重
    price_discrimination_index = np.max(prices) - np.min(prices)
    
    # 理论预期：
    # - anonymization="anonymized" → price_variance = 0, index = 0
    # - anonymization="identified" + Common Preferences → price_variance ≈ 0（后验相近）
    # - anonymization="identified" + Common Experience → price_variance > 0（后验异质）
    
    return MarketOutcome(
        participation=participation,
        participation_rate=np.mean(participation),
        num_participants=int(np.sum(participation)),
        prices=prices,
        quantities=quantities,
        mu_consumers=mu_consumers,
        mu_producer=mu_producer,
        utilities=utilities,
        consumer_surplus=consumer_surplus,
        producer_profit=producer_profit,
        intermediary_profit=intermediary_profit,
        social_welfare=social_welfare,
        learning_quality_participants=learning_quality_participants,
        learning_quality_rejecters=learning_quality_rejecters,
        gini_coefficient=gini_coefficient,
        acceptor_avg_utility=acceptor_avg_utility,
        rejecter_avg_utility=rejecter_avg_utility,
        price_variance=price_variance,
        price_discrimination_index=price_discrimination_index
    )


def compute_gini(utilities: np.ndarray) -> float:
    """
    计算Gini系数（对负值和零总和稳健）
    
    Args:
        utilities: 效用数组
        
    Returns:
        Gini系数 (0-1之间，越大越不平等)
    """
    n = len(utilities)
    if n == 0:
        return 0.0
    
    # 平移到正值（保持相对不平等度不变）
    # 这样处理对负效用和接近零的总和更稳健
    min_utility = np.min(utilities)
    if min_utility < 0:
        utilities_shifted = utilities - min_utility + 1e-6  # 平移到正区间
    else:
        utilities_shifted = utilities + 1e-6  # 避免除零
    
    # 排序
    sorted_utilities = np.sort(utilities_shifted)
    
    # 计算Gini系数
    total = np.sum(sorted_utilities)
    if total <= 0:
        return 0.0  # 如果总和仍为0或负，返回0
    
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_utilities)) / (n * total) - (n + 1) / n
    
    return max(0.0, min(1.0, gini))  # 确保在[0,1]范围内


def compute_expected_utility_ex_ante(
    consumer_id: int,
    participates: bool,
    others_participation_rate: float,
    params: ScenarioCParams,
    num_world_samples: int = 30,
    num_market_samples: int = 20,
    base_seed: int = None
) -> float:
    """
    计算Ex Ante期望效用：对所有随机性取平均
    
    这是学术上正确的做法，与论文时序对齐：
    - 消费者在不知道(w, s)实现的情况下决策
    - 期望对所有随机性取平均：信号、偏好、参与者集合、价格
    
    两层Monte Carlo：
    - 外层：抽取世界状态(w, s)
    - 内层：抽取参与者集合
    
    Args:
        consumer_id: 消费者ID
        participates: 该消费者是否参与
        others_participation_rate: 其他消费者的参与概率
        params: 场景参数
        num_world_samples: 世界状态采样数
        num_market_samples: 市场采样数（每个世界状态）
        base_seed: 基础随机种子
        
    Returns:
        Ex Ante期望效用
    """
    if base_seed is None:
        base_seed = params.seed
    
    total_utility = 0.0
    N = params.N
    
    # 外层循环：遍历可能的世界状态
    for world_idx in range(num_world_samples):
        # 生成一个可能的世界状态
        world_seed = base_seed + world_idx * 10000 + consumer_id * 100
        data = generate_consumer_data(
            ScenarioCParams(**{**params.to_dict(), 'seed': world_seed})
        )
        
        # 内层循环：在这个世界状态下，遍历可能的参与者集合
        world_utility = 0.0
        for market_idx in range(num_market_samples):
            # 采样他人的参与决策
            np.random.seed(world_seed + market_idx + 50000)
            participation = np.zeros(N, dtype=bool)
            for j in range(N):
                if j == consumer_id:
                    participation[j] = participates
                else:
                    participation[j] = np.random.rand() < others_participation_rate
            
            # 模拟市场结果
            outcome = simulate_market_outcome(data, participation, params)
            
            # 累加效用
            world_utility += outcome.utilities[consumer_id]
        
        total_utility += world_utility / num_market_samples
    
    return total_utility / num_world_samples


def compute_expected_utility_given_participation(
    consumer_id: int,
    participates: bool,
    others_participation_rate: float,
    data: ConsumerData,
    params: ScenarioCParams,
    num_samples: int = 100
) -> float:
    """
    计算消费者i在给定参与决策和他人参与率下的期望效用
    
    使用蒙特卡洛方法：
    1. 采样他人的参与向量（以概率r独立参与）
    2. 对每个样本，模拟市场结果并计算消费者i的效用
    3. 平均得到期望效用
    
    Args:
        consumer_id: 消费者ID
        participates: 消费者i是否参与
        others_participation_rate: 其他消费者的参与概率
        data: 消费者数据
        params: 场景参数
        num_samples: 蒙特卡洛样本数
        
    Returns:
        期望效用
    """
    N = params.N
    total_utility = 0.0
    
    for _ in range(num_samples):
        # 采样他人的参与决策
        participation = np.zeros(N, dtype=bool)
        for j in range(N):
            if j == consumer_id:
                participation[j] = participates
            else:
                participation[j] = np.random.rand() < others_participation_rate
        
        # 模拟市场结果
        outcome = simulate_market_outcome(data, participation, params)
        
        # 累加消费者i的效用
        total_utility += outcome.utilities[consumer_id]
    
    return total_utility / num_samples


def compute_rational_participation_rate_ex_ante(
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_world_samples: int = 30,
    num_market_samples: int = 20,
    compute_per_consumer: bool = False
) -> Tuple[float, List[float], float, np.ndarray, np.ndarray]:
    """
    Ex Ante固定点：消费者在不知道信号实现时决策
    
    支持两种模式：
    1. 无异质性（tau_dist="none"）：r*通常为0或1
    2. 有异质性（tau_dist!="none"）：r* = F_τ(ΔU(r*))，可产生内点
    
    ✅ 新增：支持个性化补偿m_i的真正计算
    - 如果compute_per_consumer=True，为每个消费者单独计算ΔU_i(m_i)
    - 返回每个消费者的ΔU_i和参与概率p_i
    
    Args:
        params: 场景参数（不需要传入fixed data！）
        max_iter: 最大迭代次数
        tol: 收敛容差
        num_world_samples: 世界状态采样数
        num_market_samples: 市场采样数
        compute_per_consumer: 是否为每个消费者单独计算（个性化补偿时需要）
        
    Returns:
        (收敛的参与率, 参与率历史, 平均ΔU, ΔU向量, 参与概率向量)
    """
    N = params.N
    r = 0.5  # 初始参与率
    r_history = [r]
    delta_u_avg = 0.0  # 平均ΔU（向后兼容）
    delta_u_vector = np.zeros(N)  # 每个消费者的ΔU
    p_vector = np.zeros(N)  # 每个消费者的参与概率
    
    for iteration in range(max_iter):
        if compute_per_consumer:
            # 个性化模式：为每个消费者单独计算ΔU_i(m_i)
            delta_u_vector = np.zeros(N)
            for i in range(N):
                utility_accept_i = compute_expected_utility_ex_ante(
                    consumer_id=i,
                    participates=True,
                    others_participation_rate=r,
                    params=params,
                    num_world_samples=num_world_samples,
                    num_market_samples=num_market_samples
                )
                
                utility_reject_i = compute_expected_utility_ex_ante(
                    consumer_id=i,
                    participates=False,
                    others_participation_rate=r,
                    params=params,
                    num_world_samples=num_world_samples,
                    num_market_samples=num_market_samples
                )
                
                delta_u_vector[i] = utility_accept_i - utility_reject_i
            
            delta_u_avg = np.mean(delta_u_vector)
            
            # 计算每个消费者的参与概率
            if params.tau_dist == "none":
                p_vector = (delta_u_vector > 0).astype(float)
            elif params.tau_dist == "normal":
                from scipy.stats import norm
                p_vector = norm.cdf(delta_u_vector, loc=params.tau_mean, scale=params.tau_std)
            elif params.tau_dist == "uniform":
                a = params.tau_mean - np.sqrt(3) * params.tau_std
                b = params.tau_mean + np.sqrt(3) * params.tau_std
                p_vector = np.clip((delta_u_vector - a) / (b - a), 0, 1)
            else:
                raise ValueError(f"Unsupported tau_dist: {params.tau_dist}")
            
            # 平均参与率
            r_new = np.mean(p_vector)
            
        else:
            # 代表性消费者模式（向后兼容，用于统一补偿）
            utility_accept = compute_expected_utility_ex_ante(
                consumer_id=0,  # 代表性消费者
                participates=True,
                others_participation_rate=r,
                params=params,
                num_world_samples=num_world_samples,
                num_market_samples=num_market_samples
            )
            
            utility_reject = compute_expected_utility_ex_ante(
                consumer_id=0,
                participates=False,
                others_participation_rate=r,
                params=params,
                num_world_samples=num_world_samples,
                num_market_samples=num_market_samples
            )
            
            delta_u_avg = utility_accept - utility_reject
            delta_u_vector = np.full(N, delta_u_avg)  # 所有人相同
            
            # 根据异质性设置计算新参与率
            if params.tau_dist == "none":
                r_new = 1.0 if delta_u_avg > 0 else 0.0
                p_vector = np.full(N, r_new)
            elif params.tau_dist == "normal":
                from scipy.stats import norm
                r_new = norm.cdf(delta_u_avg, loc=params.tau_mean, scale=params.tau_std)
                p_vector = np.full(N, r_new)
            elif params.tau_dist == "uniform":
                a = params.tau_mean - np.sqrt(3) * params.tau_std
                b = params.tau_mean + np.sqrt(3) * params.tau_std
                r_new = np.clip((delta_u_avg - a) / (b - a), 0, 1)
                p_vector = np.full(N, r_new)
            else:
                raise ValueError(f"Unsupported tau_dist: {params.tau_dist}")
        
        r_history.append(r_new)
        
        # 检查收敛
        if abs(r_new - r) < tol:
            mode_str = "个性化" if compute_per_consumer else "代表性消费者"
            print(f"  Ex Ante固定点收敛于迭代 {iteration + 1} ({mode_str}), r* = {r_new:.4f}, ΔU_avg = {delta_u_avg:.4f}")
            if compute_per_consumer:
                print(f"    ΔU范围: [{np.min(delta_u_vector):.4f}, {np.max(delta_u_vector):.4f}], std={np.std(delta_u_vector):.4f}")
            return r_new, r_history, delta_u_avg, delta_u_vector, p_vector
        
        # 平滑更新
        r = 0.6 * r_new + 0.4 * r
    
    # 未收敛
    raise RuntimeError(
        f"Ex Ante固定点未在{max_iter}次迭代内收敛！\n"
        f"当前 r = {r:.4f}, 最后ΔU_avg = {delta_u_avg:.4f}\n"
        f"建议：增加max_iter或放宽tol\n"
        f"历史：{[f'{x:.3f}' for x in r_history[-10:]]}"
    )


def compute_rational_participation_rate_ex_post(
    data: ConsumerData,
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50
) -> Tuple[float, List[float]]:
    """
    Ex Post固定点：消费者在看到realized (w, s)后决策（旧实现）
    
    注意：这是Ex Post/Interim决策，与论文时序不一致。
    仅用于鲁棒性/对比分析。主结果应使用Ex Ante版本。
    
    使用迭代算法：
    1. 给定realized data
    2. 计算每个消费者基于realized s_i的期望效用差
    3. 更新参与率
    4. 重复直到收敛
    
    Args:
        data: 已实现的消费者数据（w, s）
        params: 场景参数
        max_iter: 最大迭代次数
        tol: 收敛容差
        num_mc_samples: 蒙特卡洛样本数
        
    Returns:
        (收敛的参与率, 参与率历史)
    """
    N = params.N
    r = 0.5  # 初始参与率
    r_history = [r]
    
    for iteration in range(max_iter):
        # 计算每个消费者的效用差
        accept_decisions = []
        
        for i in range(N):
            # 期望效用（参与）
            utility_accept = compute_expected_utility_given_participation(
                i, True, r, data, params, num_mc_samples
            )
            
            # 期望效用（拒绝）
            utility_reject = compute_expected_utility_given_participation(
                i, False, r, data, params, num_mc_samples
            )
            
            # 决策：参与 iff 效用差 > 0
            # 注意：utility_accept 已经在 simulate_market_outcome 中包含了补偿 m
            # 因此这里不需要再加 m（否则会双重计入）
            delta_u = utility_accept - utility_reject
            should_accept = delta_u > 0
            accept_decisions.append(should_accept)
        
        # 更新参与率
        r_new = np.mean(accept_decisions)
        r_history.append(r_new)
        
        # 检查收敛
        if abs(r_new - r) < tol:
            print(f"  固定点收敛于迭代 {iteration + 1}, r* = {r_new:.4f}")
            return r_new, r_history
        
        # 平滑更新（避免震荡）
        r = 0.6 * r_new + 0.4 * r
    
    # ⚠️ P1-2修正：Ground Truth必须收敛，否则不可用
    raise RuntimeError(
        f"Ex Post固定点未在{max_iter}次迭代内收敛！\n"
        f"当前 r = {r:.4f}\n"
        f"建议：增加max_iter或放宽tol\n"
        f"历史：{[f'{x:.3f}' for x in r_history[-10:]]}"
    )


def compute_rational_participation_rate(
    params: ScenarioCParams,
    data: ConsumerData = None,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50,
    compute_per_consumer: bool = False
) -> Tuple[float, List[float], float, np.ndarray, np.ndarray]:
    """
    计算理性参与率的统一接口
    
    根据params.participation_timing自动选择：
    - "ex_ante": 论文标准时序（推荐，学术正确）
    - "ex_post": 观察到realized data后决策（鲁棒性/对比）
    
    Args:
        params: 场景参数
        data: 消费者数据（仅ex_post需要）
        max_iter: 最大迭代次数
        tol: 收敛容差
        num_mc_samples: MC采样数（含义依时序而异）
        compute_per_consumer: 是否为每个消费者单独计算（个性化补偿时需要）
        
    Returns:
        (收敛的参与率, 参与率历史, 平均ΔU, ΔU向量, 参与概率向量)
    """
    if params.participation_timing == "ex_ante":
        # Ex Ante: 对所有随机性取平均
        # num_mc_samples会被拆分为world samples和market samples
        num_world = max(10, int(np.sqrt(num_mc_samples)))
        num_market = max(10, num_mc_samples // num_world)
        
        return compute_rational_participation_rate_ex_ante(
            params=params,
            max_iter=max_iter,
            tol=tol,
            num_world_samples=num_world,
            num_market_samples=num_market,
            compute_per_consumer=compute_per_consumer
        )
    
    elif params.participation_timing == "ex_post":
        # Ex Post: 给定realized data
        if data is None:
            raise ValueError("Ex Post模式需要提供data参数")
        
        r_star, r_history = compute_rational_participation_rate_ex_post(
            data=data,
            params=params,
            max_iter=max_iter,
            tol=tol,
            num_mc_samples=num_mc_samples
        )
        # Ex Post不支持个性化计算，返回统一值
        delta_u_avg = 0.0  # Ex Post模式不返回ΔU
        delta_u_vector = np.zeros(params.N)
        p_vector = np.full(params.N, r_star)
        return r_star, r_history, delta_u_avg, delta_u_vector, p_vector
    
    else:
        raise ValueError(f"Unsupported participation_timing: {params.participation_timing}")


def generate_participation_from_tau(
    delta_u: float,
    params: ScenarioCParams,
    seed: int = None
) -> np.ndarray:
    """
    基于隐私成本τ_i生成participation决策（P2-2修正）
    
    经济学microfoundation：
    - 每个消费者i有隐私成本τ_i ~ F_τ
    - 消费者i参与当且仅当ΔU ≥ τ_i
    - 这比独立Bernoulli(r*)更符合理论结构
    
    Args:
        delta_u: 参与vs拒绝的期望效用差（对所有人相同，ex ante）
        params: 场景参数
        seed: 随机种子
        
    Returns:
        (N,) bool数组，True表示参与
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = params.N
    
    if params.tau_dist == "none":
        # 无异质性：所有人决策相同
        # 所有人参与 if ΔU > 0 else 拒绝
        participation = np.full(N, delta_u > 0, dtype=bool)
    
    elif params.tau_dist == "normal":
        # τ_i ~ N(μ_τ, σ_τ²)
        tau_i = np.random.normal(params.tau_mean, params.tau_std, size=N)
        # 参与 if τ_i ≤ ΔU
        participation = tau_i <= delta_u
    
    elif params.tau_dist == "uniform":
        # τ_i ~ Uniform[μ_τ - √3·σ_τ, μ_τ + √3·σ_τ]
        a = params.tau_mean - np.sqrt(3) * params.tau_std
        b = params.tau_mean + np.sqrt(3) * params.tau_std
        tau_i = np.random.uniform(a, b, size=N)
        # 参与 if τ_i ≤ ΔU
        participation = tau_i <= delta_u
    
    else:
        raise ValueError(f"Unsupported tau_dist: {params.tau_dist}")
    
    return participation


def generate_conditional_equilibrium(
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50,
    num_outcome_samples: int = 20
) -> Dict:
    """
    条件均衡：给定中介策略 (m, anonymization) 下的市场均衡
    
    ⚠️ 注意：这不是完整的Ground Truth！
    完整的GT需要求解最优策略，见 generate_ground_truth()
    
    用途：
    - 调试特定策略
    - 反事实分析（"如果中介选择m=2会怎样？"）
    - 研究策略空间的局部行为
    
    输出包含两层指标：
    1. 理论指标：
       - r*（固定点收敛值）
       - E[outcome | r*]（期望市场结果，MC平均）
    2. 示例指标：
       - 一次参与抽样realization
       - 对应的市场结果
    
    Args:
        params: 场景参数（包含给定的 m 和 anonymization）
        max_iter: 固定点迭代最大次数
        tol: 收敛容差
        num_mc_samples: 蒙特卡洛样本数（固定点计算）
        num_outcome_samples: outcome期望化的MC样本数
        
    Returns:
        条件均衡字典（给定策略下的均衡）
    """
    print(f"\n{'='*60}")
    print(f"生成场景C Ground Truth")
    print(f"{'='*60}")
    print(f"参数:")
    print(f"  N = {params.N}")
    print(f"  数据结构 = {params.data_structure}")
    print(f"  匿名化 = {params.anonymization}")
    # 支持标量和向量m的打印
    if isinstance(params.m, np.ndarray):
        if np.all(params.m == params.m[0]):
            print(f"  补偿 m = {params.m[0]:.2f} (统一)")
        else:
            print(f"  补偿 m = 向量 (均值={np.mean(params.m):.2f})")
    else:
        print(f"  补偿 m = {float(params.m):.2f}")
    print(f"  噪声水平 σ = {params.sigma:.2f}")
    
    # 计算理性参与率（根据时序模式）
    print(f"\n计算理性参与率（{params.participation_timing}模式）...")
    
    delta_u = None  # 保存ΔU（用于基于τ_i的participation生成）
    
    # 计算理性参与率（使用统一接口）
    rational_rate, r_history, delta_u, delta_u_vector, p_vector = compute_rational_participation_rate(
        params,
        data=None,
        max_iter=max_iter,
        tol=tol,
        num_mc_samples=num_mc_samples,
        compute_per_consumer=False  # optimize_intermediary_policy使用代表性消费者
    )
    
    # ========================================================================
    # 新增：计算内生m_0（生产者对数据的支付意愿）
    # ========================================================================
    print(f"\n计算内生m_0（生产者支付意愿，MC-200次）...")
    
    # 定义参与规则（与GT的参与生成一致）
    def participation_rule(p: ScenarioCParams, world: ConsumerData, rng: np.random.Generator) -> np.ndarray:
        """生成参与决策（与GT一致的规则）"""
        if delta_u is None:
            # Ex Post: 使用Bernoulli(r*)
            return rng.random(p.N) < rational_rate
        
        # Ex Ante: 使用τ阈值规则
        if p.tau_dist == "none":
            return np.full(p.N, delta_u > 0, dtype=bool)
        elif p.tau_dist == "normal":
            tau = rng.normal(p.tau_mean, p.tau_std, p.N)
            return tau <= delta_u
        elif p.tau_dist == "uniform":
            low = p.tau_mean - np.sqrt(3) * p.tau_std
            high = p.tau_mean + np.sqrt(3) * p.tau_std
            tau = rng.uniform(low, high, p.N)
            return tau <= delta_u
        else:
            raise ValueError(f"Unknown tau_dist: {p.tau_dist}")
    
    # 调用estimate_m0_mc计算内生m_0
    m_0_estimated, delta_profit_mean, delta_profit_std, e_num_participants = estimate_m0_mc(
        params=params,
        participation_rule=participation_rule,
        T=200,  # MC样本数
        beta=1.0,  # 中介完全议价能力
        seed=params.seed + 777
    )
    
    print(f"  m_0 = {m_0_estimated:.4f} (期望利润增量: {delta_profit_mean:.4f} ± {delta_profit_std:.4f})")
    print(f"  期望参与人数: {e_num_participants:.2f}")
    
    # 计算理论口径的中介期望利润
    # R = m_0 - E[Σ m_i·a_i]（论文式(4)）
    # ✅ 支持个性化补偿
    if np.all(params.m == params.m[0]):
        # 统一补偿（优化路径）
        expected_intermediary_cost = params.m[0] * e_num_participants
    else:
        # 个性化补偿：使用均值补偿×期望参与人数作为近似
        expected_intermediary_cost = np.mean(params.m) * e_num_participants
    expected_intermediary_profit = m_0_estimated - expected_intermediary_cost
    
    print(f"  期望中介利润: {expected_intermediary_profit:.4f}")
    
    # ========================================================================
    # 第一部分：计算期望outcome（理论严格，不受抽样波动影响）
    # ========================================================================
    print(f"\n计算期望market outcome（MC平均，{num_outcome_samples}次采样）...")
    
    expected_metrics = {
        'consumer_surplus': 0.0,
        'producer_profit': 0.0,
        'intermediary_profit': 0.0,
        'social_welfare': 0.0,
        'gini_coefficient': 0.0,
        'price_discrimination_index': 0.0,
        'participation_rate_realized': 0.0,  # 实际参与率的期望
    }
    
    for sample_idx in range(num_outcome_samples):
        # 每次重新生成数据和参与决策
        sample_seed = params.seed + 1000 + sample_idx
        sample_rng = np.random.default_rng(sample_seed)
        sample_data = generate_consumer_data(params, rng=sample_rng)
        
        # 使用统一的participation_rule生成参与
        sample_participation = participation_rule(params, sample_data, sample_rng)
        
        # ⭐ 计算市场结果（注入内生m_0）
        sample_outcome = simulate_market_outcome(
            sample_data, sample_participation, params,
            producer_info_mode="with_data",
            m0=m_0_estimated,
            rng=sample_rng
        )
        
        # 累加
        expected_metrics['consumer_surplus'] += sample_outcome.consumer_surplus
        expected_metrics['producer_profit'] += sample_outcome.producer_profit
        expected_metrics['intermediary_profit'] += sample_outcome.intermediary_profit
        expected_metrics['social_welfare'] += sample_outcome.social_welfare
        expected_metrics['gini_coefficient'] += sample_outcome.gini_coefficient
        expected_metrics['price_discrimination_index'] += sample_outcome.price_discrimination_index
        expected_metrics['participation_rate_realized'] += sample_outcome.participation_rate
    
    # 平均
    for key in expected_metrics:
        expected_metrics[key] /= num_outcome_samples
    
    # ========================================================================
    # 第二部分：生成一次示例outcome（用于LLM评估）
    # ========================================================================
    print(f"\n生成示例market outcome（单次抽样，用于LLM评估）...")
    
    # 使用统一的participation_rule生成参与
    sample_rng = np.random.default_rng(params.seed + 10000)
    sample_data_llm = generate_consumer_data(params, rng=sample_rng)
    sample_participation = participation_rule(params, sample_data_llm, sample_rng)
    
    # ⭐ 计算市场结果（注入内生m_0）
    sample_outcome = simulate_market_outcome(
        sample_data_llm, sample_participation, params,
        producer_info_mode="with_data",
        m0=m_0_estimated,
        rng=sample_rng
    )
    
    # ========================================================================
    # 打印结果（显示两套指标）
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Ground Truth 结果:")
    print(f"{'='*60}")
    print(f"\n【理论指标】（r* = {rational_rate:.4f}）")
    print(f"  内生m_0（生产者支付）: {m_0_estimated:.4f}")
    print(f"  期望参与人数: {e_num_participants:.2f}")
    print(f"  期望中介利润: {expected_intermediary_profit:.4f}")
    print(f"  期望参与率（实际）: {expected_metrics['participation_rate_realized']:.4f}")
    print(f"  期望消费者剩余: {expected_metrics['consumer_surplus']:.4f}")
    print(f"  期望生产者利润: {expected_metrics['producer_profit']:.4f}")
    print(f"  期望社会福利: {expected_metrics['social_welfare']:.4f}")
    print(f"\n【示例指标】（单次抽样）")
    print(f"  参与率: {sample_outcome.participation_rate:.2%} ({sample_outcome.num_participants}/{params.N})")
    print(f"  消费者剩余: {sample_outcome.consumer_surplus:.4f}")
    print(f"  生产者利润: {sample_outcome.producer_profit:.4f}")
    print(f"  社会福利: {sample_outcome.social_welfare:.4f}")
    
    # 构建返回结果（P1-1：区分理论和示例）
    result = {
        "params": params.to_dict(),
        
        # 理论指标（严格的Ground Truth）
        "rational_participation_rate": float(rational_rate),  # r*（固定点）
        "r_history": [float(x) for x in r_history],
        
        # ⭐ 内生m_0估计（新增）
        "m0_estimation": {
            "m_0": float(m_0_estimated),
            "delta_profit_mean": float(delta_profit_mean),
            "delta_profit_std": float(delta_profit_std),
            "expected_num_participants": float(e_num_participants),
            "expected_intermediary_cost": float(expected_intermediary_cost),
            "expected_intermediary_profit": float(expected_intermediary_profit),
            "method": "estimate_m0_mc (Ex-Ante期望)",
            "mc_samples": 200,
            "beta": 1.0
        },
        
        # 期望outcome（MC平均，理论基准）
        "expected_outcome": {
            "participation_rate_realized": float(expected_metrics['participation_rate_realized']),
            "consumer_surplus": float(expected_metrics['consumer_surplus']),
            "producer_profit": float(expected_metrics['producer_profit']),
            "intermediary_profit": float(expected_metrics['intermediary_profit']),
            "social_welfare": float(expected_metrics['social_welfare']),
            "gini_coefficient": float(expected_metrics['gini_coefficient']),
            "price_discrimination_index": float(expected_metrics['price_discrimination_index']),
        },
        
        # 示例数据和outcome（用于LLM评估）
        "sample_data": {
            "w": sample_data_llm.w.tolist(),
            "s": sample_data_llm.s.tolist(),
            "theta": float(sample_data_llm.theta) if sample_data_llm.theta is not None else None,
            "epsilon": float(sample_data_llm.epsilon) if sample_data_llm.epsilon is not None else None,
        },
        "sample_participation": sample_participation.tolist(),
        "sample_outcome": {
            "participation_rate": float(sample_outcome.participation_rate),
            "num_participants": int(sample_outcome.num_participants),
            "consumer_surplus": float(sample_outcome.consumer_surplus),
            "producer_profit": float(sample_outcome.producer_profit),
            "intermediary_profit": float(sample_outcome.intermediary_profit),
            "social_welfare": float(sample_outcome.social_welfare),
            "gini_coefficient": float(sample_outcome.gini_coefficient),
            "price_variance": float(sample_outcome.price_variance),
            "price_discrimination_index": float(sample_outcome.price_discrimination_index),
            "acceptor_avg_utility": float(sample_outcome.acceptor_avg_utility),
            "rejecter_avg_utility": float(sample_outcome.rejecter_avg_utility),
            "learning_quality_participants": float(sample_outcome.learning_quality_participants),
            "learning_quality_rejecters": float(sample_outcome.learning_quality_rejecters),
        },
        "sample_detailed_results": {
            "prices": sample_outcome.prices.tolist(),
            "quantities": sample_outcome.quantities.tolist(),
            "utilities": sample_outcome.utilities.tolist(),
            "mu_consumers": sample_outcome.mu_consumers.tolist(),
        },
        
        # 向后兼容（指向expected_outcome）
        "outcome": {
            "participation_rate": float(expected_metrics['participation_rate_realized']),
            "num_participants": int(expected_metrics['participation_rate_realized'] * params.N),
            "consumer_surplus": float(expected_metrics['consumer_surplus']),
            "producer_profit": float(expected_metrics['producer_profit']),
            "intermediary_profit": float(expected_metrics['intermediary_profit']),
            "social_welfare": float(expected_metrics['social_welfare']),
            "gini_coefficient": float(expected_metrics['gini_coefficient']),
            "price_variance": 0.0,  # expected不易计算
            "price_discrimination_index": float(expected_metrics['price_discrimination_index']),
            "acceptor_avg_utility": 0.0,  # expected不易计算
            "rejecter_avg_utility": 0.0,  # expected不易计算
            "learning_quality_participants": 0.0,  # expected不易计算
            "learning_quality_rejecters": 0.0,  # expected不易计算
        },
        "data": {  # 向后兼容（与sample_data相同）
            "w": sample_data_llm.w.tolist(),
            "s": sample_data_llm.s.tolist(),
            "theta": float(sample_data_llm.theta) if sample_data_llm.theta is not None else None,
            "epsilon": float(sample_data_llm.epsilon) if sample_data_llm.epsilon is not None else None,
        },
        "rational_participation": sample_participation.tolist(),  # 向后兼容
    }
    
    return result


def generate_ground_truth(
    params_base: Dict,
    m_grid: Optional[np.ndarray] = None,
    policies: Optional[List[str]] = None,
    num_mc_samples: int = 50,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_outcome_samples: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Ground Truth：完整博弈的均衡解（论文理论解）
    
    这是论文《The Economics of Social Data》的完整博弈均衡：
    
    博弈序列（Stackelberg博弈）：
    ┌─────────────────────────────────────────────────┐
    │ 第0阶段：中介优化（Stackelberg Leader）⭐       │
    │   max_{m, anonymization} R                      │
    │   = m_0(m, anonymization) - m·E[N(m, a)]       │
    └──────────────┬──────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────┐
    │ 第1阶段：中介发布合约                            │
    │   offer = (m*, anonymization*)                  │
    └──────────────┬──────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────┐
    │ 第2阶段：消费者反应                              │
    │   r*(m*, anonymization*)                        │
    └──────────────┬──────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────┐
    │ 第3阶段：数据交易                                │
    │   m_0* = E[π_producer(Y_0*) - π_producer(∅)]   │
    └──────────────┬──────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────┐
    │ 第4阶段：产品市场                                │
    │   生产者定价，消费者购买                         │
    └─────────────────────────────────────────────────┘
    
    这才是真正的Ground Truth！
    
    Args:
        params_base: 基础参数字典（N, 数据结构, tau分布等）
                    ⚠️ 不应包含 m 和 anonymization
        m_grid: 中介补偿的候选值网格（默认：0到3，31个点）
        policies: 匿名化策略候选（默认：['identified', 'anonymized']）
        num_mc_samples: Monte Carlo样本数（用于Ex Ante计算）
        max_iter: 固定点迭代最大次数
        tol: 收敛容差
        num_outcome_samples: outcome期望化的MC样本数
        verbose: 是否打印详细信息
    
    Returns:
        完整的Ground Truth字典：
        {
            "optimal_strategy": {  # 最优策略（博弈的均衡）
                "m_star": float,
                "anonymization_star": str,
                "intermediary_profit_star": float,
                "r_star": float,
                ...
            },
            "equilibrium": {  # 最优策略下的市场结果
                "consumer_surplus": float,
                "producer_profit": float,
                "intermediary_profit": float,
                "social_welfare": float,
                ...
            },
            "data_transaction": {  # 数据交易信息
                "m_0": float,
                "producer_profit_gain": float,
                ...
            },
            "all_candidates": [...],  # 所有候选策略的结果
            "sample_data": {...},  # 示例数据（用于LLM评估）
        }
    """
    print(f"\n{'='*70}")
    print(f"生成Ground Truth - 完整博弈均衡（论文理论解）")
    print(f"{'='*70}")
    
    # 验证params_base不包含m和anonymization
    if 'm' in params_base:
        raise ValueError("params_base不应包含'm'！m由中介优化求解。")
    if 'anonymization' in params_base:
        raise ValueError("params_base不应包含'anonymization'！anonymization由中介优化求解。")
    
    # ⭐ 第1步：中介优化（博弈的起点，Stackelberg Leader）
    print(f"\n第1步：求解中介最优策略（Stackelberg Leader）...")
    optimal_policy = optimize_intermediary_policy(
        params_base=params_base,
        m_grid=m_grid if m_grid is not None else np.linspace(0, 3, 31),
        policies=policies if policies is not None else ['identified', 'anonymized'],
        num_mc_samples=num_mc_samples,
        max_iter=max_iter,
        tol=tol,
        seed=params_base.get('seed'),  # ⭐ 传递seed确保可重复性
        verbose=verbose
    )
    
    # 第2步：提取最优策略
    m_star = optimal_policy.optimal_m
    anonymization_star = optimal_policy.optimal_anonymization
    optimal_result = optimal_policy.optimal_result
    
    print(f"\n{'='*70}")
    print(f"最优策略（博弈均衡）:")
    print(f"{'='*70}")
    print(f"  m* = {m_star:.4f}")
    print(f"  anonymization* = {anonymization_star}")
    print(f"  r* = {optimal_result.r_star:.4f}")
    print(f"  m_0* = {optimal_result.m_0:.4f}")
    print(f"  中介利润* = {optimal_result.intermediary_profit:.4f}")
    print(f"  社会福利* = {optimal_result.social_welfare:.4f}")
    
    # 第3步：生成最优策略下的示例数据（用于LLM评估）
    print(f"\n第2步：生成示例数据（用于LLM评估）...")
    params_optimal = ScenarioCParams(
        m=m_star,
        anonymization=anonymization_star,
        **params_base
    )
    
    # 生成示例world和participation
    rng_sample = np.random.default_rng(params_optimal.seed + 9999)
    sample_data = generate_consumer_data(params_optimal, rng=rng_sample)
    
    # 使用与optimal_result一致的participation_rule
    delta_u = optimal_result.delta_u
    if params_optimal.tau_dist == "normal":
        tau_samples = rng_sample.normal(params_optimal.tau_mean, params_optimal.tau_std, params_optimal.N)
        sample_participation = tau_samples <= delta_u
    elif params_optimal.tau_dist == "uniform":
        low = params_optimal.tau_mean - np.sqrt(3) * params_optimal.tau_std
        high = params_optimal.tau_mean + np.sqrt(3) * params_optimal.tau_std
        tau_samples = rng_sample.uniform(low, high, params_optimal.N)
        sample_participation = tau_samples <= delta_u
    else:
        # 回退到Bernoulli
        sample_participation = rng_sample.random(params_optimal.N) < optimal_result.r_star
    
    # 第4步：构建完整输出
    result = {
        # ⭐ 最优策略（博弈的均衡）
        "optimal_strategy": {
            "m_star": float(m_star),
            "anonymization_star": anonymization_star,
            "intermediary_profit_star": float(optimal_result.intermediary_profit),
            "r_star": float(optimal_result.r_star),
            "delta_u_star": float(optimal_result.delta_u) if optimal_result.delta_u is not None else None,
            "m_0_star": float(optimal_result.m_0)
        },
        
        # 最优策略下的市场均衡结果
        "equilibrium": {
            "consumer_surplus": float(optimal_result.consumer_surplus),
            "producer_profit": float(optimal_result.producer_profit_with_data),  # 有数据的生产者利润
            "intermediary_profit": float(optimal_result.intermediary_profit),
            "social_welfare": float(optimal_result.social_welfare),
            "gini_coefficient": float(optimal_result.gini_coefficient),
            "price_discrimination_index": float(optimal_result.price_discrimination_index)
        },
        
        # 数据交易信息
        "data_transaction": {
            "m_0": float(optimal_result.m_0),
            "producer_profit_with_data": float(optimal_result.producer_profit_with_data),
            "producer_profit_no_data": float(optimal_result.producer_profit_no_data),
            "producer_profit_gain": float(optimal_result.producer_profit_gain),
            "expected_num_participants": float(optimal_result.num_participants),
            "intermediary_cost": float(optimal_result.intermediary_cost)
        },
        
        # 所有候选策略（用于分析）
        "all_candidates": [
            {
                "m": float(r.m),
                "anonymization": r.anonymization,
                "intermediary_profit": float(r.intermediary_profit),
                "r_star": float(r.r_star),
                "m_0": float(r.m_0),
                "social_welfare": float(r.social_welfare),
                "consumer_surplus": float(r.consumer_surplus),
                "producer_profit": float(r.producer_profit_with_data)
            }
            for r in optimal_policy.all_results
        ],
        
        # 示例数据（用于LLM评估）
        "sample_data": {
            "w": sample_data.w.tolist(),
            "s": sample_data.s.tolist(),
            "theta": float(sample_data.theta) if sample_data.theta is not None else None,
            "epsilon": float(sample_data.epsilon) if sample_data.epsilon is not None else None
        },
        "sample_participation": sample_participation.tolist(),
        
        # 基础参数
        "params_base": params_base,
        
        # 元数据
        "metadata": {
            "generation_method": "optimize_intermediary_policy",
            "is_optimal_strategy": True,
            "optimization_grid": {
                "m_min": float(m_grid[0] if m_grid is not None else 0.0),
                "m_max": float(m_grid[-1] if m_grid is not None else 3.0),
                "m_steps": int(len(m_grid) if m_grid is not None else 31),
                "policies": policies if policies is not None else ['identified', 'anonymized']
            }
        }
    }
    
    print(f"\n{'='*70}")
    print(f"Ground Truth 生成完成！")
    print(f"{'='*70}")
    
    return result


# ============================================================================
# 中介最优化（Intermediary Optimization）
# ============================================================================
# 
# 本节实现论文完整的三层博弈框架的外层：中介最优化
# 
# 对应论文：
# - Section 5.2-5.3: 中介的最优信息设计
# - Proposition 2: 何时选择匿名化
# - Theorem 1: 最优补偿水平的刻画
# ============================================================================

@dataclass
class IntermediaryOptimizationResult:
    """
    中介优化结果
    
    记录给定策略组合(m, anonymization)下的完整市场均衡
    """
    # 策略参数
    m: float                      # 向消费者支付的数据补偿
    anonymization: str            # 匿名化策略：'identified' or 'anonymized'
    
    # 内层均衡：消费者反应
    r_star: float                 # 均衡参与率（固定点）
    delta_u: float                # 期望效用差（参与 vs 拒绝）
    num_participants: int         # 参与人数（实现值）
    
    # 中层均衡：生产者反应
    producer_profit_with_data: float    # 生产者利润（有数据）
    producer_profit_no_data: float      # 生产者利润（无数据，baseline）
    producer_profit_gain: float         # 数据带来的利润增益
    
    # 外层：中介利润
    m_0: float                    # 生产者向中介支付的费用（收入）
    intermediary_cost: float      # 中介向消费者支付的总成本
    intermediary_profit: float    # 中介净利润 R = m_0 - cost
    
    # 福利指标
    consumer_surplus: float
    social_welfare: float
    gini_coefficient: float
    price_discrimination_index: float


@dataclass
class OptimalPolicy:
    """
    中介的最优策略
    
    通过遍历所有候选策略，选择使中介利润最大化的策略
    """
    # 最优策略
    optimal_m: float
    optimal_anonymization: str
    
    # 最优策略下的均衡
    optimal_result: IntermediaryOptimizationResult
    
    # 优化过程
    all_results: List[IntermediaryOptimizationResult]  # 所有候选策略的结果
    optimization_summary: Dict  # 优化摘要统计


def simulate_market_outcome_no_data(
    data: ConsumerData,
    params: ScenarioCParams,
    seed: Optional[int] = None
) -> MarketOutcome:
    """
    模拟"无数据"情况下的市场结果（Counterfactual Baseline）
    
    场景：中介不存在，生产者只能依赖先验信息定价
    
    信息结构：
      - 生产者后验 = 先验：μ_producer[i] = μ_θ for all i
      - 消费者后验 = 只基于自己信号（无他人数据）：μ_consumer[i] = E[w_i | s_i]
      
    定价策略：
      - 必然是统一定价（因为生产者无法区分个体）
      - p* = argmax Σ(p-c)·max(μ_θ-p, 0)
      
    用途：
      - 计算生产者从数据中获得的利润增益
      - 确定生产者对数据的支付意愿 m_0
      
    对应论文：
      - Section 2.3: Producer problem
      - 用于计算数据的价值增量
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = params.N
    
    # 生产者后验 = 先验（无数据学习）
    mu_producer = np.full(N, params.mu_theta)
    
    # 消费者后验 = 基于私人信号的贝叶斯更新
    if params.data_structure == "common_preferences":
        # E[θ | s_i] 的贝叶斯更新
        prior_precision = 1.0 / (params.sigma_theta ** 2)
        signal_precision = 1.0 / (params.sigma ** 2)
        posterior_precision = prior_precision + signal_precision
        
        mu_consumer = (
            (prior_precision * params.mu_theta + signal_precision * data.s)
            / posterior_precision
        )
    
    elif params.data_structure == "common_experience":
        # 无他人数据，无法识别共同噪声ε，只能用先验
        mu_consumer = np.full(N, params.mu_theta)
    
    else:
        raise ValueError(f"Unknown data_structure: {params.data_structure}")
    
    # 生产者统一定价（无个体信息）
    p_optimal, _ = compute_optimal_price_uniform(mu_producer, params.c)
    prices = np.full(N, p_optimal)
    
    # 消费者购买决策
    quantities = np.maximum(mu_consumer - prices, 0)
    
    # 效用与利润实现
    utilities = data.w * quantities - prices * quantities - 0.5 * quantities ** 2
    producer_profit = np.sum((prices - params.c) * quantities)
    consumer_surplus = np.sum(utilities)
    intermediary_profit = 0.0
    social_welfare = consumer_surplus + producer_profit + intermediary_profit
    
    # 不平等指标（统一定价）
    gini_coefficient = 0.0
    price_discrimination_index = 0.0
    
    # 返回市场结果
    participation = np.zeros(N, dtype=bool)
    
    return MarketOutcome(
        participation=participation,
        participation_rate=0.0,
        num_participants=0,
        prices=prices,
        quantities=quantities,
        mu_consumers=mu_consumer,
        mu_producer=mu_producer,
        utilities=utilities,
        consumer_surplus=consumer_surplus,
        producer_profit=producer_profit,
        intermediary_profit=intermediary_profit,
        social_welfare=social_welfare,
        learning_quality_participants=0.0,
        learning_quality_rejecters=0.0,
        gini_coefficient=gini_coefficient,
        acceptor_avg_utility=0.0,
        rejecter_avg_utility=np.mean(utilities),
        price_variance=0.0,
        price_discrimination_index=price_discrimination_index
    )


def estimate_m0_mc(
    params: ScenarioCParams,
    participation_rule: Callable,
    T: int = 200,
    beta: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    使用Monte Carlo方法估计数据信息价值m_0（Ex-Ante期望）
    
    ========================================================================
    理论基础（对应论文机制设计框架）
    ========================================================================
    
    数据中介向生产者收取的费用m_0等于"生产者从中介信息中获得的
    期望利润增量（可提取部分）"：
    
        m_0 = β × max(0, E[π_with_data] - E[π_no_data])
    
    其中：
    - π_with_data: 生产者在获得中介数据Y_0后的产品市场利润
    - π_no_data: 生产者无中介数据（Y_0=∅）时的基准利润
    - β ∈ [0,1]: 中介可提取比例（默认1.0，即提取全部增量）
    - E[·]: Ex-Ante期望（在世界状态和参与实现上平均）
    
    ========================================================================
    关键原则（Common Random Numbers）
    ========================================================================
    
    为了确保m_0度量的是**纯信息价值**，必须遵守：
    
    1. **同一个world state**:
       - with和no使用相同的(w, s, τ)实现
       - 否则差分会混入"世界状态差异"
    
    2. **同一个participation**:
       - with和no使用相同的参与集合A
       - 否则差分会混入"参与变化效应"
    
    3. **Ex-Ante期望**:
       - 必须用MC平均（T次）估计期望
       - 单次realization不稳定，不能作为理论m_0
    
    这确保：Δπ = π_with(w,A) - π_no(w,A) 只反映信息差异
    
    ========================================================================
    Args:
        params: ScenarioCParams - 所有模型参数
        
        participation_rule: Callable - 参与决策规则
                           签名：(params, world_data, rng) -> participation
                           输入：参数、世界状态、随机数生成器
                           输出：(N,) bool数组，True表示参与
                           
                           常见实现：
                           - Bernoulli(r*): lambda p,w,rng: rng.random(N) < r_star
                           - Threshold(τ): lambda p,w,rng: w.tau <= delta_u
        
        T: int - Monte Carlo样本数（默认200）
               理论求解器建议≥100以确保稳定估计
        
        beta: float - 中介可提取比例（默认1.0）
                     = 1.0: 提取全部生产者剩余（中介完全议价能力）
                     < 1.0: 部分提取（中介有竞争或监管约束）
        
        seed: Optional[int] - 随机种子（确保可复现）
    
    Returns:
        (m_0, delta_mean, delta_std): Tuple[float, float, float]
        
        m_0: 数据信息价值（中介可收取的费用）
             = β × max(0, delta_mean)
        
        delta_mean: 利润增量的期望（可能为负）
                   = E[π_with - π_no]
        
        delta_std: 利润增量的标准差（衡量不确定性）
                  = std(π_with - π_no)
    
    ========================================================================
    实现细节
    ========================================================================
    
    伪代码：
    for t = 1 to T:
        1. 生成同一份世界状态 (w, s, τ)
        2. 在该状态下生成参与集合 A
        3. 计算 π_with(w, A, Y_0)  # 生产者有中介数据
        4. 计算 π_no(w, A, Y_0=∅) # 生产者无中介数据
        5. 记录 Δπ_t = π_with - π_no
    
    输出:
        m_0 = β × max(0, mean(Δπ))
        delta_mean = mean(Δπ)
        delta_std = std(Δπ)
    
    ========================================================================
    与旧实现的区别
    ========================================================================
    
    旧方法（不正确）：
    - 生成不同world，分别计算π_with和π_no
    - 用单次或少量realization
    - 无法分离信息价值 vs 随机波动
    
    新方法（正确）：
    - 同一world，同一participation，只改变信息
    - 大量MC平均（T=200）
    - 精确度量纯信息价值
    
    差异影响：
    - 旧方法：m_0不稳定，可能为负，难以解释
    - 新方法：m_0稳定，总≥0，经济含义清晰
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    deltas = []
    num_parts = []  # 记录每次的参与人数，用于计算E[#participants]
    
    for _ in range(T):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤1：生成世界状态（使用统一的rng流）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        world_data = generate_consumer_data(params, rng=rng)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2：在该世界状态下生成参与集合（同一个A）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        participation = participation_rule(params, world_data, rng)
        num_parts.append(int(np.sum(participation)))
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3：计算with-data利润（生产者有中介信息Y_0）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        outcome_with = simulate_market_outcome(
            world_data, participation, params,
            producer_info_mode="with_data", rng=rng
        )
        pi_with = outcome_with.producer_profit
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤4：计算no-data利润（生产者无中介信息，Y_0=∅）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 关键：使用同一个world_data和同一个participation
        outcome_no = simulate_market_outcome(
            world_data, participation, params,
            producer_info_mode="no_data", rng=rng
        )
        pi_no = outcome_no.producer_profit
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤5：记录利润差（纯信息价值）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        delta = pi_with - pi_no
        deltas.append(delta)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤6：计算Ex-Ante期望和标准差
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas, ddof=1)) if T > 1 else 0.0
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤7：计算m_0和期望参与人数
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # max(0, ·)：如果数据降低生产者利润，中介收取0（不卖数据）
    # β：中介的议价能力（β=1表示提取全部剩余）
    m_0 = beta * max(0.0, delta_mean)
    e_num_participants = float(np.mean(num_parts))  # 期望参与人数
    
    return m_0, delta_mean, delta_std, e_num_participants


def evaluate_intermediary_strategy(
    m: Union[float, np.ndarray],
    anonymization: str,
    params_base: Dict,
    num_mc_samples: int = 50,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None
) -> IntermediaryOptimizationResult:
    """
    评估给定策略(m, anonymization)下的完整市场均衡
    
    执行逆向归纳：
      1. 内层：求解消费者均衡 r*(m, anonymization)
      2. 中层：计算生产者利润 π*(r*, anonymization)
      3. 外层：计算中介利润 R = m_0 - m·r*·N
      
    参数：
      m: 补偿水平
      anonymization: 匿名化策略
      params_base: 基础市场参数（dict）
      num_mc_samples: Monte Carlo样本数
      max_iter: 固定点最大迭代次数
      tol: 收敛容差
      seed: 随机种子
      
    返回：
      IntermediaryOptimizationResult 包含完整均衡信息
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 构建完整参数
    params = ScenarioCParams(
        m=m,
        anonymization=anonymization,
        **params_base
    )
    
    # 内层：求解消费者均衡
    r_star, r_history, delta_u, delta_u_vector, p_vector = compute_rational_participation_rate(
        params,
        max_iter=max_iter,
        tol=tol,
        num_mc_samples=num_mc_samples,
        compute_per_consumer=False  # 生成equilibrium结果时使用代表性消费者
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 外层：计算生产者对数据的支付意愿 m_0（新方法）
    # 
    # 改进方案（GPT建议，理论严格）：
    #   m_0 = β × max(0, E[π_with_data] - E[π_no_data])
    # 
    # 关键原则（Common Random Numbers）：
    #   1. 同一个world state (w, s, τ)
    #   2. 同一个participation A
    #   3. 只改变生产者信息集 Y_0
    #   4. 用MC估计Ex-Ante期望（T=200）
    # 
    # 这确保m_0度量的是**纯信息价值**，不混入：
    #   - 世界状态差异
    #   - 参与变化效应
    #   - 单次realization的随机波动
    # 
    # 优点（相比旧方法）：
    #   - ✅ 理论严格（对应论文机制设计框架）
    #   - ✅ 稳定可靠（MC平均，不受单次抽样影响）
    #   - ✅ 经济含义清晰（纯信息价值）
    #   - ✅ 总是非负（max(0, ·)保证）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # 定义参与决策规则（基于τ阈值）
    def participation_rule(p: ScenarioCParams, world: ConsumerData, rng) -> np.ndarray:
        """
        根据τ_i阈值生成参与决策
        
        参与条件：τ_i ≤ ΔU（隐私成本 ≤ 参与净收益）
        """
        if p.tau_dist == "none":
            # 同质τ：全体同一决策
            return np.full(p.N, delta_u > 0, dtype=bool)
        elif p.tau_dist == "normal":
            # 正态分布τ
            tau_samples = rng.normal(p.tau_mean, p.tau_std, p.N)
            return tau_samples <= delta_u
        elif p.tau_dist == "uniform":
            # 均匀分布τ
            tau_low = p.tau_mean - np.sqrt(3) * p.tau_std
            tau_high = p.tau_mean + np.sqrt(3) * p.tau_std
            tau_samples = rng.uniform(tau_low, tau_high, p.N)
            return tau_samples <= delta_u
        else:
            raise ValueError(f"Unknown tau_dist: {p.tau_dist}")
    
    # 使用新方法估计m_0（Ex-Ante期望）
    m_0, delta_profit_mean, delta_profit_std, e_num_participants = estimate_m0_mc(
        params=params,
        participation_rule=participation_rule,
        T=200,  # MC样本数（理论建议≥100）
        beta=1.0,  # 中介提取全部剩余
        seed=seed
    )
    
    # 为了兼容性，也生成一次市场实现用于其他指标
    rng_sample = np.random.default_rng(seed)
    data = generate_consumer_data(params, rng=rng_sample)
    participation = participation_rule(params, data, rng_sample)
    num_participants = int(np.sum(participation))
    
    # 计算市场结果（用于消费者剩余、福利等指标）
    # ⭐ 注入估计的m_0
    outcome_with_data = simulate_market_outcome(
        data, participation, params,
        producer_info_mode="with_data",
        m0=m_0,
        rng=rng_sample
    )
    producer_profit_with_data = outcome_with_data.producer_profit
    
    # 计算无数据基准（用于记录，虽然m_0已经由MC估计）
    outcome_no_data = simulate_market_outcome(
        data, participation, params,
        producer_info_mode="no_data",
        m0=0.0,
        rng=rng_sample
    )
    producer_profit_no_data = outcome_no_data.producer_profit
    
    # 记录单次实现的利润增益（用于对比）
    producer_profit_gain_sample = producer_profit_with_data - producer_profit_no_data
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 计算中介利润（使用Ex-Ante期望）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 理论口径：
    #   R = m_0 - m × E[#participants]
    # 
    # 其中：
    #   m_0 = β × E[π_with - π_no]（由estimate_m0_mc计算）
    #   E[#participants] 由estimate_m0_mc返回
    # 
    # 使用期望参与数（而非单次实现）
    # ✅ 支持个性化补偿
    if isinstance(m, np.ndarray):
        # 个性化补偿：使用均值补偿作为近似
        intermediary_cost = float(np.mean(m)) * e_num_participants
    else:
        intermediary_cost = float(m) * e_num_participants
    intermediary_profit = float(m_0 - intermediary_cost)
    
    return IntermediaryOptimizationResult(
        m=float(m) if isinstance(m, (int, float, np.integer, np.floating)) else float(np.mean(m)),  # ✅ 向量时返回均值
        anonymization=anonymization,
        r_star=float(r_star),
        delta_u=float(delta_u),
        num_participants=int(num_participants),
        producer_profit_with_data=float(producer_profit_with_data),
        producer_profit_no_data=float(producer_profit_no_data),
        producer_profit_gain=float(producer_profit_gain_sample),  # 单次实现（用于对比）
        m_0=float(m_0),  # Ex-Ante期望（MC估计）
        intermediary_cost=float(intermediary_cost),
        intermediary_profit=float(intermediary_profit),
        consumer_surplus=float(outcome_with_data.consumer_surplus),
        social_welfare=float(outcome_with_data.social_welfare),
        gini_coefficient=float(outcome_with_data.gini_coefficient),
        price_discrimination_index=float(outcome_with_data.price_discrimination_index)
    )


def optimize_intermediary_policy(
    params_base: Dict,
    m_grid: np.ndarray = None,
    policies: List[str] = None,
    num_mc_samples: int = 50,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    verbose: bool = True
) -> OptimalPolicy:
    """
    求解中介的最优策略组合 (m*, anonymization*)
    
    通过网格搜索遍历所有候选策略，选择使中介利润最大化的策略
    
    对应论文：
      - Section 5.2-5.3: 中介的最优信息设计
      - 通过逆向归纳求解Stackelberg均衡
      
    参数：
      params_base: 基础市场参数（dict，不含m和anonymization）
      m_grid: 补偿候选值（默认：[0, 0.1, ..., 3.0]）
      policies: 匿名化策略候选（默认：['identified', 'anonymized']）
      num_mc_samples: Monte Carlo样本数
      max_iter: 固定点最大迭代次数
      tol: 收敛容差
      seed: 随机种子
      verbose: 是否打印优化过程
      
    返回：
      OptimalPolicy 包含最优策略及所有候选策略的评估结果
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 默认参数
    if m_grid is None:
        m_grid = np.linspace(0, 3.0, 31)  # [0, 0.1, 0.2, ..., 3.0]
    
    if policies is None:
        policies = ['identified', 'anonymized']
    
    if verbose:
        print("\n" + "="*80)
        print("🎯 中介最优策略求解（Intermediary Optimal Policy）")
        print("="*80)
        print(f"\n策略空间：{len(m_grid)} 个补偿候选 × {len(policies)} 个匿名化策略")
        print(f"总计：{len(m_grid) * len(policies)} 个候选策略")
        print(f"\n市场参数：")
        print(f"  - N = {params_base['N']}")
        print(f"  - 数据结构 = {params_base['data_structure']}")
        print(f"  - μ_θ = {params_base['mu_theta']:.2f}")
        print(f"  - σ_θ = {params_base['sigma_theta']:.2f}")
        print(f"  - σ = {params_base['sigma']:.2f}")
        print(f"  - tau分布 = {params_base.get('tau_dist', 'none')}")
        print("\n" + "-"*80)
        print(f"{'补偿m':>8} | {'策略':>12} | {'r*':>6} | {'m_0':>8} | {'成本':>8} | {'中介利润R':>10}")
        print("-"*80)
    
    # 遍历所有候选策略
    all_results = []
    skipped_count = 0
    
    for m in m_grid:
        for anonymization in policies:
            try:
                # 评估该策略
                result = evaluate_intermediary_strategy(
                    m=m,
                    anonymization=anonymization,
                    params_base=params_base,
                    num_mc_samples=num_mc_samples,
                    max_iter=max_iter,
                    tol=tol,
                    seed=seed
                )
                
                all_results.append(result)
                
                if verbose:
                    print(f"{m:8.2f} | {anonymization:>12} | "
                          f"{result.r_star:5.1%} | "
                          f"{result.m_0:8.2f} | "
                          f"{result.intermediary_cost:8.2f} | "
                          f"{result.intermediary_profit:10.2f}")
            
            except RuntimeError as e:
                # 捕获固定点不收敛错误，跳过该候选策略
                skipped_count += 1
                if verbose:
                    print(f"{m:8.2f} | {anonymization:>12} | {'SKIP':>6} | "
                          f"{'--':>8} | {'--':>8} | {'--':>10}  (不收敛)")
    
    # 检查是否有成功的候选策略
    if not all_results:
        raise RuntimeError(
            f"所有 {len(m_grid) * len(policies)} 个候选策略都未收敛！\n"
            f"建议：\n"
            f"  1. 增加 max_iter（当前：{max_iter}）\n"
            f"  2. 放宽 tol（当前：{tol}）\n"
            f"  3. 调整 m_grid 范围\n"
            f"  4. 增加 tau_std 以增加异质性"
        )
    
    if verbose and skipped_count > 0:
        print(f"\n⚠️  跳过 {skipped_count} 个不收敛的候选策略")
    
    # ============================================================
    # ✅ 新增：过滤亏损策略（理性参与约束）
    # 
    # 经济学原理：
    # - 理性中介不会选择R < 0的策略
    # - 如果所有策略都亏损，应该选择不参与市场（R = 0）
    # 
    # 论文依据：
    # - Proposition 4: "profitable intermediation"
    # - 隐含假设：中介具有outside option（不参与）
    # ============================================================
    profitable_results = [
        r for r in all_results 
        if r.intermediary_profit > 0.0  # 严格正利润
    ]
    
    if not profitable_results:
        # 所有策略都亏损 → 中介选择不参与市场
        if verbose:
            print("\n" + "="*80)
            print("⚠️  所有策略均亏损，中介选择不参与市场")
            print("="*80)
            max_loss = max(r.intermediary_profit for r in all_results)
            max_loss_result = max(all_results, key=lambda x: x.intermediary_profit)
            print(f"最小亏损: R = {max_loss:.4f}")
            print(f"  对应策略: m={max_loss_result.m:.2f}, {max_loss_result.anonymization}")
            print(f"理性选择: 不参与市场（outside option, R=0）")
            print("="*80)
        
        # 返回"不参与"策略
        # 创建零利润的dummy result
        dummy_result = IntermediaryOptimizationResult(
            m=0.0,
            anonymization="no_participation",
            r_star=0.0,
            delta_u=0.0,
            num_participants=0,
            producer_profit_with_data=0.0,
            producer_profit_no_data=0.0,
            producer_profit_gain=0.0,
            m_0=0.0,
            intermediary_cost=0.0,
            intermediary_profit=0.0,  # 不参与 = 零利润
            consumer_surplus=0.0,
            social_welfare=0.0,
            gini_coefficient=0.0,
            price_discrimination_index=0.0
        )
        
        optimization_summary = {
            'num_candidates_total': len(m_grid) * len(policies),
            'num_candidates_converged': len(all_results),
            'num_candidates_skipped': skipped_count,
            'num_candidates_profitable': 0,  # ✅ 新增字段
            'participation_feasible': False,  # ✅ 新增字段
            'max_profit': 0.0,
            'profit_range': [
                min(r.intermediary_profit for r in all_results),
                0.0  # 不参与是零利润
            ],
            'optimal_is_anonymized': False
        }
        
        return OptimalPolicy(
            optimal_m=0.0,
            optimal_anonymization="no_participation",
            optimal_result=dummy_result,
            all_results=all_results,
            optimization_summary=optimization_summary
        )
    
    # ✅ 从盈利策略中选择最优（而非所有策略）
    optimal_result = max(profitable_results, key=lambda x: x.intermediary_profit)
    
    if verbose:
        print("\n" + "="*80)
        print(f"✅ 共{len(profitable_results)}个盈利策略")
        print(f"❌ 淘汰{len(all_results) - len(profitable_results)}个亏损策略")
        print("="*80)
    
    if verbose:
        print("-"*80)
        print("\n🎯 最优策略：")
        print(f"  - 最优补偿：m* = {optimal_result.m:.2f}")
        print(f"  - 最优策略：{optimal_result.anonymization}")
        print(f"  - 均衡参与率：r* = {optimal_result.r_star:.1%}")
        print(f"  - 生产者支付：m_0 = {optimal_result.m_0:.2f}")
        print(f"  - 中介成本：{optimal_result.intermediary_cost:.2f}")
        print(f"  - 中介利润：R* = {optimal_result.intermediary_profit:.2f}")
        print(f"  - 社会福利：SW = {optimal_result.social_welfare:.2f}")
        print("="*80)
    
    # 优化摘要
    optimization_summary = {
        'num_candidates_total': len(m_grid) * len(policies),
        'num_candidates_converged': len(all_results),
        'num_candidates_skipped': skipped_count,
        'num_candidates_profitable': len(profitable_results),  # ✅ 新增
        'participation_feasible': True,  # ✅ 新增（此分支表示有盈利策略）
        'max_profit': optimal_result.intermediary_profit,
        'profit_range': [
            min(r.intermediary_profit for r in all_results),
            max(r.intermediary_profit for r in all_results)
        ],
        'optimal_is_anonymized': optimal_result.anonymization == 'anonymized'
    }
    
    return OptimalPolicy(
        optimal_m=optimal_result.m,
        optimal_anonymization=optimal_result.anonymization,
        optimal_result=optimal_result,
        all_results=all_results,
        optimization_summary=optimization_summary
    )


def verify_proposition_2(
    params_base: Dict,
    N_values: List[int] = None,
    m_fixed: float = 1.0,
    seed: Optional[int] = None
) -> Dict:
    """
    验证论文Proposition 2：市场规模对匿名化策略的影响
    
    命题：N足够大时，anonymized最优
    
    原因：
      - N大时，聚合数据仍能精确估计θ（大数定律）
      - anonymized降低消费者价格歧视担忧 → r*更高
      - 成本降低（或相同m下r*更高）→ R更高
      
    对应论文：
      - Section 5.2, Proposition 2
      
    返回：
      Dict包含不同N下的对比结果
    """
    if N_values is None:
        N_values = [10, 20, 50, 100]
    
    print("\n" + "="*80)
    print("📊 验证Proposition 2：市场规模 N 对匿名化策略的影响")
    print("="*80)
    print(f"\n固定补偿：m = {m_fixed:.2f}")
    print(f"对比策略：identified vs anonymized")
    print("\n" + "-"*80)
    print(f"{'N':>5} | {'策略':>12} | {'r*':>6} | {'m_0':>8} | {'R':>10} | {'SW':>10}")
    print("-"*80)
    
    results = {}
    
    for N in N_values:
        # 更新参数
        params_N = params_base.copy()
        params_N['N'] = N
        
        results[N] = {}
        
        for policy in ['identified', 'anonymized']:
            result = evaluate_intermediary_strategy(
                m=m_fixed,
                anonymization=policy,
                params_base=params_N,
                seed=seed
            )
            
            results[N][policy] = result
            
            print(f"{N:5d} | {policy:>12} | "
                  f"{result.r_star:5.1%} | "
                  f"{result.m_0:8.2f} | "
                  f"{result.intermediary_profit:10.2f} | "
                  f"{result.social_welfare:10.2f}")
    
    print("-"*80)
    print("\n分析：")
    
    for N in N_values:
        R_iden = results[N]['identified'].intermediary_profit
        R_anon = results[N]['anonymized'].intermediary_profit
        
        if R_anon > R_iden:
            print(f"  ✅ N={N:3d}: anonymized占优 "
                  f"(R_anon={R_anon:6.2f} > R_iden={R_iden:6.2f}, "
                  f"差距={R_anon-R_iden:+6.2f})")
        else:
            print(f"  ❌ N={N:3d}: identified占优 "
                  f"(R_iden={R_iden:6.2f} > R_anon={R_anon:6.2f}, "
                  f"差距={R_iden-R_anon:+6.2f})")
    
    print("="*80)
    
    return results


def analyze_optimal_compensation_curve(
    optimal_policy: OptimalPolicy,
    save_path: Optional[str] = None
) -> Dict:
    """
    分析最优补偿曲线：R(m), r*(m), m_0(m)
    
    可视化中介的trade-off：
      - 提高m → 提高r* → 提高m_0
      - 但成本也增加（m·r*·N）
      - 最优m*在边际收益 = 边际成本处
      
    对应论文：
      - Theorem 1: 最优补偿的一阶条件
    """
    results_by_policy = {}
    
    for policy in ['identified', 'anonymized']:
        policy_results = [r for r in optimal_policy.all_results 
                         if r.anonymization == policy]
        policy_results.sort(key=lambda x: x.m)
        
        results_by_policy[policy] = {
            'm': [r.m for r in policy_results],
            'r_star': [r.r_star for r in policy_results],
            'm_0': [r.m_0 for r in policy_results],
            'intermediary_profit': [r.intermediary_profit for r in policy_results],
            'social_welfare': [r.social_welfare for r in policy_results]
        }
    
    # 打印关键点
    print("\n" + "="*80)
    print("📈 最优补偿曲线分析")
    print("="*80)
    
    for policy in ['identified', 'anonymized']:
        data = results_by_policy[policy]
        max_profit_idx = np.argmax(data['intermediary_profit'])
        
        print(f"\n{policy.capitalize()}:")
        print(f"  最优补偿：m* = {data['m'][max_profit_idx]:.2f}")
        print(f"  最大利润：R* = {data['intermediary_profit'][max_profit_idx]:.2f}")
        print(f"  对应r*：{data['r_star'][max_profit_idx]:.1%}")
        print(f"  对应m_0：{data['m_0'][max_profit_idx]:.2f}")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results_by_policy, f, indent=2)
        print(f"\n💾 曲线数据已保存到：{save_path}")
    
    print("="*80)
    
    return results_by_policy


def export_optimization_results(
    optimal_policy: OptimalPolicy,
    output_path: str
):
    """
    导出优化结果到JSON文件
    """
    from dataclasses import asdict
    
    output = {
        'optimal_policy': {
            'm': optimal_policy.optimal_m,
            'anonymization': optimal_policy.optimal_anonymization,
            'result': asdict(optimal_policy.optimal_result)
        },
        'optimization_summary': optimal_policy.optimization_summary,
        'all_results': [asdict(r) for r in optimal_policy.all_results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 优化结果已保存到：{output_path}")


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    # 示例：生成最优Ground Truth（论文理论解）
    params_base = {
        'N': 20,
        'data_structure': 'common_preferences',
        # ⚠️ 不包含 m 和 anonymization，由中介优化求解
        'mu_theta': 5.0,
        'sigma_theta': 1.0,
        'sigma': 1.0,
        'tau_dist': 'normal',
        'tau_mean': 1.0,
        'tau_std': 0.3,
        'c': 0.0,
        'participation_timing': 'ex_ante',
        'seed': 42
    }
    
    # 生成Ground Truth（完整博弈均衡）
    gt = generate_ground_truth(
        params_base=params_base,
        max_iter=20,
        num_mc_samples=30
    )
    
    # 保存到文件
    from pathlib import Path
    output_path = "data/ground_truth/scenario_c_optimal.json"
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 最优Ground Truth已保存到: {output_path}")
    print(f"   最优策略: m*={gt['optimal_strategy']['m_star']:.2f}, "
          f"{gt['optimal_strategy']['anonymization_star']}")
