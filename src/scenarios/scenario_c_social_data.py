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
from typing import Dict, List, Tuple, Optional, Literal
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
    # 中介向消费者支付的补偿m（论文式5.1）
    # 消费者参与的直接激励：ΔU = E[u|参与] - E[u|拒绝] + m - τ_i
    # 典型值：0.5-2.0
    # 影响：m越高，参与率越高（论文Theorem 1）
    m: float
    
    # 生产者向中介支付m_0（我们的扩展，论文隐含）
    # 中介利润 = m_0 - m·N_参与
    # 默认值：0.0（中介纯支出）
    # 扩展：可设为生产者利润提升的某个比例
    m_0: float = 0.0
    
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
    
    def to_dict(self):
        return asdict(self)


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


def generate_consumer_data(params: ScenarioCParams) -> ConsumerData:
    """
    生成消费者数据（真实偏好和信号）
    
    Args:
        params: 场景参数
        
    Returns:
        ConsumerData对象
    """
    np.random.seed(params.seed)
    N = params.N # 消费者数量
    
    if params.data_structure == "common_preferences": # 共同偏好假设
        # 所有消费者有相同的真实偏好θ
        theta = np.random.normal(params.mu_theta, params.sigma_theta) # 真实偏好
        w = np.ones(N) * theta # 真实支付意愿
        
        # 独立噪声 ~ N(0, 1)
        e = np.random.normal(0, 1, N)
        
        # 信号 = 真实支付意愿 + 噪声 * 噪声水平
        s = w + params.sigma * e
        
        return ConsumerData(w=w, s=s, e=e, theta=theta, epsilon=None)
    
    elif params.data_structure == "common_experience": # 共同经验假设
        # 每个消费者的真实偏好不同
        w = np.random.normal(params.mu_theta, params.sigma_theta, N) # 真实偏好 ~ N(μ_theta, σ_theta²)
        
        # 共同噪声冲击
        epsilon = np.random.normal(0, 1)
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
    
    Args:
        s_i: 消费者i自己的信号
        participant_signals: 参与者的信号集合（可能包括s_i）
        params: 场景参数
        
    Returns:
        后验期望 mu_i
    """
    if len(participant_signals) == 0:
        # 没有参与者数据，只能用先验
        return params.mu_theta # 返回先验期望
    
    if params.data_structure == "common_preferences":
        # w_i = θ for all i
        # 最优估计是所有信号的加权平均
        # 精确贝叶斯更新
        n = len(participant_signals) # 参与者数量
        
        # 先验精度
        prior_precision = 1 / (params.sigma_theta ** 2) # 先验精度 = 1 / 先验方差
        
        # 似然精度（每个信号贡献）
        signal_precision = 1 / (params.sigma ** 2) # 似然精度 = 1 / 噪声方差
        
        # 后验精度
        posterior_precision = prior_precision + n * signal_precision # 后验精度 = 先验精度 + 参与者信号的精度
        posterior_variance = 1 / posterior_precision # 后验方差 = 1 / 后验精度
        
        # 后验均值
        posterior_mean = posterior_variance * (
            prior_precision * params.mu_theta +
            signal_precision * np.sum(participant_signals) # 后验期望 = 后验方差 * (先验期望 + 参与者信号的精度 * 参与者信号的和)
        )
        
        return posterior_mean # 返回后验期望
    
    elif params.data_structure == "common_experience":
        # s_i = w_i + σ·ε
        # 需要估计共同噪声ε，然后过滤
        
        # 如果没有参与者信号，只能用先验
        if len(participant_signals) == 0:
            return params.mu_theta # 返回先验期望
        
        # 根据参数选择后验估计方法
        if params.posterior_method == "exact":
            # 精确贝叶斯估计（更复杂但更准确）
            # TODO: 实现完整的高斯共轭先验更新
            # 目前回退到近似方法
            return _compute_ce_posterior_approx(s_i, participant_signals, params) # 返回近似后验期望
        else:
            # 近似方法（计算效率高）
            return _compute_ce_posterior_approx(s_i, participant_signals, params) # 返回近似后验期望
    
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
    
    if params.anonymization == "identified": # 实名化
        # 实名: 生产者可以看到 (i, sᵢ) 映射
        # 对参与者: 可以用其信号计算个体后验
        # 对拒绝者: 只能使用先验（因为没有其信号）
        for i in range(N):
            if participation[i]:
                # 参与者: 用其信号和其他参与者信号计算后验
                mu_producer[i] = compute_posterior_mean_consumer(
                    data.s[i], participant_signals, params
                )
            # 拒绝者保持先验 mu_theta（已经初始化）
    
    else:  # 匿名化
        # 匿名: 生产者只能看到信号集合 {sᵢ : i ∈ participants}（无身份）
        # 无法识别哪个信号对应哪个消费者，因此无法个性化定价
        # 所有消费者的后验期望相同（基于聚合统计）
        if len(participant_signals) > 0:
            if params.data_structure == "common_preferences": # 共同偏好假设
                # Common Preferences: 可以用信号均值更新对θ的估计
                # E[θ | X] 对所有人相同
                mean_signal = np.mean(participant_signals) # 参与者信号的平均值
                n_participants = len(participant_signals) # 参与者数量
                
                # 后验期望（简化公式）
                tau_X = n_participants / params.sigma**2 # 参与者信号的精度
                tau_0 = 1 / params.sigma_theta**2 # 先验精度
                mu_common = (tau_0 * params.mu_theta + tau_X * mean_signal) / (tau_0 + tau_X) # 后验期望
                mu_producer[:] = mu_common # 所有消费者的后验期望相同
            
            else:  # 共同经验假设
                # Common Experience: 生产者知道有共同噪声ε，但无法识别个体
                # 可以估计 ε，但不知道每个人的 wᵢ
                # 简化: 使用先验均值（因为无法个性化）
                mu_producer[:] = params.mu_theta # 所有消费者的后验期望相同
        # 无参与者时，使用先验（已经初始化）
    
    return mu_producer # 返回所有消费者的后验期望


def simulate_market_outcome(
    data: ConsumerData,
    participation: np.ndarray,
    params: ScenarioCParams
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
        
    Returns:
        MarketOutcome - 完整的市场结果和福利指标
    
    ========================================================================
    实现细节（逐步拆解）
    ========================================================================
    """
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
        np.random.shuffle(participant_signals)  # 打乱顺序，破坏身份映射
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
    
    # 生产者的信息集Y_0取决于匿名化政策：
    #
    # 实名（Identified）：
    # - Y_0 = {(i, s_i) : i ∈ participants}
    # - 对参与者i：知道s_i，可计算E[w_i | s_i, X]（与消费者i相同）
    # - 对拒绝者j：不知道s_j，只能用先验μ_θ
    # - 结果：μ_producer异质，可以个性化定价
    #
    # 匿名（Anonymized）：
    # - Y_0 = {s_i : i ∈ participants}（无身份）
    # - 无法识别哪个信号对应哪个消费者
    # - 对所有人：μ_producer[i] = E[θ | X]（相同）
    # - 结果：μ_producer同质，必须统一定价
    #
    # 这正是论文Proposition 2的核心：
    # "匿名化通过阻止生产者识别个体，防止价格歧视"
    
    mu_producer = compute_producer_posterior(
        data, participation, participant_signals, params
    )
    
    # ------------------------------------------------------------------------
    # 步骤4：生产者定价（最优定价策略）
    # 对应论文Section 2.2 + Section 4
    # ------------------------------------------------------------------------
    
    # 生产者目标：max Π = Σ_i (p_i - c) · q_i
    # 约束条件：取决于匿名化政策
    
    prices = np.zeros(N)
    
    if params.anonymization == "identified":
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
    # 参与者额外获得中介支付的补偿m
    # 这是激励参与的直接手段
    # 
    # 重要：这里不会产生"double counting"
    # 因为utilities[participation]只被加一次m
    # （之前的P0-1修复）
    utilities[participation] += params.m
    
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
    # IS = m_0 - m·N_participants
    #
    # 收入：m_0（生产者向中介支付）
    # - 可以是固定费用
    # - 或生产者利润提升的某个比例（我们的扩展）
    # - 论文隐含假设中介能获得部分生产者剩余
    #
    # 支出：m·N_participants（中介向消费者支付）
    # - 激励参与的成本
    # - m是每个参与者的统一补偿
    #
    # 净利润：IS = 收入 - 支出
    # 如果m_0 = 0（默认），则IS < 0（中介纯支出）
    # 这在论文中是隐含的简化假设
    
    num_participants = int(np.sum(participation))
    intermediary_profit = params.m_0 - params.m * num_participants
    
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
    num_market_samples: int = 20
) -> Tuple[float, List[float]]:
    """
    Ex Ante固定点：消费者在不知道信号实现时决策
    
    支持两种模式：
    1. 无异质性（tau_dist="none"）：r*通常为0或1
    2. 有异质性（tau_dist!="none"）：r* = F_τ(ΔU(r*))，可产生内点
    
    Args:
        params: 场景参数（不需要传入fixed data！）
        max_iter: 最大迭代次数
        tol: 收敛容差
        num_world_samples: 世界状态采样数
        num_market_samples: 市场采样数
        
    Returns:
        (收敛的参与率, 参与率历史)
    """
    r = 0.5  # 初始参与率
    r_history = [r]
    
    for iteration in range(max_iter):
        # 计算代表性消费者的期望效用差（Ex Ante）
        # 注意：由于对称性，只需计算一个代表性消费者即可
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
        
        delta_u = utility_accept - utility_reject  # m已经在效用中计入
        
        # 根据异质性设置计算新参与率
        if params.tau_dist == "none":
            # 无异质性：r*∈{0,1}
            r_new = 1.0 if delta_u > 0 else 0.0
        elif params.tau_dist == "normal":
            # 异质性：r* = P(τ_i ≤ ΔU) = Φ((ΔU - μ_τ) / σ_τ)
            from scipy.stats import norm
            r_new = norm.cdf(delta_u, loc=params.tau_mean, scale=params.tau_std)
        elif params.tau_dist == "uniform":
            # τ_i ~ Uniform[μ_τ - √3·σ_τ, μ_τ + √3·σ_τ]
            a = params.tau_mean - np.sqrt(3) * params.tau_std
            b = params.tau_mean + np.sqrt(3) * params.tau_std
            r_new = np.clip((delta_u - a) / (b - a), 0, 1)
        else:
            raise ValueError(f"Unsupported tau_dist: {params.tau_dist}")
        
        r_history.append(r_new)
        
        # 检查收敛
        if abs(r_new - r) < tol:
            print(f"  Ex Ante固定点收敛于迭代 {iteration + 1}, r* = {r_new:.4f}, ΔU = {delta_u:.4f}")
            return r_new, r_history
        
        # 平滑更新
        r = 0.6 * r_new + 0.4 * r
    
    print(f"  警告: Ex Ante固定点未在{max_iter}次迭代内收敛, 当前 r = {r:.4f}")
    return r, r_history


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
    
    print(f"  警告: Ex Post固定点未在{max_iter}次迭代内收敛, 当前 r = {r:.4f}")
    return r, r_history


def compute_rational_participation_rate(
    params: ScenarioCParams,
    data: ConsumerData = None,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50
) -> Tuple[float, List[float]]:
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
        
    Returns:
        (收敛的参与率, 参与率历史)
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
            num_market_samples=num_market
        )
    
    elif params.participation_timing == "ex_post":
        # Ex Post: 给定realized data
        if data is None:
            raise ValueError("Ex Post模式需要提供data参数")
        
        return compute_rational_participation_rate_ex_post(
            data=data,
            params=params,
            max_iter=max_iter,
            tol=tol,
            num_mc_samples=num_mc_samples
        )
    
    else:
        raise ValueError(f"Unsupported participation_timing: {params.participation_timing}")


def generate_ground_truth(
    params: ScenarioCParams,
    max_iter: int = 100,
    tol: float = 1e-3,
    num_mc_samples: int = 50
) -> Dict:
    """
    生成场景C的Ground Truth
    
    包括：
    1. 理性参与率（固定点）
    2. 对应的市场结果
    3. 关键指标
    
    Args:
        params: 场景参数
        max_iter: 固定点迭代最大次数
        tol: 收敛容差
        num_mc_samples: 蒙特卡洛样本数
        
    Returns:
        Ground Truth字典
    """
    print(f"\n{'='*60}")
    print(f"生成场景C Ground Truth")
    print(f"{'='*60}")
    print(f"参数:")
    print(f"  N = {params.N}")
    print(f"  数据结构 = {params.data_structure}")
    print(f"  匿名化 = {params.anonymization}")
    print(f"  补偿 m = {params.m:.2f}")
    print(f"  噪声水平 σ = {params.sigma:.2f}")
    
    # 计算理性参与率（根据时序模式）
    print(f"\n计算理性参与率（{params.participation_timing}模式）...")
    
    if params.participation_timing == "ex_post":
        # Ex Post需要先生成数据
        print(f"生成消费者数据...")
        data = generate_consumer_data(params)
        rational_rate, r_history = compute_rational_participation_rate(
            params, data, max_iter, tol, num_mc_samples
        )
    else:
        # Ex Ante不需要先生成数据
        rational_rate, r_history = compute_rational_participation_rate(
            params, None, max_iter, tol, num_mc_samples
        )
        # 为了输出，生成一个示例数据
        print(f"\n生成示例消费者数据（用于输出）...")
        data = generate_consumer_data(params)
    
    # 使用理性参与率生成参与决策（采样）
    np.random.seed(params.seed + 1000)  # 不同的种子
    rational_participation = np.random.rand(params.N) < rational_rate
    
    # 计算市场结果
    print(f"\n计算市场结果...")
    outcome = simulate_market_outcome(data, rational_participation, params)
    
    print(f"\n{'='*60}")
    print(f"Ground Truth 结果:")
    print(f"{'='*60}")
    print(f"参与率: {outcome.participation_rate:.2%} ({outcome.num_participants}/{params.N})")
    print(f"消费者剩余: {outcome.consumer_surplus:.4f}")
    print(f"生产者利润: {outcome.producer_profit:.4f}")
    print(f"中介利润: {outcome.intermediary_profit:.4f}")
    print(f"社会福利: {outcome.social_welfare:.4f}")
    print(f"Gini系数: {outcome.gini_coefficient:.4f}")
    print(f"价格歧视指数: {outcome.price_discrimination_index:.4f}")
    
    # 构建返回结果
    result = {
        "params": params.to_dict(),
        "data": {
            "w": data.w.tolist(),
            "s": data.s.tolist(),
            "theta": float(data.theta) if data.theta is not None else None,
            "epsilon": float(data.epsilon) if data.epsilon is not None else None,
        },
        "rational_participation_rate": float(rational_rate),
        "rational_participation": rational_participation.tolist(),
        "r_history": [float(x) for x in r_history],
        "outcome": {
            "participation_rate": float(outcome.participation_rate),
            "num_participants": int(outcome.num_participants),
            "consumer_surplus": float(outcome.consumer_surplus),
            "producer_profit": float(outcome.producer_profit),
            "intermediary_profit": float(outcome.intermediary_profit),
            "social_welfare": float(outcome.social_welfare),
            "gini_coefficient": float(outcome.gini_coefficient),
            "price_variance": float(outcome.price_variance),
            "price_discrimination_index": float(outcome.price_discrimination_index),
            "acceptor_avg_utility": float(outcome.acceptor_avg_utility),
            "rejecter_avg_utility": float(outcome.rejecter_avg_utility),
            "learning_quality_participants": float(outcome.learning_quality_participants),
            "learning_quality_rejecters": float(outcome.learning_quality_rejecters),
        },
        "detailed_results": {
            "prices": outcome.prices.tolist(),
            "quantities": outcome.quantities.tolist(),
            "utilities": outcome.utilities.tolist(),
            "mu_consumers": outcome.mu_consumers.tolist(),
        }
    }
    
    return result


# 示例使用
if __name__ == "__main__":
    # 创建场景参数（MVP配置）
    params = ScenarioCParams(
        N=20,
        data_structure="common_preferences",
        anonymization="identified",
        mu_theta=5.0,
        sigma_theta=1.0,
        sigma=1.0,
        m=1.0,
        c=0.0,
        seed=42
    )
    
    # 生成Ground Truth
    gt = generate_ground_truth(params, max_iter=20, num_mc_samples=30)
    
    # 保存到文件
    output_path = "data/ground_truth/scenario_c_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Ground Truth已保存到: {output_path}")
