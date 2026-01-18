## 0. 你们 benchmark 的定位（与 TMD 一致）

*   LLM 是博弈参与者：这里 **LLM 作为消费者**，做“是否参与（是否把自己的信号交给中介）”的选择；（可选扩展：LLM 作为中介选择匿名/实名；LLM 作为生产者定价）
*   理论对照（ground truth）：用同一环境下的“理性参与规则/机制设计近似解”作为基线
*   评估：参与率偏离、价格歧视强度、消费者剩余/生产者利润/中介利润、以及不平等

---

## 1. 环境与符号（你实现时必须固定的数学对象）

### 1.1 产品市场（必须按论文的线性需求形式）

对每个消费者 \(i=1,\dots,N\)：

*   真实偏好（愿付）：\(w_i\)
*   生产者设单位价格：\(p_i\)
*   消费者选数量：\(q_i\ge 0\)

效用：
\[
u_i = w_i q_i - p_i q_i - \frac{1}{2}q_i^2
\]
最优需求（标准一阶条件 + 截断）：
\[
q_i^*(\mathbb{E}[w_i\mid \mathcal{I}_i],p_i)=\max\{\mathbb{E}[w_i\mid \mathcal{I}_i]-p_i,\ 0\}
\]
这里 \(\mathcal{I}_i\) 是消费者在购买时拥有的信息集合（由中介的数据政策决定）。

生产者利润（设边际成本 \(c\ge 0\)，benchmark 常用 \(c=0\)）：
\[
\pi = \sum_{i=1}^N (p_i-c)\ q_i
\]

> 重要：在这个线性—二次模型里，生产者的最优定价通常等价于“对每个消费者，按其条件期望 \(\mu_i=\mathbb{E}[w_i\mid \mathcal{I}_0]\) 做一个线性定价规则”。你实现时不必推闭式公式也能跑：可以用蒙特卡洛/网格搜索在每个信息结构下求最优 \(p_i\)。

---

### 1.2 数据环境（社会数据：跨个体相关是核心）

每个消费者有噪声信号：
\[
s_i = w_i + \sigma e_i,\quad \sigma>0
\]
并允许 \(w=(w_1,\dots,w_N)\) 与 \(e=(e_1,\dots,e_N)\) 在个体间存在相关结构。论文强调两类典型结构来展示社会数据的不同“可学习性”。

你至少实现两类（这能让 benchmark 很“论文味”）：

*   **Common Preferences（共同偏好）**
    \[
    w_i = \theta \ \forall i,\quad e_i \ \text{i.i.d.}
    \Rightarrow s_i=\theta+\sigma e_i
    \]
    多人数据能滤掉噪声，识别共同 \(\theta\)。

*   **Common Experience（共同经历/共同冲击）**
    \[
    w_i \ \text{i.i.d.},\quad e_i = \varepsilon\ \forall i
    \Rightarrow s_i=w_i+\sigma \varepsilon
    \]
    多人数据能识别共同噪声项，从而更准确估计每个 \(w_i\)。

---

## 2. 合同、数据政策与信息流（实现中最容易混乱的部分）

论文把中介（intermediary）看作信息设计者：决定“收什么、怎么匿名、发给谁”。你的 benchmark 要把它**程序化**成两个政策：

### 2.1 流入政策 \(X\)：实名 vs 匿名（关键变量）

消费者若“参与”，中介收集其信号 \(s_i\)。

*   实名（Identified）\(X=S\)：中介保存映射 \((i\mapsto s_i)\)
*   匿名（Anonymized）\(X=A\)：中介对索引做随机置换 \(\delta\)，仅保存 \((\delta(i)\mapsto s_i)\)

实现上，匿名化就是：**打乱 identity-key，但不丢掉数值信息**。

### 2.2 流出政策 \(Y\)：发给生产者与消费者什么信息

论文在均衡分析中强调一个关键性质：中介在最优流出下，会让生产者与消费者获得一致的信息（消费者至少不比生产者更“无知”）。

为了让 benchmark 清晰、可落地，建议你从“固定最强流出政策”开始：

**Baseline（固定）流出：完全披露数据库 \(X\)**

*   给生产者：\(Y_0 = X\)
*   给每个消费者：\(Y_i = X\)（包括拒绝者也能得到“别人”的数据，形成搭便车外部性）

也就是你程序里实现：
\[
Y_0(X)=Y_i(X)=X,\quad \forall i
\]

这一步非常关键：它把复杂性集中到了消费者的参与决策上——消费者接受合同的边际影响，主要来自“生产者能否把我的信号用于对我定价”，而不是“我能不能学习”。这正是论文强调的社会数据楔子。

---

## 3. 博弈时序（你代码主循环应严格等同这个）

我给你一个“单轮静态博弈”的时序（benchmark 重复多 seed）：

1.  参数初始化：生成 \(w,e,s\) 或至少生成 \(s\)（\(w\) 仅用于结算）
2.  中介发布消费者合同：\((m_i, X)\)（先把 \(X\) 当作 exogenous 机制变量更清晰）
3.  消费者同时决定参与 \(a_i\in\{0,1\}\)（这里由 LLM 给出）
4.  中介收集参与者信号集合并执行匿名/实名映射，形成数据库 \(X\)
5.  中介发布生产者合同并执行流出：\(Y_0, Y_i\)
6.  生产者基于 \(Y_0\) 设价（可个性化：得到的是每个“身份/匿名桶”的信息）
7.  每个消费者基于 \(\mathcal{I}_i=\{s_i\ \text{(自己可见)}, Y_i\}\) 形成 \(\mu_i=\mathbb{E}[w_i\mid \mathcal{I}_i]\)
8.  消费者计算 \(q_i^*=\max\{\mu_i-p_i,\ 0\}\)，得到效用 \(u_i\)
9.  结算：消费者剩余、生产者利润、中介利润、不平等指标

> 注意：消费者是否参与，会影响生产者的 \(\mathcal{I}_0\)（从而影响 \(p_i\)），也会影响所有消费者的学习质量（通过 \(Y_i\)）。这就是“社会数据外部性”。

---

## 4. “理性”基线如何算（否则你无法定义偏离）

你说你要 benchmark，核心就是要有 ground truth。这里给你两层可实现的“理论基线”，从容易到难。

### 4.1 基线 A（推荐你先做）：外生支付 \(m_i\)，理性消费者做 best response

你把 \(m_i\) 当作实验变量（sweep），然后用理性规则判定是否参与。

理性消费者参与条件：
\[
\text{Accept iff } \mathbb{E}[u_i \mid a_i=1,a_{-i}] + m_i \ \ge\ \mathbb{E}[u_i \mid a_i=0,a_{-i}]
\]

但消费者在做决策时并不知道 \(a_{-i}\)，所以你要定义“理性预期”：

*   最简单（与你们 TMD 的做法一致）：把 \(a_{-i}\) 视为 i.i.d. 参与概率 \(r\)，求一个固定点
    \[
    r = \Pr(\text{Accept} \mid r)
    \]
    实现时用迭代：
    *   初始化 \(r^{(0)}=0.5\)
    *   迭代：在参与概率 \(r^{(t)}\) 下，用蒙特卡洛估计左右两边的期望效用差 \(\Delta U(r^{(t)})\)，得出新的参与概率 \(r^{(t+1)}\)
    *   收敛后得到 \(r^*\) 作为理论参与率

这种“离散化 + 固定点”与你们 TMD 的“枚举/迭代”是同一路线，只是 payoff 计算换成了信息结构与定价模型。

### 4.2 基线 B（更贴论文但更难）：把 \(m_i\) 设为“边际补偿”

你用机制设计近似，让消费者在边际无差异：
\[
m_i = \mathbb{E}[u_i \mid a_i=0,a_{-i}] - \mathbb{E}[u_i \mid a_i=1,a_{-i}] + \varepsilon
\]
这会让理性消费者倾向接受（取 \(\varepsilon>0\) 小正数确保接受）。这个基线用于构造“理论上应高参与”的条件，以便检验 LLM 是否仍然拒绝（风险厌恶、对价格歧视恐惧等）。

---

## 5. 生产者定价：给你一个“可编码、无需推导闭式”的确定规则

你觉得不清晰，往往卡在“生产者怎么定价”。我给你一个确定可执行的数值规则，不需要推闭式。

### 5.1 生产者观察到的信息

*   若实名：生产者可能看到“每个 \(i\) 的信号或其统计量”
*   若匿名：生产者看到“信号集合但无法匹配到 \(i\)”，只能做市场级/分组定价（由你设定约束）

为了把 benchmark 做得清楚，建议你明确两种定价制度：

**(P1) 个性化定价（Identified pricing）**
生产者对每个消费者 \(i\) 单独选择 \(p_i\)，目标最大化期望利润：
\[
p_i^* \in \arg\max_{p\ge 0}\ \mathbb{E}\big[(p-c)\max(\mu_i-p,0)\mid Y_0\big]
\]
其中 \(\mu_i=\mathbb{E}[w_i\mid Y_0]\)。

**(P2) 统一定价（Market-level pricing）**（对应匿名化后“无法区分个体”）
\[
p^* \in \arg\max_{p\ge 0}\ \sum_{i=1}^N \mathbb{E}\big[(p-c)\max(\mu_i-p,0)\mid Y_0\big]
\]

### 5.2 数值实现（网格搜索/一维优化）

对每个 \(i\)（或对统一价），你都可以用网格搜索：

*   取价格网格 \(p\in\{0,\Delta,2\Delta,\dots,p_{\max}\}\)
*   用蒙特卡洛抽样估计 \(\mu_i\)（或直接由模型计算）
*   选使利润最大的 \(p\)

这使得你的 benchmark 不会因推导闭式公式而卡住，而且“信息结构变化导致最优价格变化”会自然体现在数值解里。

---

## 6. LLM 消费者的 prompt 该如何写（确保信息结构正确）

你们在 TMD 已经做到“LLM 不知道他人当期决策”，这里要同样硬约束：

LLM 消费者 \(i\) 在决定 \(a_i\) 时，只能知道：

*   自己的信号（或自己对 \(w_i\) 的粗估）
*   机制变量：\(N,\sigma,X\in\{A,S\}\)、以及支付 \(m_i\)
*   环境规则：若参与，生产者可能对你更精准定价；若不参与，你仍可能从他人数据中学习（baseline 里这是事实）

LLM 输出：

*   Accept / Reject
*   简短理由（必须写入日志，供 top-k 解释）

---

## 7. 你最终要输出哪些指标（benchmark 的“评分表”）

### 7.1 行为偏离

*   参与率偏离：
    \[
    \Delta PR = PR^{LLM} - PR^{Theory}
    \]

### 7.2 市场结果偏离

*   价格歧视强度：例如价格方差
    \[
    \mathrm{Var}(p_i)
    \]
*   消费者总剩余（可直接用效用和）：
    \[
    CS=\sum_i u_i
    \]
*   生产者利润 \(\pi\)
*   中介利润（若你把 \(m_0\) 外生，也可定义为“净现金流”）：
    \[
    R = m_0 - \sum_{i:a_i=1} m_i
    \]
*   社会福利（你可以选择是否把中介利润也计入；建议都报）：
    \[
    W = CS + \pi \quad (\text{或 } CS+\pi+R)
    \]

### 7.3 不平等与外部性分解

*   参与者 vs 拒绝者平均效用差
*   Gini（对 \(\{u_i\}\)）
*   Bottom-quantile welfare loss（如 bottom 20%）

---

## 8. 关键设计决策确定（Q1–Q4）

本节基于三个判定标准来确定实现细节：
1. **是否忠实于论文的机制与信息结构**
2. **是否有利于benchmark的可识别性**（把偏差归因到LLM的参与决策，而不是复杂性扩散）
3. **是否在实现上可控、可复现**

### 8.1 Q1：匿名化下生产者如何定价？

**决策：选择方案A作为Baseline**

**方案A（推荐）：完全统一价格**
```
if anonymization == "anonymized":
    # 所有消费者同一价格
    p* = argmax_p Σ_i E[(p-c)·max(μ_i-p, 0) | Y_0]
    
if anonymization == "identified":
    # 每个消费者个性化定价
    p_i* = argmax_p E[(p-c)·max(μ_i-p, 0) | Y_0^(i)]
```

**理由**：
1. **论文对应**：匿名化的本质是切断"信息→身份"映射，使生产者无法逐人定价。论文强调匿名化对应"market-level information"与"aggregate pricing"。
2. **可识别性**：统一价 vs 个性化价的对比最清晰，LLM消费者能直观理解"匿名=价格保护，实名=精准歧视"。
3. **实现简单**：避免了分桶机制的复杂性（桶的可观察性、匹配规则等）。

**可选扩展**：方案B（基于信号分桶定价）可作为鲁棒性检验，但需要额外设计桶机制。

---

### 8.2 Q2：消费者如何计算后验期望与需求？

**决策：Baseline用理性贝叶斯，LLM只负责参与决策**

**实现方案**：
```python
# 后验估计（独立模块，不涉及LLM）
μ_i = compute_posterior_mean(s_i, Y_i, data_structure)
# 使用贝叶斯更新或LMMSE（在高斯下等价）

# 需求决策（理性最优）
q_i* = max(μ_i - p_i, 0)

# LLM任务（唯一干预点）
a_i = llm_decide_participation(s_i, m_i, X, environment_info)
# 返回 "accept" 或 "reject"
```

**理由**：
1. **论文对应**：论文的重点是数据外部性如何影响市场结果，而非消费者的计算能力。消费者与生产者都被建模为理性决策者。
2. **可识别性**：保持需求决策理性，使偏差完全归因于"LLM是否理解参与的trade-off"，而非"LLM是否会算贝叶斯"。
3. **可复现性**：后验估计有确定性公式（在Common Preferences和Common Experience两种结构下都有闭式解），结果不依赖LLM的数值能力。

**后验估计公式**：
- **Common Preferences** \(w_i = \theta, e_i \sim \text{i.i.d.}\)：
  \[
  \mu_i = \mathbb{E}[\theta \mid s_i, Y_i] = \text{所有观测信号的加权平均}
  \]
  
- **Common Experience** \(w_i \sim \text{i.i.d.}, e_i = \varepsilon\)：
  \[
  \mu_i = \mathbb{E}[w_i \mid s_i, Y_i] = s_i - \hat{\sigma}\hat{\varepsilon}
  \]
  其中\(\hat{\varepsilon}\)是共同噪声的估计。

---

### 8.3 Q3：搭便车（拒绝者能否学习）如何建模？

**决策：选择方案A作为Baseline**

**方案A（推荐）：拒绝者也能获得数据库信息**
```python
# 信息流出（完全披露）
for i in range(N):
    Y_0 = X  # 生产者看到参与者数据库
    Y_i = X  # 消费者i也看到（包括拒绝者！）

# 拒绝者的学习
if consumer_i.rejected:
    μ_i = E[w_i | s_i, X]  # 仍能从他人数据中学习
    # 但不会被生产者用于定价（如果匿名化）
```

**理由**：
1. **论文对应**：论文强调在最优信息流出下，消费者至少与生产者一样知情。这产生强烈的"搭便车+外部性"张力，是论文要解释的核心现象之一。
2. **可识别性**：最大化"拒绝仍收益"的对比，使参与决策的trade-off更明显：
   - **参与的代价**：可能被价格歧视（实名下）
   - **拒绝的收益**：仍能学习（搭便车）
3. **外部性度量**：能够测量搭便车收益——拒绝者的\(\mu_i\)精度提升是否接近参与者。

**可选扩展**：方案C（只给聚合统计）可作为敏感性分析，调节搭便车强度。

---

### 8.4 Q4：中介支付合同是ex-ante还是ex-post？

**决策：选择方案A（ex-ante承诺）**

**方案A（推荐）：支付在决策前承诺**
```
时序：
1. 中介发布合同 (m_i, X)  ← 承诺支付金额
2. 消费者同时决定 a_i     ← LLM决策点
3. 参与者获得补偿 m_i      ← 兑现承诺
4. 后续市场交易...
```

**理由**：
1. **论文对应**：论文明确把合同设定为"ex ante、在需求冲击实现之前"的双边合同。
2. **可识别性**：决策点单一且清晰，LLM在明确信息下做决策。可以通过扫描\(m_i\)得到参与率曲线，便于与理论对照。
3. **实现简单**：避免了动态机制设计的复杂性（预期调整、重复博弈等）。

**可选扩展**：方案B（ex-post根据参与人数调整）会引入动态机制设计问题，可作为高级扩展，但不建议第一版。

---

## 9. Baseline配置锁定版

基于上述设计决策，确定第一版benchmark的配置：

### 9.1 核心配置

```python
BASELINE_CONFIG = {
    # 市场规模
    "N": 20,  # 中等市场（足够展示外部性，计算可控）
    
    # 数据结构（先实现一种）
    "data_structure": "common_preferences",
    # w_i = θ for all i, e_i ~ N(0,1) i.i.d.
    # s_i = θ + σ·e_i
    
    # 匿名化政策（核心对比维度）
    "anonymization": ["identified", "anonymized"],
    
    # 噪声水平
    "sigma": 1.0,  # 中等噪声
    
    # 支付水平（扫描）
    "m_i": [0, 0.5, 1.0, 2.0, 3.0, 5.0],
    
    # 定价机制（由匿名化政策决定）
    "pricing_mode": {
        "identified": "personalized",   # 每人一价
        "anonymized": "uniform"          # 统一价格
    },
    
    # 信息流出（固定）
    "information_disclosure": "full",  # Y_0 = Y_i = X
    
    # 合同时序
    "contract_timing": "ex_ante",  # 支付m_i事前承诺
    
    # 需求决策
    "demand_model": "rational_bayesian",  # q_i* = max(μ_i - p_i, 0)
    
    # 后验估计
    "posterior_estimator": "bayesian",  # 精确贝叶斯或LMMSE
    
    # 边际成本
    "marginal_cost": 0,  # c = 0（标准设定）
    
    # 随机种子
    "seeds": 10  # 每个配置10次重复
}
```

### 9.2 实验矩阵

#### Phase 1: MVP（最小可行版本）
- **目标**：验证框架可行性，得到初步结果
- **配置**：
  ```
  N = 20
  data_structure = "common_preferences"（一种）
  anonymization = ["identified", "anonymized"]（核心对比）
  sigma = 1.0（固定）
  m_i = [0, 0.5, 1.0, 2.0, 3.0]（5个点）
  seeds = 10
  
  总计：1 × 1 × 2 × 1 × 5 × 10 = 100 runs
  ```

#### Phase 2: 核心扩展
- **目标**：覆盖两种数据结构，探索噪声影响
- **配置**：
  ```
  N = 20
  data_structure = ["common_preferences", "common_experience"]（两种）
  anonymization = ["identified", "anonymized"]
  sigma = [0.5, 1.0, 2.0]（噪声强度）
  m_i = [0, 0.5, 1.0, 2.0, 3.0]
  seeds = 20
  
  总计：1 × 2 × 2 × 3 × 5 × 20 = 1200 runs
  ```

#### Phase 3: 完整benchmark
- **目标**：探索市场规模效应，更细粒度扫描
- **配置**：
  ```
  N = [10, 20, 50, 100]（市场规模效应）
  data_structure = ["common_preferences", "common_experience"]
  anonymization = ["identified", "anonymized"]
  sigma = [0.5, 1.0, 2.0]
  m_i = linspace(0, 5, 11)（11个点）
  seeds = 50
  
  可选择性运行关键配置组合
  ```

### 9.3 理性Baseline计算方法

**固定点迭代**（与场景B的TMD做法一致）：

```
输入：m_i, X, data_structure, sigma, N
输出：理论参与率 r*

算法：
1. 初始化 r^(0) = 0.5
2. for t = 1, 2, ..., max_iter:
   a) 在参与概率r^(t-1)下，蒙特卡洛估计：
      - E[u_i | a_i=1, r^(t-1)]  # 参与的期望效用
      - E[u_i | a_i=0, r^(t-1)]  # 拒绝的期望效用
   b) 计算效用差：
      Δu = E[u_i | a_i=1, r] - E[u_i | a_i=0, r]
   c) 更新参与概率：
      r^(t) = Pr(Δu + m_i > 0)
           = 1 if Δu + m_i > 0 else 0  # 确定性情况
           或使用分布（如果有异质性）
   d) 检查收敛：
      if |r^(t) - r^(t-1)| < tol:
          break
3. 返回 r* = r^(t)
```

**蒙特卡洛估计细节**：
- 对每个参与率r，抽样K个"他人参与向量" \(a_{-i}\)
- 每个向量中每人独立以概率r参与
- 对每个样本，计算消费者i在该配置下的效用
- 平均得到期望效用

### 9.4 评估指标完整列表

```python
METRICS = {
    # === 行为偏离 ===
    "participation_rate_llm": float,         # LLM实际参与率
    "participation_rate_theory": float,       # 理论参与率(固定点)
    "participation_deviation": float,         # PR_LLM - PR_theory
    "participation_deviation_abs": float,     # |PR_LLM - PR_theory|
    
    # === 市场结果 ===
    "price_mean": float,                      # 平均价格
    "price_variance": float,                  # 价格方差（歧视强度）
    "price_discrimination_index": float,      # max(p_i) - min(p_i)
    "consumer_surplus": float,                # Σ u_i
    "producer_profit": float,                 # π
    "intermediary_profit": float,             # m_0 - Σ m_i（如果m_0外生）
    "social_welfare": float,                  # CS + π
    "social_welfare_total": float,            # CS + π + R（含中介）
    
    # === 不平等与分配 ===
    "acceptor_avg_utility": float,            # 参与者平均效用
    "rejecter_avg_utility": float,            # 拒绝者平均效用
    "acceptor_rejecter_gap": float,           # 参与者 - 拒绝者效用差
    "gini_coefficient": float,                # 效用的Gini系数
    "bottom_20_welfare": float,               # 底部20%平均福利
    "top_20_welfare": float,                  # 顶部20%平均福利
    "welfare_ratio_top_bottom": float,        # 顶部/底部比率
    
    # === 外部性与学习 ===
    "free_rider_benefit": float,              # 拒绝者从Y_i学习的收益
    "learning_quality_acceptors": float,      # 参与者: mean |μ_i - w_i|
    "learning_quality_rejecters": float,      # 拒绝者: mean |μ_i - w_i|
    "learning_quality_ratio": float,          # 拒绝者/参与者学习质量比
    "data_externality_strength": float,       # 量化外部性强度
    
    # === 价格影响 ===
    "acceptor_avg_price": float,              # 参与者平均支付价格
    "rejecter_avg_price": float,              # 拒绝者平均支付价格
    "price_discrimination_cost": float,       # 参与者相比拒绝者多付的价格
    
    # === 额外分析 ===
    "num_participants": int,                  # 参与人数
    "participation_count": int,               # 同上（别名）
    "avg_quantity": float,                    # 平均购买量
    "market_coverage": float,                 # 购买者比例（q_i > 0）
}
```

### 9.5 数据结构参数设定

#### Common Preferences（共同偏好）
```python
# 所有消费者有相同的真实偏好θ，但观测有噪声
theta = np.random.normal(loc=5.0, scale=1.0)  # 共同偏好
w = np.ones(N) * theta                         # w_i = θ for all i
e = np.random.normal(loc=0, scale=1.0, size=N) # i.i.d. 噪声
s = w + sigma * e                              # 信号

# 参数说明：
# - theta ~ N(5, 1): 期望价值在5左右，标准差1
# - sigma: 噪声水平（可调：0.5, 1.0, 2.0）
# - 学习价值：多个信号可以滤掉e_i，更准确估计θ
```

#### Common Experience（共同经历）
```python
# 每个消费者有不同偏好，但受共同噪声冲击
w = np.random.normal(loc=5.0, scale=1.0, size=N)  # 异质偏好
epsilon = np.random.normal(loc=0, scale=1.0)      # 共同噪声
e = np.ones(N) * epsilon                          # e_i = ε for all i
s = w + sigma * e                                 # 信号

# 参数说明：
# - w_i ~ N(5, 1) i.i.d.: 真实偏好异质
# - epsilon ~ N(0, 1): 共同经历/测量误差
# - sigma: 噪声水平
# - 学习价值：多个信号可以识别共同ε，过滤后估计各自w_i
```

### 9.6 LLM Prompt模板（对应Q1-Q4决策）

```python
PROMPT_TEMPLATE = """
你是{N}个消费者中的消费者{consumer_id}。

【产品与市场】
市场上有一个产品，你对它的初步评估（信号）是：{signal:.2f}
（这个评估可能有噪声，真实价值范围通常在0-10之间）

【数据环境】
{data_structure_description}
- 共同偏好型：大家对产品的真实价值相近，但各自的初步评估有随机误差
- 共同经历型：每人的真实偏好不同，但都受到相同的市场信息噪声影响

当前环境：{data_structure_name}

【数据中介的提议】
一个数据中介愿意支付你 {payment:.2f} 元来获取你的信号数据。

数据政策：{anonymization_policy}
{policy_details}

【你需要权衡】

✓ 如果你接受（参与数据共享）：
  • 你会立即获得 {payment:.2f} 元补偿
  • 你可以看到其他参与者的数据，帮助你更准确判断产品价值
  • {participation_cost}

✗ 如果你拒绝：
  • 你不会获得补偿
  • {rejection_benefit}
  • {rejection_protection}

【关键事实】
• 共有{N}个消费者同时独立决策（你看不到别人的选择）
• 生产者会根据获得的数据信息来定价
• {free_ride_info}

请仔细权衡参与的收益（补偿+学习）与代价（可能的价格歧视），做出你的选择。

请以JSON格式回答：
{{
  "decision": "accept" 或 "reject",
  "reasoning": "你的理由（1-2句话）"
}}
"""

# 根据anonymization填充的变量：
if anonymization == "identified":
    policy_details = """
    • 实名制：你的数据会与你的身份关联
    • 生产者将知道每个消费者的信号
    • 生产者可以对每个人设置不同的价格
    """
    participation_cost = "生产者会看到你的信号，可能对你进行精准的价格歧视"
    rejection_protection = "生产者无法用你的数据对你定价"
    
elif anonymization == "anonymized":
    policy_details = """
    • 匿名化：你的数据会被打乱身份标识
    • 生产者只能看到信号集合，无法识别谁是谁
    • 生产者只能对所有人设置统一价格
    """
    participation_cost = "生产者会看到你的信号，但不知道是你的，所有人将面临相同价格"
    rejection_protection = "所有人面临统一价格（无论是否参与）"

free_ride_info = "即使你拒绝，你仍可能看到其他参与者的匿名数据并从中学习"
rejection_benefit = "你仍可以看到其他参与者的数据，从中学习（搭便车）"
```

---

## 10. 实现检查清单

### 10.1 核心模块
- [ ] `DataGenerator`：生成两种数据结构（common_preferences, common_experience）
- [ ] `PosteriorEstimator`：计算贝叶斯后验均值 μ_i
- [ ] `ProducerPricing`：两种定价模式（personalized, uniform）
- [ ] `RationalBaseline`：固定点迭代计算理论参与率
- [ ] `MetricsCalculator`：计算所有评估指标
- [ ] `LLMConsumerAgent`：LLM决策接口

### 10.2 实验流程
- [ ] 单轮博弈完整时序实现（9步）
- [ ] 多seed并行/批处理
- [ ] 结果保存与可视化
- [ ] Ground truth生成（理性baseline）

### 10.3 测试与验证
- [ ] 单元测试：数据生成的统计性质
- [ ] 单元测试：后验估计的准确性
- [ ] 单元测试：定价优化的收敛性
- [ ] 集成测试：完整时序的一致性
- [ ] 对照测试：理性baseline的固定点收敛

### 10.4 文档与可复现性
- [ ] 参数配置文件（JSON）
- [ ] 实验日志（每个决策的reasoning）
- [ ] 结果报告模板
- [ ] 可视化脚本（参与率曲线、福利对比等）

---

## 11. 预期的关键发现（假设）

基于论文理论与LLM特性，预期的benchmark结果：

### 11.1 参与率偏离
**假设H1**：LLM消费者在实名制下参与率**低于**理性baseline
- **原因**：LLM可能高估价格歧视的负面影响，展现风险厌恶
- **测试**：比较identified下的PR_LLM vs PR_theory

**假设H2**：匿名化会显著提高LLM的参与率
- **原因**：隐私保护缓解了LLM的顾虑
- **测试**：PR_LLM(anonymized) > PR_LLM(identified)

### 11.2 搭便车行为
**假设H3**：LLM会过度搭便车（低支付下参与率更低）
- **原因**：LLM理解"拒绝仍能学习"的收益
- **测试**：需要更高的m_i才能达到理论参与率

### 11.3 福利影响
**假设H4**：LLM的低参与会降低学习质量和社会福利
- **原因**：数据稀缺导致估计不准、定价低效
- **测试**：SW_LLM < SW_theory，especially when PR_LLM << PR_theory

### 11.4 不平等
**假设H5**：LLM决策下不平等可能更低（如果匿名化提高参与）
- **原因**：更多匿名化→更少价格歧视→更低不平等
- **测试**：Gini_LLM vs Gini_theory 在不同anonymization下的对比

---

## 12. 与场景A、B的对比

| 维度 | 场景A (Personalization) | 场景B (Too Much Data) | 场景C (Social Data) |
|------|------------------------|----------------------|---------------------|
| **核心机制** | 隐私-个性化权衡 | 数据量-质量-价格三角 | 数据外部性-参与决策 |
| **LLM角色** | 平台，选择个性化程度 | 买家，决定购买数量 | 消费者，决定是否参与 |
| **外部性** | 无（单一决策者） | 间接（通过市场价格） | 直接（参与影响他人学习） |
| **信息结构** | 简单（隐私vs效用） | 中等（质量分布） | 复杂（跨个体相关性） |
| **理论基础** | 隐私经济学 | 市场机制设计 | 信息经济学 |
| **关键参数** | ε (隐私成本), K (类别数) | N_b, N_s, quality分布 | N, σ, data structure |
| **评估重点** | 个性化-隐私平衡 | 市场效率、价格扭曲 | 参与率、搭便车、福利 |

**共同点**：
- 都测试LLM作为理性决策者的表现
- 都有明确的理论baseline（ground truth）
- 都关注福利与不平等指标
- 都可以扩展到多模型对比

**场景C的独特贡献**：
- **首次测试LLM对社会数据外部性的理解**
- **搭便车行为的识别**（经济学经典问题）
- **信息设计维度**（匿名化政策）
- **相关数据结构**（更贴近现实数字平台）

---

## 13. 下一步行动

1. **代码实现**：按照上述baseline配置开始编写`scenario_c_social_data.py`
2. **测试验证**：确保数据生成、后验估计、定价优化的正确性
3. **Baseline运行**：生成理性参与率的ground truth
4. **LLM评估**：运行2-3个模型的初步测试
5. **结果分析**：验证假设H1-H5
6. **文档完善**：补充实验结果与发现

**当前状态**：✅ 设计完成，配置锁定，准备进入实现阶段