# Phase 2 设计文档：智能化研究框架

> 范围：场景B（Acemoglu et al. 2022 "Too Much Data"）
> 目标：论文PDF输入 → 假设验证报告输出，人类只在三个Gate把关

---

## 一、总体目标与原则

### 核心目标
研究者提供一篇数据市场/隐私外部性论文的PDF，系统自动完成：
1. 论文理解与场景提取
2. 求解器构建与Ground Truth生成
3. 实验设计与执行
4. 结果分析与假设验证报告

人类研究者只需在三个Gate审核并放行。

### 四条设计原则

**1. Paradigm-as-Contract**
研究者定义"测什么、为什么测"；Agent实现"怎么测"。`paradigm.yaml` 是研究者意图与Agent行为之间的稳定契约。

**2. 协作网络，非流水线**
Agent之间点对点通信，多Agent并行迭代。Orchestrator是元协调者，只在跨层资源决策或冲突时介入，不是每条消息都经过的中央路由。

**3. 输入即论文PDF**
系统从学术论文出发。Agent提取场景、形式化假设、构建求解器。Ground Truth由确定性求解器生成，其数学正确性由人类研究者负责（Gate 1审核）。

**4. Test Registry可扩展性**
测试类型是注册抽象，不是硬编码逻辑。已注册三种类型；新场景若需要新测试形式，由Agent驱动设计并经人工批准后永久注册。

---

## 二、系统架构：8个Agent，4层

```
Layer 1 (理解层)：Paper Agent ↔ Scenario Extractor ↔ Solver Builder → Gate 1
Layer 2 (设计层)：Market Designer ↔ Orchestrator（含Runner/Analysis成本协商）→ Gate 2
Layer 3 (执行层)：Runner ↔ Analysis（动态调整）→ 触发 Gate 3
Layer 4 (优化层)：Analysis ↔ Market Designer ↔ Prompt Engineer → Gate 3
```

所有Agent共享 `phase2/workspace/` 作为共享内存。

---

## 三、Workspace（共享内存）结构

```
phase2/
├── workspace/
│   ├── input/
│   │   └── paper.pdf                  # 研究者提供
│   ├── paper_parse.json               # Paper Agent 输出
│   ├── paradigm.yaml                  # Scenario Extractor 输出
│   ├── market_config.yaml             # Market Designer 输出（Layer 2）
│   ├── solver/
│   │   └── solver_b.py                # Solver Builder 输出
│   ├── raw_results/                   # Runner 输出（Layer 3）
│   ├── metrics/                       # Analysis 输出（Layer 3）
│   ├── hypothesis_status.json         # 各假设实时状态
│   ├── messages/                      # Agent间通信（Plan A）
│   │   └── {from}_{to}_{ts}.json
│   └── gate1_request.json             # Layer 1 联合签字，触发 Gate 1
│   └── gate1_approved.json            # 人工审核通过标志
├── agents/
│   ├── paper_agent.py
│   ├── scenario_extractor.py
│   ├── solver_builder.py
│   ├── market_designer.py             # Layer 2
│   ├── orchestrator.py                # Layer 2
│   ├── runner.py                      # Layer 3
│   ├── analysis.py                    # Layer 3
│   └── prompt_engineer.py             # Layer 4
├── tools/
│   ├── pdf_reader.py                  # PDF解析工具
│   ├── file_io.py                     # workspace读写
│   └── llm_client.py                  # Claude API封装
├── tests/
│   ├── base_test.py                   # BaseTest抽象类
│   ├── prompt_ladder_test.py          # knowledge_dependent
│   ├── sensitivity_test.py            # comparative_static
│   └── fictitious_play_test.py        # dynamic_convergence
├── layer1_coordinator.py              # Layer 1 协调器（Plan A）
└── DESIGN.md                          # 本文档
```

---

## 四、Layer 1：理解层设计

### 4.1 Paper Agent

**核心问题**：这篇论文讲了什么？

**输入**：`workspace/input/paper.pdf`

**工具**：

| 工具 | 签名 | 说明 |
|------|------|------|
| `read_pdf_full` | `() → dict` | 提取全文，按节结构化存储 |
| `search_paper` | `(query: str) → list[str]` | 在全文中语义检索相关段落 |
| `answer_query` | `(question: str, sections: list) → str` | 响应其他Agent的提问，定位原文回答 |

**主动输出**：`workspace/paper_parse.json`

```json
{
  "title": "Too Much Data: Prices and Inefficiencies in Data Markets",
  "sections": {
    "model_setup": "...",
    "equilibrium_definition": "...",
    "comparative_statics": "...",
    "numerical_examples": ["example1_text", "example2_text"],
    "appendix": "..."
  },
  "key_variables": {
    "N": "number of users",
    "rho": "inter-user type correlation coefficient",
    "v": "privacy preference value",
    "alpha": "platform's marginal value of information"
  },
  "payoff_functions": "..."
}
```

**被动职责**：在 Gate 1 之前始终保持响应，处理来自 Extractor 和 Solver Builder 的 `query` 消息。

---

### 4.2 Scenario Extractor Agent

**核心问题**：这是个什么博弈？H1/H2/H3 是什么？

**输入**：`paper_parse.json`

**工具**：

| 工具 | 签名 | 说明 |
|------|------|------|
| `read_file` | `(path: str) → str` | 读取 workspace 任意文件 |
| `write_paradigm` | `(content: dict) → None` | 写入 `paradigm.yaml` |
| `query_paper_agent` | `(question: str) → str` | 向 Paper Agent 发消息请求补充信息 |
| `notify_solver_builder` | `(paradigm_draft: dict) → None` | 发送假设草稿让 Solver Builder 评估可测性 |

**核心产出**：`workspace/paradigm.yaml`

```yaml
scenario_id: scenario_b
paper: "Acemoglu et al. 2022 - Too Much Data"
game_type: static_bayesian_game

players:
  - role: user
    count: N
    action_space: [share, not_share]
  - role: platform
    count: 1
    action_space: continuous_price_vector

equilibrium_concept: Nash_BNE

key_parameters:
  - name: rho
    description: "inter-user type correlation"
    range: [0, 1]
  - name: v
    description: "privacy preference value"
    range: continuous

hypotheses:
  - id: H1
    statement: "rho increases → equilibrium share_rate decreases"
    nature: comparative_static
    preferred_test: SensitivityTest
    parameters_to_vary: [rho]
    success_criterion: "EAS < 0 for all rho grid points"
  - id: H2
    statement: "More externality context in prompt → better mechanism understanding"
    nature: knowledge_dependent
    preferred_test: PromptLadderTest
    success_criterion: "Jaccard monotone increasing v1→v6"
  - id: H3
    statement: "LLM converges to BNE through fictitious play belief updating"
    nature: dynamic_convergence
    preferred_test: FictitiousPlayTest
    success_criterion: "strategy_delta < 0.01 within 50 rounds"
```

**迭代终止条件**：`paradigm.yaml` 写定 **且** Solver Builder 确认所有假设均可数值验证。

---

### 4.3 Solver Builder Agent

**核心问题**：如何计算均衡？

**输入**：`paper_parse.json` + `paradigm.yaml`

**工具**：

| 工具 | 签名 | 说明 |
|------|------|------|
| `read_file` | `(path: str) → str` | 读取 workspace 文件 |
| `write_solver` | `(code: str) → None` | 写入 `solver/solver_b.py` |
| `run_solver_test` | `(test_params: dict) → dict` | **执行**求解器，捕获输出用于自验证 |
| `query_paper_agent` | `(question: str) → str` | 澄清数学符号/边界条件 |
| `query_extractor` | `(question: str) → str` | 协商假设是否可数值验证 |

**自验证流程**（写完求解器必须全部通过）：

1. **边界条件**：ρ=0 时各用户独立，均衡应退化为单用户情形
2. **数值对照**：与论文中给出的数值算例核对（如存在）
3. **方向性检验**：ρ 从 0.3 → 0.9，share_rate 应单调下降（验证 H1 方向）

**迭代终止条件**：三项自验证全部通过，自验证报告写入 `gate1_request.json`。

---

### 4.4 Layer 1 迭代协议（Plan A）

#### 消息格式

```json
// workspace/messages/{from}_{to}_{timestamp}.json
{
  "from": "extractor",
  "to": "paper_agent",
  "type": "query",
  "content": "What is the exact definition of I(S) in Section 3?",
  "reply_to": null,
  "status": "pending"
}
```

`type` 取值：`query` / `reply` / `notify` / `negotiate`

#### 协调器运行逻辑（`layer1_coordinator.py`）

```
1. 启动 Paper Agent → 产出 paper_parse.json
2. 并行启动 Scenario Extractor + Solver Builder（均可独立读 paper_parse.json）
3. 主循环（每 N 秒）：
   a. 扫描 messages/，将 status=pending 的消息路由给对应 Agent
   b. Agent 处理消息，写回 reply
   c. 检测 gate1_request.json 是否出现三方签字
4. Gate 1 触发 → 暂停协调器 → 等待人工写入 gate1_approved.json
5. Gate 1 通过 → 解锁 Layer 2
```

---

### 4.5 Gate 1：联合签字与人工审核

三方就绪后联合写入：

```json
// workspace/gate1_request.json
{
  "status": "ready_for_review",
  "signed_by": ["paper_agent", "extractor", "solver_builder"],
  "timestamp": "...",
  "summary": {
    "hypotheses_count": 3,
    "solver_validation": "passed",
    "open_issues": []
  },
  "review_checklist": [
    "solver correctly implements Kalman-like posterior covariance?",
    "H1 success criterion direction (EAS < 0) is correct?",
    "boundary condition rho=0 produces expected single-user result?",
    "paradigm.yaml hypothesis nature tags are appropriate?"
  ]
}
```

人工审核通过后写入：

```json
// workspace/gate1_approved.json
{
  "approved": true,
  "reviewer_notes": "...",
  "timestamp": "..."
}
```

---

## 五、Test Registry

每种测试类型实现 `BaseTest` 接口：

```python
class BaseTest:
    def setup(self, params) -> TestEnv: ...
    def run(self, env, model) -> RawResult: ...
    def evaluate(self, raw, ground_truth) -> Metrics: ...
    def verify_hypothesis(self, metrics, hypothesis) -> HypothesisResult: ...
```

目前已注册三种类型：

| 抽象类 | 场景B实例 | 假设Nature标签 |
|--------|-----------|----------------|
| Knowledge Injection Test | Prompt Ladder (v1→v6) | `knowledge_dependent` |
| Comparative Static Test | Sensitivity Analysis (3×3 grid) | `comparative_static` |
| Dynamic Convergence Test | Fictitious Play (50轮) | `dynamic_convergence` |

---

## 六、Plan A → Plan B 迁移路径

| 组件 | Plan A（当前） | Plan B（目标） |
|------|----------------|----------------|
| Agent实体 | Python类，同进程 | Claude Code sub-agent，独立进程 |
| 消息传递 | 文件轮询 (`messages/`) | `SendMessage` 工具 |
| 并发模型 | 伪并发（轮询） | 真并发 |
| 工具签名 | **不变** | **不变** |
| 业务逻辑 | **不变** | **不变** |

**迁移原则**：Agent的工具函数签名在 Plan A 和 Plan B 中完全相同。迁移只替换消息路由机制，不修改任何业务逻辑。

---

## 七、后续层（Layer 2/3/4）概要

> 详细设计待 Layer 1 实现并通过 Gate 1 后展开。

- **Layer 2（设计层）**：Market Designer + Orchestrator，协商实验参数（模型数量、网格范围、trial数），产出 `market_config.yaml`，通过 Gate 2。
- **Layer 3（执行层）**：Runner 并发执行 LLM 实验 + Analysis 实时分析，动态调整 trial 数，输出假设 PASS/FAIL/INCONCLUSIVE 报告。
- **Layer 4（优化层）**：仅当 Analysis 判定 prompt 质量是瓶颈时触发，Prompt Engineer 提出新版本，通过 Gate 3 后进入 role_prompts/。
