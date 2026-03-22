# Phase 2 工作汇报（更新于 2026-03-22）

---

## 一、系统架构设计（保留）

### 1.1 设计原则

1. Paradigm-as-Contract
- `paradigm.yaml` 作为研究意图与执行系统之间的契约。
- 研究者定义“测什么/为什么测”，系统负责“怎么测”。

2. 交互式协作网络（非流水线）
- 各层 Agent 通过消息通信协作，支持多轮协商与反馈。
- 不采用固定 A->B->C 单向流水线。

3. 输入即论文 PDF
- 从论文出发自动抽取场景、假设、测试与执行配置。

4. Test Registry 可扩展
- 测试类型通过注册机制扩展，不与单一场景硬绑定。

### 1.2 四层多 Agent 架构

- Layer 1（理解层）：`paper_agent`、`scenario_extractor`、`solver_builder`，产出论文结构化结果、假设契约、求解器。
- Layer 2（设计层）：`market_designer`、`orchestrator`，产出 `market_config.yaml` 和 Gate2 审核请求。
- Layer 3（执行层）：`runner`、`analysis`，执行测试并汇总假设验证结果。
- Layer 4（优化层）：`analysis_l4`、`prompt_engineer`、`market_designer_review`，在需要时生成提示词优化与 Gate3 请求。

### 1.3 Gate 机制

- Gate 1：审阅 L1 产物（解析、假设、求解器）。
- Gate 2：审阅实验设计（模型池、测试配置、预算约束）。
- Gate 3：审阅提示词优化变更（可比性与风险）。

### 1.4 共享内存与通信

- 共享内存：`phase2/workspace/`。
- 消息队列：`phase2/workspace/messages/`。
- Gate 文件：`gate{n}_request.json` 与 `gate{n}_approved.json`。

![Phase 2 Agent Architecture](phase2_architecture.png)

---

## 二、当前实现进展

### 2.1 已完成

1. Layer 1
- 三个 Agent 可启动并协作。
- 可产出 `paper_parse.json`、`paradigm.yaml`、`solver/solver_b.py`、`gate1_request.json`。
- 已支持真实 API 调用重试（含连接错误重试与超时配置）。

2. Layer 2
- 可从 `paradigm.yaml` 生成 `market_config_draft.json` 与 `market_config.yaml`。
- 可产出 `gate2_request.json`，并等待 Gate2 审批。

3. Layer 3
- `runner` 与 `analysis` 交互循环已实现。
- 可生成 `metrics/layer3_summary.json`。

4. Layer 4
- 交互循环已实现。
- 可产出 `optimization/root_cause_diagnosis.json`、`prompt_change_proposal_*.json`、`comparability_review.json`。

5. 审批模板
- G1/G2/G3 审批模板和示例文件已补齐。

### 2.2 最新联调结果（2026-03-22）

- 已完成一次从 L1 到 L4 的真实 API 全链路测试。
- L1/L2/L4 均有有效产物输出。
- L3 本次执行 `jobs_total=0`（未形成有效测试任务）。

---

## 三、遇到的主要问题（仅问题，不含方案）

1. `paradigm.yaml` 在部分运行中缺失 `hypotheses`。
- 直接导致 Layer 2 生成的 `hypothesis_map/tests` 为空。

2. Layer 3 任务队列为空。
- 表现为 `runner` 无可执行 job，`layer3_summary.json` 显示 `jobs_total=0`。

3. 真实 API 环境存在不稳定性。
- 出现过连接类错误，影响长链路稳定执行。

4. 运行结果对 Gate 文件依赖较强。
- 审批文件状态会显著影响层间推进与自动化测试连贯性。

---

## 四、下一步计划

1. 修复并收紧 Layer 1 的假设产出质量门禁。
- 确保 `paradigm.yaml` 始终包含可执行的 `hypotheses` 列表。

2. 重新跑一次 L1->L3 真实 API 回归。
- 目标：Layer 3 出现非零 `jobs_total`，并产出有效假设判定。

3. 增加联调可观测性。
- 强化各层关键产物完整性检查与运行日志标识。

4. 基于稳定结果更新 Gate 审批与复测基线。

---

## 五、当前文件状态（按模块）

### 5.1 关键入口与设计文档

- `phase2/DESIGN.md`：架构设计文档。
- `phase2/PROGRESS.md`：当前进展文档（本文件）。
- `phase2/layer1_coordinator.py`：L1 协调入口。
- `phase2/layer2_coordinator.py`：L2 协调入口。
- `phase2/layer3_coordinator.py`：L3 协调入口。
- `phase2/layer4_coordinator.py`：L4 协调入口。

### 5.2 agents/

- `paper_agent.py`、`scenario_extractor.py`、`solver_builder.py`。
- `market_designer.py`、`orchestrator.py`。
- `runner.py`、`analysis.py`。
- `analysis_optimizer.py`、`prompt_engineer.py`、`market_designer_review.py`。

### 5.3 tools/

- `file_io.py`（workspace 与消息读写）。
- `agent_api_client.py`（统一 API 调用层）。
- `pdf_reader.py`（论文文本抽取与分段）。

### 5.4 tests/

- `base_test.py`、`sensitivity_test.py`、`prompt_ladder_test.py`、`fictitious_play_test.py`。

### 5.5 templates/

- `gate1_approved.template.json`
- `gate2_approved.template.json`
- `gate3_approved.template.json`

### 5.6 workspace 当前关键产物（最新一轮）

- 已有：
  - `input/paper.pdf`
  - `paper_parse.json`
  - `paradigm.yaml`
  - `market_config.yaml`
  - `metrics/layer3_summary.json`
  - `optimization/root_cause_diagnosis.json`
  - `optimization/prompt_change_proposal_1.json`
  - `optimization/comparability_constraints.json`
  - `optimization/comparability_review.json`
  - `gate1_request.json`、`gate2_request.json`、`gate3_request.json`

- 状态备注：
  - Layer 3 summary 显示本轮 `jobs_total=0`，需在下一轮回归中重点验证。
