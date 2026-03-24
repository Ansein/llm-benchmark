# Phase 2 多智能体系统运行复盘（L1-L3）

本文基于 2026-03-23 到 2026-03-24 这次真实 API 运行结果整理，目的不是讲设计理想图，而是把系统这次实际怎么跑的、各层 agent 做了什么、为什么这么做、产出了什么，说清楚。

## 1. 先给结论

这次系统已经完整跑通了：

- L1 完成并通过 Gate 1
- L2 完成并生成 Gate 2 审批材料
- L3 完成并写出完整 summary

这说明系统主链路已经通了，不再只是框架代码。

但这次研究结果不是“成功验证论文”，而是：

- H1: FAIL
- H2: FAIL
- H3: FAIL

同时，L3 的 H1 执行过程中还有 4 个 job 以 timeout 结束，说明执行稳定性还不够好。

## 2. 这套系统这次是怎么跑的

从运行逻辑上看，当前系统不是简单串行流水线，而是“层内交互、层间门控”的结构。

实际运行顺序是：

1. L1 读取论文，产出 `paper_parse.json`、`paradigm.yaml`、`solver/solver_b.py`
2. 人工审批 Gate 1
3. L2 读取 `paradigm.yaml`，产出 `market_config.yaml`
4. 人工审批 Gate 2
5. L3 读取 `market_config.yaml` 和 `paradigm.yaml`，展开实验任务，执行并分析

层内不是单 agent 独跑，而是通过 `workspace/messages/` 互发消息。

共享状态主要通过 `phase2/workspace/` 落盘实现：

- 结构化中间件：`paper_parse.json`、`paradigm.yaml`、`market_config.yaml`
- agent 间消息：`workspace/messages/*.json`
- 运行状态：`workspace/metrics/*.json`
- 实验原始产物：`workspace/raw_results/**`

## 3. L1 复盘：Paper Agent / Scenario Extractor / Solver Builder

### 3.1 L1 的目标

L1 的任务是把“论文文本”转成“后续层可执行的研究对象”。

具体要产出三样东西：

1. `paper_parse.json`
2. `paradigm.yaml`
3. `solver/solver_b.py`

只有这三样都完成并 sign-off，Gate 1 才能放行。

### 3.2 Paper Agent 做了什么

入口代码：[`layer1_coordinator.py`](/d:/benchmark/phase2/layer1_coordinator.py)

核心实现：[`paper_agent.py`](/d:/benchmark/phase2/agents/paper_agent.py)

它做了两件事：

1. 解析 PDF
2. 在别的 agent 提问时回答论文内容

这次运行里，它先调用 `pdf_reader` 提取论文文本，再让模型整理成结构化输出，最后写出：

- [`paper_parse.json`](/d:/benchmark/phase2/workspace/paper_parse.json)
- [`paper_signoff.json`](/d:/benchmark/phase2/workspace/paper_signoff.json)

此外还写了 PDF 质量报告：

- [`paper_quality_report.json`](/d:/benchmark/phase2/workspace/input/paper_quality_report.json)

### 3.3 Scenario Extractor 做了什么

核心实现：[`scenario_extractor.py`](/d:/benchmark/phase2/agents/scenario_extractor.py)

它的职责是把论文解释成“可测试假设 + 游戏结构”。

这次它最终写出了：

- [`paradigm.yaml`](/d:/benchmark/phase2/workspace/paradigm.yaml)
- [`extractor_signoff.json`](/d:/benchmark/phase2/workspace/extractor_signoff.json)

`paradigm.yaml` 里最关键的是 `hypotheses`，这次产出了 3 条：

- `H1`: `comparative_static -> SensitivityTest`
- `H2`: `knowledge_dependent -> PromptLadderTest`
- `H3`: `dynamic_convergence -> FictitiousPlayTest`

这一步的意义是：把论文里的研究主张压缩成系统后续可执行的实验索引。

### 3.4 Scenario Extractor 为什么这么做

这个“为什么”是有显式记录的，不是我猜的。

它的 system prompt 里明确要求：

- 提取游戏结构
- 产出 H1/H2/H3
- 先问 Paper Agent
- 再通知 Solver Builder 检查可测试性
- 最后才能写 `paradigm.yaml`

对应代码见 [`scenario_extractor.py`](/d:/benchmark/phase2/agents/scenario_extractor.py)

这次真实运行里，Extractor 确实这么做了。

它先发消息问 Paper Agent：

- 文件：[`scenario_extractor_paper_agent_1774276438632_308766.json`](/d:/benchmark/phase2/workspace/messages/scenario_extractor_paper_agent_1774276438632_308766.json)

问题是：

- 这篇论文的 equilibrium concept 是什么
- 哪些 comparative statics 参数驱动 equilibrium share rate
- 能否确认是 Bayes-Nash equilibrium

Paper Agent 回了一个结构化答复：

- 文件：[`paper_agent_scenario_extractor_1774276447650_c3df76.json`](/d:/benchmark/phase2/workspace/messages/paper_agent_scenario_extractor_1774276447650_c3df76.json)

然后 Extractor 把 H1/H2/H3 草案发给 Solver Builder，请它确认是否数值可测试：

- 文件：[`scenario_extractor_solver_builder_1774276453840_13765a.json`](/d:/benchmark/phase2/workspace/messages/scenario_extractor_solver_builder_1774276453840_13765a.json)

Solver Builder 回复：

- H1: TESTABLE
- H2: PARTIALLY TESTABLE，需要数值代理指标
- H3: TESTABLE

对应文件：

- [`solver_builder_scenario_extractor_1774276465118_6b3b7b.json`](/d:/benchmark/phase2/workspace/messages/solver_builder_scenario_extractor_1774276465118_6b3b7b.json)

所以 L1 里，Extractor 不是孤立写文件，而是先问论文、再和 Solver 协商，再定稿。

### 3.5 Solver Builder 做了什么

核心实现：[`solver_builder.py`](/d:/benchmark/phase2/agents/solver_builder.py)

它的职责是：

1. 根据论文和 paradigm 写 solver
2. 对 solver 做自检
3. 只有自检通过才 sign-off

这次它产出了：

- [`solver_b.py`](/d:/benchmark/phase2/workspace/solver/solver_b.py)
- [`solver_signoff.json`](/d:/benchmark/phase2/workspace/solver_signoff.json)

自检结果：

- `test1`: true
- `test2`: true
- `test3`: true
- `all_passed`: true

### 3.6 Solver Builder 为什么这么做

这个“为什么”也有显式规则。

它的 prompt 明确要求 solver 必须：

- 接受 `N, rho, v_lo, v_hi, alpha, seed`
- 枚举 sharing sets
- 计算 posterior covariance
- 计算 leakage
- 找 platform-optimal set
- 跑 3 个验证测试

也就是说，Solver Builder 的角色不是“解释论文”，而是把论文写成一个可被 L3 调用的可执行求解器。

### 3.7 这次 L1 的关键现实结论

这次 L1 结构上是成功的，但有一个语义警报：

- `H1` 说：`rho` 增加，`share_rate` 下降
- 但 [`solver_signoff.json`](/d:/benchmark/phase2/workspace/solver_signoff.json) 里 `test2_detail` 写的是：
  - `rho=0.3 share=0.500`
  - `rho=0.9 share=0.500`

也就是说：

- L1 的格式产物没问题
- 但 H1 和 solver 自检之间存在张力

这个矛盾没有在 Gate 1 自动阻断，而是被留到了 L3 才暴露得更明显。

## 4. L2 复盘：Market Designer / Orchestrator

### 4.1 L2 的目标

L2 的任务不是跑实验，而是把 L1 的研究对象转成可执行实验计划。

它的核心输出是：

- [`market_config_draft.json`](/d:/benchmark/phase2/workspace/market_config_draft.json)
- [`market_config.yaml`](/d:/benchmark/phase2/workspace/market_config.yaml)
- [`gate2_request.json`](/d:/benchmark/phase2/workspace/gate2_request.json)

### 4.2 Market Designer 做了什么

核心实现：[`market_designer.py`](/d:/benchmark/phase2/agents/market_designer.py)

它读取：

- `paradigm.yaml`
- 可选的 `paper_parse.json`
- `configs/model_configs.json`

然后生成一份实验草案，包含：

1. 用哪些模型
2. 每个 hypothesis 对应什么 test
3. 每个 test 的参数配置
4. 全局预算限制

这次它选了 3 个模型：

- `gpt-5.2`
- `gpt-5.1`
- `gpt-5.1-2025-11-13`

并生成了 3 个测试配置：

- `SensitivityTest`
- `PromptLadderTest`
- `FictitiousPlayTest`

### 4.3 Orchestrator 做了什么

核心实现：[`orchestrator.py`](/d:/benchmark/phase2/agents/orchestrator.py)

它的职责是审核草案，而不是自己重新设计。

主要检查：

1. hypothesis 覆盖是否完整
2. budget 是否可行
3. 是否需要回退给 Market Designer 做 revision

这次真实运行里，Orchestrator 没要求 revision，而是直接接受草案并 finalize。

消息证据：

- Market Designer 通知 draft ready：
  - [`market_designer_orchestrator_1774277265341_c50b63.json`](/d:/benchmark/phase2/workspace/messages/market_designer_orchestrator_1774277265341_c50b63.json)
- Orchestrator 回复 final accept：
  - [`orchestrator_market_designer_1774277265353_574978.json`](/d:/benchmark/phase2/workspace/messages/orchestrator_market_designer_1774277265353_574978.json)

### 4.4 L2 为什么这么快

因为它本质上只是“生成计划并做一致性检查”，不是执行层。

它没有大规模模型调用，没有展开实验，所以几秒钟完成是正常的。

### 4.5 这次 L2 最终产物是什么

[`market_config.yaml`](/d:/benchmark/phase2/workspace/market_config.yaml) 这次的关键信息是：

- `hypothesis_map`
  - `H1 -> SensitivityTest`
  - `H2 -> PromptLadderTest`
  - `H3 -> FictitiousPlayTest`
- `estimated_total_calls = 222`
- `budget_feasible = true`
- `fail_fast = false`
- `retry_on_error = 1`
- `resume_from_artifacts = true`

这意味着 L2 认为：

- 覆盖是全的
- 预算是够的
- L3 允许错误重试一次

## 5. L3 复盘：Runner / Analysis

### 5.1 L3 的目标

L3 是真正的实验执行层。

它不再讨论“该不该测”，而是：

1. 展开实验 job
2. 执行 job
3. 实时分析结果
4. 必要时回调 Runner 做重试或调整
5. 输出最终 hypothesis 结论

核心文件：

- [`runner.py`](/d:/benchmark/phase2/agents/runner.py)
- [`analysis.py`](/d:/benchmark/phase2/agents/analysis.py)

### 5.2 Runner 做了什么

Runner 读取：

- `market_config.yaml`
- `paradigm.yaml`

然后按“每条 hypothesis × 每个模型”建初始 job 队列。

这次初始队列其实是 9 个 job：

- H1 × 3 models
- H2 × 3 models
- H3 × 3 models

你当时在 `runner_state.json` 里看到：

- `queue_len = 8`
- `active_job = H1 / SensitivityTest / gpt-5.2`

原因不是少一个，而是：

- 1 个 job 正在执行
- 剩下 8 个还在队列里

### 5.3 Analysis 做了什么

Analysis 持续监听 Runner 发来的 `job_result` 事件，然后更新：

- [`hypothesis_status.json`](/d:/benchmark/phase2/workspace/metrics/hypothesis_status.json)
- [`layer3_summary.json`](/d:/benchmark/phase2/workspace/metrics/layer3_summary.json)

它还会在运行中给 Runner 发 directive。

最典型的这次就是：

- 当 H1 某个 job timeout 报错后
- Analysis 自动发 `retry_failed`

证据：

- Runner 报 H1 超时：
  - [`runner_analysis_1774279301884_2d6569.json`](/d:/benchmark/phase2/workspace/messages/runner_analysis_1774279301884_2d6569.json)
- Analysis 回发 retry 指令：
  - [`analysis_runner_1774279301924_69997d.json`](/d:/benchmark/phase2/workspace/messages/analysis_runner_1774279301924_69997d.json)

这说明 L3 不是“Runner 跑完再统一分析”，而是边跑边交互。

### 5.4 这次 L3 到底跑了多少东西

最终汇总文件：

- [`layer3_summary.json`](/d:/benchmark/phase2/workspace/metrics/layer3_summary.json)
- [`run_manifest.json`](/d:/benchmark/phase2/workspace/raw_results/run_manifest.json)

最终结果：

- `events_processed = 12`
- `jobs_total = 12`
- `jobs_ok = 8`
- `jobs_error = 4`
- `estimated_calls_used = 12`

为什么不是 9，而是 12：

因为 H1 初始 3 个 job 先后报错，Analysis 触发了 retry。

总 job 构成是：

1. 初始 H1: 3 个
2. 初始 H2: 3 个
3. 初始 H3: 3 个
4. H1 重试: 3 个

合计 12 个。

### 5.5 这次 L3 的实验原始产物有多少

这次 `raw_results/` 下面生成了大量实验文件：

- `sensitivity`: 6 个目录，2041 个文件
- `prompt_ladder`: 1 个目录，381 个文件
- `fictitious_play`: 3 个目录，200 个文件

这说明系统不是只写一份 summary，而是把大量原始过程保留下来了。

## 6. 三个 hypothesis 为什么失败

### 6.1 H1 为什么失败

H1 的判据是：

- `rho` 增加时，`share_rate` 应该单调不增

这次 H1 有两类结果：

1. 一部分 job 直接超时
2. 成功跑完的 job 也没有满足单调性

例如 [`H1_SensitivityTest_gpt-5.1_1774289980427.json`](/d:/benchmark/phase2/workspace/raw_results/H1_SensitivityTest_gpt-5.1_1774289980427.json)：

- `monotonic_non_increasing_ratio = 0.3333`
- `all_combinations_non_increasing = false`

具体看其中一组：

- `v=0.6-0.9`: `1.0 -> 0.975 -> 1.0`

这不是单调下降。

另一个成功完成的模型 [`H1_SensitivityTest_gpt-5.1-2025-11-13_1774291516400.json`](/d:/benchmark/phase2/workspace/raw_results/H1_SensitivityTest_gpt-5.1-2025-11-13_1774291516400.json) 也一样：

- `monotonic_non_increasing_ratio = 0.6666`
- 仍然不是全组合满足单调性

所以 H1 的最终失败，不只是因为 timeout，而是即使跑完也没验证出 hypothesis。

### 6.2 H2 为什么失败

H2 的判据是：

- prompt detail level 提高后，机制理解分数应单调上升

这次 H2 没有执行错误，但结果不满足单调性。

例如 [`H2_PromptLadderTest_gpt-5.2_1774283830571.json`](/d:/benchmark/phase2/workspace/raw_results/H2_PromptLadderTest_gpt-5.2_1774283830571.json)：

- `jaccard_similarity_mean = [0.9375, 1.0, 1.0, 0.9375, 0.0625, 1.0]`
- `monotone_non_decreasing = false`

也就是说，版本越高不等于理解越好，曲线有明显回落。

所以 H2 是“正常执行后 FAIL”，不是工程错误。

### 6.3 H3 为什么失败

H3 的判据是：

- fictitious play 过程中策略应收敛向 BNE

这次 H3 也没有执行错误，但没有收敛。

例如 [`H3_FictitiousPlayTest_gpt-5.2_1774285732791.json`](/d:/benchmark/phase2/workspace/raw_results/H3_FictitiousPlayTest_gpt-5.2_1774285732791.json)：

- `converged = false`
- `strategy_delta = 0.3333`
- `actual_rounds = 4`

所以 H3 同样是“正常执行后 FAIL”。

## 7. Gate 3 为什么没有触发

Gate 3 的触发逻辑在 [`analysis.py`](/d:/benchmark/phase2/agents/analysis.py) 里写死了：

- 当 H2 失败
- 并且 H1 或 H3 至少有一个 PASS

才怀疑是“prompt bottleneck”，进入 Layer 4。

这次不是这种情况，因为：

- H2 FAIL
- H1 FAIL
- H3 FAIL

所以 Analysis 判断这不是“只有 prompt 出问题”，而是整体假设都没过，因此不触发 Gate 3。

## 8. 这次运行里，agent 的“为什么”哪些是有记录的，哪些没有

### 8.1 有明确记录的

这些可以直接从 prompt、message、result 里读出来：

1. Extractor 为什么问 Paper Agent
2. Extractor 为什么问 Solver Builder
3. Analysis 为什么给 Runner 发 retry
4. Gate 3 为什么没触发

### 8.2 没有单独写成自然语言思维链的

这些系统没有保存 agent 的完整内心推理，只能从代码和产物反推：

1. Paper Agent 为什么挑某些段落写入 `paper_parse.json`
2. Market Designer 为什么选这 3 个模型
3. Orchestrator 为什么这次没有要求 revision
4. Solver Builder 为什么写成当前这版 solver

这些不是完全没依据，而是：

- 有行为结果
- 有规则约束
- 但没有保存逐步自然语言推理过程

所以如果你要看“它怎么做的”，可以看代码和产物。
如果你要看“它一步一步是怎么想的”，当前系统没有完整保留。

## 9. 你现在应该怎么看这些文件

如果你要重新理解一次系统运行逻辑，建议按这个顺序看。

### 9.1 先看层与门

1. [`layer1_coordinator.py`](/d:/benchmark/phase2/layer1_coordinator.py)
2. [`layer2_coordinator.py`](/d:/benchmark/phase2/layer2_coordinator.py)
3. [`layer3_coordinator.py`](/d:/benchmark/phase2/layer3_coordinator.py)

### 9.2 再看本次 L1 产物

1. [`paper_parse.json`](/d:/benchmark/phase2/workspace/paper_parse.json)
2. [`paradigm.yaml`](/d:/benchmark/phase2/workspace/paradigm.yaml)
3. [`solver_signoff.json`](/d:/benchmark/phase2/workspace/solver_signoff.json)
4. [`gate1_request.json`](/d:/benchmark/phase2/workspace/gate1_request.json)

### 9.3 再看本次 L2 产物

1. [`market_config_draft.json`](/d:/benchmark/phase2/workspace/market_config_draft.json)
2. [`market_config.yaml`](/d:/benchmark/phase2/workspace/market_config.yaml)
3. [`gate2_request.json`](/d:/benchmark/phase2/workspace/gate2_request.json)

### 9.4 最后看本次 L3 结果

1. [`layer3_summary.json`](/d:/benchmark/phase2/workspace/metrics/layer3_summary.json)
2. [`hypothesis_status.json`](/d:/benchmark/phase2/workspace/metrics/hypothesis_status.json)
3. [`run_manifest.json`](/d:/benchmark/phase2/workspace/raw_results/run_manifest.json)
4. H1/H2/H3 对应的 job 结果 JSON

### 9.5 如果你想看“交互是否真的发生了”

直接看：

- [`workspace/messages/`](/d:/benchmark/phase2/workspace/messages)

这里是最能证明“不是流水线空壳”的目录。

## 10. 这次运行说明了什么

这次运行最重要的不是“三个 hypothesis 都失败了”，而是下面三点：

1. 多智能体主链路已经真实跑通
2. 层内交互不是摆设，消息机制真的参与了运行
3. 系统已经能把失败保存为结构化证据，而不是只在终端里打印一句报错

但同时也暴露了两个现实问题：

1. H1 的理论假设、solver 自检和 L3 结果之间存在不一致
2. H1 的 `SensitivityTest` 太重，worker timeout 仍然是 L3 的主要执行瓶颈

这两个问题决定了系统下一步更像是“研究语义校准 + 执行稳定性优化”，而不是继续加新层。
