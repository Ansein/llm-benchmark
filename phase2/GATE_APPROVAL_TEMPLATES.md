# Gate 审批模板（G1 / G2 / G3）

用途：为人工审核提供标准化审批记录格式。  
写入位置：`phase2/workspace/gate{N}_approved.json`。

## 通用最小模板

```json
{
  "approved": true,
  "reviewer_notes": "结论与关键理由",
  "timestamp": 1773980000.0
}
```

## Gate 1（求解器与可测性）

文件：`phase2/workspace/gate1_approved.json`

```json
{
  "approved": true,
  "reviewer_notes": "Solver数学正确，边界条件与方向性检验通过；同意进入Layer 2。",
  "reviewer": "your_name",
  "checklist_result": {
    "solver_math_correct": true,
    "hypothesis_testability_ok": true,
    "boundary_condition_ok": true,
    "open_issues": []
  },
  "timestamp": 1773980000.0
}
```

## Gate 2（实验设计与覆盖）

文件：`phase2/workspace/gate2_approved.json`

```json
{
  "approved": true,
  "reviewer_notes": "实验设计覆盖H1/H2/H3，预算可接受；同意进入Layer 3。",
  "reviewer": "your_name",
  "checklist_result": {
    "market_design_faithful_to_paper": true,
    "hypothesis_test_mapping_ok": true,
    "budget_runtime_ok": true,
    "reproducibility_policy_ok": true,
    "open_issues": []
  },
  "timestamp": 1773980000.0
}
```

## Gate 3（提示词变更可比性）

文件：`phase2/workspace/gate3_approved.json`

```json
{
  "approved": true,
  "reviewer_notes": "提示词改动针对根因且不破坏跨版本可比性；批准进入复测。",
  "reviewer": "your_name",
  "checklist_result": {
    "root_cause_match": true,
    "comparability_guardrails_ok": true,
    "rollback_plan_defined": true,
    "open_issues": []
  },
  "timestamp": 1773980000.0
}
```

## 说明

- 系统只强制读取 `approved` 字段；其余字段用于审计追踪与复盘。
- 若拒绝放行，将 `approved` 改为 `false`，并在 `reviewer_notes` 与 `open_issues` 中写明原因。
- 建议保留 `timestamp`（Unix 秒）以便串联实验日志。

