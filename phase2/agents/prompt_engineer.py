"""
Prompt Engineer Agent — Layer 4

Responsibilities:
- Generate prompt-change proposal from diagnosis
- Revise proposal based on analysis feedback and constraints
- Produce proposal artifact for Gate 3 review
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "prompt_engineer"
ANALYSIS_AGENT = "analysis_l4"


class PromptEngineerAgent:
    def __init__(self):
        self._diagnosis: dict[str, Any] = {}
        self._constraints: dict[str, Any] = {}
        self._proposal_version = 0

    def handle_pending_messages(self) -> int:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                payload = json.loads(msg.get("content", "{}"))
                kind = payload.get("kind")
                if kind == "root_cause_diagnosis":
                    self._diagnosis = payload.get("diagnosis", {})
                    self._emit_proposal()
                elif kind == "revision_request":
                    self._constraints = payload.get("constraints", self._constraints)
                    self._emit_proposal(issues=payload.get("issues", []), round_id=payload.get("round", 0))
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def _emit_proposal(self, issues: list[str] | None = None, round_id: int = 0) -> None:
        self._proposal_version += 1
        proposal = self._build_proposal(issues=issues or [], round_id=round_id)
        artifact_path = self._write_artifact(proposal)
        proposal["artifact_path"] = artifact_path

        payload = {
            "kind": "prompt_change_proposal",
            "proposal": proposal,
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=ANALYSIS_AGENT,
            msg_type="reply",
            content=json.dumps(payload, ensure_ascii=False),
        )

    def _build_proposal(self, issues: list[str], round_id: int) -> dict[str, Any]:
        preserve_versions = self._constraints.get(
            "must_keep_prompt_versions", ["b.v1", "b.v2", "b.v3", "b.v4", "b.v5", "b.v6"]
        )
        target_test = self._diagnosis.get("target_test", "PromptLadderTest")

        changes = [
            {
                "id": "chg-1",
                "type": "instruction_reordering",
                "description": "Move mechanism explanation before action checklist.",
                "rationale": "Reduce reasoning omission under long prompt contexts.",
            },
            {
                "id": "chg-2",
                "type": "decision_scaffold",
                "description": "Add explicit 3-step utility comparison scaffold before final JSON output.",
                "rationale": "Stabilize mechanism-grounded decisions for H2.",
            },
            {
                "id": "chg-3",
                "type": "output_schema_lock",
                "description": "Keep JSON schema unchanged (share/reason fields).",
                "rationale": "Preserve comparability and downstream parser compatibility.",
            },
        ]

        if issues:
            changes.append(
                {
                    "id": "chg-revision",
                    "type": "revision_response",
                    "description": "Address review issues from analysis/market designer.",
                    "issues": issues,
                    "round": round_id,
                }
            )

        return {
            "version": f"b.v7.proposal.{self._proposal_version}",
            "target_test": target_test,
            "diagnosis_type": self._diagnosis.get("type", "prompt_quality_bottleneck"),
            "changes": changes,
            "comparability_guardrails": {
                "preserve_versions": preserve_versions,
                "unchanged_output_schema": True,
                "no_solver_or_gt_change": True,
                "same_model_pool_for_retest": True,
            },
            "retest_plan": {
                "tests": ["PromptLadderTest", "SensitivityTest", "FictitiousPlayTest"],
                "priority": ["PromptLadderTest", "SensitivityTest"],
                "acceptance": "H2 improves without H1/H3 regression",
            },
            "timestamp": time.time(),
        }

    def _write_artifact(self, proposal: dict[str, Any]) -> str:
        base = fio.ws("optimization")
        base.mkdir(parents=True, exist_ok=True)
        p = base / f"prompt_change_proposal_{self._proposal_version}.json"
        p.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

        prompts_dir = fio.ws("role_prompts")
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompts_dir / "scenario_b_v7_candidate.md"
        prompt_file.write_text(self._render_prompt_candidate(proposal), encoding="utf-8")
        return str(p)

    @staticmethod
    def _render_prompt_candidate(proposal: dict[str, Any]) -> str:
        return (
            f"# Prompt Candidate {proposal.get('version')}\n\n"
            "## Goal\n"
            "- Improve mechanism understanding in PromptLadderTest (H2)\n"
            "- Preserve output schema and cross-version comparability\n\n"
            "## User Prompt Additions\n"
            "1. First restate mechanism assumptions in one sentence.\n"
            "2. Then compute utility gap: compensation - marginal privacy cost.\n"
            "3. Finally return JSON only with keys `share` and `reason`.\n"
        )

