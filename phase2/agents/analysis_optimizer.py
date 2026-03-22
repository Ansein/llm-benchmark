"""
Optimization Analysis Agent — Layer 4

Responsibilities:
- Root-cause diagnosis based on Layer 3 artifacts
- Drive interactive negotiation with MarketDesigner and PromptEngineer
- Validate comparability and trigger Gate 3 review request
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "analysis_l4"
PROMPT_AGENT = "prompt_engineer"
MARKET_AGENT = "market_designer_l4"


class Layer4AnalysisAgent:
    def __init__(self):
        self.done = False
        self._diagnosis: dict[str, Any] = {}
        self._constraints: dict[str, Any] = {}
        self._proposal: dict[str, Any] = {}
        self._revision_round = 0
        self._started = time.time()

    def start_session(self, force: bool = False) -> dict[str, Any]:
        if (not force) and (not fio.file_exists("gate3_request.json")):
            raise RuntimeError("gate3_request.json not found; Layer 4 should be triggered by Layer 3.")

        summary = fio.read_json("metrics/layer3_summary.json") if fio.file_exists("metrics/layer3_summary.json") else {}
        status = fio.read_json("metrics/hypothesis_status.json") if fio.file_exists("metrics/hypothesis_status.json") else {}
        gate3 = fio.read_json("gate3_request.json") if fio.file_exists("gate3_request.json") else {}

        self._diagnosis = self._build_diagnosis(summary, status, gate3)
        fio.write_json("optimization/root_cause_diagnosis.json", self._diagnosis)

        seed_message = {
            "kind": "root_cause_diagnosis",
            "diagnosis": self._diagnosis,
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=PROMPT_AGENT,
            msg_type="notify",
            content=json.dumps(seed_message, ensure_ascii=False),
        )
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=MARKET_AGENT,
            msg_type="notify",
            content=json.dumps(seed_message, ensure_ascii=False),
        )
        logger.info(f"[{AGENT_NAME}] session started; diagnosis broadcast to PromptEngineer + MarketDesigner")
        return self._diagnosis

    def handle_pending_messages(self) -> int:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                payload = json.loads(msg.get("content", "{}"))
                kind = payload.get("kind")
                if kind == "comparability_constraints":
                    self._constraints = payload.get("constraints", {})
                    logger.info(f"[{AGENT_NAME}] received comparability constraints")
                    # Ask PromptEngineer to align proposal with constraints if already present.
                    if self._proposal:
                        self._request_revision_if_needed()
                elif kind == "prompt_change_proposal":
                    self._proposal = payload.get("proposal", {})
                    logger.info(f"[{AGENT_NAME}] received prompt proposal v{self._proposal.get('version', '?')}")
                    self._evaluate_and_progress()
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def is_done(self) -> bool:
        return self.done

    def _evaluate_and_progress(self) -> None:
        if not self._proposal:
            return
        if not self._constraints:
            # Ask market designer to provide constraints if not yet available.
            fio.send_message(
                from_agent=AGENT_NAME,
                to_agent=MARKET_AGENT,
                msg_type="query",
                content=json.dumps({"kind": "need_constraints", "reason": "proposal_received_no_constraints"}),
            )
            return
        self._request_revision_if_needed()

    def _request_revision_if_needed(self) -> None:
        ok, issues = self._validate_proposal(self._proposal, self._constraints)
        review = {
            "status": "ready_for_gate3_review" if ok else "needs_revision",
            "issues": issues,
            "diagnosis_path": str(fio.ws("optimization/root_cause_diagnosis.json")),
            "proposal_path": self._proposal.get("artifact_path"),
            "constraints_path": str(fio.ws("optimization/comparability_constraints.json")),
            "timestamp": time.time(),
        }
        fio.write_json("optimization/comparability_review.json", review)

        if ok:
            self._emit_gate3_request(review)
            self.done = True
            return

        self._revision_round += 1
        directive = {
            "kind": "revision_request",
            "round": self._revision_round,
            "issues": issues,
            "constraints": self._constraints,
            "diagnosis": self._diagnosis,
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=PROMPT_AGENT,
            msg_type="negotiate",
            content=json.dumps(directive, ensure_ascii=False),
        )

    @staticmethod
    def _validate_proposal(proposal: dict[str, Any], constraints: dict[str, Any]) -> tuple[bool, list[str]]:
        issues: list[str] = []
        required_fields = ["version", "target_test", "changes", "comparability_guardrails"]
        for rf in required_fields:
            if rf not in proposal:
                issues.append(f"missing required field: {rf}")

        # Comparability checks
        keep_versions = constraints.get("must_keep_prompt_versions", [])
        if keep_versions and not set(keep_versions).issubset(set(proposal.get("comparability_guardrails", {}).get("preserve_versions", []))):
            issues.append("proposal does not preserve required prompt versions")

        forbidden = constraints.get("forbidden_changes", [])
        proposal_change_text = json.dumps(proposal.get("changes", []), ensure_ascii=False).lower()
        for f in forbidden:
            if str(f).lower() in proposal_change_text:
                issues.append(f"contains forbidden change pattern: {f}")

        ok = len(issues) == 0
        return ok, issues

    def _emit_gate3_request(self, review: dict[str, Any]) -> None:
        proposal = self._proposal
        fio.request_gate(
            gate_num=3,
            signed_by=[AGENT_NAME, MARKET_AGENT, PROMPT_AGENT],
            summary={
                "trigger": "prompt_change_review",
                "target_test": proposal.get("target_test", "PromptLadderTest"),
                "proposal_version": proposal.get("version"),
                "open_issues": review.get("issues", []),
            },
            checklist=[
                "Does the prompt change directly address diagnosed root cause?",
                "Do comparability guardrails preserve cross-version analysis validity?",
                "Is rollback plan defined if new prompt regresses H1/H3 behavior?",
            ],
        )

    @staticmethod
    def _build_diagnosis(summary: dict[str, Any], status: dict[str, Any], gate3: dict[str, Any]) -> dict[str, Any]:
        hypotheses = status.get("hypotheses", [])
        h_map = {h.get("id"): h for h in hypotheses}
        h2 = h_map.get("H2", {})
        diagnosis = {
            "type": "prompt_quality_bottleneck",
            "evidence": {
                "layer3_overall": summary.get("overall", {}),
                "h2_status": h2.get("status"),
                "h2_samples": h2.get("samples", 0),
                "gate3_hint": gate3.get("summary", {}),
            },
            "target_test": "PromptLadderTest",
            "objective": "Improve H2 without degrading H1/H3 comparability",
            "suggested_actions": [
                "Refine user prompt to sharpen mechanism explanation order.",
                "Keep output JSON schema unchanged.",
                "Preserve baseline prompt versions for cross-version comparability.",
            ],
            "timestamp": time.time(),
        }
        return diagnosis

