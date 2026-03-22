"""
Market Designer Review Agent — Layer 4

Responsibilities:
- Translate diagnosis into comparability and setup constraints
- Review prompt proposal impact on experimental comparability
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "market_designer_l4"
ANALYSIS_AGENT = "analysis_l4"


class MarketDesignerReviewAgent:
    def __init__(self):
        self._constraints: dict[str, Any] = {}

    def handle_pending_messages(self) -> int:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                payload = json.loads(msg.get("content", "{}"))
                kind = payload.get("kind")
                if kind == "root_cause_diagnosis":
                    self._constraints = self._build_constraints(payload.get("diagnosis", {}))
                    fio.write_json("optimization/comparability_constraints.json", self._constraints)
                    self._send_constraints()
                elif kind == "need_constraints":
                    if not self._constraints:
                        self._constraints = self._build_constraints({})
                        fio.write_json("optimization/comparability_constraints.json", self._constraints)
                    self._send_constraints()
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def _send_constraints(self) -> None:
        payload = {
            "kind": "comparability_constraints",
            "constraints": self._constraints,
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=ANALYSIS_AGENT,
            msg_type="reply",
            content=json.dumps(payload, ensure_ascii=False),
        )

    @staticmethod
    def _build_constraints(diagnosis: dict[str, Any]) -> dict[str, Any]:
        return {
            "must_keep_prompt_versions": ["b.v1", "b.v2", "b.v3", "b.v4", "b.v5", "b.v6"],
            "forbidden_changes": [
                "change output JSON schema",
                "modify payoff definition",
                "modify solver or ground-truth logic",
            ],
            "required_ablation": {
                "baseline_versions": ["b.v4", "b.v6"],
                "new_version_slot": "b.v7",
            },
            "evaluation_guardrails": {
                "same_models_as_layer3": True,
                "same_test_type": diagnosis.get("target_test", "PromptLadderTest"),
            },
            "timestamp": time.time(),
        }

