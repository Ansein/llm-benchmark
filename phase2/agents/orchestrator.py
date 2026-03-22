"""
Orchestrator Agent — Layer 2

Responsibilities:
- Validate and finalize market_config from market_config_draft
- Enforce budget/runtime limits and coverage checks
- Raise Gate 2 review request
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "orchestrator"

REQUIRED_TESTS = {"SensitivityTest", "PromptLadderTest", "FictitiousPlayTest"}


class OrchestratorAgent:
    def __init__(self, session_id: str | None = None, max_revision_rounds: int = 3):
        self.session_id = session_id
        self.max_revision_rounds = max_revision_rounds
        self._revision_round = 0
        self._ready = False

    def finalize(self) -> dict[str, Any]:
        logger.info(f"[{AGENT_NAME}] Finalizing Layer 2 plan")
        draft = fio.read_json("market_config_draft.json")
        paradigm = fio.read_yaml("paradigm.yaml")

        finalized = self._apply_constraints(draft)
        checks = self._run_consistency_checks(finalized, paradigm)

        fio.write_json("market_config.yaml", finalized)
        self.sign_off(checks)
        self._build_gate2_request(finalized, checks)

        logger.info(f"[{AGENT_NAME}] market_config.yaml and gate2_request.json written")
        return finalized

    def is_ready(self) -> bool:
        return self._ready

    def handle_pending_messages(self) -> int:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                payload = json.loads(msg.get("content", "{}"))
                if self.session_id and payload.get("session_id") not in (None, self.session_id):
                    fio.mark_message(msg["id"], "done")
                    continue

                kind = payload.get("kind")
                if kind == "draft_ready":
                    self._on_draft_ready(reply_to=msg["id"])
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def sign_off(self, checks: dict[str, Any]) -> None:
        fio.write_json(
            "orchestrator_signoff.json",
            {
                "agent": AGENT_NAME,
                "signed": True,
                "timestamp": time.time(),
                "checks": checks,
            },
        )

    def _on_draft_ready(self, reply_to: str | None = None) -> None:
        draft = fio.read_json("market_config_draft.json")
        paradigm = fio.read_yaml("paradigm.yaml")
        preview = self._apply_constraints(dict(draft))
        checks = self._run_consistency_checks(preview, paradigm)

        if checks.get("coverage_ok", False) and checks.get("budget_ok", False):
            self.finalize()
            self._ready = True
            ack = {
                "kind": "final_accept",
                "session_id": self.session_id,
                "timestamp": time.time(),
            }
            fio.send_message(
                from_agent=AGENT_NAME,
                to_agent="market_designer",
                msg_type="reply",
                content=json.dumps(ack, ensure_ascii=False),
                reply_to=reply_to,
            )
            return

        if self._revision_round >= self.max_revision_rounds:
            # Finalize even if imperfect; Gate 2 open_issues will capture unresolved items.
            self.finalize()
            self._ready = True
            return

        self._revision_round += 1
        changes = self._propose_changes(preview, checks)
        req = {
            "kind": "revision_request",
            "session_id": self.session_id,
            "round": self._revision_round,
            "changes": changes,
            "checks": checks,
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent="market_designer",
            msg_type="negotiate",
            content=json.dumps(req, ensure_ascii=False),
            reply_to=reply_to,
        )

    def _apply_constraints(self, cfg: dict[str, Any]) -> dict[str, Any]:
        # Deep-ish copy via json-compatible transformation not needed: cfg from file.
        tests = cfg.get("tests", {})
        models = cfg.get("models", [])
        limits = cfg.get("global_limits", {})

        # Lightweight budget enforcement:
        # estimate_calls = models * (sens_grid*trials + prompt_versions*rounds + fp_rounds*trials)
        sens = tests.get("SensitivityTest", {})
        prompt = tests.get("PromptLadderTest", {})
        fp = tests.get("FictitiousPlayTest", {})

        sens_grid = len(sens.get("rho_values", [])) * len(sens.get("v_ranges", []))
        sens_trials = int(sens.get("num_trials", 1))
        prompt_versions = len(prompt.get("versions", []))
        prompt_rounds = int(prompt.get("num_rounds", 1))
        fp_rounds = int(fp.get("max_rounds", 0))
        fp_trials = int(fp.get("num_trials", 1))

        est_calls = len(models) * (
            sens_grid * sens_trials + prompt_versions * prompt_rounds + fp_rounds * fp_trials
        )
        budget = int(limits.get("max_total_calls", 2500))

        if est_calls > budget and fp.get("enabled", False):
            # First trim FP rounds to fit budget before trimming model count.
            overflow = est_calls - budget
            per_round_cost = max(len(models) * fp_trials, 1)
            reduce_rounds = min(fp_rounds - 10, (overflow // per_round_cost) + 1) if fp_rounds > 10 else 0
            if reduce_rounds > 0:
                fp["max_rounds"] = fp_rounds - reduce_rounds
                est_calls = len(models) * (
                    sens_grid * sens_trials
                    + prompt_versions * prompt_rounds
                    + int(fp["max_rounds"]) * fp_trials
                )

        cfg["global_limits"]["estimated_total_calls"] = est_calls
        cfg["global_limits"]["budget_feasible"] = est_calls <= budget
        cfg["orchestrated_at"] = time.time()
        return cfg

    def _run_consistency_checks(self, cfg: dict[str, Any], paradigm: dict[str, Any]) -> dict[str, Any]:
        hypotheses = paradigm.get("hypotheses", [])
        tests = cfg.get("tests", {})
        models = cfg.get("models", [])

        preferred = {h.get("preferred_test") for h in hypotheses if h.get("preferred_test")}
        missing_preferred = sorted([t for t in preferred if t not in tests or not tests[t].get("enabled", False)])

        missing_required = sorted([t for t in REQUIRED_TESTS if t not in tests])
        mapping_ok = len(missing_preferred) == 0

        return {
            "hypotheses_count": len(hypotheses),
            "models_count": len(models),
            "missing_required_tests": missing_required,
            "missing_preferred_tests": missing_preferred,
            "coverage_ok": mapping_ok and len(missing_required) == 0,
            "budget_ok": bool(cfg.get("global_limits", {}).get("budget_feasible", False)),
        }

    def _build_gate2_request(self, cfg: dict[str, Any], checks: dict[str, Any]) -> None:
        signed_by = []
        if fio.file_exists("market_designer_signoff.json"):
            signed_by.append("market_designer")
        if fio.file_exists("orchestrator_signoff.json"):
            signed_by.append(AGENT_NAME)

        open_issues: list[str] = []
        if not checks.get("coverage_ok", False):
            open_issues.append("hypothesis-to-test coverage incomplete")
        if not checks.get("budget_ok", False):
            open_issues.append("estimated calls exceed budget")

        fio.request_gate(
            gate_num=2,
            signed_by=signed_by,
            summary={
                "scenario_id": cfg.get("scenario_id"),
                "tests_count": len(cfg.get("tests", {})),
                "models_count": len(cfg.get("models", [])),
                "estimated_total_calls": cfg.get("global_limits", {}).get("estimated_total_calls"),
                "open_issues": open_issues,
            },
            checklist=[
                "market_config faithfully represents paper assumptions and paradigm hypotheses?",
                "test coverage includes H1/H2/H3 with correct preferred test mapping?",
                "budget/runtime limits are acceptable for current API and model pool?",
                "execution policy (retry/resume/fail-fast) aligns with reproducibility requirements?",
            ],
        )

    @staticmethod
    def _propose_changes(cfg: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        limits = cfg.get("global_limits", {})
        tests = cfg.get("tests", {})
        budget = int(limits.get("max_total_calls", 2500))
        est = int(limits.get("estimated_total_calls", budget))

        if not checks.get("budget_ok", False) and est > budget:
            fp = tests.get("FictitiousPlayTest", {})
            fp_rounds = int(fp.get("max_rounds", 50))
            # Trim heavy fp first.
            changes["fp_max_rounds"] = max(10, int(fp_rounds * 0.7))
            changes["models_limit"] = max(1, len(cfg.get("models", [])) - 1)

        if not checks.get("coverage_ok", False):
            # Preserve all models, relax trial intensity for quick convergence.
            changes.setdefault("sensitivity_trials", 1)
            changes.setdefault("prompt_rounds", 1)

        return changes
