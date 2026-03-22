"""
Market Designer Agent — Layer 2

Responsibilities:
- Read paradigm.yaml (+ optional paper_parse.json metadata)
- Build a draft market_config with test plans for each hypothesis
- Sign off for Gate 2 preparation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "market_designer"


class MarketDesignerAgent:
    def __init__(self, config_file: str = "configs/model_configs.json", session_id: str | None = None):
        self.config_file = config_file
        self.session_id = session_id
        self._revision = 0

    def design(self) -> dict[str, Any]:
        logger.info(f"[{AGENT_NAME}] Building market config draft")
        paradigm = fio.read_yaml("paradigm.yaml")
        paper_parse = fio.read_json("paper_parse.json") if fio.file_exists("paper_parse.json") else {}

        hypotheses = paradigm.get("hypotheses", [])
        selected_models = self._select_models(limit=3)

        tests: dict[str, dict[str, Any]] = {}
        hypothesis_map: dict[str, dict[str, Any]] = {}
        for h in hypotheses:
            h_id = h.get("id", "H?")
            preferred_test = h.get("preferred_test")
            test_cfg = self._build_test_config(preferred_test)
            tests[preferred_test] = test_cfg
            hypothesis_map[h_id] = {
                "nature": h.get("nature"),
                "preferred_test": preferred_test,
                "success_criterion": h.get("success_criterion", ""),
            }

        draft = {
            "version": "phase2.layer2.v1",
            "scenario_id": paradigm.get("scenario_id", "unknown_scenario"),
            "paper": paradigm.get("paper", paper_parse.get("title", "unknown_paper")),
            "created_at": time.time(),
            "created_by": AGENT_NAME,
            "models": selected_models,
            "global_limits": {
                "max_total_calls": 2500,
                "max_runtime_minutes": 180,
                "seed": 42,
                "max_parallel_jobs": 2,
            },
            "hypothesis_map": hypothesis_map,
            "tests": tests,
            "execution_policy": {
                "fail_fast": False,
                "retry_on_error": 1,
                "resume_from_artifacts": True,
            },
        }

        fio.write_json("market_config_draft.json", draft)
        self.sign_off(draft)
        logger.info(f"[{AGENT_NAME}] market_config_draft.json written")
        return draft

    def start_design(self) -> dict[str, Any]:
        draft = self.design()
        self._notify_orchestrator_draft_ready(draft)
        return draft

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
                if kind == "revision_request":
                    self._revision += 1
                    draft = fio.read_json("market_config_draft.json")
                    revised = self._apply_revision(draft, payload.get("changes", {}))
                    revised["revision"] = self._revision
                    revised["revised_at"] = time.time()
                    fio.write_json("market_config_draft.json", revised)
                    self._notify_orchestrator_draft_ready(revised)
                elif kind == "query_status":
                    reply = {
                        "kind": "designer_status",
                        "session_id": self.session_id,
                        "revision": self._revision,
                        "timestamp": time.time(),
                    }
                    fio.send_message(
                        from_agent=AGENT_NAME,
                        to_agent="orchestrator",
                        msg_type="reply",
                        content=json.dumps(reply, ensure_ascii=False),
                        reply_to=msg["id"],
                    )
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def sign_off(self, draft: dict[str, Any]) -> None:
        fio.write_json(
            "market_designer_signoff.json",
            {
                "agent": AGENT_NAME,
                "signed": True,
                "timestamp": time.time(),
                "summary": {
                    "scenario_id": draft.get("scenario_id"),
                    "models_count": len(draft.get("models", [])),
                    "tests_count": len(draft.get("tests", {})),
                },
            },
        )

    def _notify_orchestrator_draft_ready(self, draft: dict[str, Any]) -> None:
        payload = {
            "kind": "draft_ready",
            "session_id": self.session_id,
            "revision": draft.get("revision", self._revision),
            "draft_path": str(fio.ws("market_config_draft.json")),
            "summary": {
                "models_count": len(draft.get("models", [])),
                "tests_count": len(draft.get("tests", {})),
            },
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent="orchestrator",
            msg_type="notify",
            content=json.dumps(payload, ensure_ascii=False),
        )

    def _select_models(self, limit: int = 3) -> list[str]:
        p = Path(self.config_file)
        if not p.exists():
            logger.warning(f"[{AGENT_NAME}] model config not found: {self.config_file}")
            return ["gpt-5.2"]
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            names = [x.get("config_name") for x in data if x.get("config_name")]
            return names[:limit] if names else ["gpt-5.2"]
        except Exception as e:
            logger.warning(f"[{AGENT_NAME}] failed to parse {self.config_file}: {e}")
            return ["gpt-5.2"]

    @staticmethod
    def _build_test_config(test_name: str) -> dict[str, Any]:
        if test_name == "SensitivityTest":
            return {
                "enabled": True,
                "rho_values": [0.3, 0.6, 0.9],
                "v_ranges": [[0.3, 0.6], [0.6, 0.9], [0.9, 1.2]],
                "prompt_version": "b.v4",
                "num_trials": 2,
                "output_dir": "phase2/workspace/raw_results/sensitivity",
            }
        if test_name == "PromptLadderTest":
            return {
                "enabled": True,
                "versions": ["b.v1", "b.v2", "b.v3", "b.v4", "b.v5", "b.v6"],
                "num_rounds": 1,
                "use_theory_platform": True,
                "output_dir": "phase2/workspace/raw_results/prompt_ladder",
            }
        if test_name == "FictitiousPlayTest":
            return {
                "enabled": True,
                "max_rounds": 50,
                "belief_window": 10,
                "num_trials": 1,
                "output_dir": "phase2/workspace/raw_results/fictitious_play",
            }
        return {"enabled": False, "note": f"unrecognized test type: {test_name}"}

    @staticmethod
    def _apply_revision(draft: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
        cfg = draft
        limits = cfg.setdefault("global_limits", {})
        tests = cfg.setdefault("tests", {})

        if "max_total_calls" in changes:
            limits["max_total_calls"] = int(changes["max_total_calls"])
        if "models_limit" in changes:
            limit = max(1, int(changes["models_limit"]))
            cfg["models"] = list(cfg.get("models", []))[:limit]
        if "fp_max_rounds" in changes and "FictitiousPlayTest" in tests:
            tests["FictitiousPlayTest"]["max_rounds"] = max(5, int(changes["fp_max_rounds"]))
        if "sensitivity_trials" in changes and "SensitivityTest" in tests:
            tests["SensitivityTest"]["num_trials"] = max(1, int(changes["sensitivity_trials"]))
        if "prompt_rounds" in changes and "PromptLadderTest" in tests:
            tests["PromptLadderTest"]["num_rounds"] = max(1, int(changes["prompt_rounds"]))

        cfg.setdefault("design_notes", []).append(
            {
                "type": "revision_applied",
                "changes": changes,
                "timestamp": time.time(),
            }
        )
        return cfg
