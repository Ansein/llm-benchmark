"""
Analysis Agent — Layer 3 (interactive loop)

Responsibilities:
- Consume Runner job events continuously
- Update rolling hypothesis_status metrics
- Send directives back to Runner for mid-run adjustment
- Produce Layer 3 summary and optional Gate 3 request
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "analysis"
RUNNER_AGENT = "runner"


class AnalysisAgent:
    def __init__(self):
        self._started_at = time.time()
        self._runner_done = False
        self._runner_summary: dict[str, Any] = {}
        self._statuses: dict[str, dict[str, Any]] = {}
        self._raw_events: list[dict[str, Any]] = []
        self._issued_rerun: set[str] = set()

    def run_loop(self, max_idle_cycles: int = 50, idle_sleep: float = 0.3) -> dict[str, Any]:
        logger.info(f"[{AGENT_NAME}] Starting analysis loop")
        paradigm = fio.read_yaml("paradigm.yaml")
        market_cfg = fio.read_json("market_config.yaml")
        hypotheses = paradigm.get("hypotheses", [])
        models = market_cfg.get("models", [])

        for h in hypotheses:
            h_id = h.get("id", "H?")
            self._statuses[h_id] = {
                "hypothesis": h,
                "samples": 0,
                "pass_count": 0,
                "fail_count": 0,
                "error_count": 0,
                "last_verdict": "INCONCLUSIVE",
                "status": "INCONCLUSIVE",
                "models_seen": [],
                "artifacts": [],
            }

        idle_cycles = 0
        while True:
            processed = self._consume_events(models=models)
            if processed == 0:
                idle_cycles += 1
                time.sleep(idle_sleep)
            else:
                idle_cycles = 0

            self._write_hypothesis_status()

            if self._runner_done and idle_cycles >= 3:
                break
            if idle_cycles >= max_idle_cycles and self._runner_done:
                break

        summary = self._finalize_summary()
        self._maybe_trigger_gate3(summary)
        logger.info(f"[{AGENT_NAME}] Analysis loop completed")
        return summary

    def _consume_events(self, models: list[str]) -> int:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        processed = 0
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                ev = json.loads(msg.get("content", "{}"))
                kind = ev.get("kind")
                if kind == "job_result":
                    self._on_job_result(ev, models=models)
                elif kind == "runner_complete":
                    self._runner_done = True
                    self._runner_summary = ev.get("summary", {})
                fio.mark_message(msg["id"], "done")
                processed += 1
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to process message: {e}")
                fio.mark_message(msg["id"], "error")
        return processed

    def _on_job_result(self, ev: dict[str, Any], models: list[str]) -> None:
        self._raw_events.append(ev)
        h_id = ev.get("hypothesis_id", "H?")
        st = self._statuses.get(h_id)
        if st is None:
            return

        st["samples"] += 1
        st["artifacts"].append(ev.get("artifact_path"))
        model = ev.get("model")
        if model and model not in st["models_seen"]:
            st["models_seen"].append(model)

        if ev.get("status") == "error":
            st["error_count"] += 1
            st["last_verdict"] = "ERROR"
            # Ask runner for retry on this failed job.
            self._send_directive(
                {
                    "action": "retry_failed",
                    "job_id": ev.get("job_id"),
                    "reason": "job execution error",
                }
            )
        else:
            if ev.get("passed", False):
                st["pass_count"] += 1
            else:
                st["fail_count"] += 1
            st["last_verdict"] = ev.get("verdict", "UNKNOWN")

        # Determine provisional status.
        if st["pass_count"] > 0 and st["fail_count"] == 0 and st["error_count"] == 0:
            st["status"] = "PASS"
        elif st["fail_count"] > 0 and st["pass_count"] == 0 and st["samples"] >= len(models):
            st["status"] = "FAIL"
        else:
            st["status"] = "INCONCLUSIVE"

        # Mid-run dynamic adjustment: unresolved with mixed outcomes.
        if (
            st["pass_count"] > 0
            and st["fail_count"] > 0
            and h_id not in self._issued_rerun
            and st["samples"] >= max(2, len(models))
        ):
            self._issued_rerun.add(h_id)
            h = st["hypothesis"]
            self._send_directive(
                {
                    "action": "adjust_trials",
                    "test_name": h.get("preferred_test"),
                    "num_trials": 2,
                    "reason": f"{h_id} mixed outcomes; increase robustness",
                }
            )
            self._send_directive(
                {
                    "action": "rerun_hypothesis",
                    "hypothesis": h,
                    "models": models,
                    "reason": f"{h_id} inconclusive after first batch",
                }
            )

        # Early stop if all hypotheses resolved and no errors.
        if self._all_resolved_no_error():
            self._send_directive({"action": "early_stop", "reason": "all hypotheses resolved"})

    def _all_resolved_no_error(self) -> bool:
        if not self._statuses:
            return False
        for st in self._statuses.values():
            if st["status"] == "INCONCLUSIVE":
                return False
            if st["error_count"] > 0:
                return False
        return True

    def _send_directive(self, directive: dict[str, Any]) -> None:
        payload = {"kind": "directive", **directive, "timestamp": time.time()}
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=RUNNER_AGENT,
            msg_type="notify",
            content=json.dumps(payload, ensure_ascii=False),
        )

    def _write_hypothesis_status(self) -> None:
        hypotheses = []
        pass_count = fail_count = inconclusive_count = 0
        for h_id, st in self._statuses.items():
            status = st["status"]
            if status == "PASS":
                pass_count += 1
            elif status == "FAIL":
                fail_count += 1
            else:
                inconclusive_count += 1

            hypotheses.append(
                {
                    "id": h_id,
                    "preferred_test": st["hypothesis"].get("preferred_test"),
                    "success_criterion": st["hypothesis"].get("success_criterion"),
                    "status": status,
                    "samples": st["samples"],
                    "pass_count": st["pass_count"],
                    "fail_count": st["fail_count"],
                    "error_count": st["error_count"],
                    "models_seen": st["models_seen"],
                    "artifacts": st["artifacts"],
                }
            )

        fio.write_json(
            "metrics/hypothesis_status.json",
            {
                "updated_at": time.time(),
                "hypotheses": hypotheses,
                "overall": {
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "inconclusive_count": inconclusive_count,
                },
            },
        )

    def _finalize_summary(self) -> dict[str, Any]:
        status_data = fio.read_json("metrics/hypothesis_status.json")
        summary = {
            "agent": AGENT_NAME,
            "started_at": self._started_at,
            "finished_at": time.time(),
            "runner_summary": self._runner_summary,
            "overall": status_data.get("overall", {}),
            "hypotheses": status_data.get("hypotheses", []),
            "events_processed": len(self._raw_events),
        }
        fio.write_json("metrics/layer3_summary.json", summary)
        return summary

    def _maybe_trigger_gate3(self, summary: dict[str, Any]) -> None:
        # Root-cause trigger v1:
        # prompt bottleneck likely if H2 fails but at least one of H1/H3 passes.
        st_map = {h["id"]: h["status"] for h in summary.get("hypotheses", [])}
        h2_fail = st_map.get("H2") == "FAIL"
        h1_or_h3_pass = st_map.get("H1") == "PASS" or st_map.get("H3") == "PASS"
        if not (h2_fail and h1_or_h3_pass):
            return

        fio.request_gate(
            gate_num=3,
            signed_by=[AGENT_NAME],
            summary={
                "trigger": "prompt_bottleneck_suspected",
                "status_map": st_map,
                "reason": "H2 failed while mechanism/dynamic hypotheses partially passed",
            },
            checklist=[
                "Is H2 failure due to prompt quality rather than mechanism mismatch?",
                "Can prompt updates preserve cross-version comparability?",
                "Should Prompt Engineer be activated for a controlled prompt revision?",
            ],
        )

