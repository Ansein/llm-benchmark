"""
Layer 3 Coordinator — interactive Runner <-> Analysis loop

Usage:
  python phase2/layer3_coordinator.py --dry-run
  python phase2/layer3_coordinator.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.analysis import AnalysisAgent
from agents.runner import RunnerAgent
from tools import file_io as fio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _print_layer3_summary() -> None:
    if not fio.file_exists("metrics/layer3_summary.json"):
        print("[Layer 3] summary not found.")
        return
    s = fio.read_json("metrics/layer3_summary.json")
    overall = s.get("overall", {})
    print("\n" + "=" * 60)
    print("  LAYER 3 SUMMARY")
    print("=" * 60)
    print(f"Events processed: {s.get('events_processed', 0)}")
    print(
        f"Hypothesis status: PASS={overall.get('pass_count', 0)}, "
        f"FAIL={overall.get('fail_count', 0)}, "
        f"INCONCLUSIVE={overall.get('inconclusive_count', 0)}"
    )
    runner_summary = s.get("runner_summary", {})
    print(
        f"Runner jobs: total={runner_summary.get('jobs_total', 0)}, "
        f"ok={runner_summary.get('jobs_ok', 0)}, "
        f"error={runner_summary.get('jobs_error', 0)}"
    )
    print(f"Calls used: {runner_summary.get('estimated_calls_used', 0)}/{runner_summary.get('budget_limit', 0)}")
    if fio.file_exists("gate3_request.json"):
        print("Gate 3 request generated: review is required before prompt change.")
    else:
        print("Gate 3 not triggered.")
    print("=" * 60 + "\n")


def _write_emergency_summary(runner_error: Exception | None = None, analysis_error: Exception | None = None) -> None:
    if fio.file_exists("metrics/layer3_summary.json"):
        return

    runner_summary = {}
    if fio.file_exists("raw_results/run_manifest.json"):
        runner_summary = fio.read_json("raw_results/run_manifest.json").get("summary", {})
    elif fio.file_exists("metrics/runner_state.json"):
        state = fio.read_json("metrics/runner_state.json")
        runner_summary = {
            "jobs_total": state.get("jobs_recorded", 0),
            "jobs_ok": 0,
            "jobs_error": 0,
            "estimated_calls_used": state.get("estimated_calls_used", 0),
            "budget_limit": None,
            "status": state.get("status"),
        }

    status_data = {
        "overall": {"pass_count": 0, "fail_count": 0, "inconclusive_count": 0},
        "hypotheses": [],
    }
    if fio.file_exists("metrics/hypothesis_status.json"):
        status_data = fio.read_json("metrics/hypothesis_status.json")

    errors = []
    if runner_error:
        errors.append(f"runner_error: {runner_error}")
    if analysis_error:
        errors.append(f"analysis_error: {analysis_error}")

    fio.write_json(
        "metrics/layer3_summary.json",
        {
            "agent": "analysis",
            "started_at": time.time(),
            "finished_at": time.time(),
            "runner_summary": runner_summary,
            "overall": status_data.get("overall", {}),
            "hypotheses": status_data.get("hypotheses", []),
            "events_processed": 0,
            "incomplete": True,
            "errors": errors,
        },
    )


def run(dry_run: bool = False, ignore_gate: bool = False) -> None:
    if (not ignore_gate) and (not fio.is_gate_approved(2)):
        raise RuntimeError("Gate 2 not approved. Please complete Layer 2 first.")
    if not fio.file_exists("market_config.yaml"):
        raise RuntimeError("market_config.yaml not found.")

    logger.info(f"=== Layer 3 start (dry_run={dry_run}) ===")
    runner = RunnerAgent(dry_run=dry_run)
    analysis = AnalysisAgent()

    runner_result = {}
    analysis_result = {}
    runner_exc = [None]
    analysis_exc = [None]

    def run_analysis():
        try:
            analysis_result["data"] = analysis.run_loop()
        except Exception as e:
            analysis_exc[0] = e
            logger.error(f"Analysis loop failed: {e}")

    def run_runner():
        try:
            runner_result["data"] = runner.run_loop()
        except Exception as e:
            runner_exc[0] = e
            logger.error(f"Runner loop failed: {e}")

    t_analysis = threading.Thread(target=run_analysis, daemon=True)
    t_runner = threading.Thread(target=run_runner, daemon=True)
    t_analysis.start()
    # Ensure analysis starts consuming before runner emits events.
    time.sleep(0.2)
    t_runner.start()

    t_runner.join()
    t_analysis.join(timeout=30)

    if runner_exc[0] or analysis_exc[0] or t_analysis.is_alive():
        _write_emergency_summary(runner_error=runner_exc[0], analysis_error=analysis_exc[0])

    if runner_exc[0]:
        raise RuntimeError(f"Runner failed: {runner_exc[0]}")
    if analysis_exc[0]:
        raise RuntimeError(f"Analysis failed: {analysis_exc[0]}")
    if t_analysis.is_alive():
        raise RuntimeError("Analysis thread did not converge in timeout window.")

    _print_layer3_summary()
    logger.info("=== Layer 3 completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 3 Coordinator — interactive mode")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling external model APIs.")
    parser.add_argument("--ignore-gate", action="store_true", help="Skip Gate 2 approval check (dev only).")
    args = parser.parse_args()
    run(dry_run=args.dry_run, ignore_gate=args.ignore_gate)
