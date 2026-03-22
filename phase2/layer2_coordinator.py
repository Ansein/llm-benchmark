"""
Layer 2 Coordinator — interactive network mode

Loop:
1. Validate Gate 1 approval exists
2. MarketDesigner publishes draft_ready message
3. Orchestrator reviews and either:
   - requests revision from MarketDesigner, or
   - finalizes market_config + gate2_request
4. Wait for human Gate 2 approval (optional)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.market_designer import MarketDesignerAgent
from agents.orchestrator import OrchestratorAgent
from tools import file_io as fio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_WAIT_GATE2 = 60 * 30
MAX_LAYER2_INTERACTIVE_SECONDS = 120


def _print_gate2_summary() -> None:
    print("\n" + "=" * 60)
    print("  GATE 2 — HUMAN REVIEW REQUIRED")
    print("=" * 60)
    if fio.file_exists("gate2_request.json"):
        req = fio.read_json("gate2_request.json")
        print(f"\nSigned by: {', '.join(req.get('signed_by', []))}")
        summary = req.get("summary", {})
        print(f"Scenario: {summary.get('scenario_id', '?')}")
        print(f"Tests: {summary.get('tests_count', '?')}, Models: {summary.get('models_count', '?')}")
        print(f"Estimated calls: {summary.get('estimated_total_calls', '?')}")
        if summary.get("open_issues"):
            print(f"Open issues: {summary['open_issues']}")

        print("\nReview checklist:")
        for item in req.get("review_checklist", []):
            print(f"  [ ] {item}")

    print("\nFiles to review:")
    print(f"  {fio.ws('market_config_draft.json')}")
    print(f"  {fio.ws('market_config.yaml')}")
    print(f"  {fio.ws('gate2_request.json')}")
    print("\nTo approve:")
    approve_path = fio.ws("gate2_approved.json")
    print(f"  Write to {approve_path}:")
    print('  {"approved": true, "reviewer_notes": "your notes here"}')
    print("=" * 60 + "\n")


def run(skip_wait: bool = False) -> None:
    if not fio.is_gate_approved(1):
        raise RuntimeError("Gate 1 not approved. Please complete Layer 1 first.")

    session_id = f"layer2-{uuid.uuid4().hex[:8]}"
    logger.info(f"=== Layer 2 interactive session: {session_id} ===")
    designer = MarketDesignerAgent(session_id=session_id)
    orchestrator = OrchestratorAgent(session_id=session_id)

    designer.start_design()

    deadline = time.time() + MAX_LAYER2_INTERACTIVE_SECONDS
    while time.time() < deadline:
        processed = 0
        processed += designer.handle_pending_messages()
        processed += orchestrator.handle_pending_messages()
        if orchestrator.is_ready() and fio.file_exists("gate2_request.json"):
            break
        if processed == 0:
            time.sleep(0.3)
    else:
        raise RuntimeError("Layer 2 interactive loop timed out before producing gate2_request.json")

    _print_gate2_summary()
    if skip_wait:
        logger.info("skip_wait=True, exiting after generating Gate 2 request.")
        return

    logger.info("Waiting for human to write gate2_approved.json ...")
    deadline = time.time() + MAX_WAIT_GATE2
    while time.time() < deadline:
        if fio.is_gate_approved(2):
            logger.info("Gate 2 approved! Layer 2 complete.")
            print("\n[Layer 2 complete] Gate 2 approved. Ready for Layer 3.")
            return
        time.sleep(10)

    logger.warning("Gate 2 approval timed out. Exiting.")
    print("[Warning] Gate 2 was not approved within the timeout. Re-run when ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 2 Coordinator — Phase 2")
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Generate gate2_request.json and exit without waiting for manual approval.",
    )
    args = parser.parse_args()
    run(skip_wait=args.skip_wait)
