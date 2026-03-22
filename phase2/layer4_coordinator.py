"""
Layer 4 Coordinator — interactive optimization loop

Loop participants:
- analysis_l4 (root-cause diagnosis + gate decision)
- market_designer_l4 (comparability constraints)
- prompt_engineer (proposal + revisions)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.analysis_optimizer import Layer4AnalysisAgent
from agents.market_designer_review import MarketDesignerReviewAgent
from agents.prompt_engineer import PromptEngineerAgent
from tools import file_io as fio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(force: bool = False, max_cycles: int = 200, idle_sleep: float = 0.25) -> None:
    if (not force) and (not fio.file_exists("gate3_request.json")):
        raise RuntimeError("gate3_request.json not found. Layer 4 should run only when Gate 3 is triggered.")
    if not fio.file_exists("metrics/layer3_summary.json"):
        raise RuntimeError("metrics/layer3_summary.json not found.")

    analysis = Layer4AnalysisAgent()
    market = MarketDesignerReviewAgent()
    prompt = PromptEngineerAgent()

    analysis.start_session(force=force)

    idle_cycles = 0
    for _ in range(max_cycles):
        processed = 0
        processed += market.handle_pending_messages()
        processed += prompt.handle_pending_messages()
        processed += analysis.handle_pending_messages()

        if analysis.is_done():
            break
        if processed == 0:
            idle_cycles += 1
            if idle_cycles >= 30:
                logger.warning("Layer 4 loop idle timeout reached.")
                break
            time.sleep(idle_sleep)
        else:
            idle_cycles = 0

    _print_summary()


def _print_summary() -> None:
    print("\n" + "=" * 60)
    print("  LAYER 4 SUMMARY")
    print("=" * 60)
    diagnosis = fio.ws("optimization/root_cause_diagnosis.json")
    constraints = fio.ws("optimization/comparability_constraints.json")
    review = fio.ws("optimization/comparability_review.json")
    proposal_files = sorted(fio.ws("optimization").glob("prompt_change_proposal_*.json"))

    print(f"Diagnosis: {diagnosis if diagnosis.exists() else 'missing'}")
    print(f"Constraints: {constraints if constraints.exists() else 'missing'}")
    if proposal_files:
        print(f"Proposal count: {len(proposal_files)} (latest: {proposal_files[-1]})")
    else:
        print("Proposal count: 0")
    print(f"Comparability review: {review if review.exists() else 'missing'}")
    if fio.file_exists("gate3_request.json"):
        print("Gate 3 request is present (ready for human review).")
    else:
        print("Gate 3 request not present.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 4 Coordinator — interactive mode")
    parser.add_argument("--force", action="store_true", help="Run even if gate3_request.json does not exist.")
    args = parser.parse_args()
    run(force=args.force)

