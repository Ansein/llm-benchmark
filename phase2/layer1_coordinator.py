"""
Layer 1 Coordinator — Plan A

Event loop that:
1. Starts Paper Agent → paper_parse.json
2. Starts Scenario Extractor + Solver Builder in parallel (pseudo-concurrent via polling)
3. Routes inter-agent messages
4. Detects Gate 1 readiness and prompts human review

Usage:
    python layer1_coordinator.py path/to/paper.pdf
    python layer1_coordinator.py  # uses workspace/input/paper.pdf if it exists
"""

import argparse
import logging
import shutil
import sys
import threading
import time
from pathlib import Path

# Ensure phase2/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from agents.paper_agent import PaperAgent, AGENT_NAME as PAPER
from agents.scenario_extractor import ScenarioExtractorAgent, AGENT_NAME as EXTRACTOR
from agents.solver_builder import SolverBuilderAgent, AGENT_NAME as SOLVER
from tools import file_io as fio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 3   # seconds between message routing loops
MAX_WAIT_GATE1 = 60 * 30  # 30 minutes before giving up on Gate 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_signed_off() -> bool:
    """Return True when all three agents have written their sign-off files."""
    return (
        fio.file_exists("paper_signoff.json")
        and fio.file_exists("extractor_signoff.json")
        and fio.file_exists("solver_signoff.json")
    )


def _build_gate1_request(solver_results: dict) -> None:
    """Assemble and write gate1_request.json."""
    signed_by = []
    if fio.file_exists("paper_signoff.json"):
        signed_by.append(PAPER)
    if fio.file_exists("extractor_signoff.json"):
        signed_by.append(EXTRACTOR)
    if fio.file_exists("solver_signoff.json"):
        signed_by.append(SOLVER)
        solver_results = fio.read_json("solver_signoff.json").get("validation_results", {})

    fio.request_gate(
        gate_num=1,
        signed_by=signed_by,
        summary={
            "hypotheses_count": _count_hypotheses(),
            "solver_validation": "passed" if solver_results.get("all_passed") else "failed",
            "open_issues": _collect_open_issues(),
        },
        checklist=[
            "solver correctly implements Kalman-like posterior covariance?",
            "H1 success criterion direction (EAS < 0) is correct?",
            "boundary condition rho=0 produces expected single-user result?",
            "paradigm.yaml hypothesis nature tags are appropriate?",
        ],
    )


def _count_hypotheses() -> int:
    if fio.file_exists("paradigm.yaml"):
        paradigm = fio.read_yaml("paradigm.yaml")
        return len(paradigm.get("hypotheses", []))
    return 0


def _collect_open_issues() -> list[str]:
    issues = []
    # Check if any messages are still pending
    pending = fio.list_messages(status="pending")
    if pending:
        issues.append(f"{len(pending)} messages still pending")
    return issues


def _print_gate1_summary() -> None:
    print("\n" + "=" * 60)
    print("  GATE 1 — HUMAN REVIEW REQUIRED")
    print("=" * 60)

    if fio.file_exists("gate1_request.json"):
        req = fio.read_json("gate1_request.json")
        print(f"\nSigned by: {', '.join(req.get('signed_by', []))}")
        summary = req.get("summary", {})
        print(f"Hypotheses: {summary.get('hypotheses_count', '?')}")
        print(f"Solver validation: {summary.get('solver_validation', '?')}")
        if summary.get("open_issues"):
            print(f"Open issues: {summary['open_issues']}")

        print("\nReview checklist:")
        for item in req.get("review_checklist", []):
            print(f"  [ ] {item}")

    print("\nFiles to review:")
    print(f"  {fio.ws('paper_parse.json')}")
    print(f"  {fio.ws('paradigm.yaml')}")
    print(f"  {fio.ws('solver/solver_b.py')}")
    if fio.file_exists("solver_signoff.json"):
        signoff = fio.read_json("solver_signoff.json")
        val = signoff.get("validation_results", {})
        print(f"\nValidation results:")
        for key, val_item in val.items():
            print(f"  {key}: {val_item}")

    print("\nTo approve:")
    approve_path = fio.ws("gate1_approved.json")
    print(f"  Write to {approve_path}:")
    print('  {"approved": true, "reviewer_notes": "your notes here"}')
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main coordinator loop
# ---------------------------------------------------------------------------

def run(pdf_path: Path, model: str = "claude-opus-4-6") -> None:
    """
    Run the Layer 1 coordination loop.

    Phase A: Paper Agent parses the PDF.
    Phase B: Extractor and Solver Builder run (with message routing between all three).
    Phase C: Wait for Gate 1 sign-off and human approval.
    """

    # ── Setup workspace ───────────────────────────────────────────────
    fio.WORKSPACE.mkdir(parents=True, exist_ok=True)
    (fio.WORKSPACE / "messages").mkdir(exist_ok=True)
    (fio.WORKSPACE / "solver").mkdir(exist_ok=True)

    # Copy PDF into workspace/input/
    dest = fio.ws("input/paper.pdf")
    dest.parent.mkdir(exist_ok=True)
    if pdf_path.resolve() != dest.resolve():
        shutil.copy2(pdf_path, dest)
    logger.info(f"PDF ready at {dest}")

    # Instantiate agents
    paper_agent = PaperAgent(model=model)
    extractor = ScenarioExtractorAgent(model=model)
    solver_builder = SolverBuilderAgent(model=model)

    # ── Phase A: Paper Agent ─────────────────────────────────────────
    logger.info("=== Phase A: Paper Agent parsing ===")
    paper_agent.parse_paper(dest)

    # Paper Agent doesn't have a separate sign-off file — write one now
    fio.write_json("paper_signoff.json", {
        "agent": PAPER,
        "signed": True,
        "timestamp": time.time(),
    })
    logger.info("Paper Agent complete, paper_signoff.json written")

    # ── Phase B: Extractor first, then Solver Builder ────────────────
    # Sequential to avoid rate-limit contention on shared API endpoint.
    # Paper Agent stays active throughout to answer queries from both.
    logger.info("=== Phase B-1: Scenario Extractor ===")

    extractor_done = threading.Event()
    extractor_error = [None]

    def run_extractor():
        try:
            extractor.extract()
        except Exception as e:
            extractor_error[0] = e
            logger.error(f"Extractor failed: {e}")
        finally:
            extractor_done.set()

    t_extractor = threading.Thread(target=run_extractor, daemon=True)
    t_extractor.start()

    while not extractor_done.is_set():
        paper_agent.handle_pending_messages()
        time.sleep(POLL_INTERVAL)
    paper_agent.handle_pending_messages()

    if extractor_error[0]:
        logger.error(f"Extractor error: {extractor_error[0]}")

    logger.info("=== Phase B-2: Solver Builder ===")

    solver_done = threading.Event()
    solver_error = [None]

    def run_solver():
        try:
            solver_builder.build()
        except Exception as e:
            solver_error[0] = e
            logger.error(f"Solver Builder failed: {e}")
        finally:
            solver_done.set()

    t_solver = threading.Thread(target=run_solver, daemon=True)
    t_solver.start()

    while not solver_done.is_set():
        paper_agent.handle_pending_messages()
        time.sleep(POLL_INTERVAL)
    paper_agent.handle_pending_messages()

    if solver_error[0]:
        logger.error(f"Solver Builder error: {solver_error[0]}")

    # ── Phase C: Gate 1 ───────────────────────────────────────────────
    logger.info("=== Phase C: Gate 1 ===")

    solver_val = {}
    if fio.file_exists("solver_signoff.json"):
        solver_val = fio.read_json("solver_signoff.json").get("validation_results", {})

    _build_gate1_request(solver_val)
    _print_gate1_summary()

    # Wait for human approval
    logger.info("Waiting for human to write gate1_approved.json ...")
    deadline = time.time() + MAX_WAIT_GATE1
    while time.time() < deadline:
        if fio.is_gate_approved(1):
            logger.info("Gate 1 approved! Layer 1 complete.")
            print("\n[Layer 1 complete] Gate 1 approved. Ready for Layer 2.")
            return
        time.sleep(10)

    logger.warning("Gate 1 approval timed out. Exiting.")
    print("[Warning] Gate 1 was not approved within the timeout. Re-run when ready.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 1 Coordinator — Phase 2")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(fio.ws("input/paper.pdf")),
        help="Path to the paper PDF (default: workspace/input/paper.pdf)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Claude model to use for all agents",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Usage: python layer1_coordinator.py path/to/paper.pdf")
        sys.exit(1)

    run(pdf_path=pdf_path, model=args.model)
