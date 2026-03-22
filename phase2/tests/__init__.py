"""
Phase 2 Test Registry.

Provides:
  - test class registration
  - lookup by preferred_test or hypothesis nature
  - convenience helper to run one hypothesis end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base_test import BaseTest, HypothesisResult, Metrics, RawResult, TestEnv
from .fictitious_play_test import FictitiousPlayTest
from .prompt_ladder_test import PromptLadderTest
from .sensitivity_test import SensitivityTest

# Ensure project root is importable when running `python phase2/layer3_coordinator.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TEST_REGISTRY_BY_NAME = {
    "SensitivityTest": SensitivityTest,
    "PromptLadderTest": PromptLadderTest,
    "FictitiousPlayTest": FictitiousPlayTest,
}

TEST_REGISTRY_BY_NATURE = {
    "comparative_static": SensitivityTest,
    "knowledge_dependent": PromptLadderTest,
    "dynamic_convergence": FictitiousPlayTest,
}


def resolve_test_class(hypothesis: dict[str, Any]) -> type[BaseTest]:
    preferred = hypothesis.get("preferred_test")
    nature = hypothesis.get("nature")
    if preferred in TEST_REGISTRY_BY_NAME:
        return TEST_REGISTRY_BY_NAME[preferred]
    if nature in TEST_REGISTRY_BY_NATURE:
        return TEST_REGISTRY_BY_NATURE[nature]
    raise ValueError(
        f"No registered test for preferred_test={preferred!r}, nature={nature!r}"
    )


def run_hypothesis_test(
    hypothesis: dict[str, Any],
    model: str,
    params: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    End-to-end helper for one hypothesis:
      setup -> run -> evaluate -> verify
    """
    test_cls = resolve_test_class(hypothesis)
    test = test_cls()
    env: TestEnv = test.setup(params or {})
    raw: RawResult = test.run(env, model=model)
    metrics: Metrics = test.evaluate(raw, ground_truth=ground_truth)
    result: HypothesisResult = test.verify_hypothesis(metrics, hypothesis=hypothesis)
    return {
        "hypothesis": hypothesis,
        "test_name": test.test_name,
        "nature": test.nature,
        "env": env.__dict__,
        "metrics": metrics.values,
        "result": result.__dict__,
    }
