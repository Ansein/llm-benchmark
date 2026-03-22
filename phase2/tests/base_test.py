"""
Phase 2 Test Registry base abstractions.

These classes standardize how test types are set up, executed, evaluated,
and converted into hypothesis PASS/FAIL decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestEnv:
    """Prepared environment for a test run."""

    name: str
    params: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RawResult:
    """Raw execution output returned by a test implementation."""

    test_name: str
    payload: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class Metrics:
    """Normalized metrics extracted from RawResult."""

    test_name: str
    values: dict[str, Any]


@dataclass
class HypothesisResult:
    """Final hypothesis verification result."""

    hypothesis_id: str
    passed: bool
    verdict: str
    details: dict[str, Any]


class BaseTest(ABC):
    """
    Base interface for all test types in Phase 2.

    Contract:
      - setup() prepares a TestEnv
      - run() executes the experiment and returns RawResult
      - evaluate() computes standardized Metrics
      - verify_hypothesis() turns metrics into PASS/FAIL
    """

    test_name: str = "base_test"
    nature: str = "unknown"

    @abstractmethod
    def setup(self, params: dict[str, Any]) -> TestEnv:
        raise NotImplementedError

    @abstractmethod
    def run(self, env: TestEnv, model: str) -> RawResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, raw: RawResult, ground_truth: dict[str, Any] | None = None) -> Metrics:
        raise NotImplementedError

    @abstractmethod
    def verify_hypothesis(self, metrics: Metrics, hypothesis: dict[str, Any]) -> HypothesisResult:
        raise NotImplementedError

    @staticmethod
    def _ts() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _ensure_dir(path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

