"""
Dynamic-convergence test implementation for Scenario B.

Reuses Phase 1 fictitious-play evaluator (ScenarioBEvaluator.simulate_fictitious_play).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .base_test import BaseTest, HypothesisResult, Metrics, RawResult, TestEnv


class FictitiousPlayTest(BaseTest):
    test_name = "FictitiousPlayTest"
    nature = "dynamic_convergence"

    def setup(self, params: dict[str, Any]) -> TestEnv:
        merged = {
            "ground_truth_path": "data/ground_truth/scenario_b_result.json",
            "config_file": "configs/model_configs.json",
            "max_rounds": 50,
            "belief_window": 10,
            "num_trials": 1,
            "output_dir": "phase2/workspace/raw_results/fictitious_play",
        }
        merged.update(params or {})
        return TestEnv(name=self.test_name, params=merged)

    def run(self, env: TestEnv, model: str) -> RawResult:
        from src.evaluators.evaluate_scenario_b import ScenarioBEvaluator
        from src.evaluators.llm_client import LLMClient, load_model_configs

        cfgs = load_model_configs(env.params["config_file"])
        if model not in cfgs:
            raise ValueError(f"Model '{model}' not found in {env.params['config_file']}")

        out_dir = self._ensure_dir(env.params["output_dir"])
        log_dir = out_dir / f"{model}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        llm_client = LLMClient(config=cfgs[model], log_dir=str(log_dir))
        evaluator = ScenarioBEvaluator(
            llm_client=llm_client,
            ground_truth_path=env.params["ground_truth_path"],
            use_theory_platform=True,
        )
        results = evaluator.simulate_fictitious_play(
            max_rounds=int(env.params.get("max_rounds", 50)),
            belief_window=int(env.params.get("belief_window", 10)),
            num_trials=int(env.params.get("num_trials", 1)),
        )
        return RawResult(
            test_name=self.test_name,
            payload={"results": results},
            artifacts={"log_dir": str(Path(log_dir))},
        )

    def evaluate(self, raw: RawResult, ground_truth: dict[str, Any] | None = None) -> Metrics:
        results = raw.payload.get("results", {})
        conv = results.get("convergence_analysis", {})
        strategy_delta = float(conv.get("avg_hamming_distance_last10", 1.0))

        return Metrics(
            test_name=self.test_name,
            values={
                "converged": bool(conv.get("converged", False)),
                "convergence_round": conv.get("convergence_round"),
                "strategy_delta": strategy_delta,
                "final_similarity_to_equilibrium": float(conv.get("final_similarity_to_equilibrium", 0.0)),
                "actual_rounds": int(results.get("actual_rounds", 0)),
                "artifacts": raw.artifacts,
            },
        )

    def verify_hypothesis(self, metrics: Metrics, hypothesis: dict[str, Any]) -> HypothesisResult:
        # Progress criterion: strategy_delta < 0.01 within 50 rounds.
        strategy_delta = float(metrics.values.get("strategy_delta", 1.0))
        convergence_round = metrics.values.get("convergence_round")
        converged = bool(metrics.values.get("converged", False))
        round_ok = convergence_round is not None and int(convergence_round) <= 50
        delta_ok = strategy_delta < 0.01
        passed = converged and round_ok and delta_ok

        return HypothesisResult(
            hypothesis_id=hypothesis.get("id", "H3"),
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            details={
                "test_name": self.test_name,
                "success_criterion": hypothesis.get(
                    "success_criterion",
                    "strategy_delta < 0.01 within 50 rounds",
                ),
                "checks": {
                    "converged": converged,
                    "round_ok": round_ok,
                    "delta_ok": delta_ok,
                },
                "metrics": metrics.values,
            },
        )

