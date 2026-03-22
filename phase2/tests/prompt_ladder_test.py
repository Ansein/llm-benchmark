"""
Knowledge-injection test implementation for Scenario B.

Reuses Phase 1 prompt ladder experiment controller (run_prompt_experiments.py).
"""

from __future__ import annotations

from typing import Any

from .base_test import BaseTest, HypothesisResult, Metrics, RawResult, TestEnv


class PromptLadderTest(BaseTest):
    test_name = "PromptLadderTest"
    nature = "knowledge_dependent"

    def setup(self, params: dict[str, Any]) -> TestEnv:
        merged = {
            "versions": ["b.v1", "b.v2", "b.v3", "b.v4", "b.v5", "b.v6"],
            "num_rounds": 1,
            "output_dir": "phase2/workspace/raw_results/prompt_ladder",
            "ground_truth_path": "data/ground_truth/scenario_b_result.json",
            "config_file": "configs/model_configs.json",
            "use_theory_platform": True,
        }
        merged.update(params or {})
        return TestEnv(name=self.test_name, params=merged)

    def run(self, env: TestEnv, model: str) -> RawResult:
        from run_prompt_experiments import PromptExperimentController

        controller = PromptExperimentController(
            model_name=model,
            ground_truth_path=env.params["ground_truth_path"],
            output_dir=env.params["output_dir"],
            use_theory_platform=bool(env.params.get("use_theory_platform", True)),
            config_file=env.params["config_file"],
        )
        all_results = controller.run_all_experiments(
            versions=env.params["versions"],
            num_rounds=int(env.params.get("num_rounds", 1)),
        )
        return RawResult(
            test_name=self.test_name,
            payload={"results_by_version": all_results, "versions": list(env.params["versions"])},
            artifacts={"experiment_dir": env.params["output_dir"]},
        )

    def evaluate(self, raw: RawResult, ground_truth: dict[str, Any] | None = None) -> Metrics:
        versions = raw.payload.get("versions", [])
        results = raw.payload.get("results_by_version", {})

        jaccards: list[float] = []
        available_versions: list[str] = []
        for v in versions:
            data = results.get(v, {})
            if "error" in data:
                continue
            metric = data.get("metrics", {}).get("jaccard_similarity_mean")
            if metric is None:
                continue
            jaccards.append(float(metric))
            available_versions.append(v)

        non_decreasing = all(jaccards[i + 1] >= jaccards[i] - 1e-9 for i in range(len(jaccards) - 1)) if len(jaccards) >= 2 else False

        return Metrics(
            test_name=self.test_name,
            values={
                "available_versions": available_versions,
                "jaccard_similarity_mean": jaccards,
                "monotone_non_decreasing": non_decreasing,
                "num_versions_evaluated": len(jaccards),
                "artifacts": raw.artifacts,
            },
        )

    def verify_hypothesis(self, metrics: Metrics, hypothesis: dict[str, Any]) -> HypothesisResult:
        passed = bool(metrics.values.get("monotone_non_decreasing", False))
        return HypothesisResult(
            hypothesis_id=hypothesis.get("id", "H2"),
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            details={
                "test_name": self.test_name,
                "success_criterion": hypothesis.get(
                    "success_criterion",
                    "Jaccard should be monotone non-decreasing along prompt ladder",
                ),
                "metrics": metrics.values,
            },
        )

