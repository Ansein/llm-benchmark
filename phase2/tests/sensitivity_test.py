"""
Comparative-static test implementation for Scenario B.

Reuses Phase 1 sensitivity experiment pipeline (run_sensitivity_b.py).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .base_test import BaseTest, HypothesisResult, Metrics, RawResult, TestEnv


class SensitivityTest(BaseTest):
    test_name = "SensitivityTest"
    nature = "comparative_static"

    def setup(self, params: dict[str, Any]) -> TestEnv:
        merged = {
            "rho_values": [0.3, 0.6, 0.9],
            "v_ranges": [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)],
            "prompt_version": "b.v4",
            "num_trials": 1,
            "output_dir": "phase2/workspace/raw_results/sensitivity",
            "config_file": "configs/model_configs.json",
        }
        merged.update(params or {})

        # Ensure tuple shape for stable keys.
        merged["v_ranges"] = [tuple(vr) for vr in merged["v_ranges"]]
        return TestEnv(name=self.test_name, params=merged)

    def run(self, env: TestEnv, model: str) -> RawResult:
        from run_sensitivity_b import run_sensitivity_experiment

        results, exp_dir = run_sensitivity_experiment(
            rho_values=env.params["rho_values"],
            v_ranges=env.params["v_ranges"],
            prompt_version=env.params["prompt_version"],
            model_names=[model],
            num_trials=int(env.params.get("num_trials", 1)),
            output_dir=env.params["output_dir"],
            config_file=env.params["config_file"],
        )
        return RawResult(
            test_name=self.test_name,
            payload={"results": results},
            artifacts={"experiment_dir": str(Path(exp_dir))},
        )

    def evaluate(self, raw: RawResult, ground_truth: dict[str, Any] | None = None) -> Metrics:
        grouped_llm: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        grouped_gt: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        records = raw.payload.get("results", [])

        for rec in records:
            params = rec.get("sensitivity_params", {})
            rho = float(params.get("rho", 0.0))
            key = (
                rec.get("experiment_meta", {}).get("model_name", "unknown_model"),
                f"{params.get('v_min', 'x')}-{params.get('v_max', 'y')}",
            )
            llm_sr = rec.get("metrics", {}).get("llm", {}).get("share_rate")
            gt_sr = rec.get("metrics", {}).get("ground_truth", {}).get("share_rate")
            if llm_sr is not None:
                grouped_llm[key][rho].append(float(llm_sr))
            if gt_sr is not None:
                grouped_gt[key][rho].append(float(gt_sr))

        monotonic_checks: list[bool] = []
        combo_details: dict[str, Any] = {}

        for key in sorted(grouped_llm.keys()):
            by_rho = grouped_llm[key]
            rhos = sorted(by_rho.keys())
            if len(rhos) < 2:
                continue
            means = [sum(by_rho[r]) / len(by_rho[r]) for r in rhos]
            is_non_increasing = all(means[i + 1] <= means[i] + 1e-9 for i in range(len(means) - 1))
            monotonic_checks.append(is_non_increasing)
            combo_details[f"{key[0]}|v={key[1]}"] = {
                "rho_grid": rhos,
                "llm_share_rate_mean": means,
                "non_increasing": is_non_increasing,
            }

        monotonic_ratio = (sum(1 for x in monotonic_checks if x) / len(monotonic_checks)) if monotonic_checks else 0.0
        return Metrics(
            test_name=self.test_name,
            values={
                "num_records": len(records),
                "num_combinations": len(monotonic_checks),
                "monotonic_non_increasing_ratio": monotonic_ratio,
                "all_combinations_non_increasing": all(monotonic_checks) if monotonic_checks else False,
                "details": combo_details,
                "artifacts": raw.artifacts,
            },
        )

    def verify_hypothesis(self, metrics: Metrics, hypothesis: dict[str, Any]) -> HypothesisResult:
        passed = bool(metrics.values.get("all_combinations_non_increasing", False))
        return HypothesisResult(
            hypothesis_id=hypothesis.get("id", "H1"),
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            details={
                "test_name": self.test_name,
                "success_criterion": hypothesis.get(
                    "success_criterion",
                    "share_rate should be non-increasing as rho increases",
                ),
                "metrics": metrics.values,
            },
        )

