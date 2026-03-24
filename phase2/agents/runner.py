"""
Runner Agent — Layer 3 (interactive loop)

Responsibilities:
- Read market_config + paradigm hypotheses
- Execute hypothesis tests in batches
- Publish per-job events to Analysis Agent
- Consume mid-run directives from Analysis Agent and adapt execution
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tools import file_io as fio

logger = logging.getLogger(__name__)

AGENT_NAME = "runner"
ANALYSIS_AGENT = "analysis"


class RunnerAgent:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._stop_requested = False
        self._queue: list[dict[str, Any]] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        self._results: list[dict[str, Any]] = []
        self._retry_on_error = 1
        self._max_total_calls = 2500
        self._estimated_calls_used = 0
        self._tests_cfg: dict[str, Any] = {}
        self._active_job: dict[str, Any] | None = None

    def run_loop(self, max_cycles: int = 2000, idle_sleep: float = 0.2) -> dict[str, Any]:
        logger.info(f"[{AGENT_NAME}] Starting Layer 3 run loop (dry_run={self.dry_run})")
        config = fio.read_json("market_config.yaml")
        paradigm = fio.read_yaml("paradigm.yaml")

        self._tests_cfg = config.get("tests", {})
        self._retry_on_error = int(config.get("execution_policy", {}).get("retry_on_error", 1))
        self._max_total_calls = int(config.get("global_limits", {}).get("max_total_calls", 2500))

        hypotheses = paradigm.get("hypotheses", [])
        models = config.get("models", [])
        self._build_initial_queue(hypotheses=hypotheses, models=models)
        self._write_state(status="running")

        cycle = 0
        while cycle < max_cycles:
            cycle += 1

            self._consume_directives()
            if self._stop_requested:
                logger.info(f"[{AGENT_NAME}] Stop requested by Analysis")
                break

            if self._estimated_calls_used >= self._max_total_calls:
                logger.warning(
                    f"[{AGENT_NAME}] budget reached ({self._estimated_calls_used}/{self._max_total_calls}), stopping"
                )
                break

            if self._queue:
                job = self._queue.pop(0)
                self._execute_job(job)
                self._write_state(status="running")
                continue

            # No queued jobs: if analysis has no more directives soon, we can finish.
            if not self._has_pending_directives():
                break
            time.sleep(idle_sleep)

        summary = self._finalize()
        self._notify_analysis_runner_complete(summary)
        self._write_state(status="completed")
        logger.info(f"[{AGENT_NAME}] Completed run loop")
        return summary

    def _build_initial_queue(self, hypotheses: list[dict[str, Any]], models: list[str]) -> None:
        from tests import resolve_test_class

        for h in hypotheses:
            h_id = h.get("id", "H?")
            try:
                test_name = resolve_test_class(h).test_name
            except Exception as e:
                raise RuntimeError(f"Unable to resolve test class for hypothesis {h_id}: {e}") from e
            if test_name not in self._tests_cfg:
                raise RuntimeError(
                    f"Resolved test {test_name!r} for hypothesis {h_id} is missing from market_config tests"
                )
            for model in models:
                self._enqueue_job(
                    {
                        "hypothesis": h,
                        "hypothesis_id": h_id,
                        "test_name": test_name,
                        "model": model,
                        "retry_count": 0,
                        "source": "initial",
                    }
                )

    def _enqueue_job(self, job: dict[str, Any]) -> str:
        jid = f"{job['hypothesis_id']}|{job['test_name']}|{job['model']}|r{job.get('retry_count', 0)}|{int(time.time()*1000)}"
        job["job_id"] = jid
        self._queue.append(job)
        self._jobs[jid] = job
        return jid

    def _execute_job(self, job: dict[str, Any]) -> None:
        hypothesis = job["hypothesis"]
        test_name = job["test_name"]
        model = job["model"]
        params = self._tests_cfg.get(test_name, {})
        started = time.time()

        logger.info(f"[{AGENT_NAME}] job {job['job_id']} started ({test_name}, model={model})")
        try:
            if self.dry_run:
                result = self._run_hypothesis_test_dry(hypothesis, model, params)
            else:
                result = self._run_hypothesis_test_subprocess(hypothesis=hypothesis, model=model, params=params)

            status = "ok"
            error = None
        except Exception as e:
            status = "error"
            error = str(e)
            result = {
                "hypothesis": hypothesis,
                "test_name": test_name,
                "result": {"hypothesis_id": hypothesis.get("id", "H?"), "passed": False, "verdict": "ERROR"},
                "error": error,
            }

        finished = time.time()
        self._active_job = None
        self._estimated_calls_used += 1
        artifact = self._write_job_artifact(job, result, status=status, error=error, started=started, finished=finished)
        event = {
            "kind": "job_result",
            "job_id": job["job_id"],
            "hypothesis_id": hypothesis.get("id", "H?"),
            "test_name": test_name,
            "model": model,
            "status": status,
            "error": error,
            "artifact_path": artifact,
            "passed": bool(result.get("result", {}).get("passed", False)),
            "verdict": result.get("result", {}).get("verdict", "UNKNOWN"),
            "retry_count": int(job.get("retry_count", 0)),
            "timestamp": finished,
        }
        self._results.append(event)
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=ANALYSIS_AGENT,
            msg_type="notify",
            content=json.dumps(event, ensure_ascii=False),
        )

    def _run_hypothesis_test_dry(self, hypothesis: dict[str, Any], model: str, params: dict[str, Any]) -> dict[str, Any]:
        h_id = hypothesis.get("id", "H?")
        key = f"{h_id}|{model}|{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        score = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        passed = (score % 100) >= 35  # deterministic pseudo pass-rate ~65%
        verdict = "PASS" if passed else "FAIL"
        metrics = {"dry_score": score % 100, "dry_run": True}
        return {
            "hypothesis": hypothesis,
            "test_name": job_test_name(hypothesis),
            "metrics": metrics,
            "result": {"hypothesis_id": h_id, "passed": passed, "verdict": verdict, "details": {"metrics": metrics}},
        }

    def _run_hypothesis_test_subprocess(
        self,
        hypothesis: dict[str, Any],
        model: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="phase2_job_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "job_input.json"
            output_path = tmp_path / "job_output.json"
            stdout_path = tmp_path / "worker_stdout.log"
            stderr_path = tmp_path / "worker_stderr.log"
            input_payload = {
                "hypothesis": hypothesis,
                "model": model,
                "params": params,
            }
            input_path.write_text(json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            cmd = [
                sys.executable,
                "-m",
                "phase2.tests.job_worker",
                str(input_path),
                str(output_path),
            ]
            expected_units = self._estimate_subtasks(test_name=job_test_name(hypothesis), params=params)
            self._active_job = {
                "hypothesis_id": hypothesis.get("id", "H?"),
                "test_name": job_test_name(hypothesis),
                "model": model,
                "started_at": time.time(),
                "expected_units": expected_units,
                "completed_units": 0,
                "recent_artifact_time": None,
                "progress_note": "worker_started",
            }

            with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_f:
                with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_f:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(project_root),
                        stdout=stdout_f,
                        stderr=stderr_f,
                        env=env,
                    )
                    deadline = time.time() + 1800
                    while True:
                        ret = proc.poll()
                        progress = self._collect_subprogress(
                            test_name=job_test_name(hypothesis),
                            model=model,
                            params=params,
                            started_at=self._active_job["started_at"],
                        )
                        self._active_job.update(progress)
                        self._write_state(status="running")
                        if ret is not None:
                            break
                        if time.time() > deadline:
                            proc.kill()
                            proc.wait(timeout=10)
                            raise RuntimeError("worker timeout (>1800s)")
                        time.sleep(5)

            stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-400:]
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-400:]

            if output_path.exists():
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                if payload.get("ok"):
                    return payload["result"]
                raise RuntimeError(
                    "worker exception: "
                    f"{payload.get('error', 'unknown error')}\n"
                    f"{payload.get('traceback', '').strip()}"
                )

            raise RuntimeError(
                "worker exited without structured output: "
                f"returncode={proc.returncode}, "
                f"stdout_tail={stdout_tail!r}, stderr_tail={stderr_tail!r}"
            )

    def _write_job_artifact(
        self,
        job: dict[str, Any],
        result: dict[str, Any],
        status: str,
        error: str | None,
        started: float,
        finished: float,
    ) -> str:
        out_dir = fio.ws("raw_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{job['hypothesis_id']}_{job['test_name']}_{job['model']}_{int(finished*1000)}.json"
        p = out_dir / fname
        payload = {
            "job": job,
            "status": status,
            "error": error,
            "started_at": started,
            "finished_at": finished,
            "result": result,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def _consume_directives(self) -> None:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        for msg in msgs:
            fio.mark_message(msg["id"], "processing")
            try:
                directive = json.loads(msg.get("content", "{}"))
                self._handle_directive(directive)
                fio.mark_message(msg["id"], "done")
            except Exception as e:
                logger.error(f"[{AGENT_NAME}] failed to handle directive: {e}")
                fio.mark_message(msg["id"], "error")

    def _handle_directive(self, directive: dict[str, Any]) -> None:
        kind = directive.get("kind")
        if kind != "directive":
            return

        action = directive.get("action")
        if action == "early_stop":
            self._stop_requested = True
            return

        if action == "retry_failed":
            failed_job_id = directive.get("job_id")
            if failed_job_id and failed_job_id in self._jobs:
                prev = self._jobs[failed_job_id]
                if int(prev.get("retry_count", 0)) < self._retry_on_error:
                    nxt = dict(prev)
                    nxt["retry_count"] = int(prev.get("retry_count", 0)) + 1
                    nxt["source"] = "retry_failed"
                    self._enqueue_job(nxt)
            return

        if action == "adjust_trials":
            test_name = directive.get("test_name")
            new_trials = directive.get("num_trials")
            if test_name in self._tests_cfg and isinstance(new_trials, int) and new_trials >= 1:
                self._tests_cfg[test_name]["num_trials"] = new_trials
            return

        if action == "rerun_hypothesis":
            from tests import resolve_test_class

            hypothesis = directive.get("hypothesis")
            models = directive.get("models", [])
            if not hypothesis:
                return
            test_name = resolve_test_class(hypothesis).test_name
            for model in models:
                self._enqueue_job(
                    {
                        "hypothesis": hypothesis,
                        "hypothesis_id": hypothesis.get("id", "H?"),
                        "test_name": test_name,
                        "model": model,
                        "retry_count": 0,
                        "source": "rerun_hypothesis",
                    }
                )

    def _has_pending_directives(self) -> bool:
        msgs = fio.list_messages(to_agent=AGENT_NAME, status="pending")
        return len(msgs) > 0

    def _notify_analysis_runner_complete(self, summary: dict[str, Any]) -> None:
        event = {
            "kind": "runner_complete",
            "summary": summary,
            "timestamp": time.time(),
        }
        fio.send_message(
            from_agent=AGENT_NAME,
            to_agent=ANALYSIS_AGENT,
            msg_type="notify",
            content=json.dumps(event, ensure_ascii=False),
        )

    def _finalize(self) -> dict[str, Any]:
        out = {
            "agent": AGENT_NAME,
            "completed_at": time.time(),
            "jobs_total": len(self._results),
            "jobs_ok": sum(1 for x in self._results if x["status"] == "ok"),
            "jobs_error": sum(1 for x in self._results if x["status"] == "error"),
            "estimated_calls_used": self._estimated_calls_used,
            "budget_limit": self._max_total_calls,
            "stop_requested": self._stop_requested,
        }
        fio.write_json("raw_results/run_manifest.json", {"summary": out, "jobs": self._results})
        return out

    def _write_state(self, status: str) -> None:
        payload = {
            "status": status,
            "queue_len": len(self._queue),
            "jobs_recorded": len(self._results),
            "estimated_calls_used": self._estimated_calls_used,
            "timestamp": time.time(),
        }
        if self._active_job is not None:
            payload["active_job"] = self._active_job
        fio.write_json("metrics/runner_state.json", payload)

    def _estimate_subtasks(self, test_name: str, params: dict[str, Any]) -> int | None:
        if test_name == "SensitivityTest":
            rho_values = params.get("rho_values", [])
            v_ranges = params.get("v_ranges", [])
            num_trials = int(params.get("num_trials", 1))
            return max(len(rho_values) * len(v_ranges) * num_trials, 1)
        if test_name == "PromptLadderTest":
            versions = params.get("versions", [])
            num_rounds = int(params.get("num_rounds", 1))
            return max(len(versions) * num_rounds, 1)
        if test_name == "FictitiousPlayTest":
            return max(int(params.get("num_trials", 1)) * int(params.get("max_rounds", 50)), 1)
        return None

    def _collect_subprogress(
        self,
        test_name: str,
        model: str,
        params: dict[str, Any],
        started_at: float,
    ) -> dict[str, Any]:
        output_dir = params.get("output_dir")
        if not output_dir:
            return {"completed_units": 0, "recent_artifact_time": None, "progress_note": "no_output_dir"}

        base = Path(output_dir)
        if not base.exists():
            return {"completed_units": 0, "recent_artifact_time": None, "progress_note": "output_dir_not_created"}

        if test_name == "SensitivityTest":
            pattern = f"**/{model}/result_*.json"
            files = [p for p in base.glob(pattern) if p.stat().st_mtime >= started_at - 2]
            latest = max((p.stat().st_mtime for p in files), default=None)
            return {
                "completed_units": len(files),
                "recent_artifact_time": latest,
                "progress_note": "counting_sensitivity_result_files",
            }

        if test_name == "PromptLadderTest":
            files = [
                p for p in base.glob(f"*_{model}_*.json")
                if p.is_file() and not p.name.startswith("summary_") and p.stat().st_mtime >= started_at - 2
            ]
            latest = max((p.stat().st_mtime for p in files), default=None)
            return {
                "completed_units": len(files),
                "recent_artifact_time": latest,
                "progress_note": "counting_prompt_version_result_files",
            }

        if test_name == "FictitiousPlayTest":
            files = [p for p in base.glob(f"**/{model}_*/*") if p.is_file() and p.stat().st_mtime >= started_at - 2]
            latest = max((p.stat().st_mtime for p in files), default=None)
            return {
                "completed_units": len(files),
                "recent_artifact_time": latest,
                "progress_note": "counting_fictitious_play_artifacts",
            }

        files = [p for p in base.glob("**/*") if p.is_file() and p.stat().st_mtime >= started_at - 2]
        latest = max((p.stat().st_mtime for p in files), default=None)
        return {
            "completed_units": len(files),
            "recent_artifact_time": latest,
            "progress_note": "counting_generic_artifacts",
        }


def job_test_name(hypothesis: dict[str, Any]) -> str:
    from tests import resolve_test_class

    return resolve_test_class(hypothesis).test_name
