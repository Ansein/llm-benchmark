from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

from . import run_hypothesis_test


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m phase2.tests.job_worker <input_json> <output_json>", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    hypothesis = payload["hypothesis"]
    model = payload["model"]
    params = payload.get("params", {})

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            result = run_hypothesis_test(hypothesis=hypothesis, model=model, params=params)
        out = {
            "ok": True,
            "result": result,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as e:
        out = {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
