"""
Workspace file I/O utilities.
All agents read/write through these helpers to ensure consistent paths and formats.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Workspace root
# ---------------------------------------------------------------------------

_PHASE2_DIR = Path(__file__).parent.parent
WORKSPACE = _PHASE2_DIR / "workspace"


def ws(relative: str) -> Path:
    """Resolve a path relative to workspace/."""
    return WORKSPACE / relative


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def read_json(relative: str) -> Any:
    p = ws(relative)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(relative: str, data: Any, indent: int = 2) -> None:
    p = ws(relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def file_exists(relative: str) -> bool:
    return ws(relative).exists()


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def read_yaml(relative: str) -> Any:
    p = ws(relative)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(relative: str, data: Any) -> None:
    p = ws(relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def read_text(relative: str) -> str:
    p = ws(relative)
    return p.read_text(encoding="utf-8")


def write_text(relative: str, content: str) -> None:
    p = ws(relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Inter-agent message queue  (Plan A: file-based)
# ---------------------------------------------------------------------------

MSG_DIR = "messages"


def send_message(
    from_agent: str,
    to_agent: str,
    msg_type: str,
    content: str,
    reply_to: str | None = None,
) -> str:
    """Write a message file. Returns the message id."""
    msg_id = f"{from_agent}_{to_agent}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    payload = {
        "id": msg_id,
        "from": from_agent,
        "to": to_agent,
        "type": msg_type,        # query / reply / notify / negotiate
        "content": content,
        "reply_to": reply_to,
        "status": "pending",
        "timestamp": time.time(),
    }
    write_json(f"{MSG_DIR}/{msg_id}.json", payload)
    return msg_id


def list_messages(
    to_agent: str | None = None,
    status: str | None = "pending",
) -> list[dict]:
    """List messages, optionally filtered by recipient and status."""
    msg_path = WORKSPACE / MSG_DIR
    msg_path.mkdir(parents=True, exist_ok=True)
    messages = []
    for f in sorted(msg_path.glob("*.json")):
        try:
            msg = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if to_agent and msg.get("to") != to_agent:
            continue
        if status and msg.get("status") != status:
            continue
        messages.append(msg)
    return messages


def mark_message(msg_id: str, status: str) -> None:
    """Update a message's status field (e.g. 'processing', 'done')."""
    relative = f"{MSG_DIR}/{msg_id}.json"
    if not file_exists(relative):
        return
    msg = read_json(relative)
    msg["status"] = status
    write_json(relative, msg)


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def request_gate(gate_num: int, signed_by: list[str], summary: dict, checklist: list[str]) -> None:
    write_json(f"gate{gate_num}_request.json", {
        "status": "ready_for_review",
        "signed_by": signed_by,
        "timestamp": time.time(),
        "summary": summary,
        "review_checklist": checklist,
    })


def is_gate_approved(gate_num: int) -> bool:
    key = f"gate{gate_num}_approved.json"
    if not file_exists(key):
        return False
    data = read_json(key)
    return bool(data.get("approved"))
