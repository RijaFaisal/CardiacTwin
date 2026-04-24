"""
storage/session_store.py

Persist analysis results to disk so export and simulate routes can
retrieve them without the client re-sending the full JSON body.

Files are stored as JSON in storage/sessions/{session_id}.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DIR = Path("storage/sessions")
_DIR.mkdir(parents=True, exist_ok=True)


def save(session_id: str, data: Any) -> None:
    payload = data.model_dump() if hasattr(data, "model_dump") else data
    try:
        (_DIR / f"{session_id}.json").write_text(
            json.dumps(payload, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not persist session %s: %s", session_id, exc)


def load(session_id: str) -> dict | None:
    p = _DIR / f"{session_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read session %s: %s", session_id, exc)
        return None
