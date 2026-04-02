from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


def human_approval_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "human_approval.json"


def load_human_approval(run_dir: str | Path) -> dict | None:
    path = human_approval_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_human_approval(
    run_dir: str | Path,
    decision: Literal["approved", "rejected"],
    notes: str | None = None,
) -> Path:
    path = human_approval_path(run_dir)
    payload = {
        "status": decision,
        "notes": notes or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
