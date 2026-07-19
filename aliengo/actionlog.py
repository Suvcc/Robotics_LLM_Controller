import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class ActionLog:
    """Append-only JSONL log. Pass path=None to disable (e.g. in tests)."""

    def __init__(self, path: str | Path | None):
        self._path = Path(path) if path else None
        self._lock = threading.RLock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **entry) -> None:
        if not self._path:
            return
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
