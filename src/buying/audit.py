"""秘密値を含めないJSON Lines監査ログ。"""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, log_dir: Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{dt.date.today():%Y%m%d}_buying.jsonl"
        self.secrets = tuple(value for value in secrets if value)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        for secret in self.secrets:
            raw = raw.replace(secret, "***")
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(raw + "\n")

