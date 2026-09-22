"""同一アカウントの多重起動を防ぐOSファイルロック。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class AlreadyRunningError(RuntimeError):
    pass


class ProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"0")
            self._stream.flush()
        self._stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self._stream.close()
                self._stream = None
                raise AlreadyRunningError(f"別プロセスが実行中です: {self.path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._stream.close()
                self._stream = None
                raise AlreadyRunningError(f"別プロセスが実行中です: {self.path}") from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(str(os.getpid()).encode("ascii"))
        self._stream.flush()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._stream is None:
            return
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            try:
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None
