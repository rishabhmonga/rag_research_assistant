"""memory_store.py
Lightweight persistence helpers:
- save_json(path, obj): atomic write
- load_json(path, default): safe load with default
- rotate_backup(path, keep=3): keep rolling backups (path.bak1, bak2, ...)
"""
from __future__ import annotations

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

__all__ = [
    "save_json",
    "load_json",
    "rotate_backup",
]

def _atomic_write(path: Path, data: bytes) -> None:
    """Atomically write bytes to a file (safe across crashes).
    Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def save_json(path: str | Path, obj: Any) -> None:
    """Safely write a JSON file with UTF-8 encoding."""
    path = Path(path)
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def load_json(path: str | Path, default: Any) -> Any:
    """Load JSON if present/valid, otherwise return default."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def rotate_backup(path: str | Path, keep: int = 3) -> None:
    """Keep rolling backups: file.bak1 (latest), bak2, ... up to `keep`.
    If the main file exists, copy it to .bak1 and shift older backups.
    """
    path = Path(path)
    base = path.name
    dirn = path.parent
    if keep < 1:
        return

    # Shift old backups
    for idx in range(keep, 0, -1):
        src = dirn / f"{base}.bak{idx}"
        dst = dirn / f"{base}.bak{idx+1}"
        if src.exists():
            try:
                os.replace(src, dst)
            except Exception:
                pass

    # Copy current file to .bak1
    if path.exists():
        try:
            shutil.copy2(path, dirn / f"{base}.bak1")
        except Exception:
            pass
