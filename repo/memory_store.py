# memory_store.py
import json, os, tempfile, shutil
from typing import Any

def _atomic_write(path: str, data: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def rotate_backup(path: str, keep: int = 3):
    import glob
    base = os.path.basename(path)
    dirn = os.path.dirname(path) or "."
    # shift old backups
    for idx in range(keep, 0, -1):
        src = os.path.join(dirn, f"{base}.bak{idx}")
        dst = os.path.join(dirn, f"{base}.bak{idx+1}")
        if os.path.exists(src):
            try: os.replace(src, dst)
            except Exception: pass
    if os.path.exists(path):
        try: shutil.copy2(path, f"{path}.bak1")
        except Exception:
            pass


def save_json(path: str, obj: Any):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def load_json(path: str, default):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except Exception:
        return default
