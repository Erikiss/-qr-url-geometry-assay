from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO


def open_text(path: str | Path, mode: str) -> IO[str]:
    target = Path(path)
    if "b" in mode:
        raise ValueError("open_text only accepts text modes")
    if target.suffix.lower() == ".gz":
        kwargs = {"encoding": "utf-8", "newline": ""}
        if "w" in mode or "a" in mode or "x" in mode:
            kwargs["compresslevel"] = 1
        return gzip.open(target, mode, **kwargs)
    return target.open(mode, encoding="utf-8", newline="")
