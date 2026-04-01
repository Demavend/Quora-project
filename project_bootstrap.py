from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()

    for p in (start, *start.parents):
        if (p / "src").exists() and (p / "data").exists():
            return p

    raise FileNotFoundError("Could not find project root with 'src' and 'data' folders.")


def setup_project(start: Path | None = None) -> Path:
    project_root = find_project_root(start)
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return project_root