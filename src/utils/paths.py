from __future__ import annotations

from pathlib import Path


def project_root_from_file(file_path: str | Path, levels_up: int = 1) -> Path:
    path = Path(file_path).resolve()
    return path.parents[levels_up]


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
