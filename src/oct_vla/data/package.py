"""Portable manifest generation for finalized LeRobot datasets."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


def build_dataset_manifest(root: str | Path) -> dict:
    """Describe every portable dataset file without changing the dataset."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a finalized LeRobot dataset: {info_path} is missing")
    info = json.loads(info_path.read_text())
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "format": "oct-vla-lerobot-package-v1",
        "lerobot_codebase_version": info.get("codebase_version"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "features": info.get("features"),
        "files": files,
    }
