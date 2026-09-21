#!/usr/bin/env python
"""Write a deterministic dry-run manifest for the bounded experiment plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oct_vla.experiments import (  # noqa: E402
    ExperimentCell,
    ThroughputMeasurement,
    build_manifest,
)


def _read_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        if "cells" in value:
            value = value["cells"]
        elif "throughput_measurements" in value:
            value = value["throughput_measurements"]
    if not isinstance(value, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True, type=Path, help="JSON list of experiment cells")
    parser.add_argument(
        "--throughput",
        required=True,
        type=Path,
        help="JSON list of measured per-stage GPU-hour records",
    )
    parser.add_argument("--output", type=Path, help="Write manifest JSON here")
    args = parser.parse_args()

    cells = [ExperimentCell.from_dict(row) for row in _read_json_list(args.cells)]
    measurements = [
        ThroughputMeasurement.from_dict(row) for row in _read_json_list(args.throughput)
    ]
    manifest = build_manifest(cells, measurements)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
