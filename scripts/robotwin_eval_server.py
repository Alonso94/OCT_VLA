#!/usr/bin/env python
"""Run the shelf-restock simulator as a server. RoboTwin Python 3.10 only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.serve.server import ShelfRestockEvalServer
    from oct_vla.tasks.shelf_restock.collect import DEFAULT_HZ

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="Written once the socket is listening, so a launcher can wait on it",
    )
    args = parser.parse_args()

    def make_port(task_class: str) -> RoboTwinNativePort:
        return RoboTwinNativePort(
            args.robotwin_root,
            task_name=f"oct_vla.tasks.shelf_restock.robotwin_env:{task_class}",
            task_config="demo_clean",
        )

    server = ShelfRestockEvalServer(make_port, hz=args.hz)
    try:
        server.serve_forever(
            args.host, args.port, ready_file=str(args.ready_file) if args.ready_file else None
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
