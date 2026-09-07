"""One CLI; only implemented commands are exposed."""

import argparse
import json
from pathlib import Path

from .config import ENV_KEYS, resolve_paths
from .doctor import diagnose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="octvla")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect external dependencies without installing")
    doctor.add_argument("--config", type=Path, help="JSON path configuration")
    doctor.add_argument("--env-file", type=Path, help="Explicit dotenv path file (never executed)")
    doctor.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    for key, variable in ENV_KEYS.items():
        doctor.add_argument("--" + key.replace("_", "-"), help=f"Overrides {variable}")
    args = parser.parse_args(argv)
    try:
        paths = resolve_paths(
            {key: getattr(args, key) for key in ENV_KEYS},
            config=args.config,
            env_file=args.env_file,
        )
        report = diagnose(paths)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for check in report.checks:
            print(f"{check.name:26} {check.status:5}  {check.detail}")
    return report.exit_code
