"""Explicit external paths; loading configuration never installs or creates files."""

import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

ENV_KEYS = {
    "robotwin_root": "ROBOTWIN_ROOT",
    "robotwin_python": "OCTVLA_ROBOTWIN_PYTHON",
    "policy_python": "OCTVLA_POLICY_PYTHON",
    "data_root": "OCTVLA_DATA_ROOT",
    "asset_root": "OCTVLA_ASSET_ROOT",
    "output_root": "OCTVLA_OUTPUT_ROOT",
}


@dataclass(frozen=True)
class ExternalPaths:
    robotwin_root: Path | None = None
    robotwin_python: Path | None = None
    policy_python: Path | None = None
    data_root: Path | None = None
    asset_root: Path | None = None
    output_root: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            field.name: str(value) if (value := getattr(self, field.name)) else None
            for field in fields(self)
        }


def read_env_file(path: Path) -> dict[str, str]:
    """Read a small dotenv subset without shell execution or variable expansion."""
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if not separator or key not in ENV_KEYS.values():
            raise ValueError(f"{path}:{number}: expected a supported path variable assignment")
        tokens = shlex.split(value, comments=True, posix=True)
        if len(tokens) != 1 or not tokens[0].strip():
            raise ValueError(f"{path}:{number}: expected one nonempty path (quote spaces)")
        result[key] = tokens[0]
    return result


def resolve_paths(
    overrides: Mapping[str, str | None],
    *,
    config: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> ExternalPaths:
    """Precedence: CLI > process environment > explicit env file > JSON config.

    File-relative values resolve against that file's directory. CLI/environment
    values resolve against cwd. Interpreter symlinks are NOT dereferenced: doing
    so can bypass a virtual environment's pyvenv.cfg.
    """
    cwd = Path.cwd() if cwd is None else cwd.absolute()
    environment = os.environ if environ is None else environ
    defaults = {}
    config_base = cwd
    if config is not None:
        config = (cwd / config).absolute()
        defaults = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(defaults, dict) or set(defaults) - ENV_KEYS.keys():
            raise ValueError("Config must be a JSON object containing only supported path keys")
        config_base = config.parent
    file_values = {}
    env_base = cwd
    if env_file is not None:
        env_file = (cwd / env_file).absolute()
        file_values = read_env_file(env_file)
        env_base = env_file.parent
    if set(overrides) - ENV_KEYS.keys():
        raise ValueError("Unknown path override")
    result = {}
    for key, env_key in ENV_KEYS.items():
        candidates = (
            (overrides.get(key), cwd),
            (environment.get(env_key), cwd),
            (file_values.get(env_key), env_base),
            (defaults.get(key), config_base),
        )
        value, base = next(((v, b) for v, b in candidates if v is not None), (None, cwd))
        if value is None:
            result[key] = None
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        else:
            result[key] = Path(os.path.abspath(base / Path(value).expanduser()))
    root = result["robotwin_root"]
    if root is not None:
        if result["asset_root"] is None:
            result["asset_root"] = root / "assets"
        if result["robotwin_python"] is None:
            result["robotwin_python"] = root / ".venv-robotwin" / "bin" / "python"
    return ExternalPaths(**result)
