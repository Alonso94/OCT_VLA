"""Bounded, read-only discovery; FOUND is not proof of runtime compatibility."""

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExternalPaths


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class Report:
    paths: ExternalPaths
    checks: tuple[Check, ...]

    @property
    def exit_code(self) -> int:
        return int(any(check.status == "FAIL" for check in self.checks))

    def as_dict(self) -> dict:
        return {
            "paths": self.paths.as_dict(),
            "checks": [asdict(c) for c in self.checks],
            "exit_code": self.exit_code,
        }


def _run(command: list[str], timeout: float) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Probe timed out after {timeout:g}s") from error
    except OSError as error:
        raise ValueError(str(error)) from error
    if result.returncode:
        raise ValueError(
            (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")[:2000]
        )
    return result.stdout.strip()


def task_config_dir(root: Path) -> Path:
    """Recognize both known RoboTwin layouts without importing RoboTwin."""
    if (root / "envs").is_dir():
        for relative in ("env_cfg/task_config", "task_config"):
            directory = root / relative
            if all(
                (directory / name).is_file()
                for name in ("_embodiment_config.yml", "_camera_config.yml")
            ):
                return directory
    raise ValueError(
        "Missing envs/ or embodiment/camera registries in a supported task-config layout"
    )


def _environment(
    name: str, python: Path | None, packages: dict[str, str], expected_minor: str, timeout: float
) -> list[Check]:
    if python is None:
        return [Check(name, "FAIL", "Interpreter path not configured")]
    try:
        payload = json.loads(
            _run(
                [
                    str(python),
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("_probe.py")),
                    json.dumps(packages),
                ],
                timeout,
            )
        )
        version = payload["python"]
        checks = [
            Check(
                name,
                "FOUND" if version.startswith(expected_minor + ".") else "WARN",
                f"{payload['executable']} (Python {version}; expected {expected_minor}.x)",
            )
        ]
        for module, info in payload["packages"].items():
            detail = f"version={info['version']}; source={info['origin']}"
            if info["revision"]:
                detail += f"; revision={info['revision']}"
            checks.append(Check(f"{name}/{module}", "FOUND" if info["origin"] else "FAIL", detail))
        for stale in payload["stale_paths"]:
            checks.append(
                Check(
                    f"{name}/editable path",
                    "WARN",
                    f"{stale['file']} -> missing {stale['target']}; no repair attempted",
                )
            )
        return checks
    except (ValueError, KeyError, TypeError) as error:
        return [Check(name, "FAIL", f"Interpreter probe failed: {error}")]


def diagnose(paths: ExternalPaths, *, timeout: float = 10) -> Report:
    checks = []
    root = paths.robotwin_root
    if root is None:
        checks.append(Check("RoboTwin", "FAIL", "ROBOTWIN_ROOT not configured"))
    else:
        try:
            config = task_config_dir(root)
            checks.append(Check("RoboTwin", "FOUND", f"{root}; task configs={config}"))
        except ValueError as error:
            checks.append(Check("RoboTwin", "FAIL", str(error)))
        try:
            if not (root / ".git").exists():
                raise ValueError("No Git metadata at RoboTwin root; revision unknown")
            revision = _run(["git", "-C", str(root), "rev-parse", "HEAD"], timeout)
            dirty = _run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], timeout
            )
            checks.append(
                Check(
                    "RoboTwin revision",
                    "WARN" if dirty else "FOUND",
                    revision
                    + (
                        "; tracked modifications/submodule changes present"
                        if dirty
                        else "; tracked tree clean (untracked files not checked)"
                    ),
                )
            )
        except ValueError as error:
            checks.append(Check("RoboTwin revision", "WARN", str(error)))
        source = root / "envs/curobo/src/curobo"
        if source.is_dir():
            checks.append(
                Check(
                    "cuRobo source candidate",
                    "FOUND",
                    f"{source}; source presence does not imply interpreter resolution",
                )
            )
    for name, path in (("Dataset root", paths.data_root), ("Asset root", paths.asset_root)):
        checks.append(
            Check(
                name,
                "FOUND" if path and path.is_dir() else "FAIL",
                str(path) if path else "Not configured",
            )
        )
    if paths.output_root is not None:
        path = paths.output_root
        checks.append(
            Check(
                "Output root",
                "FAIL" if path.exists() and not path.is_dir() else "INFO",
                f"{path}; no directory or write probe created",
            )
        )
    checks.extend(
        _environment(
            "Simulator",
            paths.robotwin_python,
            {"sapien": "sapien", "curobo": "nvidia-curobo", "torch": "torch"},
            "3.10",
            timeout,
        )
    )
    checks.extend(
        _environment(
            "Policy",
            paths.policy_python,
            {"lerobot": "lerobot", "torch": "torch", "peft": "peft"},
            "3.12",
            timeout,
        )
    )
    try:
        gpu = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout
        )
        checks.append(Check("GPU driver", "FOUND" if gpu else "FAIL", gpu or "No devices reported"))
    except ValueError as error:
        checks.append(Check("GPU driver", "FAIL", str(error)))
    checks.append(
        Check(
            "Runtime validation",
            "INFO",
            "Metadata discovery only: CUDA kernels, Vulkan rendering, asset completeness, "
            "calibration and physical actuation are not tested",
        )
    )
    return Report(paths, tuple(checks))
