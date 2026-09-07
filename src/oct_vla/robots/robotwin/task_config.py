"""Resolve RoboTwin's dual-Franka setup kwargs without importing RoboTwin.

Only the embodiment is overridden here. Camera, domain-randomization, and
data_type fields all come from the named task_config file, so scene behaviour
matches what that file already specifies rather than a second, drifting copy.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

FRANKA_EMBODIMENT = "franka-panda"
FRANKA_ARM_SEPARATION_M = 0.8


def _task_config_candidates(root: Path) -> tuple[Path, Path]:
    return root / "env_cfg" / "task_config", root / "task_config"


def resolve_task_config_dir(root: Path) -> Path:
    """Return the task-config directory for modern or legacy RoboTwin layouts."""
    for candidate in _task_config_candidates(root):
        if (candidate / "_embodiment_config.yml").is_file() and (
            candidate / "_camera_config.yml"
        ).is_file():
            return candidate
    checked = " or ".join(str(candidate) for candidate in _task_config_candidates(root))
    raise FileNotFoundError(f"No RoboTwin task_config registry found at {checked}")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def build_dual_franka_setup(
    root: Path,
    *,
    task_name: str,
    task_config: str = "demo_clean",
    eval_mode: bool = False,
) -> dict[str, Any]:
    """Build the keyword arguments RoboTwin's ``setup_demo`` expects.

    Selects two Franka-Panda arms regardless of what the named task_config's
    stock embodiment is; every other field is left as that file specifies it.
    """
    config_dir = resolve_task_config_dir(root)
    config_path = config_dir / f"{task_config}.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"RoboTwin task config not found: {config_path}")

    setup = deepcopy(_load_yaml(config_path))
    setup["embodiment"] = [FRANKA_EMBODIMENT, FRANKA_EMBODIMENT, FRANKA_ARM_SEPARATION_M]
    setup["task_name"] = task_name
    setup["task_config"] = task_config
    setup["eval_mode"] = eval_mode
    setup["need_plan"] = False
    setup["save_data"] = False
    setup["render_freq"] = 0

    save_path = Path(setup.get("save_path", "data"))
    setup["save_path"] = str(save_path if save_path.is_absolute() else root / save_path)

    embodiment_types = _load_yaml(config_dir / "_embodiment_config.yml")
    try:
        robot_dir = Path(embodiment_types[FRANKA_EMBODIMENT]["file_path"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"RoboTwin does not define embodiment {FRANKA_EMBODIMENT!r}") from error
    robot_dir = (robot_dir if robot_dir.is_absolute() else root / robot_dir).resolve()
    embodiment_config = _load_yaml(robot_dir / "config.yml")

    setup["left_robot_file"] = str(robot_dir)
    setup["right_robot_file"] = str(robot_dir)
    setup["left_embodiment_config"] = deepcopy(embodiment_config)
    setup["right_embodiment_config"] = deepcopy(embodiment_config)
    setup["dual_arm_embodied"] = False
    setup["embodiment_dis"] = FRANKA_ARM_SEPARATION_M

    camera = setup.get("camera")
    if not isinstance(camera, dict) or "head_camera_type" not in camera:
        raise ValueError(f"Missing camera.head_camera_type in {config_path}")
    return setup
