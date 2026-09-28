"""The appearance-shift tier changes appearance only, never scene geometry."""

import os
from pathlib import Path

import pytest

from oct_vla.robots.robotwin.task_config import build_dual_franka_setup

ROOT = Path(os.environ.get("OCTVLA_ROBOTWIN_ROOT", Path.home() / "octvla" / "RoboTwin"))
pytestmark = pytest.mark.skipif(not (ROOT / "env_cfg").is_dir() and not (ROOT / "task_config").is_dir(),
                                reason="RoboTwin checkout not available")


def test_visual_shift_randomises_backgrounds_and_lights_but_not_geometry():
    clean = build_dual_franka_setup(ROOT, task_name="t", task_config="demo_clean")
    shifted = build_dual_franka_setup(ROOT, task_name="t", task_config="demo_clean",
                                      visual_shift=True)
    rand = shifted["domain_randomization"]
    assert rand["random_background"] and rand["random_light"] and shifted["eval_mode"]
    assert rand["clean_background_rate"] == 0.0
    # Geometry the oracle, the shelf layout and the head-camera crops rely on.
    assert not rand["cluttered_table"]
    assert rand["random_table_height"] == 0 and rand["random_head_camera_dis"] == 0
    assert not clean["domain_randomization"]["random_background"]
    assert clean["camera"] == shifted["camera"]
