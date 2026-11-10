"""The collection guard on the Franka cuRobo limits (RoboTwin asset pack drift)."""

from __future__ import annotations

import pytest

from oct_vla.robots.robotwin.task_config import (
    FRANKA_PLANNER_LIMITS,
    franka_planner_limits,
    require_franka_planner_limits,
)

TEMPLATE = """robot_cfg:
  kinematics:
    cspace:
      joint_names: ["panda_joint1"]
      max_acceleration: {acc}
      max_jerk: {jerk}
planner:
  frame_bias: [0., 0., 0.]
"""


def _write(root, tmp=(15.0, 500.0), gen=(15.0, 500.0)):
    d = root / "assets" / "embodiments" / "franka-panda"
    d.mkdir(parents=True, exist_ok=True)
    (d / "curobo_tmp.yml").write_text(TEMPLATE.format(acc=tmp[0], jerk=tmp[1]))
    (d / "curobo.yml").write_text(TEMPLATE.format(acc=gen[0], jerk=gen[1]))
    return root


def test_corpus_limits_pass(tmp_path):
    assert require_franka_planner_limits(_write(tmp_path)) == FRANKA_PLANNER_LIMITS


def test_upstream_unified_limits_are_refused(tmp_path):
    with pytest.raises(ValueError, match="planned with"):
        require_franka_planner_limits(_write(tmp_path, (3.0, 100.0), (3.0, 100.0)))


def test_generated_and_source_must_agree(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        franka_planner_limits(_write(tmp_path, (15.0, 500.0), (3.0, 100.0)))
