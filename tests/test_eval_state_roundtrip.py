"""The evaluation client rebuilds `observation.state` from the live simulator;
the exporter builds it from a recording. They are separate code paths that must
produce byte-identical vectors, or the policy meets a state at inference that
never appeared in training -- a failure that shows up only as an unexplained
success rate, never as an error."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_policy", ROOT / "scripts/eval_shelf_restock_policy.py"
)
eval_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_policy)

from oct_vla.data.lerobot_export import _joint_state_vector  # noqa: E402


def joints(left, right):
    """A live `JointState` as the bridge reports it."""
    return NS(
        left=NS(positions=tuple(left[:-1]), gripper=left[-1]),
        right=NS(positions=tuple(right[:-1]), gripper=right[-1]),
    )


def recorded(*configurations):
    """The same configurations as a recorded episode the exporter reads."""
    samples = []
    for left, right in configurations:
        j = joints(left, right)
        j.to_vector = (lambda k: lambda: (
            *k.left.positions, k.left.gripper, *k.right.positions, k.right.gripper
        ))(j)
        samples.append(NS(observation=NS(joints=j)))
    return NS(samples=tuple(samples), seed=1)


TRAJECTORY = (
    ((0.0, 1.0, 0.83), (2.0, 3.0, 0.17)),
    ((0.1, 1.5, 0.83), (2.0, 3.25, 0.17)),
    ((0.15, 1.4, 0.55), (1.9, 3.30, 0.80)),
)


def test_position_state_matches_the_exporter():
    episode = recorded(*TRAJECTORY)
    for index, (left, right) in enumerate(TRAJECTORY):
        assert eval_policy.joint_vector(joints(left, right)) == list(
            _joint_state_vector(episode, index)
        )


def test_velocity_state_matches_the_exporter_along_a_trajectory():
    """The whole point of the round trip: replay the recording through the
    client's own carry-the-previous-frame logic and require every step to agree."""
    episode = recorded(*TRAJECTORY)
    previous = None
    for index, (left, right) in enumerate(TRAJECTORY):
        live = joints(left, right)
        positions = eval_policy.joint_vector(live)
        built = eval_policy.joint_vector(live, previous if previous is not None else positions)
        assert built == pytest.approx(list(_joint_state_vector(episode, index,
                                                               "position_velocity")))
        previous = positions


def test_the_first_step_after_a_reset_has_zero_velocity():
    """There is no previous observation yet, and the exporter writes zeros for a
    recording's first frame. A mismatch here would put every episode's opening
    step off-distribution."""
    live = joints(*TRAJECTORY[0])
    positions = eval_policy.joint_vector(live)
    assert eval_policy.joint_vector(live, positions)[len(positions):] == [0.0] * len(positions)


def test_a_width_mismatch_is_refused_rather_than_broadcast():
    live = joints(*TRAJECTORY[0])
    with pytest.raises(ValueError, match="velocity block would be misaligned"):
        eval_policy.joint_vector(live, [0.0, 0.0])
