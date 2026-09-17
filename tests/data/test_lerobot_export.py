from types import SimpleNamespace

import pytest

from oct_vla.data.lerobot_export import CAMERA_FEATURES, _features, _state_vector
from oct_vla.data.object_tokens import ObjectTokenSpec


def test_state_vector_and_camera_mapping_match_lerobot_contract():
    """The adapter's non-optional mapping stays testable without LeRobot installed."""
    arm = SimpleNamespace(
        pose=SimpleNamespace(position=(1.0, 2.0, 3.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        gripper=0.5,
    )
    eef = SimpleNamespace(left=arm, right=arm)
    sample = SimpleNamespace(observation=SimpleNamespace(eef=eef))
    episode = SimpleNamespace(samples=(sample,))
    assert _state_vector(episode, 0) == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5) * 2
    assert CAMERA_FEATURES == {
        "head_rgb": "observation.images.head",
        "left_wrist_rgb": "observation.images.left_wrist",
        "right_wrist_rgb": "observation.images.right_wrist",
    }


def test_object_feature_schema_is_fixed_and_numeric():
    frame = SimpleNamespace(height=240, width=320)
    observation = SimpleNamespace(
        head_rgb=frame,
        left_wrist_rgb=frame,
        right_wrist_rgb=frame,
        joints=None,
    )
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=observation),))
    features = _features(episode, object_token_spec=ObjectTokenSpec(max_objects=8))

    assert features["observation.object_tokens"] == {"dtype": "float32", "shape": (8, 15)}
    assert features["observation.object_token_mask"] == {"dtype": "float32", "shape": (8,)}


def _observation_with_joints(joints_per_arm=7):
    from oct_vla.core.state import ArmJoints, JointState

    frame = SimpleNamespace(height=240, width=320)
    return SimpleNamespace(
        head_rgb=frame,
        left_wrist_rgb=frame,
        right_wrist_rgb=frame,
        joints=JointState(
            ArmJoints((0.1,) * joints_per_arm, 0.8),
            ArmJoints((0.2,) * joints_per_arm, 0.3),
        ),
    )


def test_joint_columns_are_absent_when_joints_were_not_recorded():
    """Recordings predating joint capture must still export."""
    frame = SimpleNamespace(height=240, width=320)
    observation = SimpleNamespace(
        head_rgb=frame, left_wrist_rgb=frame, right_wrist_rgb=frame, joints=None
    )
    features = _features(SimpleNamespace(samples=(SimpleNamespace(observation=observation),)))
    assert "observation.joint_state" not in features
    assert "action.joint_position" not in features


def test_joint_columns_carry_one_entry_per_motor():
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=_observation_with_joints()),))
    features = _features(episode)
    # 7 arm joints + 1 gripper, per arm.
    assert features["observation.joint_state"]["shape"] == (16,)
    assert features["action.joint_position"]["shape"] == (16,)
    names = features["observation.joint_state"]["names"]["motors"]
    assert names[:2] == ["left_arm.j0", "left_arm.j1"]
    assert names[7] == "left_arm.gripper"
    assert names[15] == "right_arm.gripper"


def test_joint_motor_names_follow_the_embodiment_not_a_constant():
    """A different joint count must export without editing the exporter."""
    episode = SimpleNamespace(
        samples=(SimpleNamespace(observation=_observation_with_joints(joints_per_arm=6)),)
    )
    assert _features(episode)["observation.joint_state"]["shape"] == (14,)


def test_joint_control_space_makes_state_and_action_both_joints():
    """A joint-space policy must be proprioceptive in the space it commands.
    Giving it a Cartesian state while asking for joint targets would make it
    learn inverse kinematics as a side job -- the round-trip this removes."""
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=_observation_with_joints()),))
    features = _features(episode, control_space="joint")
    assert features["observation.state"]["shape"] == (16,)
    assert features["action"]["shape"] == (16,)
    assert features["observation.state"]["names"]["motors"][0] == "left_arm.j0"
    # The separate joint columns are redundant once they ARE state and action.
    assert "observation.joint_state" not in features
    assert "action.joint_position" not in features


def test_cartesian_remains_the_default():
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=_observation_with_joints()),))
    features = _features(episode)
    assert features["action"]["shape"] == (14,)
    assert features["observation.state"]["names"]["motors"][0] == "left_eef.x"


def test_joint_control_space_refuses_a_recording_without_joints():
    """Failing here is far cheaper than a dataset whose action column is
    silently the wrong quantity."""
    frame = SimpleNamespace(height=240, width=320)
    observation = SimpleNamespace(
        head_rgb=frame, left_wrist_rgb=frame, right_wrist_rgb=frame, joints=None
    )
    episode = SimpleNamespace(seed=7, samples=(SimpleNamespace(observation=observation),))
    with pytest.raises(ValueError, match="needs recorded joint positions"):
        _features(episode, control_space="joint")


def test_an_unknown_control_space_is_rejected():
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=_observation_with_joints()),))
    with pytest.raises(ValueError, match="control_space must be one of"):
        _features(episode, control_space="torque")


def _two_frame_episode():
    """Two frames whose joints differ by a known increment."""
    from oct_vla.core.state import ArmJoints, JointState

    def obs(offset):
        frame = SimpleNamespace(height=240, width=320)
        return SimpleNamespace(
            head_rgb=frame, left_wrist_rgb=frame, right_wrist_rgb=frame,
            joints=JointState(
                ArmJoints(tuple(0.1 + offset for _ in range(7)), 0.8),
                ArmJoints(tuple(0.2 + offset for _ in range(7)), 0.3),
            ),
        )

    return SimpleNamespace(
        seed=1,
        samples=(SimpleNamespace(observation=obs(0.0)), SimpleNamespace(observation=obs(0.05))),
    )


def test_joint_delta_encodes_the_increment_not_the_target():
    """Absolute targets are badly conditioned here: one oracle step is 0.03
    sigma of the joint spread, finer than the model's own error, so a trained
    checkpoint commanded 3.6x too much motion. The increment is 0.40 sigma."""
    from oct_vla.data.lerobot_export import _joint_delta_action_vector

    action = _joint_delta_action_vector(_two_frame_episode(), 0)
    assert action[:7] == pytest.approx((0.05,) * 7)
    assert action[8:15] == pytest.approx((0.05,) * 7)


def test_joint_delta_keeps_the_gripper_absolute():
    """The gripper is a binary actuator state decoded through a threshold, not
    a position to integrate -- an increment on it means nothing."""
    from oct_vla.data.lerobot_export import _joint_delta_action_vector

    action = _joint_delta_action_vector(_two_frame_episode(), 0)
    assert action[7] == pytest.approx(0.8)
    assert action[15] == pytest.approx(0.3)


def test_the_last_frame_of_a_delta_episode_asks_for_no_motion():
    from oct_vla.data.lerobot_export import _joint_delta_action_vector

    action = _joint_delta_action_vector(_two_frame_episode(), 1)
    assert action[:7] == pytest.approx((0.0,) * 7)


def test_joint_delta_shares_the_joint_feature_layout():
    features = _features(_two_frame_episode(), control_space="joint_delta")
    assert features["action"]["shape"] == (16,)
    assert features["observation.state"]["shape"] == (16,)
