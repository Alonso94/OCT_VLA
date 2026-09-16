from types import SimpleNamespace

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
