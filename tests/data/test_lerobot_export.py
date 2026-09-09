from types import SimpleNamespace

from oct_vla.data.lerobot_export import CAMERA_FEATURES, _state_vector


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
