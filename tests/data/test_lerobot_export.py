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
    )
    episode = SimpleNamespace(samples=(SimpleNamespace(observation=observation),))
    features = _features(episode, object_token_spec=ObjectTokenSpec(max_objects=8))

    assert features["observation.object_tokens"] == {"dtype": "float32", "shape": (8, 15)}
    assert features["observation.object_token_mask"] == {"dtype": "float32", "shape": (8,)}
