from oct_vla.data.lerobot_export import CAMERA_FEATURES
from oct_vla.data.policy_inputs import PI05_RENAME_MAP


def test_rename_map_covers_exactly_the_exported_cameras():
    """Every camera the exporter writes must have a mapping, and no mapping may
    name a feature that is not exported. A key that matches nothing is silently
    ignored by LeRobot's rename step, so a typo here would leave the policy
    reading an unrenamed feature rather than failing."""
    assert set(PI05_RENAME_MAP) == set(CAMERA_FEATURES.values())


def test_rename_map_targets_are_distinct():
    """Two cameras mapping to one target would drop a view without erroring."""
    assert len(set(PI05_RENAME_MAP.values())) == len(PI05_RENAME_MAP)


def test_head_maps_to_the_base_view():
    """The head camera is the scene view; pi0.5 calls that `base_0_rgb`. Getting
    this pair wrong is the failure inference cannot detect -- shapes match, so
    the policy just sees a wrist close-up where it expects the shelf."""
    assert PI05_RENAME_MAP["observation.images.head"] == "observation.images.base_0_rgb"
    assert PI05_RENAME_MAP["observation.images.left_wrist"].endswith("left_wrist_0_rgb")
    assert PI05_RENAME_MAP["observation.images.right_wrist"].endswith("right_wrist_0_rgb")
