import pytest

from oct_vla.data.lerobot_export import CAMERA_FEATURES
from oct_vla.data.policy_inputs import (
    PI05_RENAME_MAP,
    SMOLVLA_RENAME_MAP,
    rename_map_for,
)


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


def test_rename_map_is_resolved_from_the_policy_type():
    """Evaluation reads the type back from the checkpoint and looks the map up
    here, so training and inference cannot drift onto different maps."""
    assert rename_map_for("pi05") is PI05_RENAME_MAP
    assert rename_map_for("control_pi05") is PI05_RENAME_MAP


def test_smolvla_map_covers_exactly_the_exported_cameras():
    """`lerobot/smolvla_base` declares camera1/2/3, and the RGB arm loads that
    checkpoint's own input_features -- so a missing or partial map stops the
    run outright."""
    assert set(SMOLVLA_RENAME_MAP) == set(CAMERA_FEATURES.values())
    assert len(set(SMOLVLA_RENAME_MAP.values())) == len(SMOLVLA_RENAME_MAP)


def test_smolvla_head_maps_to_the_first_camera():
    """Same decision as pi0.5's head -> base_0_rgb: the scene view takes the
    primary slot, wrists follow left then right. Swapping these is invisible to
    inference and shows up only as a worse success rate."""
    assert SMOLVLA_RENAME_MAP["observation.images.head"].endswith("camera1")
    assert SMOLVLA_RENAME_MAP["observation.images.left_wrist"].endswith("camera2")
    assert SMOLVLA_RENAME_MAP["observation.images.right_wrist"].endswith("camera3")


def test_both_arms_of_one_backbone_share_a_map():
    """The RGB arm resolves features from the base checkpoint and the object arm
    from the dataset. They must still land on identical camera names, or the
    arms differ in something other than their conditioning."""
    for stock, plugin in (("pi05", "control_pi05"), ("smolvla", "control_smolvla")):
        assert rename_map_for(stock) == rename_map_for(plugin)


def test_the_two_backbones_do_not_share_a_map():
    """The failure this guards against: handing SmolVLA pi0.5's openpi names,
    which inference would accept silently because every camera shares shape and
    dtype -- surfacing only as an unexplained drop in success rate."""
    assert rename_map_for("control_smolvla") != rename_map_for("control_pi05")


def test_an_unknown_backbone_is_refused_rather_than_defaulted():
    """Defaulting to pi0.5's map for a new backbone would be the silent version
    of exactly that bug."""
    with pytest.raises(ValueError, match="No rename map registered"):
        rename_map_for("octo")


def test_a_from_scratch_backbone_takes_the_datasets_own_camera_names():
    """ACT has no pretrained camera names to match, so its map is empty --
    recorded explicitly, because the refusal for unknown types is what keeps a
    new policy from inheriting pi0.5's openpi names by accident."""
    assert rename_map_for("act") == {}
    assert rename_map_for("act") != rename_map_for("pi05")
