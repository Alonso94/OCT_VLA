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


def test_privileged_export_has_env_state_and_no_cameras():
    """ACT accepts 'at least one image or the environment state', so a dataset
    with privileged scene state and no cameras trains a vision-free policy with
    no new policy class."""
    features = _features(
        _two_frame_episode(),
        control_space="joint_delta",
        object_token_spec=ObjectTokenSpec(max_objects=8),
        privileged=True,
    )
    assert features["observation.environment_state"]["shape"] == (8 * 15,)
    assert not [k for k in features if "images" in k], "cameras must be absent"
    assert features["action"]["shape"] == (16,)
    # The per-slot token columns are redundant once the tokens ARE the env state.
    assert "observation.object_tokens" not in features


def test_privileged_export_requires_an_object_spec():
    with pytest.raises(ValueError, match="privileged export needs an object token spec"):
        _features(_two_frame_episode(), control_space="joint_delta", privileged=True)


def test_a_normal_export_still_carries_cameras():
    features = _features(
        _two_frame_episode(),
        control_space="joint_delta",
        object_token_spec=ObjectTokenSpec(max_objects=8),
    )
    assert [k for k in features if "images" in k]
    assert "observation.environment_state" not in features


def _run_clips(seeds, per_run=3):
    from pathlib import Path
    return [Path(f"seed_{s}/episode_0{s}_{i}") for s in seeds for i in range(per_run)]


def _splits_module():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "bs", root / "scripts/build_shelf_restock_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_budget_counts_runs_not_clips():
    """A run is one scene played to the end, stored as its atomic transfer
    clips. Asking for 25 runs must give 25 scenes and every clip they contain,
    not 25 clips -- otherwise a data-scaling curve measures a third of what its
    axis claims."""
    m = _splits_module()
    kept = m.take_runs(_run_clips([102, 105, 108, 110]), 2)
    assert len(kept) == 6
    assert sorted({m.seed_of(c) for c in kept}) == [102, 105]


def test_growing_the_budget_only_adds_scenes():
    """Ascending seed, so a larger budget is a superset of a smaller one and
    the points of a scaling curve stay comparable."""
    m = _splits_module()
    clips = _run_clips([102, 105, 108, 110])
    assert set(m.take_runs(clips, 2)) < set(m.take_runs(clips, 3))


def test_an_unmet_budget_fails_rather_than_quietly_shrinking():
    m = _splits_module()
    with pytest.raises(SystemExit, match="only 4 are available"):
        m.take_runs(_run_clips([102, 105, 108, 110]), 9)


def test_no_budget_keeps_everything():
    m = _splits_module()
    clips = _run_clips([102, 105])
    assert m.take_runs(clips, None) == clips


def test_a_split_can_own_several_seed_ranges(tmp_path):
    """The train block was extended into 350-399 once the budget started being
    counted in runs. The extension is recorded in docs/dataset_protocol.md; this
    pins the code to it, because a split that silently ignores its extension
    trains on a third of the data the protocol says it has."""
    m = _splits_module()
    for seed in (105, 205, 355):
        clip = tmp_path / f"seed_{seed}" / "episode_0_0"
        clip.mkdir(parents=True)
        (clip / "episode.json").write_text("{}")
    grouped = m.collect_clips(tmp_path, m.SPLIT_BLOCKS["three_object"])
    assert sorted(m.seed_of(c) for c in grouped["train"]) == [105, 355]
    assert sorted(m.seed_of(c) for c in grouped["val"]) == [205]


def test_the_extension_sorts_behind_the_original_block():
    """A budget of N runs must stay a subset of a budget of N+1, so the added
    range has to be drawn from only after the original block is exhausted."""
    m = _splits_module()
    train = m.SPLIT_BLOCKS["three_object"]["train"]
    assert [r.start for r in train] == sorted(r.start for r in train)
    assert train[0].stop <= train[1].start


def _clip(tmp_path, seed, name="episode_0_0"):
    clip = tmp_path / f"seed_{seed}" / name
    clip.mkdir(parents=True, exist_ok=True)
    (clip / "episode.json").write_text("{}")
    return clip


def test_the_boundary_check_accepts_an_extension_seed_above_the_val_block(tmp_path):
    """The bug this replaced: the check compared seed *numbers* across the
    boundary, so a train scene from the 350-399 extension sitting above the
    200-249 validation block was rejected even though the layout was correct."""
    m = _splits_module()
    ordered = [_clip(tmp_path, s) for s in (105, 398, 202, 240)]
    m.check_boundary(ordered, 2, m.SPLIT_BLOCKS["three_object"])


def test_a_validation_scene_on_the_train_side_is_rejected(tmp_path):
    """What the check is actually for: LeRobot holds out a positional tail, so a
    misplaced episode means training on a validation scene."""
    m = _splits_module()
    ordered = [_clip(tmp_path, s) for s in (105, 202, 398, 240)]
    with pytest.raises(SystemExit, match="validation scene"):
        m.check_boundary(ordered, 2, m.SPLIT_BLOCKS["three_object"])


def test_a_train_scene_on_the_validation_side_is_rejected(tmp_path):
    m = _splits_module()
    ordered = [_clip(tmp_path, s) for s in (105, 202, 240, 398)]
    with pytest.raises(SystemExit, match="train scene"):
        m.check_boundary(ordered, 1, m.SPLIT_BLOCKS["three_object"])


def test_a_dataset_without_a_validation_split_has_no_boundary(tmp_path):
    m = _splits_module()
    ordered = [_clip(tmp_path, s) for s in (105, 398)]
    m.check_boundary(ordered, 2, m.SPLIT_BLOCKS["three_object"])


# ------------------------------------------------- position_velocity state


def _joint_episode(*configurations):
    """Episode whose samples carry only joints, one 2-DOF arm per side."""
    from types import SimpleNamespace as NS

    samples = []
    for left, right in configurations:
        joints = NS(
            left=NS(positions=tuple(left[:-1]), gripper=left[-1]),
            right=NS(positions=tuple(right[:-1]), gripper=right[-1]),
        )
        joints.to_vector = (lambda j: lambda: (
            *j.left.positions, j.left.gripper, *j.right.positions, j.right.gripper
        ))(joints)
        frame = NS(height=240, width=320)
        samples.append(NS(observation=NS(
            joints=joints, head_rgb=frame, left_wrist_rgb=frame, right_wrist_rgb=frame
        )))
    return NS(samples=tuple(samples), seed=1)


def test_position_only_state_is_unchanged():
    from oct_vla.data.lerobot_export import _joint_state_vector

    episode = _joint_episode(((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)))
    assert _joint_state_vector(episode, 0) == (0.0, 1.0, 0.5, 2.0, 3.0, 0.8)


def test_velocity_block_is_a_backward_difference():
    """Backward, not forward: the velocity at t must describe motion that has
    already happened, or a future frame leaks into the observation."""
    from oct_vla.data.lerobot_export import _joint_state_vector

    episode = _joint_episode(
        ((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)),
        ((0.1, 1.5, 0.5), (2.0, 3.25, 0.2)),
    )
    state = _joint_state_vector(episode, 1, "position_velocity")
    assert state[:6] == (0.1, 1.5, 0.5, 2.0, 3.25, 0.2)
    assert state[6:] == pytest.approx((0.1, 0.5, 0.0, 0.0, 0.25, -0.6))


def test_the_first_frame_reports_zero_velocity():
    """It has no predecessor. The evaluation client reports the same on the
    step after a reset, and the two must agree."""
    from oct_vla.data.lerobot_export import _joint_state_vector

    episode = _joint_episode(((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)))
    assert _joint_state_vector(episode, 0, "position_velocity")[6:] == (0.0,) * 6


def test_the_delta_action_is_unaffected_by_the_state_encoding():
    """The increment is defined against the configuration; differencing a
    position+velocity vector would put an acceleration in the action."""
    from oct_vla.data.lerobot_export import _joint_delta_action_vector

    episode = _joint_episode(
        ((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)),
        ((0.1, 1.5, 0.4), (2.0, 3.25, 0.2)),
    )
    action = _joint_delta_action_vector(episode, 0)
    assert len(action) == 6
    # Arm joints differenced, grippers absolute.
    assert action == pytest.approx((0.1, 0.5, 0.4, 0.0, 0.25, 0.2))


def test_the_state_feature_widens_but_the_action_does_not():
    from oct_vla.data.lerobot_export import _features

    episode = _joint_episode(((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)))
    features = _features(episode, control_space="joint_delta",
                         state_encoding="position_velocity")
    assert features["observation.state"]["shape"] == (12,)
    assert features["action"]["shape"] == (6,)
    names = features["observation.state"]["names"]["motors"]
    assert names[:6] == features["action"]["names"]["motors"]
    assert names[6:] == [f"{n}.vel" for n in features["action"]["names"]["motors"]]


def test_velocities_are_refused_for_a_cartesian_export():
    """observation.state is an end-effector pose there; appending a joint
    velocity would silently produce a vector nothing can interpret."""
    from oct_vla.data.lerobot_export import _features

    episode = _joint_episode(((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)))
    with pytest.raises(ValueError, match="only defined for a joint-space"):
        _features(episode, control_space="cartesian", state_encoding="position_velocity")


def test_an_unknown_state_encoding_is_refused():
    from oct_vla.data.lerobot_export import _features

    episode = _joint_episode(((0.0, 1.0, 0.5), (2.0, 3.0, 0.8)))
    with pytest.raises(ValueError, match="state_encoding must be one of"):
        _features(episode, control_space="joint", state_encoding="velocity_only")


# --------------------------------------------------- absolute end effector


def _eef_episode(*poses):
    """Episode carrying end-effector poses, the cartesian export's input."""
    from types import SimpleNamespace as NS

    samples = []
    for left, right in poses:
        def arm(p):
            return NS(pose=NS(position=p[:3], orientation=p[3:7]), gripper=p[7])
        frame = NS(height=240, width=320)
        samples.append(NS(
            observation=NS(eef=NS(left=arm(left), right=arm(right)), joints=None,
                           head_rgb=frame, left_wrist_rgb=frame, right_wrist_rgb=frame),
            action=NS(to_vector=lambda: tuple(range(14))),
        ))
    return NS(samples=tuple(samples), seed=1)


POSE_A = ((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.83),
          (0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0, 0.17))
POSE_B = ((0.15, 0.2, 0.35, 0.0, 0.0, 0.0, 1.0, 0.83),
          (0.4, 0.55, 0.6, 0.0, 0.0, 1.0, 0.0, 0.80))


def test_the_absolute_eef_action_is_the_next_frame_pose():
    """The task-space counterpart of the absolute joint target, and it has to
    be the NEXT frame or the policy is trained to hold still."""
    from oct_vla.data.lerobot_export import _eef_action_vector, _state_vector

    episode = _eef_episode(POSE_A, POSE_B)
    assert _eef_action_vector(episode, 0) == _state_vector(episode, 1)


def test_the_final_frame_repeats_its_own_pose():
    from oct_vla.data.lerobot_export import _eef_action_vector, _state_vector

    episode = _eef_episode(POSE_A, POSE_B)
    assert _eef_action_vector(episode, 1) == _state_vector(episode, 1)


def test_the_absolute_eef_action_is_16d_where_the_increment_is_14d():
    """An absolute pose carries a 4-component quaternion; an increment carries
    a 3-component rotation. The widths differ and both are declared."""
    from oct_vla.data.lerobot_export import _features

    episode = _eef_episode(POSE_A, POSE_B)
    delta = _features(episode, control_space="cartesian")
    absolute = _features(episode, control_space="cartesian_absolute")
    assert delta["action"]["shape"] == (14,)
    assert absolute["action"]["shape"] == (16,)
    # The absolute action and the observation are the same quantity one frame
    # apart, so they must carry the same column labels.
    assert absolute["action"]["names"] == absolute["observation.state"]["names"]


def test_absolute_eef_is_a_registered_control_space():
    from oct_vla.data.lerobot_export import CONTROL_SPACES

    assert "cartesian_absolute" in CONTROL_SPACES
