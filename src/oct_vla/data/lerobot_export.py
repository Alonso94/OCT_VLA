"""Export canonical OCT-VLA recordings as a local LeRobot v3 dataset.

The canonical store deliberately uses dependency-free gzip streams of packed
RGB.  This module is an optional downstream adapter: importing OCT-VLA still
does not require NumPy, PyAV, or LeRobot.  Run it from the policy environment
documented in ``docs/setup.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from oct_vla.data.episode import Episode, validate_episode
from oct_vla.data.object_tokens import (
    ObjectTokenSpec,
    object_token_ranks,
    object_tokens,
    stable_ranks,
)
from oct_vla.data.store import read_episode

CAMERA_FEATURES = {
    "head_rgb": "observation.images.head",
    "left_wrist_rgb": "observation.images.left_wrist",
    "right_wrist_rgb": "observation.images.right_wrist",
}


@dataclass(frozen=True)
class ExportReport:
    """The episodes accepted and skipped by one export invocation."""

    output: Path
    exported: tuple[Path, ...]
    skipped: tuple[tuple[Path, str], ...]


def _require_export_dependencies():
    try:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "LeRobot export requires the policy environment with NumPy, PyAV, "
            "and LeRobot installed. "
            "See docs/setup.md."
        ) from error
    return np, LeRobotDataset


def _state_vector(episode: Episode, index: int) -> tuple[float, ...]:
    eef = episode.samples[index].observation.eef
    return (
        *eef.left.pose.position,
        *eef.left.pose.orientation,
        eef.left.gripper,
        *eef.right.pose.position,
        *eef.right.pose.orientation,
        eef.right.gripper,
    )


#: Column labels for a 16-d end-effector pose, the layout `_state_vector`
#: produces. Shared by observation.state and by the absolute-EE action, which
#: are the same quantity one frame apart.
_EEF_POSE_NAMES = [
    f"{side}_eef.{axis}"
    for side in ("left", "right")
    for axis in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
]


def _joint_motor_names(observation) -> list[str] | None:
    """Per-motor labels for the joint columns, or None if joints were not
    recorded. Derived from the observation rather than hard-coded so a
    different embodiment's joint count exports without editing this file."""
    if observation.joints is None:
        return None
    names = []
    for side, arm in (("left", observation.joints.left), ("right", observation.joints.right)):
        names.extend(f"{side}_arm.j{i}" for i in range(len(arm.positions)))
        names.append(f"{side}_arm.gripper")
    return names


def _state_motor_names(joint_names: list[str], state_encoding: str) -> list[str]:
    """Labels for `observation.state`, which is wider than the action when the
    state carries velocities. Suffixed rather than renamed so the first half
    still lines up column-for-column with the action."""
    if state_encoding != "position_velocity":
        return joint_names
    return [*joint_names, *(f"{name}.vel" for name in joint_names)]


#: How `observation.state` is built for a joint-space export.
#:
#: "position_velocity" exists because of a measured failure. With `joint_delta`
#: the action is essentially a velocity, and ACT sees a single frame
#: (`n_obs_steps` is hard-capped at 1), so a static position snapshot does not
#: determine the target. Offline on the validation split, in MEAN_STD units:
#: predicting a constant scores 0.471, the trained policy 0.284-0.334, and
#: simply repeating the previous action scores 0.126. The target is far more
#: determined by its own history than by any observation, including
#: ground-truth object state -- so the history is put in the observation.
#:
#: The appended block is a finite difference of the measured configuration,
#: which is what a real arm's joint-velocity sensor reports, not a replayed
#: action label. It is still the previous action by construction for
#: `joint_delta`, so this variant is exposed to the copycat failure that
#: behaviour cloning shows when action history is observable: a policy can
#: score well offline by continuing whatever motion it is already in while
#: ignoring the scene. That is precisely what the closed-loop comparison
#: against `position` is for.
STATE_ENCODINGS = ("position", "position_velocity")


def _joint_state_vector(
    episode: Episode, index: int, state_encoding: str = "position"
) -> tuple[float, ...] | None:
    joints = episode.samples[index].observation.joints
    if joints is None:
        return None
    positions = joints.to_vector()
    if state_encoding != "position_velocity":
        return positions
    # Backward difference, so the velocity at t describes motion that already
    # happened and no future frame leaks in. The first frame has no
    # predecessor and reports zero, which is also what the evaluation client
    # reports on the step after a reset -- the two must agree or the policy
    # meets a state at inference that never appeared in training.
    if index == 0:
        return (*positions, *(0.0,) * len(positions))
    previous = episode.samples[index - 1].observation.joints
    if previous is None:
        return None
    before = previous.to_vector()
    return (*positions, *(now - was for now, was in zip(positions, before, strict=True)))


def _joint_delta_action_vector(episode: Episode, index: int) -> tuple[float, ...] | None:
    """Per-step joint *increment*, with the gripper left absolute.

    Absolute joint targets are badly conditioned for this task. A single oracle
    step moves 0.009 rad while the joint values themselves span 0.30 rad, so one
    step is 0.03 standard deviations of the normalisation scale -- finer than
    the model's own residual error. Measured on a trained checkpoint: mean
    prediction error 0.033 rad against a 0.008 rad step, i.e. it commanded 3.6x
    too much motion on frames it was fit on. Encoding the increment instead puts
    a step at 0.40 sigma, about 13x more resolution.

    The gripper stays absolute: it is a binary actuator state decoded through a
    threshold, not a position to integrate, and a delta on it means nothing.
    """
    # Positions only, whatever the state encoding is: the increment is defined
    # against the configuration, and differencing a position+velocity vector
    # would put an acceleration in the action's second half.
    current = _joint_state_vector(episode, index, "position")
    following = _joint_action_vector(episode, index)
    if current is None or following is None:
        return None
    half = len(current) // 2
    grippers = (half - 1, 2 * half - 1)
    return tuple(
        following[i] if i in grippers else following[i] - current[i]
        for i in range(len(current))
    )


def _joint_action_vector(episode: Episode, index: int) -> tuple[float, ...] | None:
    """The NEXT frame's measured joints: an absolute position target.

    Absolute rather than a delta, matching RoboTwin's own `joint_action` format
    and standard behaviour-cloning practice. An absolute target has no
    integrator to drift and, unlike the Cartesian action, needs no inverse
    kinematics to execute -- so a command the policy emits is always feasible
    up to joint limits, which is precisely the failure mode the Cartesian
    representation could not avoid.

    The final sample has no successor, so it repeats its own configuration:
    holding still is the only well-defined target there.
    """
    following = min(index + 1, len(episode.samples) - 1)
    joints = episode.samples[following].observation.joints
    return None if joints is None else joints.to_vector()


def _eef_action_vector(episode: Episode, index: int) -> tuple[float, ...]:
    """The NEXT frame's absolute end-effector pose: position, quaternion and
    gripper, per arm, in the same 16-d layout as `observation.state`.

    The task-space counterpart of `_joint_action_vector`, and it exists for the
    same reason that one does. Measured in docs/control_space_comparison.md, the
    delta/absolute axis is the largest effect in this project -- absolute joint
    targets lift an object in 13 of 20 episodes where increments manage 0-2. The
    end-effector space had only an incremental encoding, so comparing it against
    joint space would have measured the target type and reported it as a
    representation result.

    Executing this needs no reference to integrate and no leash: an absolute
    pose is solved straight through IK. The quaternion, however, is regressed
    without a unit-norm constraint, which is a real cost of the encoding and is
    normalised at execution rather than hidden here.

    The final sample has no successor and repeats its own pose, matching the
    joint path.
    """
    following = min(index + 1, len(episode.samples) - 1)
    return _state_vector(episode, following)


CONTROL_SPACES = ("cartesian", "cartesian_absolute", "joint", "joint_delta")


def _is_joint_space(control_space: str) -> bool:
    return control_space in ("joint", "joint_delta")


def _is_eef_space(control_space: str) -> bool:
    return control_space in ("cartesian", "cartesian_absolute")


def _require_joints(episode: Episode, control_space: str) -> None:
    if _is_joint_space(control_space) and episode.samples[0].observation.joints is None:
        raise ValueError(
            "control_space='joint' needs recorded joint positions, but episode "
            f"seed {episode.seed} has none. Re-collect with a build that records "
            "them (see oct_vla.core.state.JointState)."
        )


def _privileged_features(spec: ObjectTokenSpec) -> dict[str, dict]:
    """Object tokens as a flat `observation.environment_state`.

    LeRobot policies accept privileged scene state under this name, and ACT in
    particular requires "at least one image or the environment state" -- so a
    dataset carrying this and no cameras trains a vision-free, state-only
    policy with no new policy class.

    Flattened rather than kept as [N, D] because `environment_state` is a
    vector feature. The mask is not exported: a padded slot is already all
    zeros, which is what an absent object should look like to an MLP, and a
    separate mask column would need a policy that knows to read it.
    """
    return {
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (spec.max_objects * spec.token_dim,),
        }
    }


def _features(
    episode: Episode,
    *,
    object_token_spec: ObjectTokenSpec | None = None,
    control_space: str = "cartesian",
    privileged: bool = False,
    state_encoding: str = "position",
    joint_side_channel: bool = False,
) -> dict[str, dict]:
    observation = episode.samples[0].observation
    if control_space not in CONTROL_SPACES:
        raise ValueError(f"control_space must be one of {CONTROL_SPACES}, got {control_space!r}")
    if state_encoding not in STATE_ENCODINGS:
        raise ValueError(
            f"state_encoding must be one of {STATE_ENCODINGS}, got {state_encoding!r}"
        )
    if state_encoding != "position" and not _is_joint_space(control_space):
        raise ValueError(
            f"state_encoding={state_encoding!r} is only defined for a joint-space "
            f"export; control_space is {control_space!r}, whose observation.state "
            "is an end-effector pose."
        )
    if privileged and not _is_joint_space(control_space):
        # The privileged branch lives inside the joint-space block below, so a
        # privileged cartesian export would silently fall through to the camera
        # block while the dataset was created with use_videos=False -- declared
        # video features with no video written. Refused rather than repaired,
        # because the right fix is to decide what a vision-free end-effector
        # observation should be, not to guess one here.
        raise ValueError(
            f"privileged export is implemented for joint spaces only; "
            f"control_space is {control_space!r}."
        )
    _require_joints(episode, control_space)
    joint_names = _joint_motor_names(observation)
    state_names = _state_motor_names(joint_names, state_encoding) if joint_names else None
    if _is_joint_space(control_space):
        # A joint-space policy is proprioceptive in the same space it commands:
        # `observation.state` carries joints, `action` is the next joint
        # configuration. Feeding it a Cartesian state while asking for joint
        # targets would make the policy learn inverse kinematics as a side job,
        # which is the round-trip this change exists to remove. Matches
        # RoboTwin's own joint_action format.
        joint_block = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": {"motors": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": {"motors": joint_names},
            },
        }
        if privileged:
            # No cameras: this variant exists to measure what privileged scene
            # state alone can do, as an upper bound on what vision could add.
            if object_token_spec is None:
                raise ValueError("privileged export needs an object token spec")
            return {**joint_block, **_privileged_features(object_token_spec)}
        return {
            **joint_block,
            **_camera_features(observation),
            **_object_features(object_token_spec),
        }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": {"motors": _EEF_POSE_NAMES},
        },
        # 14-d for an increment (3 translation + 3 rotation + gripper, per arm);
        # 16-d for an absolute pose, which carries a 4-component quaternion
        # instead of a 3-component rotation increment.
        "action": (
            {"dtype": "float32", "shape": (16,), "names": {"motors": _EEF_POSE_NAMES}}
            if control_space == "cartesian_absolute"
            else {"dtype": "float32", "shape": (14,)}
        ),
    }
    # Joint columns appear only when the recording captured them. Older
    # episodes predate joint capture and must still export, so this is
    # conditional rather than assumed -- and a dataset either has the columns
    # for every frame or not at all, which _frame checks.
    #
    # Off by default, and that default is a fix rather than a preference.
    # `action.joint_position` starts with "action", so LeRobot's
    # `dataset_to_policy_features` types it as a second ACTION feature
    # (verified: a cartesian export from a joints-carrying collection yields
    # {'action': (16,), 'action.joint_position': (16,)}), which is not what any
    # policy here expects. Nothing in this repository reads either column, so
    # they are analysis-only and must be asked for.
    joint_names = _joint_motor_names(observation)
    if joint_names is not None and joint_side_channel:
        features["observation.joint_state"] = {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"motors": joint_names},
        }
        features["action.joint_position"] = {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"motors": joint_names},
        }
    features.update(_camera_features(observation))
    features.update(_object_features(object_token_spec))
    return features


def _camera_features(observation) -> dict[str, dict]:
    return {
        feature_name: {
            "dtype": "video",
            "shape": (getattr(observation, attr).height, getattr(observation, attr).width, 3),
            "names": ["height", "width", "channel"],
        }
        for attr, feature_name in CAMERA_FEATURES.items()
    }


def _object_features(spec: ObjectTokenSpec | None) -> dict[str, dict]:
    if spec is None:
        return {}
    return {
        "observation.object_tokens": {
            "dtype": "float32",
            "shape": (spec.max_objects, spec.token_dim),
        },
        # Float avoids an Arrow bool/nested-array incompatibility in LeRobot v3
        # and is converted to bool by the policy branch.
        "observation.object_token_mask": {"dtype": "float32", "shape": (spec.max_objects,)},
        # The episode-stable ordering the role-stripped arm sorts by. Exported
        # rather than recomputed per frame because it is a property of the
        # episode's first scene, which a single frame cannot recover.
        "observation.object_token_rank": {"dtype": "float32", "shape": (spec.max_objects,)},
    }


def _fps(episode: Episode) -> int:
    try:
        fps = float(episode.metadata["hz"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Episode metadata must contain a numeric 'hz' value for LeRobot export"
        ) from error
    rounded = round(fps)
    if rounded <= 0 or abs(fps - rounded) > 1e-6:
        raise ValueError(f"LeRobot export requires an integral frame rate; got {fps}")
    return rounded


def _source_manifest_entry(index: int, source: Path, episode: Episode) -> dict:
    payload = (source / "episode.json").read_bytes()
    return {
        "lerobot_episode_index": index,
        "source_episode": source.name,
        "episode_json_sha256": sha256(payload).hexdigest(),
        "seed": episode.seed,
        "samples": len(episode.samples),
        "metadata": dict(episode.metadata),
    }


def export_episodes(
    sources: Iterable[str | Path],
    output: str | Path,
    *,
    repo_id: str = "local/oct_vla_shelf_restock",
    robot_type: str = "robotwin_franka_bimanual",
    include_unsuccessful: bool = False,
    append: bool = False,
    object_token_spec: ObjectTokenSpec | None = None,
    control_space: str = "cartesian",
    privileged: bool = False,
    state_encoding: str = "position",
    joint_side_channel: bool = False,
) -> ExportReport:
    """Convert canonical episode directories into a new local LeRobot dataset.

    LeRobot owns its uniform ``frame_index / fps`` timestamps.  The source
    recordings remain untouched and retain their simulator-clock timestamps.
    ``output`` must not exist unless ``append`` is explicitly requested. This
    makes a long batch export restartable one finalized episode at a time.
    """
    source_paths = tuple(Path(source) for source in sources)
    if not source_paths:
        raise ValueError("At least one canonical episode directory is required")
    destination = Path(output)
    if destination.exists() and not append:
        raise FileExistsError(f"LeRobot output already exists: {destination}")
    if append and not destination.is_dir():
        raise FileNotFoundError(f"Cannot append: LeRobot output does not exist: {destination}")

    accepted: list[tuple[Path, Episode]] = []
    skipped: list[tuple[Path, str]] = []
    for source in source_paths:
        episode = read_episode(source)
        problems = validate_episode(episode)
        if problems:
            skipped.append((source, "; ".join(problems)))
        elif not episode.success and not include_unsuccessful:
            skipped.append((source, "episode is unsuccessful"))
        else:
            accepted.append((source, episode))
    if not accepted:
        raise ValueError("No successful, validation-clean episodes to export")

    fps = _fps(accepted[0][1])
    if any(_fps(episode) != fps for _, episode in accepted):
        raise ValueError("All exported episodes must use the same frame rate")
    np, LeRobotDataset = _require_export_dependencies()
    if append:
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=destination)
        if dataset.fps != fps:
            raise ValueError(f"Cannot append {fps} Hz data to a {dataset.fps} Hz LeRobot dataset")
    else:
        features = _features(
            accepted[0][1],
            object_token_spec=object_token_spec,
            control_space=control_space,
            privileged=privileged,
            state_encoding=state_encoding,
            joint_side_channel=joint_side_channel,
        )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=destination,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=not privileged,
        )
    for _, episode in accepted:
        episode_ranks = stable_ranks(episode.samples[0].scene) if object_token_spec else {}
        for index, sample in enumerate(episode.samples):
            _require_joints(episode, control_space)
            if _is_joint_space(control_space):
                action = (
                    _joint_delta_action_vector(episode, index)
                    if control_space == "joint_delta"
                    else _joint_action_vector(episode, index)
                )
                frame = {
                    "observation.state": np.asarray(
                        _joint_state_vector(episode, index, state_encoding), dtype=np.float32
                    ),
                    "action": np.asarray(action, dtype=np.float32),
                    "task": sample.context.instruction,
                }
            else:
                frame = {
                    "observation.state": np.asarray(
                        _state_vector(episode, index), dtype=np.float32
                    ),
                    "action": np.asarray(
                        _eef_action_vector(episode, index)
                        if control_space == "cartesian_absolute"
                        else sample.action.to_vector(),
                        dtype=np.float32,
                    ),
                    "task": sample.context.instruction,
                }
            joint_state = (
                _joint_state_vector(episode, index)
                if joint_side_channel and not _is_joint_space(control_space)
                else None
            )
            if joint_state is not None:
                frame["observation.joint_state"] = np.asarray(joint_state, dtype=np.float32)
                frame["action.joint_position"] = np.asarray(
                    _joint_action_vector(episode, index), dtype=np.float32
                )
            elif "observation.joint_state" in dataset.features:
                # The feature set was declared from the first episode. A later
                # episode without joints would write a ragged dataset that only
                # fails much later, during training.
                raise ValueError(
                    "Joint columns were declared from the first episode but "
                    f"episode seed {episode.seed} has no recorded joints; "
                    "re-collect the whole set so every frame carries them."
                )
            if privileged:
                tokens, _ = object_tokens(sample.scene, sample.context, spec=object_token_spec)
                frame["observation.environment_state"] = np.asarray(
                    [value for token in tokens for value in token], dtype=np.float32
                )
            elif object_token_spec is not None:
                tokens, mask = object_tokens(sample.scene, sample.context, spec=object_token_spec)
                frame["observation.object_tokens"] = np.asarray(tokens, dtype=np.float32)
                frame["observation.object_token_mask"] = np.asarray(mask, dtype=np.float32)
                frame["observation.object_token_rank"] = np.asarray(
                    object_token_ranks(
                        sample.scene,
                        episode_ranks,
                        spec=object_token_spec,
                        context=sample.context,
                    ),
                    dtype=np.float32,
                )
            for attr, feature_name in ({} if privileged else CAMERA_FEATURES).items():
                rgb = getattr(sample.observation, attr)
                frame[feature_name] = np.frombuffer(rgb.data, dtype=np.uint8).reshape(
                    rgb.height, rgb.width, 3
                )
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    manifest = {
        "format": "oct-vla-episode-provenance-v1",
        "repo_id": repo_id,
        "episodes": [
            _source_manifest_entry(index, source, episode)
            for index, (source, episode) in enumerate(accepted)
        ],
    }
    (destination / "octvla_episode_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Absolute and incremental joint actions share a column layout and motor
    # names, so nothing in the exported schema distinguishes them. Record which
    # this dataset holds, or evaluation has to guess -- and guessing wrong means
    # adding a target to the measured position, or commanding an increment as
    # an absolute pose.
    info_path = destination / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["control_space"] = control_space
        # Recorded beside control_space for the same reason: a 16-d and a 32-d
        # observation.state are distinguishable by shape, but nothing in the
        # schema says the extra half is a velocity rather than a second pose,
        # and the evaluation client has to rebuild it exactly.
        info["state_encoding"] = state_encoding
        info_path.write_text(json.dumps(info, indent=4))

    return ExportReport(destination, tuple(source for source, _ in accepted), tuple(skipped))
