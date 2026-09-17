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


def _joint_state_vector(episode: Episode, index: int) -> tuple[float, ...] | None:
    joints = episode.samples[index].observation.joints
    return None if joints is None else joints.to_vector()


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
    current = _joint_state_vector(episode, index)
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


CONTROL_SPACES = ("cartesian", "joint", "joint_delta")


def _is_joint_space(control_space: str) -> bool:
    return control_space in ("joint", "joint_delta")


def _require_joints(episode: Episode, control_space: str) -> None:
    if _is_joint_space(control_space) and episode.samples[0].observation.joints is None:
        raise ValueError(
            "control_space='joint' needs recorded joint positions, but episode "
            f"seed {episode.seed} has none. Re-collect with a build that records "
            "them (see oct_vla.core.state.JointState)."
        )


def _features(
    episode: Episode,
    *,
    object_token_spec: ObjectTokenSpec | None = None,
    control_space: str = "cartesian",
) -> dict[str, dict]:
    observation = episode.samples[0].observation
    if control_space not in CONTROL_SPACES:
        raise ValueError(f"control_space must be one of {CONTROL_SPACES}, got {control_space!r}")
    _require_joints(episode, control_space)
    joint_names = _joint_motor_names(observation)
    if _is_joint_space(control_space):
        # A joint-space policy is proprioceptive in the same space it commands:
        # `observation.state` carries joints, `action` is the next joint
        # configuration. Feeding it a Cartesian state while asking for joint
        # targets would make the policy learn inverse kinematics as a side job,
        # which is the round-trip this change exists to remove. Matches
        # RoboTwin's own joint_action format.
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": {"motors": joint_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": {"motors": joint_names},
            },
            **_camera_features(observation),
            **_object_features(object_token_spec),
        }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "motors": [
                    "left_eef.x",
                    "left_eef.y",
                    "left_eef.z",
                    "left_eef.qx",
                    "left_eef.qy",
                    "left_eef.qz",
                    "left_eef.qw",
                    "left_eef.gripper",
                    "right_eef.x",
                    "right_eef.y",
                    "right_eef.z",
                    "right_eef.qx",
                    "right_eef.qy",
                    "right_eef.qz",
                    "right_eef.qw",
                    "right_eef.gripper",
                ]
            },
        },
        "action": {"dtype": "float32", "shape": (14,)},
    }
    # Joint columns appear only when the recording captured them. Older
    # episodes predate joint capture and must still export, so this is
    # conditional rather than assumed -- and a dataset either has the columns
    # for every frame or not at all, which _frame checks.
    joint_names = _joint_motor_names(observation)
    if joint_names is not None:
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
        )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=destination,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=True,
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
                        _joint_state_vector(episode, index), dtype=np.float32
                    ),
                    "action": np.asarray(action, dtype=np.float32),
                    "task": sample.context.instruction,
                }
            else:
                frame = {
                    "observation.state": np.asarray(
                        _state_vector(episode, index), dtype=np.float32
                    ),
                    "action": np.asarray(sample.action.to_vector(), dtype=np.float32),
                    "task": sample.context.instruction,
                }
            joint_state = (
                None if _is_joint_space(control_space) else _joint_state_vector(episode, index)
            )
            if joint_state is not None:
                frame["observation.joint_state"] = np.asarray(joint_state, dtype=np.float32)
                frame["action.joint_position"] = np.asarray(
                    _joint_action_vector(episode, index), dtype=np.float32
                )
            elif (
                not _is_joint_space(control_space)
                and "observation.joint_state" in dataset.features
            ):
                # The feature set was declared from the first episode. A later
                # episode without joints would write a ragged dataset that only
                # fails much later, during training.
                raise ValueError(
                    "Joint columns were declared from the first episode but "
                    f"episode seed {episode.seed} has no recorded joints; "
                    "re-collect the whole set so every frame carries them."
                )
            if object_token_spec is not None:
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
            for attr, feature_name in CAMERA_FEATURES.items():
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
        info_path.write_text(json.dumps(info, indent=4))

    return ExportReport(destination, tuple(source for source, _ in accepted), tuple(skipped))
