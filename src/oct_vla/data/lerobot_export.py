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


def _features(
    episode: Episode, *, object_token_spec: ObjectTokenSpec | None = None
) -> dict[str, dict]:
    observation = episode.samples[0].observation
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
    for attr, feature_name in CAMERA_FEATURES.items():
        frame = getattr(observation, attr)
        features[feature_name] = {
            "dtype": "video",
            "shape": (frame.height, frame.width, 3),
            "names": ["height", "width", "channel"],
        }
    if object_token_spec is not None:
        features["observation.object_tokens"] = {
            "dtype": "float32",
            "shape": (object_token_spec.max_objects, object_token_spec.token_dim),
        }
        # Float avoids an Arrow bool/nested-array incompatibility in LeRobot v3
        # and is converted to bool by the policy branch.
        features["observation.object_token_mask"] = {
            "dtype": "float32",
            "shape": (object_token_spec.max_objects,),
        }
        # The episode-stable ordering the role-stripped arm sorts by. Exported
        # rather than recomputed per frame because it is a property of the
        # episode's first scene, which a single frame cannot recover.
        features["observation.object_token_rank"] = {
            "dtype": "float32",
            "shape": (object_token_spec.max_objects,),
        }
    return features


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
        features = _features(accepted[0][1], object_token_spec=object_token_spec)
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
            frame = {
                "observation.state": np.asarray(_state_vector(episode, index), dtype=np.float32),
                "action": np.asarray(sample.action.to_vector(), dtype=np.float32),
                "task": sample.context.instruction,
            }
            joint_state = _joint_state_vector(episode, index)
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
    return ExportReport(destination, tuple(source for source, _ in accepted), tuple(skipped))
