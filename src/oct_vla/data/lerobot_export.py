"""Export canonical OCT-VLA recordings as a local LeRobot v3 dataset.

The canonical store deliberately uses dependency-free gzip streams of packed
RGB.  This module is an optional downstream adapter: importing OCT-VLA still
does not require NumPy, PyAV, or LeRobot.  Run it from the policy environment
documented in ``docs/setup.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from oct_vla.data.episode import Episode, validate_episode
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


def _features(episode: Episode) -> dict[str, dict]:
    observation = episode.samples[0].observation
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "motors": [
                    "left_eef.x", "left_eef.y", "left_eef.z", "left_eef.qx", "left_eef.qy",
                    "left_eef.qz", "left_eef.qw", "left_eef.gripper", "right_eef.x",
                    "right_eef.y", "right_eef.z", "right_eef.qx", "right_eef.qy", "right_eef.qz",
                    "right_eef.qw", "right_eef.gripper",
                ]
            },
        },
        "action": {"dtype": "float32", "shape": (14,)},
    }
    for attr, feature_name in CAMERA_FEATURES.items():
        frame = getattr(observation, attr)
        features[feature_name] = {
            "dtype": "video",
            "shape": (frame.height, frame.width, 3),
            "names": ["height", "width", "channel"],
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


def export_episodes(
    sources: Iterable[str | Path],
    output: str | Path,
    *,
    repo_id: str = "local/oct_vla_shelf_restock",
    robot_type: str = "robotwin_franka_bimanual",
    include_unsuccessful: bool = False,
) -> ExportReport:
    """Convert canonical episode directories into a new local LeRobot dataset.

    LeRobot owns its uniform ``frame_index / fps`` timestamps.  The source
    recordings remain untouched and retain their simulator-clock timestamps.
    ``output`` must not exist, preventing accidental replacement of an export.
    """
    source_paths = tuple(Path(source) for source in sources)
    if not source_paths:
        raise ValueError("At least one canonical episode directory is required")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"LeRobot output already exists: {destination}")

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
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=destination,
        fps=fps,
        robot_type=robot_type,
        features=_features(accepted[0][1]),
        use_videos=True,
    )
    for _, episode in accepted:
        for index, sample in enumerate(episode.samples):
            frame = {
                "observation.state": np.asarray(_state_vector(episode, index), dtype=np.float32),
                "action": np.asarray(sample.action.to_vector(), dtype=np.float32),
                "task": sample.context.instruction,
            }
            for attr, feature_name in CAMERA_FEATURES.items():
                rgb = getattr(sample.observation, attr)
                frame[feature_name] = np.frombuffer(rgb.data, dtype=np.uint8).reshape(
                    rgb.height, rgb.width, 3
                )
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return ExportReport(destination, tuple(source for source, _ in accepted), tuple(skipped))
