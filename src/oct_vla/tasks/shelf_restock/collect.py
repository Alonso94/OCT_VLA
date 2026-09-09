"""Record oracle demonstrations of this task as canonical episodes.

The glue between three things that deliberately do not know about each other:
the oracle (`oracle/expert.py`, written against `NativePort`), the recorder
(`data/recording.py`, which samples any port at a fixed cadence), and the
RoboTwin scene (which owns the actors). Keeping it here rather than inside any
of them is what lets the oracle stay unit-testable against fakes and the
recorder stay simulator-agnostic.

Simulation-only: importing this pulls in the RoboTwin port.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oct_vla.core.frames import WORKCELL_FRAME, Transform
from oct_vla.core.objects import ObjectScene
from oct_vla.core.observation import RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.episode import Episode, validate_episode
from oct_vla.data.recording import EpisodeRecorder, build_episode, label_spans
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator
from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor
from oct_vla.robots.robotwin.backend import decode_pose
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager
from oct_vla.tasks.shelf_restock.oracle.expert import ShelfRestockExpert
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec

#: The scene is authored directly in world coordinates, so workcell == world
#: for this task (docs/architecture.md). Named rather than inlined so the day
#: that stops being true has one place to change.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

#: Target policy/data cadence from the plan. The backend runs its own faster
#: inner loop; this is the rate the policy will see.
DEFAULT_HZ = 15.0


class CollectionError(RuntimeError):
    """The episode could not be recorded at all."""


def collect_episode(
    port: Any,
    *,
    seed: int,
    spec: ShelfRestockSpec = DEFAULT_SPEC,
    hz: float = DEFAULT_HZ,
    metadata: Mapping[str, str] | None = None,
) -> tuple[Episode, tuple[str, ...]]:
    """Run one oracle episode against `port`, returning it and its problems.

    Validation problems are returned rather than raised: a collection run
    should be able to record what went wrong and carry on to the next seed,
    since one malformed episode is not a reason to abandon a batch.
    """
    port.reset(seed)
    task = port.task

    tracked = {
        track_id: TrackedActor(
            actor=entry.actor,
            size_xyz=entry.size_xyz,
            upright_rotation=entry.upright_rotation,
            center_offset=entry.center_offset,
        )
        for track_id, entry in task.tracked_objects.items()
    }
    estimator = GroundTruthObjectStateEstimator(
        RoboTwinObjectEvidenceSource(tracked), WORLD_TO_WORKCELL
    )

    def observe() -> RobotObservation:
        reading = recorder.read()
        return RobotObservation(
            recorder.simulation_time,
            EEFState(
                ArmState(
                    WORLD_TO_WORKCELL.apply_pose(decode_pose(reading.left.pose_wxyz)),
                    reading.left.gripper,
                ),
                ArmState(
                    WORLD_TO_WORKCELL.apply_pose(decode_pose(reading.right.pose_wxyz)),
                    reading.right.gripper,
                ),
            ),
            *reading.cameras,
        )

    def observe_scene() -> ObjectScene:
        return estimator.estimate(observe())

    recorder = EpisodeRecorder(port, observe, observe_scene, hz=hz)

    # Each arm's rest pose, measured before anything moves: a park target that
    # is reachable by construction, unlike a hand-chosen constant.
    rest = port.read()
    home = {
        "left": WORLD_TO_WORKCELL.apply_pose(decode_pose(rest.left.pose_wxyz)),
        "right": WORLD_TO_WORKCELL.apply_pose(decode_pose(rest.right.pose_wxyz)),
    }
    body_names = {track_id: entry.actor.actor.get_name() for track_id, entry in tracked.items()}

    expert = ShelfRestockExpert(spec, recorder, observe_scene, WORLD_TO_WORKCELL, body_names, home)
    recorder.reset_capture()
    records = expert.run(ShelfRestockManager(spec))

    episode = build_episode(
        recorder.frames,
        label_spans(records),
        seed=seed,
        instruction=spec.instruction,
        # The task's own hook, not the per-transfer placement check: the
        # episode succeeded only if the lower shelf actually ended up empty.
        success=bool(task.check_success()),
        metadata={
            "task": type(task).__name__,
            "transfers": str(len(records)),
            "hz": str(hz),
            "dt": str(port.dt),
            **(metadata or {}),
        },
    )
    return episode, validate_episode(episode)


def collect_dataset(
    port: Any,
    directory: Path,
    seeds: tuple[int, ...],
    *,
    spec: ShelfRestockSpec = DEFAULT_SPEC,
    hz: float = DEFAULT_HZ,
) -> tuple[dict[int, str], ...]:
    """Record one episode per seed into `directory`, one directory each.

    Returns a per-seed report. A failing seed is recorded and skipped rather
    than aborting the batch: with the oracle's planner being stochastic, some
    seeds are expected to fail, and losing the successful ones alongside them
    would be worse than an incomplete dataset.
    """
    from oct_vla.data.store import write_episode

    reports = []
    for seed in seeds:
        try:
            episode, problems = collect_episode(port, seed=seed, spec=spec, hz=hz)
        except Exception as error:  # noqa: BLE001 - one bad seed must not end the batch
            reports.append(
                {"seed": seed, "status": "failed", "detail": f"{type(error).__name__}: {error}"}
            )
            continue
        path = write_episode(episode, directory / f"episode_{seed:04d}")
        reports.append(
            {
                "seed": seed,
                "status": "ok" if episode.success and not problems else "recorded",
                "samples": len(episode.samples),
                "success": episode.success,
                "problems": "; ".join(problems),
                "path": str(path),
            }
        )
    return tuple(reports)
