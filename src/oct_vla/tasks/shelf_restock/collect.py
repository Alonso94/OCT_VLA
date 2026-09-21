"""Record oracle demonstrations of this task as canonical episodes.

The glue between three things that deliberately do not know about each other:
the oracle (`oracle/expert.py`, written against `NativePort`), the recorder
(`data/recording.py`, which samples any port at a fixed cadence), and the
RoboTwin scene (which owns the actors). Keeping it here rather than inside any
of them is what lets the oracle stay unit-testable against fakes and the
recorder stay simulator-agnostic.

Simulation-only: importing this pulls in the RoboTwin port.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import count as count_from
from pathlib import Path
from typing import Any

from oct_vla.core.frames import WORKCELL_FRAME, Transform
from oct_vla.core.objects import ObjectScene
from oct_vla.core.observation import RobotObservation
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState
from oct_vla.data.episode import Episode, validate_episode
from oct_vla.data.recording import (
    CapturedFrame,
    EpisodeRecorder,
    build_episode,
    label_spans,
)
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator
from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor
from oct_vla.robots.robotwin.backend import decode_pose
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager
from oct_vla.tasks.shelf_restock.oracle.expert import ShelfRestockExpert, TransferRecord
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec
from oct_vla.tasks.shelf_restock.success import check_placement

#: The scene is authored directly in world coordinates, so workcell == world
#: for this task (docs/architecture.md). Named rather than inlined so the day
#: that stops being true has one place to change.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

#: Target policy/data cadence from the plan. The backend runs its own faster
#: inner loop; this is the rate the policy will see.
DEFAULT_HZ = 15.0


class CollectionError(RuntimeError):
    """The episode could not be recorded at all."""


#: Collection profile names, keyed by how many objects the scene spawns.
#: Every profile shares one geometry and differs only in object count (see
#: spec.DEFAULT_SPEC), so the count identifies the profile exactly rather than
#: standing in for it.
PROFILE_NAMES = {2: "two_object", 3: "three_object", 4: "four_object"}


def profile_name(object_count: int) -> str:
    """The collection profile a scene of `object_count` objects belongs to."""
    try:
        return PROFILE_NAMES[object_count]
    except KeyError:
        raise CollectionError(
            f"no collection profile spawns {object_count} objects; "
            f"known profiles are {sorted(PROFILE_NAMES)}"
        ) from None


def atomic_clips(
    frames: Sequence[CapturedFrame], records: Sequence[TransferRecord]
) -> tuple[tuple[TransferRecord, tuple[CapturedFrame, ...]], ...]:
    """Cut one continuous run into the atomic transfers it contains.

    The demonstration unit the study needs is a single transfer, but running a
    whole shelf and slicing it afterwards is closer to how the data would be
    gathered on hardware -- nobody resets the scene between restocks -- and it
    guarantees the mix the study wants for free: the first clip has no previous
    neighbour, every later one does, without staging an artificial scene.

    A clip keeps only frames inside its own transfer, so each is
    self-contained: its last frame is dropped for lack of a successor rather
    than borrowing the next transfer's first state.
    """
    clips = []
    start = 0
    for record in records:
        end = start + sum(motion.ticks for _, motion in record.motions)
        clips.append((record, tuple(frame for frame in frames if start <= frame.tick < end)))
        start = end
    return tuple(clips)


def collect_episode(
    port: Any,
    *,
    seed: int,
    spec: ShelfRestockSpec = DEFAULT_SPEC,
    hz: float = DEFAULT_HZ,
    metadata: Mapping[str, str] | None = None,
    episode_kind: str = "atomic",
) -> tuple[Episode, ...]:
    """Attempt one continuous shelf-emptying run for `seed`, once, cut into
    atomic clips.

    All-or-nothing: raises `CollectionError` the moment any transfer fails to
    place its target, or any resulting clip fails validation, discarding the
    whole run rather than returning a partial or unsuccessful result. A seed
    that raises here is meant to be discarded and never retried, since the
    reason it failed does not depend on how many times it is asked again --
    see `collect_dataset` and `collect_n_atomic_demonstrations`, which sample
    a fresh seed instead.
    """
    if episode_kind not in ("atomic", "full_run"):
        raise ValueError("episode_kind must be 'atomic' or 'full_run'")
    port.reset(seed)
    task = port.task

    tracked = {
        track_id: TrackedActor(
            actor=entry.actor,
            size_xyz=entry.size_xyz,
            upright_rotation=entry.upright_rotation,
            center_offset=entry.center_offset,
            category=getattr(entry, "category", None),
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
            # The oracle plans and executes in joint space; the Cartesian action
            # is derived from these. Recording them keeps the original control
            # signal, rather than only its round-trip through IK.
            joints=JointState(
                ArmJoints(reading.left.joints, reading.left.gripper),
                ArmJoints(reading.right.joints, reading.right.gripper),
            ),
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
    manager = ShelfRestockManager(spec)

    # Driven by hand rather than `expert.run`, so each transfer's placement can
    # be checked the instant it happens. The oracle has no grasp verification:
    # a transfer that never actually picks up its target completes every
    # motion without raising, leaves the object on the lower shelf, and the
    # manager reselects that same target forever -- observed live as sixteen
    # identical attempts before an incidental collision finally knocked the
    # object off-shelf and let the run move on. Checking here turns that into
    # an immediate, cheap "discard this whole run" instead of a multi-minute
    # stall, and is what makes trying each seed exactly once viable.
    records: list[TransferRecord] = []
    while True:
        context = manager.next_context(observe_scene())
        if context is None:
            break
        record = expert.transfer(context)
        manager.record_placement(context.target_track_id)
        records.append(record)
        result = check_placement(observe_scene(), context, spec)
        if not result.success:
            raise CollectionError(
                f"seed {seed}: transfer {len(records)} ({context.target_track_id}) "
                "did not place successfully -- discarding the whole run rather than "
                "retrying the same target"
            )
    if not records:
        raise CollectionError(f"seed {seed}: no lower-shelf target to demonstrate")

    # Computed once, over every record, because tick numbers in `recorder.frames`
    # are absolute across the whole run. Recomputing spans per clip from just
    # that clip's own record renumbers its ticks from 0, which does not match
    # the clip's actual (later) frames for any transfer past the first --
    # label_at then finds no span for any of them, and build_episode sees zero
    # labelled frames.
    spans = label_spans(records)

    if episode_kind == "full_run":
        # Build from the original recorder stream.  Do not reconstruct this by
        # concatenating atomic clips: atomic construction intentionally drops
        # each terminal frame, which would silently remove transfer-boundary
        # actions from a purportedly continuous demonstration.
        episode = build_episode(
            recorder.frames,
            spans,
            seed=seed,
            instruction=spec.instruction,
            success=True,
            metadata={
                "task": type(task).__name__,
                "task_profile": profile_name(len(tracked)),
                "object_count": str(len(tracked)),
                "episode_kind": "full_run",
                "transfers": str(len(records)),
                "hz": str(hz),
                "dt": str(port.dt),
                **(metadata or {}),
            },
        )
        problems = validate_episode(episode)
        if problems:
            raise CollectionError(f"seed {seed}: full run failed validation: {problems}")
        return (episode,)

    episodes: list[Episode] = []
    for index, (record, frames) in enumerate(atomic_clips(recorder.frames, records)):
        if not frames:
            raise CollectionError(f"seed {seed}: transfer {index} captured no frames")
        # success=True throughout: every transfer above already had its
        # placement verified live, before the run was allowed to continue.
        episode = build_episode(
            frames,
            spans,
            seed=seed,
            instruction=spec.instruction,
            success=True,
            metadata={
                "task": type(task).__name__,
                "task_profile": profile_name(len(tracked)),
                "object_count": str(len(tracked)),
                "episode_kind": "atomic_restock",
                "transfers": "1",
                "transfer_index": str(index),
                "transfers_in_run": str(len(records)),
                "target_track_id": record.context.target_track_id,
                "previous_neighbor_track_id": record.context.previous_neighbor_track_id or "",
                "compaction_required": str(
                    record.context.previous_neighbor_track_id is not None
                ).lower(),
                "compacted": str(record.compacted).lower(),
                "hz": str(hz),
                "dt": str(port.dt),
                **(metadata or {}),
            },
        )
        problems = validate_episode(episode)
        if problems:
            raise CollectionError(f"seed {seed}: clip {index} failed validation: {problems}")
        episodes.append(episode)
    return tuple(episodes)


def _write_clips(
    directory: Path, seed: int, episodes: Sequence[Episode]
) -> tuple[dict[str, Any], ...]:
    from oct_vla.data.store import write_episode

    reports = []
    for index, episode in enumerate(episodes):
        path = write_episode(episode, directory / f"episode_{seed:04d}_{index}")
        report = {
            "seed": seed,
            "clip": index,
            "status": "ok",
            "samples": len(episode.samples),
            "episode_kind": episode.metadata.get("episode_kind", "atomic_restock"),
            "transfers": episode.metadata.get("transfers", "1"),
            "path": str(path),
        }
        if report["episode_kind"] == "atomic_restock":
            report["neighbour"] = episode.metadata.get("previous_neighbor_track_id", "") or "-"
            report["compacted"] = episode.metadata.get("compacted", "false")
        reports.append(report)
    return tuple(reports)


def collect_dataset(
    port: Any,
    directory: Path,
    seeds: tuple[int, ...],
    *,
    spec: ShelfRestockSpec = DEFAULT_SPEC,
    hz: float = DEFAULT_HZ,
    episode_kind: str = "atomic",
) -> tuple[dict[str, Any], ...]:
    """Attempt each of `seeds` exactly once, in order, never retrying one.

    A seed whose run does not succeed end to end is discarded rather than
    retried or partially kept: `collect_episode` raises before returning
    anything for such a run, and repeating an identical seed against the same
    scene would fail for the same reason again. Use this when the exact seeds
    to attempt matter (e.g. reproducing a known batch); use
    `collect_n_atomic_demonstrations` when only the resulting count matters.
    """
    if episode_kind not in ("atomic", "full_run"):
        raise ValueError("episode_kind must be 'atomic' or 'full_run'")
    reports = []
    for seed in seeds:
        try:
            # Preserve the original call shape for atomic collection so existing
            # injectable collectors remain compatible. Full runs are opt-in.
            kwargs = {"seed": seed, "spec": spec, "hz": hz}
            if episode_kind != "atomic":
                kwargs["episode_kind"] = episode_kind
            episodes = collect_episode(port, **kwargs)
        except Exception as error:  # noqa: BLE001 - one bad seed must not end the batch
            reports.append(
                {"seed": seed, "status": "discarded", "detail": f"{type(error).__name__}: {error}"}
            )
            continue
        reports.extend(_write_clips(directory, seed, episodes))
    return tuple(reports)


def collect_n_atomic_demonstrations(
    port: Any,
    directory: Path,
    count: int,
    *,
    spec: ShelfRestockSpec = DEFAULT_SPEC,
    hz: float = DEFAULT_HZ,
    seeds: Iterable[int] | None = None,
    attempt: Callable[[int], tuple[Episode, ...]] | None = None,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    """Sample fresh seeds, one attempt each, until `count` atomic
    demonstrations have been collected.

    Each seed is tried exactly once. `collect_episode` already discards a
    seed's entire run the moment one transfer fails to place successfully, so
    retrying that seed would just fail the same way again for the same
    reason; a new seed is drawn instead. This is both faster -- no time spent
    repeating a doomed attempt -- and better data, since a demonstration from
    a run that visibly struggled is not one worth training on.

    `seeds` defaults to 0, 1, 2, ... so a given `count` reproducibly consumes
    the same seeds every time it is run. A successful run can contribute more
    than one clip (every transfer in a continuous shelf-emptying), so the
    final count may exceed `count` by up to a full run's worth of clips
    rather than trimming a verified-good demonstration to hit an exact number.

    `attempt` is injectable so the seed-consumption and stopping logic can be
    tested without a simulator; production calls leave it as `None`, which
    becomes a real `collect_episode(port, seed=seed, ...)` call.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1; got {count}")
    seed_stream = iter(seeds) if seeds is not None else count_from(0)
    if attempt is None:

        def attempt(seed: int) -> tuple[Episode, ...]:
            return collect_episode(port, seed=seed, spec=spec, hz=hz)

    written: list[Path] = []
    reports: list[dict[str, Any]] = []
    for seed in seed_stream:
        if len(written) >= count:
            break
        try:
            episodes = attempt(seed)
        except Exception as error:  # noqa: BLE001 - a bad seed is discarded, not fatal
            reports.append(
                {"seed": seed, "status": "discarded", "detail": f"{type(error).__name__}: {error}"}
            )
            continue
        clip_reports = _write_clips(directory, seed, episodes)
        reports.extend(clip_reports)
        written.extend(Path(report["path"]) for report in clip_reports)
    return tuple(written), tuple(reports)
