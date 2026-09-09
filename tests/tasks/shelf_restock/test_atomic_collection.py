"""Cutting one continuous run into atomic demonstrations.

The study's demonstration unit is a single transfer, but the run that produces
them empties a whole shelf -- which is how the data would be gathered on
hardware, and which supplies the no-neighbour and with-neighbour cases without
staging an artificial scene.
"""

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.episode import Episode
from oct_vla.data.recording import CapturedFrame, build_episode, label_spans
from oct_vla.data.store import read_episode
from oct_vla.tasks.shelf_restock.collect import (
    atomic_clips,
    collect_dataset,
    collect_n_atomic_demonstrations,
)
from oct_vla.tasks.shelf_restock.oracle.expert import TransferRecord
from oct_vla.tasks.shelf_restock.oracle.motion import ExecutedMotion


def motion(ticks: int) -> ExecutedMotion:
    rows = ((0.0,) * 7,) * ticks
    return ExecutedMotion("left", rows, rows, (0.0,) * ticks)


def observation(t: float) -> RobotObservation:
    pose = Pose((t, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)
    arm = ArmState(pose, 1.0)
    frame = RGBFrame(1, 1, b"\x00\x00\x00")
    return RobotObservation(t, EEFState(arm, arm), frame, frame, frame)


def record(target: str, neighbour: str | None, *tick_counts: int) -> TransferRecord:
    return TransferRecord(
        TaskContext("restock", target, neighbour),
        tuple((f"phase{i}", motion(n)) for i, n in enumerate(tick_counts)),
        compacted=neighbour is not None,
    )


def frames(*ticks: int) -> tuple[CapturedFrame, ...]:
    return tuple(CapturedFrame(t, t * 0.004, object(), object()) for t in ticks)


def demo_episode(seed: int, target: str, neighbour: str | None, n_ticks: int = 3) -> Episode:
    """A minimal but genuinely valid Episode, for tests of the seed-sampling
    drivers that write real episode directories rather than exercising
    `atomic_clips`'s own tick bookkeeping."""
    rec = record(target, neighbour, n_ticks)
    frames_ = real_frames(target, *range(n_ticks))
    return build_episode(
        frames_,
        label_spans((rec,)),
        seed=seed,
        instruction="restock",
        success=True,
        metadata={
            "target_track_id": target,
            "previous_neighbor_track_id": neighbour or "",
            "compacted": str(neighbour is not None).lower(),
        },
    )


def real_frames(target: str, *ticks: int) -> tuple[CapturedFrame, ...]:
    """Frames with a genuine RobotObservation/ObjectScene, for a pipeline that
    goes all the way through `build_episode` rather than only exercising
    `atomic_clips`'s bookkeeping."""
    scene = ObjectScene(
        0.0,
        (ObjectState(target, Pose((0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0)), (0.05,) * 3, 1.0, 1.0),),
    )
    return tuple(CapturedFrame(t, t * 0.004, observation(float(t)), scene) for t in ticks)


def test_each_clip_holds_only_its_own_transfers_frames():
    captured = frames(0, 5, 10, 15, 20, 25)
    clips = atomic_clips(captured, (record("obj_0", None, 12), record("obj_1", "obj_0", 14)))

    assert [f.tick for f in clips[0][1]] == [0, 5, 10]
    assert [f.tick for f in clips[1][1]] == [15, 20, 25]


def test_a_clip_is_paired_with_the_transfer_it_came_from():
    clips = atomic_clips(frames(0, 10), (record("obj_0", None, 8), record("obj_1", "obj_0", 8)))

    assert clips[0][0].context.target_track_id == "obj_0"
    assert clips[1][0].context.target_track_id == "obj_1"


def test_the_first_transfer_has_no_neighbour_and_the_next_one_does():
    """This is the mix that makes cutting a full run worthwhile: compaction
    appears without having to stage a pre-placed object."""
    clips = atomic_clips(frames(0, 10), (record("obj_0", None, 8), record("obj_1", "obj_0", 8)))

    assert clips[0][0].context.previous_neighbor_track_id is None
    assert clips[0][0].compacted is False
    assert clips[1][0].context.previous_neighbor_track_id == "obj_0"
    assert clips[1][0].compacted is True


def test_clip_boundaries_follow_the_ticks_each_transfer_commanded():
    # 3 + 4 = 7 ticks in the first transfer, so tick 7 belongs to the second.
    clips = atomic_clips(frames(6, 7), (record("obj_0", None, 3, 4), record("obj_1", "obj_0", 5)))

    assert [f.tick for f in clips[0][1]] == [6]
    assert [f.tick for f in clips[1][1]] == [7]


def test_frames_past_the_last_transfer_belong_to_no_clip():
    """Ticks after the run's motions -- anything the recorder caught while
    nothing was being commanded -- must not be attributed to a transfer."""
    clips = atomic_clips(frames(0, 5, 99), (record("obj_0", None, 10),))

    assert [f.tick for f in clips[0][1]] == [0, 5]


def test_a_run_of_one_transfer_yields_one_clip():
    clips = atomic_clips(frames(0, 1, 2), (record("obj_0", None, 10),))

    assert len(clips) == 1
    assert len(clips[0][1]) == 3


def test_build_episode_labels_a_later_clips_frames_correctly():
    """Regression: label_spans must be computed once over every record, not
    recomputed per clip from just that record. `label_spans` numbers ticks
    starting at 0 for whatever records it is given, but a later transfer's
    clip carries the run's real, absolute tick numbers -- recomputing spans
    from that record alone renumbers them from 0 and no frame ever matches,
    so build_episode saw zero labelled frames for every transfer past the
    first. This is exactly the bug that broke live collection."""
    records = (record("obj_0", None, 10), record("obj_1", "obj_0", 10))
    clips = atomic_clips(real_frames("obj_0", 0, 5) + real_frames("obj_1", 10, 15, 19), records)
    spans = label_spans(records)  # computed once, over both records

    second_clip_frames = clips[1][1]
    assert [f.tick for f in second_clip_frames] == [10, 15, 19]

    episode = build_episode(
        second_clip_frames,
        spans,
        seed=0,
        instruction="restock",
        success=True,
        metadata={},
    )
    # The final frame in a clip has no successor and is dropped, so 3 raw
    # frames give 2 samples -- just enough to clear build_episode's minimum.
    assert len(episode.samples) == 2
    assert {s.context.target_track_id for s in episode.samples} == {"obj_1"}
    assert {s.phase for s in episode.samples} == {"phase0"}


def test_each_seed_is_attempted_at_most_once(tmp_path):
    """A failing seed must never be retried: the reason it failed does not
    depend on how many times it is asked again."""
    calls: list[int] = []

    def attempt(seed):
        calls.append(seed)
        if seed in (0, 2):
            raise RuntimeError("simulated planner failure")
        return (demo_episode(seed, "obj_0", None),)

    written, reports = collect_n_atomic_demonstrations(
        port=None, directory=tmp_path, count=2, attempt=attempt
    )

    assert calls == sorted(set(calls)), "no seed was attempted twice"
    assert len(written) == 2
    discarded = [r for r in reports if r["status"] == "discarded"]
    assert {r["seed"] for r in discarded} == {0, 2}


def test_collection_stops_once_enough_demonstrations_are_written(tmp_path):
    calls: list[int] = []

    def attempt(seed):
        calls.append(seed)
        return (demo_episode(seed, "obj_0", None),)

    written, _ = collect_n_atomic_demonstrations(
        port=None, directory=tmp_path, count=3, attempt=attempt
    )

    assert len(written) == 3
    assert len(calls) == 3, "no seed was drawn beyond what was needed"


def test_a_successful_run_can_overshoot_the_requested_count(tmp_path):
    """A run's clips are never trimmed to hit an exact count: a verified-good
    demonstration is not worth discarding for a round number."""

    def attempt(seed):
        return (
            demo_episode(seed, "obj_0", None),
            demo_episode(seed, "obj_1", "obj_0"),
            demo_episode(seed, "obj_2", "obj_1"),
        )

    written, reports = collect_n_atomic_demonstrations(
        port=None, directory=tmp_path, count=2, attempt=attempt
    )

    assert len(written) == 3  # all three clips from the one run that was needed
    assert len({r["seed"] for r in reports}) == 1, "only one seed was drawn"


def test_discarded_seeds_write_no_files(tmp_path):
    def attempt(seed):
        raise RuntimeError("always fails")

    written, reports = collect_n_atomic_demonstrations(
        port=None, directory=tmp_path, count=1, seeds=(5, 6, 7), attempt=attempt
    )

    assert written == ()
    assert list(tmp_path.iterdir()) == []
    assert all(r["status"] == "discarded" for r in reports)


def test_count_below_one_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="count must be at least 1"):
        collect_n_atomic_demonstrations(port=None, directory=tmp_path, count=0)


def test_seeds_default_to_a_reproducible_sequence_starting_at_zero(tmp_path):
    calls: list[int] = []

    def attempt(seed):
        calls.append(seed)
        return (demo_episode(seed, "obj_0", None),)

    collect_n_atomic_demonstrations(port=None, directory=tmp_path, count=3, attempt=attempt)

    assert calls == [0, 1, 2]


def test_written_episodes_round_trip_from_disk(tmp_path):
    """The files collect_n_atomic_demonstrations writes are real, readable
    canonical episodes -- not just paths that happen to exist."""

    def attempt(seed):
        return (demo_episode(seed, "obj_0", None), demo_episode(seed, "obj_1", "obj_0"))

    written, _ = collect_n_atomic_demonstrations(
        port=None, directory=tmp_path, count=2, attempt=attempt
    )

    assert len(written) == 2
    for path in written:
        episode = read_episode(path)
        assert episode.success is True
        assert len(episode.samples) == 2


def test_collect_dataset_attempts_exactly_the_given_seeds_once_each(tmp_path, monkeypatch):
    """`collect_dataset` takes an explicit seed list rather than sampling, but
    must honour the same never-retry contract as `collect_n_atomic_demonstrations`."""
    calls: list[int] = []

    def fake_collect_episode(port, *, seed, spec, hz):
        calls.append(seed)
        if seed == 1:
            raise RuntimeError("simulated failure")
        return (demo_episode(seed, "obj_0", None),)

    monkeypatch.setattr("oct_vla.tasks.shelf_restock.collect.collect_episode", fake_collect_episode)

    reports = collect_dataset(port=None, directory=tmp_path, seeds=(0, 1, 2))

    assert calls == [0, 1, 2]
    statuses = {r["seed"]: r["status"] for r in reports}
    assert statuses[1] == "discarded"
    assert statuses[0] == "ok"
    assert statuses[2] == "ok"
