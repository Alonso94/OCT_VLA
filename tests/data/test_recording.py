from math import ceil

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.recording import (
    CapturedFrame,
    EpisodeRecorder,
    LabelSpan,
    build_episode,
    label_at,
    label_spans,
)
from oct_vla.tasks.shelf_restock.oracle.expert import TransferRecord
from oct_vla.tasks.shelf_restock.oracle.motion import ExecutedMotion

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _pose(x: float = 0.0) -> Pose:
    return Pose((x, 0.0, 0.0), IDENTITY, WORKCELL_FRAME)


def _frame() -> RGBFrame:
    return RGBFrame(2, 2, bytes(12))


class FakePort:
    def __init__(self, dt: float = 1.0 / 250.0) -> None:
        self.dt = dt
        self.ignored_object: str | None = None
        self.reset_calls: list[int] = []
        self.tick_calls = 0
        self.commands: list[tuple] = []
        self.closed = False

    def reset(self, seed: int) -> None:
        self.reset_calls.append(seed)

    def read(self):
        return "reading"

    def plan(self, side, pose_wxyz, constraint=None):
        return "trajectory"

    def command(self, side, q, qdot, gripper) -> None:
        self.commands.append((side, q, qdot, gripper))

    def tick(self) -> None:
        self.tick_calls += 1

    def hold(self) -> None:
        self.held = True

    def close(self) -> None:
        self.closed = True

    def contacts(self):
        return ("some contact",)


def _observation(x: float) -> RobotObservation:
    return RobotObservation(
        0.0,
        EEFState(ArmState(_pose(x), 0.0), ArmState(_pose(-x), 1.0)),
        _frame(),
        _frame(),
        _frame(),
    )


def _scene() -> ObjectScene:
    return ObjectScene(0.0, (ObjectState("box_0", _pose(), (0.05, 0.05, 0.05), 1.0, 1.0),))


def test_recorder_delegates_every_protocol_method():
    port = FakePort()
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene())

    recorder.reset(7)
    recorder.read()
    recorder.plan("left", (0.0,) * 7)
    recorder.command("left", (0.0,) * 7, (0.0,) * 7, 0.5)
    recorder.tick()
    recorder.hold()
    contacts = recorder.contacts()
    recorder.close()

    assert port.reset_calls == [7]
    assert port.commands[0] == ("left", (0.0,) * 7, (0.0,) * 7, 0.5)
    assert port.tick_calls == 1
    assert port.held is True
    assert contacts == ("some contact",)
    assert port.closed is True
    assert recorder.dt == port.dt


def test_ignored_object_round_trips_through_the_recorder():
    port = FakePort()
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene())

    assert recorder.ignored_object is None
    recorder.ignored_object = "box_0"
    assert port.ignored_object == "box_0"
    assert recorder.ignored_object == "box_0"


def test_recorder_captures_on_the_first_tick():
    port = FakePort()
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene(), hz=15.0)

    recorder.tick()

    assert len(recorder.frames) == 1
    assert recorder.frames[0].tick == 1
    assert isinstance(recorder.frames[0], CapturedFrame)


def test_recorder_captures_at_the_requested_cadence_given_a_known_dt():
    dt = 1.0 / 250.0
    hz = 15.0
    port = FakePort(dt=dt)
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene(), hz=hz)

    total_ticks = 400
    for _ in range(total_ticks):
        recorder.tick()

    captured_ticks = [frame.tick for frame in recorder.frames]

    # Independently computed expected instants: the smallest tick count m
    # such that m * dt >= n / hz, for each capture index n.
    expected = []
    n = 0
    while True:
        m = ceil(n / hz / dt - 1e-9)
        m = max(m, 1)
        if m > total_ticks:
            break
        expected.append(m)
        n += 1

    assert captured_ticks == expected


def test_recorder_does_not_drift_over_many_ticks():
    dt = 1.0 / 250.0
    hz = 15.0
    port = FakePort(dt=dt)
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene(), hz=hz)

    total_ticks = 250_000  # 1000 simulated seconds
    for _ in range(total_ticks):
        recorder.tick()

    expected_count = total_ticks * dt * hz
    assert abs(len(recorder.frames) - expected_count) <= 1

    # No drift: the last capture's simulated time should still be within one
    # dt of its exact k/hz instant, however many captures have happened.
    last = recorder.frames[-1]
    k = round(last.timestamp * hz)
    assert abs(last.timestamp - k / hz) <= dt


def test_reset_capture_clears_frames_and_restarts_cadence():
    port = FakePort()
    recorder = EpisodeRecorder(port, lambda: _observation(0.0), lambda: _scene())
    recorder.tick()
    recorder.tick()
    assert len(recorder.frames) >= 1

    recorder.reset_capture()

    assert recorder.frames == ()
    recorder.tick()
    assert recorder.frames[0].tick == 1


def _executed_motion(ticks: int) -> ExecutedMotion:
    return ExecutedMotion("left", ((0.0,) * 7,) * ticks, ((0.0,) * 7,) * ticks, (0.5,) * ticks)


def _context(target: str = "box_0") -> TaskContext:
    return TaskContext("stock the shelf", target)


def test_label_spans_assigns_contiguous_ranges_summing_to_total_ticks():
    record_a = TransferRecord(
        _context("box_0"),
        (("open", _executed_motion(3)), ("pregrasp", _executed_motion(5))),
        compacted=False,
    )
    record_b = TransferRecord(
        _context("box_1"),
        (("place", _executed_motion(4)),),
        compacted=False,
    )

    spans = label_spans((record_a, record_b))

    assert spans == (
        LabelSpan(0, 3, record_a.context, "open"),
        LabelSpan(3, 8, record_a.context, "pregrasp"),
        LabelSpan(8, 12, record_b.context, "place"),
    )
    total_ticks = sum(
        motion.ticks for record in (record_a, record_b) for _, motion in record.motions
    )
    assert spans[-1].end_tick == total_ticks


def test_label_at_returns_span_at_boundaries_and_none_past_the_end():
    context = _context()
    spans = (LabelSpan(0, 3, context, "open"), LabelSpan(3, 8, context, "pregrasp"))

    assert label_at(spans, 0) is spans[0]
    assert label_at(spans, 2) is spans[0]
    assert label_at(spans, 3) is spans[1]
    assert label_at(spans, 7) is spans[1]
    assert label_at(spans, 8) is None
    assert label_at(spans, 100) is None


def _captured(tick: int, x: float) -> CapturedFrame:
    return CapturedFrame(tick, tick * 0.004, _observation(x), _scene())


def test_build_episode_drops_final_frame_and_computes_step_to_next_sample():
    context = _context()
    spans = (LabelSpan(0, 3, context, "lift"),)
    frames = (_captured(0, 0.0), _captured(1, 0.01), _captured(2, 0.02))

    episode = build_episode(
        frames, spans, seed=1, instruction="stock the shelf", success=True, metadata={}
    )

    assert len(episode.samples) == 2
    # Step from frame 0 to frame 1 is +0.01m on the left arm's x translation.
    assert episode.samples[0].action.left.translation[0] == pytest.approx(0.01)
    assert episode.samples[1].action.left.translation[0] == pytest.approx(0.01)


def test_build_episode_skips_frames_outside_any_span():
    context = _context()
    spans = (LabelSpan(5, 8, context, "lift"),)  # frames 0-4 fall outside any span
    frames = tuple(_captured(tick, tick * 0.01) for tick in range(10))

    episode = build_episode(
        frames, spans, seed=1, instruction="stock the shelf", success=True, metadata={}
    )

    assert all(sample.phase == "lift" for sample in episode.samples)
    # Frames at ticks 5, 6, 7 fall inside the span and each has a successor
    # (the final frame, tick 9, is dropped since it has none); tick 8 is
    # outside the half-open span [5, 8) so it is skipped too.
    assert len(episode.samples) == 3


def test_build_episode_raises_when_too_few_frames_are_labelled():
    context = _context()
    spans = (LabelSpan(0, 1, context, "lift"),)
    frames = (_captured(0, 0.0), _captured(1, 0.01), _captured(2, 0.02))

    with pytest.raises(ValueError):
        build_episode(
            frames, spans, seed=1, instruction="stock the shelf", success=True, metadata={}
        )


def test_the_recorder_does_not_import_a_task_oracle():
    """This module calls itself simulator-agnostic and then imported
    `TransferRecord` from the shelf oracle, so the generic recorder could not
    be used -- or imported -- without one specific task's expert. A second task
    is blocked on exactly that."""
    import subprocess
    import sys

    probe = (
        "import oct_vla.data.recording, sys; "
        "print(any(m.startswith('oct_vla.tasks') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "recording.py pulled in a task package"


def test_single_span_labels_a_demonstration_that_reports_no_motions():
    """`label_spans` rebuilds phase boundaries from an oracle that reports how
    many ticks each motion took. A scripted demonstrator that reports nothing
    would have every frame fall outside every span and be dropped, and
    `build_episode` would then raise for having fewer than two labelled frames
    -- an empty dataset rather than a missing feature."""
    from oct_vla.core.objects import TaskContext
    from oct_vla.data.recording import label_at, single_span

    context = TaskContext(instruction="place the container on the plate", target_track_id="obj_0")
    spans = single_span(context, "demonstration", ticks=5)
    assert len(spans) == 1
    for tick in range(5):
        found = label_at(spans, tick)
        assert found is not None, tick
        assert found.context is context and found.phase == "demonstration"
    assert label_at(spans, 5) is None, "the span must not extend past the run"

    with pytest.raises(ValueError, match="at least one tick"):
        single_span(context, "demonstration", ticks=0)
