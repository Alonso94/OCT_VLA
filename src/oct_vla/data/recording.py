"""Cadence-sampled capture around a NativePort, and oracle-derived labelling.

`EpisodeRecorder` wraps a `NativePort` and implements the same protocol by
delegation, so an oracle written against `NativePort` can be handed the
recorder instead of the raw port with no other change: it ticks the
simulation exactly as before, and capture happens as a side effect of that
same `tick()` call.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from oct_vla.core.action import action_between
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RobotObservation
from oct_vla.robots.robotwin.backend import Contact, NativePort, Reading, Trajectory
from oct_vla.tasks.shelf_restock.oracle.expert import TransferRecord

from .episode import Episode, Sample

Observe = Callable[[], RobotObservation]
ObserveScene = Callable[[], ObjectScene]


@dataclass(frozen=True)
class CapturedFrame:
    """One captured instant: the tick it was captured at, and its readouts."""

    tick: int
    timestamp: float
    observation: RobotObservation
    scene: ObjectScene


class EpisodeRecorder:
    """Delegates the full `NativePort` protocol, capturing at a target cadence.

    Simulation time is `tick_count * port.dt` -- exact, because it is derived
    from the same clock the simulator itself advances on every `tick()` --
    rather than assumed from the requested `hz`. The data contract needs real
    timestamps (see `validate_episode`'s interval check) precisely because an
    FPS label can silently hide irregular capture.
    """

    def __init__(
        self,
        port: NativePort,
        observe: Observe,
        observe_scene: ObserveScene,
        *,
        hz: float = 15.0,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.port = port
        self._observe = observe
        self._observe_scene = observe_scene
        self._hz = hz
        self._tick_count = 0
        # Starting the next capture instant at 0.0 is what makes the very
        # first tick() call capture a frame: after that call, tick_count == 1
        # and 1 * port.dt >= 0.0 is always true. No separate "first tick"
        # branch is needed.
        self._next_capture = 0.0
        self._frames: list[CapturedFrame] = []

    # -- NativePort protocol, by delegation -------------------------------

    @property
    def dt(self) -> float:
        return self.port.dt

    @property
    def ignored_object(self) -> str | None:
        return self.port.ignored_object

    @ignored_object.setter
    def ignored_object(self, value: str | None) -> None:
        self.port.ignored_object = value

    def reset(self, seed: int) -> None:
        # Pure passthrough, like every other delegated method: the recorder
        # does not reset its own tick/capture bookkeeping here, so that it
        # behaves exactly like the port it wraps from the oracle's point of
        # view. Call reset_capture() explicitly when starting a new episode.
        return self.port.reset(seed)

    def read(self) -> Reading:
        return self.port.read()

    def plan(
        self,
        side: str,
        pose_wxyz: tuple[float, ...],
        constraint: tuple[float, ...] | None = None,
    ) -> Trajectory:
        return self.port.plan(side, pose_wxyz, constraint)

    def command(
        self, side: str, q: tuple[float, ...], qdot: tuple[float, ...], gripper: float
    ) -> None:
        return self.port.command(side, q, qdot, gripper)

    def tick(self) -> None:
        self.port.tick()
        self._tick_count += 1
        sim_time = self._tick_count * self.port.dt
        if sim_time >= self._next_capture:
            self._frames.append(
                CapturedFrame(self._tick_count, sim_time, self._observe(), self._observe_scene())
            )
            self._next_capture += 1.0 / self._hz

    def hold(self) -> None:
        return self.port.hold()

    def close(self) -> None:
        return self.port.close()

    def contacts(self) -> tuple[Contact, ...]:
        return self.port.contacts()

    # -- capture buffer -----------------------------------------------------

    @property
    def simulation_time(self) -> float:
        """Seconds of simulated time elapsed, from ticks rather than a clock.

        Exposed so the caller's `observe` callable can stamp its
        `RobotObservation` with the same instant the frame is stamped with;
        otherwise the observation carries a placeholder while the sample
        carries the truth, and the two silently disagree.
        """
        return self._tick_count * self.port.dt

    @property
    def frames(self) -> tuple[CapturedFrame, ...]:
        return tuple(self._frames)

    def reset_capture(self) -> None:
        """Clear the capture buffer and restart cadence bookkeeping.

        Separate from `reset()` (which is pure port delegation) so a caller
        starting a new episode has one explicit place to restart both the
        frame buffer and the tick/capture-instant counters together.
        """
        self._frames = []
        self._tick_count = 0
        self._next_capture = 0.0


@dataclass(frozen=True)
class LabelSpan:
    """A half-open tick range commanded as one oracle motion."""

    start_tick: int  # inclusive
    end_tick: int  # exclusive
    context: TaskContext
    phase: str


def label_spans(records: Iterable[TransferRecord]) -> tuple[LabelSpan, ...]:
    """Turn the oracle's own execution record into exact tick-range labels.

    The oracle already reports exactly how many ticks each named motion
    commanded (`ExecutedMotion.ticks`), and every one of those ticks is one
    `port.tick()` call -- the same calls the recorder counts. Walking the
    records in order and accumulating tick ranges therefore reproduces the
    exact tick each phase started and ended on, with no coupling between the
    recorder (which only knows about ticks) and the oracle (which only knows
    about motions): neither module needs to know how the other counts.
    """
    spans: list[LabelSpan] = []
    tick = 0
    for record in records:
        for phase, motion in record.motions:
            end = tick + motion.ticks
            spans.append(LabelSpan(tick, end, record.context, phase))
            tick = end
    return tuple(spans)


def label_at(spans: Sequence[LabelSpan], tick: int) -> LabelSpan | None:
    for span in spans:
        if span.start_tick <= tick < span.end_tick:
            return span
    return None


def build_episode(
    frames: Sequence[CapturedFrame],
    spans: Sequence[LabelSpan],
    *,
    seed: int,
    instruction: str,
    success: bool,
    metadata: Mapping[str, str],
) -> Episode:
    """Pair each labelled frame with the action that carries it to the next.

    Deliberately indexes `frames` directly (not "the next labelled frame"):
    the action for frame i is always the true step to frame i+1, whether or
    not frame i+1 itself ends up labelled.
    """
    samples: list[Sample] = []
    for index in range(len(frames) - 1):  # the final frame has no successor
        frame = frames[index]
        span = label_at(spans, frame.tick)
        if span is None:
            # This tick falls outside any commanded motion (e.g. captured
            # during a reset/settle window before the first phase began), so
            # there is no context/phase to attach it to -- drop it.
            continue
        action = action_between(frame.observation.eef, frames[index + 1].observation.eef)
        samples.append(
            Sample(
                timestamp=frame.timestamp,
                observation=frame.observation,
                scene=frame.scene,
                context=span.context,
                action=action,
                phase=span.phase,
            )
        )
    if len(samples) < 2:
        raise ValueError(f"build_episode needs at least two labelled frames, got {len(samples)}")
    return Episode(
        seed=seed,
        instruction=instruction,
        samples=tuple(samples),
        success=success,
        metadata=metadata,
    )
