"""Canonical recorded-episode schema: one Sample per capture, phase-labelled.

This is the single schema every recording path (the shelf-restock oracle
today, any future oracle or policy rollout later) is expected to produce and
every consumer (the on-disk store, a future LeRobot export adapter) is
expected to read. Keeping it here -- rather than letting each producer invent
its own lightweight tuple -- is what lets `validate_episode` catch
data-quality bugs once, for every recording path, instead of once per script.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import median
from types import MappingProxyType

from oct_vla.core.action import Action
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RobotObservation

#: Sampling-interval tolerance for validate_episode: an interval further than
#: this fraction from the episode's median interval means capture cadence
#: was not actually regular, which an FPS label alone would hide.
_INTERVAL_TOLERANCE = 0.5


@dataclass(frozen=True)
class Sample:
    """One captured instant, plus the action FROM it TO the next sample."""

    timestamp: float
    observation: RobotObservation
    scene: ObjectScene
    context: TaskContext
    action: Action
    phase: str

    def __post_init__(self) -> None:
        # Only type/shape invariants belong here -- cheap, and true of every
        # Sample regardless of whether its *episode* is well-formed. Anything
        # that depends on neighbouring samples (monotonic timestamps, regular
        # cadence) or is a business rule a collection run must be able to
        # report on rather than crash on (an empty phase, a target missing
        # from the scene) is validate_episode's job instead.
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Sample timestamp must be finite and nonnegative")
        if not isinstance(self.observation, RobotObservation):
            raise ValueError("Sample requires a RobotObservation")
        if not isinstance(self.scene, ObjectScene):
            raise ValueError("Sample requires an ObjectScene")
        if not isinstance(self.context, TaskContext):
            raise ValueError("Sample requires a TaskContext")
        if not isinstance(self.action, Action):
            raise ValueError("Sample requires an Action")
        if not isinstance(self.phase, str):
            raise ValueError("Sample phase must be a string")


@dataclass(frozen=True)
class Episode:
    """One recorded run. May still fail validate_episode's data-contract checks."""

    seed: int
    instruction: str
    samples: tuple[Sample, ...]
    success: bool
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("Episode seed must be an integer")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Episode instruction must be a nonempty string")
        samples = tuple(self.samples)
        if not all(isinstance(sample, Sample) for sample in samples):
            raise ValueError("Episode samples must all be Sample instances")
        object.__setattr__(self, "samples", samples)
        if not isinstance(self.success, bool):
            raise ValueError("Episode success must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Episode metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def validate_episode(episode: Episode) -> tuple[str, ...]:
    """Report every data-contract problem found; never raise.

    A collection run needs to keep going and report on a bad episode, not
    abort on the first problem -- so every check below is independent, and
    all of them run regardless of what earlier ones already found.
    """
    problems: list[str] = []
    samples = episode.samples
    count = len(samples)

    if count < 2:
        problems.append(
            f"episode must contain at least two samples (one sample carries no action); got {count}"
        )

    for index in range(1, count):
        if samples[index].timestamp <= samples[index - 1].timestamp:
            problems.append(
                f"timestamps not strictly increasing at index {index}: "
                f"{samples[index - 1].timestamp} -> {samples[index].timestamp}"
            )

    if count >= 2:
        intervals = [samples[i].timestamp - samples[i - 1].timestamp for i in range(1, count)]
        typical = median(intervals)
        if typical > 0:
            low, high = (1 - _INTERVAL_TOLERANCE) * typical, (1 + _INTERVAL_TOLERANCE) * typical
            for index, interval in enumerate(intervals):
                if not low <= interval <= high:
                    problems.append(
                        f"irregular sampling interval between samples {index} and {index + 1}: "
                        f"{interval} vs median {typical}"
                    )

    for index, sample in enumerate(samples):
        if not all(isfinite(value) for value in sample.action.to_vector()):
            problems.append(f"non-finite action component at sample {index}")

    # RGBFrame itself already guarantees a frame is present, positive-sized
    # and fully packed (core/observation.py), so the only thing left to check
    # here is that every frame across the whole episode shares one size --
    # a per-frame presence/emptiness check would just repeat what the
    # dataclass constructor already enforces.
    reference_size: tuple[int, int] | None = None
    for index, sample in enumerate(samples):
        observation = sample.observation
        for camera_name, frame in (
            ("head", observation.head_rgb),
            ("left_wrist", observation.left_wrist_rgb),
            ("right_wrist", observation.right_wrist_rgb),
        ):
            size = (frame.width, frame.height)
            if reference_size is None:
                reference_size = size
            elif size != reference_size:
                problems.append(
                    f"sample {index} {camera_name} frame size {size} does not match "
                    f"episode frame size {reference_size}"
                )

    for index, sample in enumerate(samples):
        if sample.scene.get(sample.context.target_track_id) is None:
            problems.append(
                f"sample {index} target_track_id {sample.context.target_track_id!r} "
                "not present in that sample's scene"
            )

    for index, sample in enumerate(samples):
        if not sample.phase.strip():
            problems.append(f"sample {index} has an empty phase")

    return tuple(problems)
