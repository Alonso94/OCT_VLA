"""Collect one demonstration from a RoboTwin built-in task.

Structurally simpler than `shelf_restock/collect.py` because the oracle, the
success check and the motion ordering all belong to RoboTwin. What remains is
observing: build the canonical scene from the task's actors, capture a frame
every time RoboTwin would have saved one, and label the whole run as one phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oct_vla.core.objects import TaskContext
from oct_vla.core.observation import RobotObservation
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState
from oct_vla.data.episode import Episode, validate_episode
from oct_vla.data.recording import CapturedFrame, build_episode, single_span
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator
from oct_vla.robots.robotwin.backend import decode_pose
from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor
from oct_vla.tasks.robotwin_builtin.adapter import (
    BuiltinTaskSpec,
    recording_task_class,
    tracked_objects,
)
from oct_vla.tasks.shelf_restock.collect import WORLD_TO_WORKCELL

#: Control steps between captures. RoboTwin steps at dt = 1/250 s, so 17 gives
#: ~14.7 Hz -- the closest integer to the 15 Hz our datasets are recorded at.
DEFAULT_SAVE_FREQ = 17


class BuiltinCollectionError(RuntimeError):
    """This seed produced no usable demonstration."""


def collect_episode(
    port: Any,
    spec: BuiltinTaskSpec,
    *,
    seed: int,
    assets_root: Path,
    save_freq: int = DEFAULT_SAVE_FREQ,
    metadata: Mapping[str, str] | None = None,
) -> Episode:
    """Run the task's own oracle once and record it canonically.

    Raises rather than returning a partial demonstration: a seed that fails is
    meant to be discarded and a fresh one sampled, exactly as the shelf-restock
    collector does, because the reason it failed does not depend on how many
    times it is asked again.
    """
    frames: list[CapturedFrame] = []

    def observe() -> RobotObservation:
        reading = port.read()
        return RobotObservation(
            len(frames) * save_freq * port.dt,
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
            joints=JointState(
                ArmJoints(reading.left.joints, reading.left.gripper),
                ArmJoints(reading.right.joints, reading.right.gripper),
            ),
        )

    def capture() -> None:
        observation = observe()
        frames.append(
            CapturedFrame(
                tick=len(frames),
                timestamp=len(frames) * save_freq * port.dt,
                observation=observation,
                scene=estimator.estimate(observation),
            )
        )

    # A factory, not a class: defining the subclass imports from `envs`, which
    # is only importable inside reset's prepared path.
    port.use_task_factory(lambda: recording_task_class(spec.task_name, capture))
    port.reset(seed)
    task = port.task

    tracked = {
        track_id: TrackedActor(
            actor=entry.actor,
            size_xyz=entry.size_xyz,
            upright_rotation=entry.upright_rotation,
            center_offset=entry.center_offset,
            category=entry.category,
        )
        for track_id, entry in tracked_objects(task, spec, assets_root).items()
    }
    estimator = GroundTruthObjectStateEstimator(
        RoboTwinObjectEvidenceSource(tracked), WORLD_TO_WORKCELL
    )

    task.save_freq = save_freq
    task.play_once()

    # `move()` opens with `if self.plan_success is False: return False`, so one
    # planning failure silently skips every later motion and `play_once` still
    # returns. Without this check the episode records a robot that stopped
    # moving as a completed demonstration.
    if not getattr(task, "plan_success", True):
        raise BuiltinCollectionError(f"seed {seed}: RoboTwin planning failed; discarding")
    if not task.check_success():
        raise BuiltinCollectionError(f"seed {seed}: the oracle did not reach success")
    if len(frames) < 2:
        raise BuiltinCollectionError(
            f"seed {seed}: captured {len(frames)} frame(s). `_take_picture` is only "
            "reached from take_dense_action when save_freq is set, so this usually "
            "means the override did not install."
        )

    context = TaskContext(instruction=spec.instruction, target_track_id=spec.target_track_id)
    episode = build_episode(
        frames,
        single_span(context, spec.phase, ticks=len(frames)),
        seed=seed,
        instruction=spec.instruction,
        success=True,
        metadata={
            "episode_kind": "full_run",
            "task": spec.task_name,
            "save_freq": str(save_freq),
            **(metadata or {}),
        },
    )
    validate_episode(episode)
    return episode
