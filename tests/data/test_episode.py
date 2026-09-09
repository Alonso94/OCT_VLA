from math import inf

import pytest

from oct_vla.core.action import Action, ArmAction
from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.episode import Episode, Sample, validate_episode

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _frame(width: int = 2, height: int = 2) -> RGBFrame:
    return RGBFrame(width, height, bytes(width * height * 3))


def _pose(x: float = 0.0) -> Pose:
    return Pose((x, 0.0, 0.0), IDENTITY, WORKCELL_FRAME)


def _eef(x: float = 0.0) -> EEFState:
    return EEFState(ArmState(_pose(x), 0.0), ArmState(_pose(-x), 1.0))


def _observation(timestamp: float, x: float = 0.0, size: int = 2) -> RobotObservation:
    return RobotObservation(timestamp, _eef(x), _frame(size), _frame(size), _frame(size))


def _scene(track_id: str = "box_0") -> ObjectScene:
    return ObjectScene(0.0, (ObjectState(track_id, _pose(), (0.05, 0.05, 0.05), 1.0, 1.0),))


def _context(target: str = "box_0") -> TaskContext:
    return TaskContext("put the box on the shelf", target)


def _zero_action() -> Action:
    return Action(ArmAction((0, 0, 0), (0, 0, 0), 0.0), ArmAction((0, 0, 0), (0, 0, 0), 1.0))


def _sample(timestamp: float, **overrides) -> Sample:
    fields = {
        "timestamp": timestamp,
        "observation": _observation(timestamp),
        "scene": _scene(),
        "context": _context(),
        "action": _zero_action(),
        "phase": "lift",
    }
    fields.update(overrides)
    return Sample(**fields)


def _good_episode(count: int = 4) -> Episode:
    samples = tuple(_sample(index * 0.1) for index in range(count))
    return Episode(
        seed=0,
        instruction="put the box on the shelf",
        samples=samples,
        success=True,
        metadata={"task": "shelf_restock"},
    )


def test_validate_episode_reports_nothing_for_a_good_episode():
    assert validate_episode(_good_episode()) == ()


def test_validate_episode_names_single_sample_episode():
    episode = Episode(
        seed=0,
        instruction="x",
        samples=(_sample(0.0),),
        success=True,
        metadata={},
    )
    problems = validate_episode(episode)
    assert any("at least two samples" in problem for problem in problems)


def test_validate_episode_names_non_monotonic_timestamps():
    samples = (_sample(0.0), _sample(0.1), _sample(0.05), _sample(0.3))
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any("timestamps not strictly increasing" in p and "index 2" in p for p in problems)


def test_validate_episode_names_irregular_interval():
    # Regular 0.1s spacing except one large 0.5s jump at index 2 -> 3.
    samples = (_sample(0.0), _sample(0.1), _sample(0.2), _sample(0.7), _sample(0.8))
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any("irregular sampling interval" in p and "samples 2 and 3" in p for p in problems)


def test_validate_episode_names_non_finite_action():
    # ArmAction's own constructor rejects non-finite components, so a
    # genuinely invalid Action can only arise from state corruption after
    # construction -- simulate that the same way the frozen dataclasses in
    # this codebase mutate themselves internally, via object.__setattr__.
    bad_action = _zero_action()
    object.__setattr__(bad_action.left, "translation", (inf, 0.0, 0.0))
    samples = (_sample(0.0), _sample(0.1, action=bad_action), _sample(0.2))
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any("non-finite action component at sample 1" in p for p in problems)


def test_validate_episode_names_mismatched_frame_size():
    samples = (
        _sample(0.0),
        _sample(0.1, observation=_observation(0.1, size=3)),
        _sample(0.2),
    )
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any("frame size" in p and "sample 1" in p for p in problems)


def test_validate_episode_names_target_missing_from_scene():
    samples = (_sample(0.0), _sample(0.1, context=_context("no_such_object")), _sample(0.2))
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any(
        "no_such_object" in p and "not present in that sample's scene" in p for p in problems
    )


def test_validate_episode_names_empty_phase():
    samples = (_sample(0.0), _sample(0.1, phase="   "), _sample(0.2))
    episode = Episode(seed=0, instruction="x", samples=samples, success=True, metadata={})
    problems = validate_episode(episode)
    assert any("sample 1 has an empty phase" in p for p in problems)


def test_sample_rejects_wrong_types_cheaply():
    with pytest.raises(ValueError):
        Sample(
            timestamp=0.0,
            observation="not an observation",
            scene=_scene(),
            context=_context(),
            action=_zero_action(),
            phase="lift",
        )


def test_episode_wraps_metadata_immutably():
    episode = _good_episode()
    assert episode.metadata == {"task": "shelf_restock"}
    with pytest.raises(TypeError):
        episode.metadata["task"] = "other"
